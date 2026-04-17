"""
Production Training Pipeline for VJepa2 Incident Detection
===========================================================

This script covers the full path from POC to production:

  Step 1: Prepare driving video dataset
  Step 2: Extract features using frozen VJepa2 backbone
  Step 3: Train a custom driving pooler + classifier
  Step 4: Evaluate the model
  Step 5: Export for edge deployment
  Step 6: Run live inference

Requirements:
  - GPU with 16GB+ VRAM (A100/V100/4090) for feature extraction
  - Training data: labeled driving clips (normal vs incident)
  - Recommended datasets:
      DoTA  — https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly (4,677 videos)
      BDD100K — https://www.bdd100k.com/ (100K driving videos)

Usage:
  # Step 1-2: Extract features from your video dataset
  python train_production.py extract --data-dir /path/to/clips --output-dir /path/to/features

  # Step 3-4: Train and evaluate
  python train_production.py train --features-dir /path/to/features --output-dir /path/to/model

  # Step 5: Export to ONNX
  python train_production.py export --model-path /path/to/model/best.pt --output-path model.onnx

  # Step 6: Live inference
  python train_production.py live --model-path /path/to/model/best.pt
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2: FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
# Expects this folder structure:
#
#   data_dir/
#   ├── normal/
#   │   ├── clip_001.mp4
#   │   ├── clip_002.mp4
#   │   └── ...
#   └── incident/
#       ├── clip_001.mp4
#       ├── clip_002.mp4
#       └── ...
#
# Each clip should be 1-5 seconds long.
# The script extracts VJepa2 encoder features (NOT pooled) and saves them.

def extract_features(data_dir: str, output_dir: str):
    from torchcodec.decoders import VideoDecoder
    from transformers import AutoModel, AutoVideoProcessor

    device = get_device()
    print(f"Device: {device}")

    # Load the BASE backbone (not the SSv2 fine-tuned one)
    # This gives us the raw encoder without SSv2 bias
    # If you don't have enough data, use the SSv2 model instead:
    #   backbone_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    backbone_repo = "facebook/vjepa2-vitl-fpc64-256"
    print(f"Loading backbone: {backbone_repo}")
    backbone = AutoModel.from_pretrained(backbone_repo, dtype=torch.float16).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(backbone_repo)
    num_frames = backbone.config.frames_per_clip

    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for label_name, label_id in [("normal", 0), ("incident", 1)]:
        label_dir = Path(data_dir) / label_name
        if not label_dir.exists():
            print(f"WARNING: {label_dir} not found, skipping")
            continue

        videos = sorted(label_dir.glob("*.mp4"))
        print(f"\n{label_name}: {len(videos)} videos")

        for i, video_path in enumerate(videos):
            try:
                vr = VideoDecoder(str(video_path))
                total = len(vr)
                fps = vr.metadata.average_fps or 30

                if total < num_frames:
                    print(f"  SKIP {video_path.name}: too short ({total} frames)")
                    continue

                # Sample frames evenly across the clip
                indices = np.linspace(0, total - 1, num_frames, dtype=int)
                frames = vr.get_frames_at(indices=indices).data

                inputs = processor(frames, return_tensors="pt").to(device=device, dtype=torch.float16)
                with torch.no_grad():
                    encoder_output = backbone(**inputs).last_hidden_state  # (1, num_tokens, 1024)

                # Save as float32 for training stability
                feature_path = Path(output_dir) / f"{label_name}_{video_path.stem}.pt"
                torch.save(encoder_output.squeeze(0).float().cpu(), feature_path)

                manifest.append({
                    "feature_path": str(feature_path),
                    "label": label_id,
                    "label_name": label_name,
                    "source_video": str(video_path),
                    "num_frames": total,
                    "fps": fps,
                })

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(videos)}")

            except Exception as e:
                print(f"  ERROR {video_path.name}: {e}")

    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(manifest)} clips processed.")
    print(f"Features saved to: {output_dir}")
    print(f"Manifest: {manifest_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: CUSTOM DRIVING POOLER + CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class DrivingPoolerClassifier(nn.Module):
    """
    Custom attention pooler trained specifically for driving incident detection.

    Takes VJepa2 encoder tokens as input and outputs incident classification.

    Architecture:
      - Learnable query tokens (like "analysts" asking questions about the scene)
      - Cross-attention: queries attend to encoder tokens to extract relevant info
      - Self-attention: queries refine their understanding
      - Classification head: produces final prediction
    """

    def __init__(self, hidden_size=1024, num_queries=16, num_heads=16, num_classes=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)

        # Cross-attention: queries attend to encoder output
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_size)

        # Self-attention: queries refine among themselves
        self.self_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(hidden_size),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.Dropout(dropout),
                ),
                "norm2": nn.LayerNorm(hidden_size),
            })
            for _ in range(3)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, encoder_tokens):
        """
        Args:
            encoder_tokens: (batch, num_tokens, 1024) from VJepa2 encoder
        Returns:
            logits: (batch, num_classes)
            embedding: (batch, 1024) — the pooled representation (useful for anomaly scoring)
        """
        batch_size = encoder_tokens.size(0)
        queries = self.queries.expand(batch_size, -1, -1)

        # Cross-attend to encoder output
        pooled, _ = self.cross_attn(queries, encoder_tokens, encoder_tokens)
        pooled = self.cross_norm(pooled + queries)

        # Self-attention refinement
        for layer in self.self_attn_layers:
            attn_out, _ = layer["attn"](pooled, pooled, pooled)
            pooled = layer["norm1"](pooled + attn_out)
            ffn_out = layer["ffn"](pooled)
            pooled = layer["norm2"](pooled + ffn_out)

        # Mean pool queries into single vector
        embedding = pooled.mean(dim=1)  # (batch, 1024)

        # Classify
        logits = self.classifier(embedding)

        return logits, embedding

    def get_embedding(self, encoder_tokens):
        """Get just the embedding without classification (for anomaly scoring)."""
        with torch.no_grad():
            _, embedding = self.forward(encoder_tokens)
        return embedding


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class DrivingFeaturesDataset(Dataset):
    def __init__(self, manifest_path: str):
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        features = torch.load(entry["feature_path"], weights_only=True)
        label = entry["label"]
        return features, label


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 & 4: TRAINING AND EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_model(features_dir: str, output_dir: str, epochs: int = 30, batch_size: int = 16, lr: float = 1e-4):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = Path(features_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {features_dir}. Run 'extract' first.")

    dataset = DrivingFeaturesDataset(str(manifest_path))
    print(f"Dataset: {len(dataset)} samples")

    # Count labels
    with open(manifest_path) as f:
        manifest = json.load(f)
    label_counts = {}
    for entry in manifest:
        name = entry["label_name"]
        label_counts[name] = label_counts.get(name, 0) + 1
    print(f"Label distribution: {label_counts}")

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = DrivingPoolerClassifier(
        hidden_size=1024,
        num_queries=16,
        num_classes=2,
        dropout=0.1,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # Handle class imbalance
    total = sum(label_counts.values())
    weights = torch.tensor([total / (2 * label_counts.get("normal", 1)),
                            total / (2 * label_counts.get("incident", 1))]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Train ───────────────────────────────────────────────────────
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits, _ = model(features)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # ── Validate ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                logits, _ = model(features)
                loss = loss_fn(logits, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.1f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(output_dir) / "best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "hidden_size": 1024,
                    "num_queries": 16,
                    "num_classes": 2,
                    "dropout": 0.1,
                },
                "epoch": epoch + 1,
                "val_acc": val_acc,
            }, save_path)
            print(f"  Saved best model (val_acc={val_acc:.1f}%)")

    # ── Final evaluation ────────────────────────────────────────────────
    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")

    # Confusion matrix
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nFinal Metrics (last epoch):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: EXPORT TO ONNX
# ─────────────────────────────────────────────────────────────────────────────

def export_model(model_path: str, output_path: str):
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    config = checkpoint["config"]

    model = DrivingPoolerClassifier(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy input matching VJepa2 encoder output shape
    # ViT-L fpc64: 8192 tokens × 1024d
    # ViT-L fpc16: 2048 tokens × 1024d
    dummy_input = torch.randn(1, 8192, 1024)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["encoder_tokens"],
        output_names=["logits", "embedding"],
        dynamic_axes={
            "encoder_tokens": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"Exported to: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: LIVE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_live(model_path: str):
    """
    Live dashcam inference using the trained driving pooler.

    This replaces the POC's cosine-distance anomaly scoring with a proper
    trained classifier that outputs incident probability directly.
    """
    import cv2
    from transformers import AutoModel, AutoVideoProcessor

    device = get_device()

    # Load VJepa2 backbone (frozen)
    backbone_repo = "facebook/vjepa2-vitl-fpc64-256"
    print(f"Loading backbone: {backbone_repo}")
    backbone = AutoModel.from_pretrained(backbone_repo, dtype=torch.float16).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(backbone_repo)
    num_frames = backbone.config.frames_per_clip

    # Load trained pooler + classifier
    print(f"Loading classifier: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    classifier = DrivingPoolerClassifier(**checkpoint["config"]).to(device).eval()
    classifier.load_state_dict(checkpoint["model_state_dict"])
    print(f"Classifier loaded (val_acc={checkpoint['val_acc']:.1f}%)")

    # Open webcam / dashcam
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    frame_buffer = []
    print("Running live inference. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Buffer frames
        small = cv2.resize(frame, (256, 256))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (C, H, W)
        frame_buffer.append(tensor)

        # Keep only the latest num_frames
        if len(frame_buffer) > num_frames:
            frame_buffer = frame_buffer[-num_frames:]

        # Run inference when we have enough frames
        if len(frame_buffer) == num_frames:
            video = torch.stack(frame_buffer)  # (T, C, H, W)
            inputs = processor(video, return_tensors="pt").to(device=device, dtype=torch.float16)

            with torch.no_grad():
                encoder_output = backbone(**inputs).last_hidden_state.float()
                logits, embedding = classifier(encoder_output)

            probs = torch.softmax(logits, dim=-1)
            incident_prob = probs[0, 1].item() * 100

            # Display
            color = (0, 255, 0) if incident_prob < 30 else (0, 255, 255) if incident_prob < 60 else (0, 0, 255)
            label = "NORMAL" if incident_prob < 30 else "CAUTION" if incident_prob < 60 else "DANGER"

            display = frame.copy()
            cv2.putText(display, f"Risk: {incident_prob:.0f}% - {label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.imshow("VJepa2 Live Incident Detection", display)

        else:
            cv2.putText(frame, f"Buffering... {len(frame_buffer)}/{num_frames}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("VJepa2 Live Incident Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VJepa2 Production Training Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Extract
    extract_parser = subparsers.add_parser("extract", help="Extract VJepa2 features from video dataset")
    extract_parser.add_argument("--data-dir", required=True, help="Path to video dataset (with normal/ and incident/ subdirs)")
    extract_parser.add_argument("--output-dir", required=True, help="Path to save extracted features")

    # Train
    train_parser = subparsers.add_parser("train", help="Train driving pooler + classifier")
    train_parser.add_argument("--features-dir", required=True, help="Path to extracted features")
    train_parser.add_argument("--output-dir", required=True, help="Path to save trained model")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    # Export
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("--model-path", required=True, help="Path to best.pt")
    export_parser.add_argument("--output-path", required=True, help="Output ONNX path")

    # Live
    live_parser = subparsers.add_parser("live", help="Run live inference with trained model")
    live_parser.add_argument("--model-path", required=True, help="Path to best.pt")

    args = parser.parse_args()

    if args.command == "extract":
        extract_features(args.data_dir, args.output_dir)
    elif args.command == "train":
        train_model(args.features_dir, args.output_dir, args.epochs, args.batch_size, args.lr)
    elif args.command == "export":
        export_model(args.model_path, args.output_path)
    elif args.command == "live":
        run_live(args.model_path)
    else:
        parser.print_help()
