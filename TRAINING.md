# Production Training Guide


## Overview

Possible Pipeline 

```
Your dashcam videos          Frozen VJepa2 backbone         Your trained classifier
  (normal + incident)              (326M params)              (~20M params)
        │                              │                            │
        ▼                              ▼                            ▼
   ┌─────────┐                  ┌─────────────┐              ┌───────────┐
   │ Step 1  │    ──────▶       │   Step 2    │   ──────▶    │  Step 3   │
   │ Prepare │   video clips    │  Extract    │  features    │  Train    │
   │ Data    │                  │  Features   │  (.pt files) │  Pooler   │
   └─────────┘                  └─────────────┘              └───────────┘
                                                                   │
                                                                   ▼
                                                             ┌───────────┐
                                                             │  Step 4   │
                                                             │ Evaluate  │
                                                             │ & Export  │
                                                             └───────────┘
                                                                   │
                                                                   ▼
                                                             ┌───────────┐
                                                             │  Step 5   │
                                                             │   Live    │
                                                             │ Inference │
                                                             └───────────┘
```

The key idea: VJepa2's backbone (the expensive part) is **frozen** — you never retrain it. You only train a small pooler + classifier head (~20M parameters) that learns what matters for driving incidents. This makes training fast and cheap.

---

## Prerequisites

```bash
conda activate vjepa2
pip install -r requirements.txt
```

Hardware requirements:
- **Feature extraction**: GPU with 16GB+ VRAM (A100, V100, 4090) or Apple Silicon Mac (slower)
- **Training**: Any GPU. Even a laptop GPU works since you're only training 20M parameters
- **Live inference**: GPU recommended for real-time. CPU works but won't be real-time

---

## Step 1: Prepare Your Data

Organize your dashcam clips into two folders:

```
my_driving_data/
├── normal/
│   ├── highway_001.mp4
│   ├── city_002.mp4
│   ├── night_003.mp4
│   └── ... (aim for 2,500+ clips)
└── incident/
    ├── rear_end_001.mp4
    ├── sideswipe_002.mp4
    ├── near_miss_003.mp4
    └── ... (aim for 2,500+ clips)
```

### Data requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Total clips | 1,000 | 10,000+ |
| Clip duration | 1-5 seconds | 2-3 seconds |
| Resolution | Any (resized automatically) | 640×360 or higher |
| Normal:Incident ratio | 1:1 | 1:1 (balanced) |

### Possible Free Dataset

If you don't have your own fleet data yet:

| Dataset | Size | Description | Link |
|---|---|---|---|
| DoTA | 4,677 videos | 13 anomaly types, temporal annotations | https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly |
| BDD100K | 100K videos | Large-scale diverse driving | https://www.bdd100k.com |
| DAD | 2,000 videos | Driver anomaly detection | https://github.com/okankop/Driver-Anomaly-Detection |
| DADA-2000 | 2,000 videos | Accident anticipation | https://github.com/JWFangit/LOTVS-DADA |

### Tips for data preparation

- Each clip should contain **one event** (either normal driving or one incident)
- For incident clips, the incident should happen **within the clip** (not before or after)
- Include diverse conditions: day, night, rain, snow, highway, city, rural
- Include diverse camera angles and dashcam models
- Remove clips with heavy watermarks or text overlays that could confuse the model

---

## Step 2: Extract Features

This runs each video clip through the frozen VJepa2 backbone and saves the encoder output (the raw tokens before pooling). This is the slowest step but only needs to run once.

```bash
python train_production.py extract \
  --data-dir /path/to/my_driving_data \
  --output-dir /path/to/features
```

### What this does

For each video clip:
1. Loads the video and samples 64 frames evenly across the clip
2. Feeds the frames through VJepa2 ViT-L encoder
3. Saves the encoder output (8192 tokens × 1024 dimensions) as a `.pt` file
4. Creates a `manifest.json` mapping each feature file to its label

### Expected output

```
features/
├── normal_highway_001.pt      (each ~32MB)
├── normal_city_002.pt
├── incident_rear_end_001.pt
├── incident_sideswipe_002.pt
├── ...
└── manifest.json              (maps files to labels)
```

### Time estimates

| Hardware | Speed per clip | 5,000 clips | 10,000 clips |
|---|---|---|---|
| A100 (CUDA) | ~0.3s | ~25 min | ~50 min |
| 4090 (CUDA) | ~0.5s | ~40 min | ~80 min |
| Mac M2 (MPS) | ~2.0s | ~2.5 hrs | ~5 hrs |
| CPU | ~5.0s | ~7 hrs | ~14 hrs |

### Storage requirements

Each feature file is ~32MB (8192 × 1024 × 4 bytes). For 5,000 clips, expect ~160GB of disk space. If this is too much, you can modify the script to use the `fpc16` model instead (2048 tokens, ~8MB per file, ~40GB total).

### Choosing the backbone

The script defaults to `facebook/vjepa2-vitl-fpc64-256` (the base backbone, 64 frames). This is the best choice because:
- It was not fine-tuned on any specific task — no domain bias
- 64 frames captures ~2 seconds of context at 30fps

If you have limited compute or storage, edit the script to use `facebook/vjepa2-vitl-fpc16-256-ssv2` instead (16 frames, 4x smaller features). This is the same model our POC uses.

---

## Step 3: Train the Driving Pooler

This trains your custom attention pooler + classifier on the extracted features. Since VJepa2 features are pre-extracted, training is fast — no GPU-heavy video encoding during training.

```bash
python train_production.py train \
  --features-dir /path/to/features \
  --output-dir /path/to/model \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4
```

### What this does

1. Loads pre-extracted VJepa2 features from disk
2. Splits data 80% train / 20% validation
3. Trains a `DrivingPoolerClassifier` (~20M parameters):
   - 16 learnable query tokens that learn to ask "is this an incident?"
   - Cross-attention layer that reads the VJepa2 encoder tokens
   - 3 self-attention layers that refine the understanding
   - Classification head that outputs normal vs incident
4. Handles class imbalance automatically with weighted loss
5. Saves the best model checkpoint based on validation accuracy
6. Prints precision, recall, F1 at the end

### Expected output

```
Dataset: 5000 samples
Label distribution: {'normal': 2500, 'incident': 2500}
Train: 4000, Val: 1000
Trainable parameters: 20,185,602 (20.2M)

Epoch   1/30 | Train Loss: 0.6821 Acc: 58.3% | Val Loss: 0.6234 Acc: 65.2%
Epoch   2/30 | Train Loss: 0.5543 Acc: 72.1% | Val Loss: 0.4891 Acc: 76.8%
...
Epoch  30/30 | Train Loss: 0.0823 Acc: 97.2% | Val Loss: 0.1245 Acc: 94.6%
  Saved best model (val_acc=94.6%)

Final Metrics (last epoch):
  Precision: 0.943
  Recall:    0.951
  F1 Score:  0.947
```

### Training time estimates

| Hardware | 5,000 samples, 30 epochs | 10,000 samples, 30 epochs |
|---|---|---|
| A100 | ~15 min | ~30 min |
| 4090 | ~20 min | ~40 min |
| Mac M2 (MPS) | ~45 min | ~90 min |
| CPU | ~2 hrs | ~4 hrs |

### Tuning tips

- **If accuracy is low (<85%)**: You likely need more data. Aim for 5,000+ clips minimum.
- **If overfitting** (train acc high, val acc low): Increase dropout (`--lr 5e-5`), reduce epochs, or add more data.
- **If underfitting** (both accuracies low): Increase learning rate (`--lr 3e-4`), increase epochs to 50.
- **For multi-class** (not just normal/incident): Edit `DrivingPoolerClassifier` to change `num_classes` and update your data folder structure with more subdirectories.

---

## Step 4: Evaluate and Export

### Check your model

After training, the best model is saved at `/path/to/model/best.pt`. The training script already prints precision, recall, and F1. For a production system, you want:

| Metric | Minimum Target | Good Target |
|---|---|---|
| Precision | >90% | >95% |
| Recall | >85% | >92% |
| F1 | >87% | >93% |

### Export to ONNX for edge deployment

```bash
python train_production.py export \
  --model-path /path/to/model/best.pt \
  --output-path /path/to/model/driving_pooler.onnx
```

This exports only the pooler + classifier (~80MB ONNX file). The VJepa2 backbone would be exported separately using HuggingFace's Optimum library for TensorRT/CoreML.

### What you deploy to edge

```
Edge device gets:
├── vjepa2_backbone.engine     (VJepa2 ViT-L, quantized INT8 via TensorRT)
└── driving_pooler.onnx        (your trained pooler, from this export step)
```

The backbone handles video → tokens. Your pooler handles tokens → incident prediction. Both run sequentially on the edge device.

---

## Step 5: Live Inference

Run real-time incident detection using your trained model with a camera:

```bash
python train_production.py live \
  --model-path /path/to/model/best.pt
```

### What this does

1. Opens the camera (webcam or dashcam)
2. Buffers frames into a sliding window (64 frames)
3. Every time the buffer is full:
   - Feeds frames through frozen VJepa2 backbone → encoder tokens
   - Feeds tokens through your trained pooler → incident probability
4. Displays the camera feed with a risk overlay:
   - Green "NORMAL" when probability < 30%
   - Yellow "CAUTION" when 30-60%
   - Red "DANGER" when > 60%
5. Press `q` to quit

### Difference from the POC

| | POC Demo | Production (this) |
|---|---|---|
| Backbone | SSv2 fine-tuned (biased toward hand actions) | Base backbone (no domain bias) |
| Scoring method | Cosine distance from baseline (hacky) | Trained classifier (proper) |
| Output | Relative risk score (0-100, normalized) | Absolute incident probability (%) |
| Pooler | SSv2 attention pooler (not trained for driving) | Your custom pooler (trained on driving data) |
| Threshold | Dynamic, needs tuning per video | Learned during training, fixed |

---

## Full Example: End to End

```bash
# 1. Activate environment
conda activate vjepa2

# 2. Prepare data (you do this manually)
#    Put normal clips in my_data/normal/
#    Put incident clips in my_data/incident/

# 3. Extract features (run once, ~25 min on A100 for 5K clips)
python train_production.py extract \
  --data-dir ./my_data \
  --output-dir ./features

# 4. Train (fast, ~15 min on A100)
python train_production.py train \
  --features-dir ./features \
  --output-dir ./model \
  --epochs 30 \
  --batch-size 16

# 5. Export for deployment
python train_production.py export \
  --model-path ./model/best.pt \
  --output-path ./model/driving_pooler.onnx

# 6. Test live with your camera
python train_production.py live \
  --model-path ./model/best.pt
```

---

## What's Next After This

Once you have a trained model with >90% accuracy:

1. **Quantize the backbone** — Use TensorRT (NVIDIA) or CoreML (Apple) to convert VJepa2 to INT8. This gives 3-5x speedup for real-time edge inference.

2. **Add temporal prediction** — Instead of detecting incidents as they happen, predict them 2-3 seconds ahead. This requires temporal annotations in your training data ("incident at frame X, but warning signs from frame Y").

3. **Multi-class expansion** — Add more incident types: rear-end, side-swipe, pedestrian, road debris, aggressive driving. Change `num_classes` in the model and add more subdirectories to your data.

4. **Fleet deployment** — Package the model into a Docker container or edge runtime. Stream predictions to a cloud dashboard via AWS Kinesis.

5. **Continuous learning** — As your fleet collects more data, retrain monthly. Use the cloud pipeline to aggregate confirmed incidents and feed them back into training.

See `ADOPTION.md` for the full production architecture and cost estimates.
