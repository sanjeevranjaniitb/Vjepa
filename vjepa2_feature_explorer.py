"""
VJepa2 Feature Explorer -- Standalone Demo
==========================================
Demonstrates VJepa2 capabilities:
  Tab 1: Video Captioning -- per-segment scene descriptions
  Tab 2: Clip Retrieval   -- find similar moments across videos
"""

import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

# -- Config -------------------------------------------------------------------
SAMPLE_DIR = Path(__file__).parent / "sample_videos"
CACHE_PATH = SAMPLE_DIR / "precomputed_cache.pt"
HF_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16

print(f"[Feature Explorer] Loading VJepa2 on {DEVICE}...")
model = AutoModelForVideoClassification.from_pretrained(HF_REPO, dtype=DTYPE).to(DEVICE).eval()
processor = AutoVideoProcessor.from_pretrained(HF_REPO)
NUM_FRAMES = model.config.frames_per_clip
LABELS = [model.config.id2label[i] for i in range(len(model.config.id2label))]
print("Model loaded.")

precomputed = torch.load(CACHE_PATH, weights_only=False) if CACHE_PATH.exists() else {}

SAMPLES = {
    "Crash - Sudden Impact": "crash_sudden_impact.mp4",
    "Crash - Intersection Collision": "crash_intersection.mp4",
    "Crash - Rear End": "crash_rear_end.mp4",
    "Normal - Night Drive": "normal_night_drive.mp4",
}
SAMPLE_NAMES = list(SAMPLES.keys())


# -- Shared helpers -----------------------------------------------------------
def extract_embedding(frames_tensor: torch.Tensor) -> torch.Tensor:
    inputs = processor(frames_tensor, return_tensors="pt").to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        hidden = model.vjepa2(**inputs).last_hidden_state
        pooled = model.pooler(hidden)
    return pooled.squeeze(0).float().cpu()


def classify_window(frames_tensor: torch.Tensor, top_k=5):
    inputs = processor(frames_tensor, return_tensors="pt").to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    topk = probs.topk(top_k)
    return [(LABELS[i], p.item()) for i, p in zip(topk.indices, topk.values)]


def get_video_data(filename):
    video_path = str(SAMPLE_DIR / filename)
    vr = VideoDecoder(video_path)
    fps = vr.metadata.average_fps or 24
    total_frames = len(vr)

    if filename in precomputed:
        cache = precomputed[filename]
        return cache["embeddings"], cache["timestamps"], vr, fps, total_frames

    stride = 8
    embeddings, timestamps = [], []
    for start in range(0, total_frames - NUM_FRAMES + 1, stride):
        indices = np.arange(start, start + NUM_FRAMES)
        frames = vr.get_frames_at(indices=indices).data
        embeddings.append(extract_embedding(frames))
        timestamps.append(start / fps)
    return torch.stack(embeddings), timestamps, vr, fps, total_frames


def get_frame_at_time(vr, t, fps, total_frames):
    idx = min(int(t * fps), total_frames - 1)
    frame = vr.get_frames_at(indices=np.array([idx])).data[0]
    return frame.permute(1, 2, 0).numpy()


# -- Tab 1: Video Captioning -------------------------------------------------
def run_captioning(sample_choice):
    if not sample_choice:
        yield None, "*Select a video above.*", "--"
        return

    filename = SAMPLES[sample_choice]
    video_path = str(SAMPLE_DIR / filename)
    vr = VideoDecoder(video_path)
    fps = vr.metadata.average_fps or 24
    total_frames = len(vr)
    stride = 8

    captions = ""
    window_idx = 0
    for start in range(0, total_frames - NUM_FRAMES + 1, stride):
        indices = np.arange(start, start + NUM_FRAMES)
        frames = vr.get_frames_at(indices=indices).data
        t = start / fps

        preds = classify_window(frames, top_k=3)
        mid_frame = get_frame_at_time(vr, t + NUM_FRAMES / (2 * fps), fps, total_frames)

        top_label = preds[0][0]
        desc = " / ".join([f"{label} ({prob:.0%})" for label, prob in preds])
        captions = f"**[{t:.1f}s]** {desc}\n\n" + captions
        window_idx += 1
        status = f"Window {window_idx} | t = {t:.1f}s | **{top_label}**"

        yield mid_frame, captions, status
        time.sleep(0.1)

    status = f"Complete -- {window_idx} windows analyzed across {total_frames/fps:.1f}s"
    yield mid_frame, captions, status


# -- Tab 2: Clip Retrieval ---------------------------------------------------
def run_retrieval(query_video, query_time):
    if not query_video:
        return "Select a query video.", []

    query_file = SAMPLES[query_video]
    embs_q, ts_q, vr_q, fps_q, total_q = get_video_data(query_file)

    query_idx = min(range(len(ts_q)), key=lambda i: abs(ts_q[i] - query_time))
    query_emb = embs_q[query_idx]

    results = []
    for name, filename in SAMPLES.items():
        if filename == query_file:
            continue
        embs, ts, vr, fps, total = get_video_data(filename)
        sims = F.cosine_similarity(query_emb.unsqueeze(0), embs)
        best_idx = sims.argmax().item()
        results.append({
            "name": name, "filename": filename,
            "time": ts[best_idx], "sim": sims[best_idx].item(),
            "vr": vr, "fps": fps, "total": total,
        })

    results.sort(key=lambda x: x["sim"], reverse=True)

    query_frame = get_frame_at_time(vr_q, ts_q[query_idx], fps_q, total_q)
    gallery = [(query_frame, f"QUERY: {query_video} @ {ts_q[query_idx]:.1f}s")]

    report = f"## Retrieval Results\n**Query:** {query_video} at t={ts_q[query_idx]:.1f}s\n\n"
    for r in results:
        frame = get_frame_at_time(r["vr"], r["time"], r["fps"], r["total"])
        gallery.append((frame, f"{r['name']} @ {r['time']:.1f}s ({r['sim']:.2%})"))
        report += f"- **{r['name']}** -- t={r['time']:.1f}s (similarity: {r['sim']:.2%})\n"

    return report, gallery


# -- Gradio App ---------------------------------------------------------------
CSS = """
.gallery-item img { border-radius: 8px; }
#caption-scroll { max-height: 400px; overflow-y: auto; padding: 12px; border: 1px solid #333; border-radius: 8px; }
"""

with gr.Blocks(title="VJepa2 Feature Explorer", css=CSS) as demo:
    gr.Markdown("# VJepa2 Feature Explorer\nExplore VJepa2 embedding capabilities across video understanding tasks.\n")

    with gr.Tab("Video Captioning"):
        gr.Markdown("Per-segment scene descriptions using VJepa2 classification head. Video plays on the left, captions scroll on the right.")
        with gr.Row():
            cap_video = gr.Dropdown(choices=SAMPLE_NAMES, label="Select video", scale=3)
            cap_btn = gr.Button("Generate Captions", variant="primary", scale=1)
        cap_status = gr.Markdown("--")
        with gr.Row(equal_height=True):
            cap_frame = gr.Image(label="Current Frame", height=400, scale=1)
            cap_report = gr.Markdown(
                value="*Captions will appear here...*",
                label="Captions",
                elem_id="caption-scroll",
            )
        cap_btn.click(run_captioning, inputs=[cap_video], outputs=[cap_frame, cap_report, cap_status])

    with gr.Tab("Clip Retrieval"):
        gr.Markdown("Pick a moment in one video. VJepa2 finds the most similar moments in all other videos via cosine similarity.")
        with gr.Row():
            ret_video = gr.Dropdown(choices=SAMPLE_NAMES, label="Query video")
            ret_time = gr.Slider(0, 30, value=5, step=0.5, label="Query time (s)")
        ret_btn = gr.Button("Search", variant="primary")
        ret_report = gr.Markdown()
        ret_gallery = gr.Gallery(label="Results", columns=4, height=250)
        ret_btn.click(run_retrieval, inputs=[ret_video, ret_time], outputs=[ret_report, ret_gallery])

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(SAMPLE_DIR)],
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
