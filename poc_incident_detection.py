"""
VJepa2 Dashcam Incident Detection — POC Demo
=============================================
A BADAS-style incident prediction demo powered by VJepa2.

The demo plays dashcam footage frame-by-frame while a live risk timeline
builds up in real time, showing exactly when VJepa2 detects anomalous events.
"""

import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

# ── Paths ───────────────────────────────────────────────────────────────────
SAMPLE_DIR = Path(__file__).parent / "sample_videos"
CACHE_PATH = SAMPLE_DIR / "precomputed_cache.pt"

# ── Model setup ─────────────────────────────────────────────────────────────
HF_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16

print(f"Loading VJepa2 on {DEVICE}...")
model = AutoModelForVideoClassification.from_pretrained(HF_REPO, dtype=DTYPE).to(DEVICE).eval()
processor = AutoVideoProcessor.from_pretrained(HF_REPO)
NUM_FRAMES = model.config.frames_per_clip
print("Model loaded.")

# ── Load precomputed cache ──────────────────────────────────────────────────
precomputed = torch.load(CACHE_PATH, weights_only=False) if CACHE_PATH.exists() else {}
print(f"Loaded cache for {list(precomputed.keys())}")

# ── Sample video config ─────────────────────────────────────────────────────
SAMPLES = {
    "Crash - Sudden Impact": "crash_sudden_impact.mp4",
    "Crash - Intersection Collision": "crash_intersection.mp4",
    "Crash - Rear End": "crash_rear_end.mp4",
    "Normal - Night Drive (no incident)": "normal_night_drive.mp4",
}


# ── Core functions ──────────────────────────────────────────────────────────
def extract_embedding(frames_tensor: torch.Tensor) -> torch.Tensor:
    inputs = processor(frames_tensor, return_tensors="pt").to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        hidden = model.vjepa2(**inputs).last_hidden_state
        pooled = model.pooler(hidden)
    return pooled.squeeze(0).float().cpu()


def compute_risk_scores(embeddings, baseline_count=3):
    baseline = embeddings[:baseline_count].mean(dim=0)
    scores = []
    for emb in embeddings:
        sim = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), baseline.unsqueeze(0))
        scores.append(1.0 - sim.item())
    scores = np.array(scores)
    if scores.max() > scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min()) * 100
    return np.zeros_like(scores)


def get_risk_color(risk):
    if risk < 30:
        return "#22c55e"  # green
    elif risk < 60:
        return "#facc15"  # yellow
    else:
        return "#ef4444"  # red


def get_risk_label(risk):
    if risk < 30:
        return "NORMAL"
    elif risk < 60:
        return "CAUTION"
    else:
        return "⚠️ DANGER"


def build_risk_gauge(risk_value):
    color = get_risk_color(risk_value)
    label = get_risk_label(risk_value)
    return f"""
    <div style="text-align:center; padding:20px;">
        <div style="font-size:64px; font-weight:bold; color:{color}; font-family:monospace;">
            {risk_value:.0f}
        </div>
        <div style="font-size:20px; color:{color}; font-weight:bold; margin-top:4px;">
            {label}
        </div>
        <div style="
            margin:12px auto 0;
            width:200px; height:16px;
            background:linear-gradient(to right, #22c55e 0%, #facc15 40%, #ef4444 70%, #dc2626 100%);
            border-radius:8px;
            position:relative;
        ">
            <div style="
                position:absolute;
                left:{min(risk_value, 100):.0f}%;
                top:-4px;
                width:4px; height:24px;
                background:white;
                border-radius:2px;
                transform:translateX(-2px);
            "></div>
        </div>
        <div style="color:#888; font-size:12px; margin-top:8px;">
            VJepa2 Anomaly Risk Score
        </div>
    </div>
    """


def build_timeline(timestamps, risk_values, current_idx, threshold):
    fig = go.Figure()

    # Show only up to current index (progressive reveal)
    ts_show = timestamps[: current_idx + 1]
    risk_show = risk_values[: current_idx + 1]

    # Color segments by risk level
    colors = [get_risk_color(r) for r in risk_show]

    fig.add_trace(go.Scatter(
        x=ts_show, y=risk_show,
        mode="lines+markers",
        name="Risk",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=4, color=colors),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.1)",
    ))

    # Current position marker
    if current_idx < len(timestamps):
        fig.add_trace(go.Scatter(
            x=[timestamps[current_idx]],
            y=[risk_values[current_idx]],
            mode="markers",
            name="Now",
            marker=dict(size=14, color="white", symbol="diamond",
                        line=dict(width=2, color=get_risk_color(risk_values[current_idx]))),
            showlegend=False,
        ))

    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#facc15", opacity=0.7,
        annotation_text=f"Threshold ({threshold:.0f})",
        annotation_position="top right",
        annotation_font_color="#facc15",
    )

    # Full timeline extent (grayed out future)
    fig.update_layout(
        xaxis=dict(title="Time (s)", range=[0, max(timestamps) * 1.02]),
        yaxis=dict(title="Risk Score", range=[0, 105]),
        template="plotly_dark",
        height=280,
        margin=dict(l=50, r=20, t=30, b=40),
        showlegend=False,
    )
    return fig


# ── Main playback generator ────────────────────────────────────────────────
def run_demo(sample_choice):
    if sample_choice is None or sample_choice not in SAMPLES:
        yield (
            None,
            build_risk_gauge(0),
            go.Figure().update_layout(template="plotly_dark", height=280),
            "Select a sample video above to start the demo.",
        )
        return

    filename = SAMPLES[sample_choice]
    video_path = str(SAMPLE_DIR / filename)

    # Load video
    vr = VideoDecoder(video_path)
    fps = vr.metadata.average_fps or 24
    total_frames = len(vr)

    # Get embeddings (from cache or compute)
    if filename in precomputed:
        cache = precomputed[filename]
        embeddings = cache["embeddings"]
        timestamps = cache["timestamps"]
    else:
        # Compute on the fly
        stride = 8
        embeddings = []
        timestamps = []
        for start in range(0, total_frames - NUM_FRAMES + 1, stride):
            indices = np.arange(start, start + NUM_FRAMES)
            frames = vr.get_frames_at(indices=indices).data
            emb = extract_embedding(frames)
            embeddings.append(emb)
            timestamps.append(start / fps)
        embeddings = torch.stack(embeddings)

    # Compute risk scores
    baseline_count = max(1, sum(1 for t in timestamps if t <= 2.0))
    risk = compute_risk_scores(embeddings, baseline_count)
    threshold = max(15, risk[:baseline_count].mean() + 2.5 * risk[:baseline_count].std())

    # Find incidents for summary
    incidents = []
    in_inc = False
    inc_start = 0
    for i, r in enumerate(risk):
        if r > threshold and not in_inc:
            in_inc = True
            inc_start = i
        elif r <= threshold and in_inc:
            in_inc = False
            incidents.append((timestamps[inc_start], timestamps[i], float(risk[inc_start:i].max())))
    if in_inc:
        incidents.append((timestamps[inc_start], timestamps[-1], float(risk[inc_start:].max())))

    # Playback: step through each analysis window
    for idx in range(len(timestamps)):
        t = timestamps[idx]
        r = risk[idx]

        # Get the video frame at this timestamp
        frame_idx = min(int(t * fps) + NUM_FRAMES // 2, total_frames - 1)
        frame = vr.get_frames_at(indices=np.array([frame_idx])).data[0]
        frame_rgb = frame.permute(1, 2, 0).numpy()

        # Build outputs
        gauge_html = build_risk_gauge(r)
        timeline_fig = build_timeline(timestamps, risk, idx, threshold)

        # Status text
        status = f"**Analyzing:** t = {t:.1f}s | Window {idx+1}/{len(timestamps)}"
        if r > threshold:
            status += f" | 🚨 **INCIDENT DETECTED** (risk: {r:.0f})"

        yield frame_rgb, gauge_html, timeline_fig, status

        # Pace the playback
        time.sleep(0.15)

    # Final summary
    summary = f"**Analysis complete!** {len(timestamps)} windows analyzed.\n\n"
    if incidents:
        summary += f"### ⚠️ {len(incidents)} Incident(s) Found\n"
        for i, (s, e, p) in enumerate(incidents):
            summary += f"- **Incident {i+1}:** {s:.1f}s – {e:.1f}s (peak risk: {p:.0f})\n"
    else:
        summary += "### ✅ No Incidents — Normal Driving"

    final_fig = build_timeline(timestamps, risk, len(timestamps) - 1, threshold)

    yield frame_rgb, gauge_html, final_fig, summary


# ── Gradio UI ───────────────────────────────────────────────────────────────
CSS = """
.risk-gauge { min-height: 180px; }
.main-video img { border-radius: 8px; }
"""

with gr.Blocks(title="VJepa2 Incident Detection") as demo:
    gr.Markdown("""
# Experiments & Insights from Open Source World Models (VJepa2)



""")

    with gr.Row():
        sample_dropdown = gr.Dropdown(
            choices=list(SAMPLES.keys()),
            label="Select a dashcam video",
            scale=3,
        )
        run_btn = gr.Button("▶️  Run Analysis", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=2):
            video_frame = gr.Image(label="Dashcam Feed", height=360, elem_classes=["main-video"])
        with gr.Column(scale=1):
            risk_gauge = gr.HTML(value=build_risk_gauge(0), label="Risk", elem_classes=["risk-gauge"])

    timeline = gr.Plot(label="Risk Timeline")
    status_md = gr.Markdown("Select a video and click **Run Analysis** to start.")

    run_btn.click(
        fn=run_demo,
        inputs=[sample_dropdown],
        outputs=[video_frame, risk_gauge, timeline, status_md],
    )

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(SAMPLE_DIR)],
        theme=gr.themes.Base(primary_hue="red", neutral_hue="slate"),
        css=CSS,
    )
