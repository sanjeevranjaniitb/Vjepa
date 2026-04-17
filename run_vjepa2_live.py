import threading
import time
from collections import deque

import cv2
import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

# ── Load model ──────────────────────────────────────────────────────────────
HF_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model on {DEVICE}...")
model = AutoModelForVideoClassification.from_pretrained(
    HF_REPO, torch_dtype=torch.float16
).to(DEVICE).eval()
processor = AutoVideoProcessor.from_pretrained(HF_REPO)
NUM_FRAMES = model.config.frames_per_clip
print("Model loaded.")

# ── Open webcam on main thread ──────────────────────────────────────────────
print("Opening webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")
print("Webcam opened.")

# ── Config ──────────────────────────────────────────────────────────────────
MOTION_AREA_THRESHOLD = 0.05  # 5% of frame must have movement
MOTION_SUSTAIN_FRAMES = 5     # motion must persist for 5 consecutive frames
COOLDOWN_SECS = 2.0           # wait between predictions

# ── Shared state ────────────────────────────────────────────────────────────
frame_buffer = deque(maxlen=NUM_FRAMES)
latest_frame = None
latest_predictions = "Waiting — perform an action to get a prediction."
reference_gray = None
motion_active = False
consecutive_motion = 0
action_in_progress = False
last_inference_time = 0.0
running = True


def compute_motion_ratio(current_gray):
    global reference_gray
    if reference_gray is None:
        reference_gray = current_gray.copy()
        return 0.0
    diff = cv2.absdiff(reference_gray, current_gray)
    blur = cv2.GaussianBlur(diff, (21, 21), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    return thresh.sum() / 255.0 / thresh.size


# ── Webcam capture thread ──────────────────────────────────────────────────
def capture_loop():
    global latest_frame, motion_active, consecutive_motion, action_in_progress
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        latest_frame = frame

        gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        ratio = compute_motion_ratio(gray)

        if ratio > MOTION_AREA_THRESHOLD:
            consecutive_motion += 1
        else:
            if action_in_progress and consecutive_motion > 0:
                # motion just stopped — action ended
                action_in_progress = False
            consecutive_motion = 0

        if consecutive_motion >= MOTION_SUSTAIN_FRAMES:
            motion_active = True
            action_in_progress = True
            small = cv2.resize(frame, (320, 240))
            frame_buffer.append(small)
        else:
            motion_active = False

        time.sleep(0.033)


# ── Inference thread ────────────────────────────────────────────────────────
def inference_loop():
    global latest_predictions, last_inference_time
    while running:
        now = time.time()
        cooldown_ok = (now - last_inference_time) > COOLDOWN_SECS
        has_frames = len(frame_buffer) >= NUM_FRAMES

        if not (action_in_progress and has_frames and cooldown_ok):
            time.sleep(0.1)
            continue

        frames = list(frame_buffer)
        video = torch.stack([
            torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            for f in frames
        ])

        inputs = processor(video, return_tensors="pt").to(device=DEVICE, dtype=torch.float16)
        with torch.no_grad():
            logits = model(**inputs).logits.float()

        probs = torch.softmax(logits, dim=-1)
        top5 = probs.topk(5)
        lines = []
        for i, (idx, prob) in enumerate(zip(top5.indices[0], top5.values[0]), 1):
            label = model.config.id2label[idx.item()]
            lines.append(f"{i}. {label} — {prob:.0%}")
        latest_predictions = "\n".join(lines)
        last_inference_time = time.time()


threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()


# ── Gradio UI ───────────────────────────────────────────────────────────────
def reset_reference():
    global reference_gray, latest_predictions, consecutive_motion, action_in_progress
    reference_gray = None
    frame_buffer.clear()
    consecutive_motion = 0
    action_in_progress = False
    latest_predictions = "Reference reset — waiting for action..."
    return latest_predictions


def get_all():
    frame = latest_frame
    if frame is None:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if action_in_progress:
        status = f"🟢 Action detected ({len(frame_buffer)}/{NUM_FRAMES} frames buffered)"
    else:
        status = "⚪ Idle — perform an action"
    return img, latest_predictions, status


with gr.Blocks(title="VJepa2 Live") as demo:
    gr.Markdown("## VJepa2 Live Action Recognition")
    gr.Markdown(
        "Sit still to set a reference, then perform an action. "
        "The model only predicts when it detects sustained deliberate movement."
    )
    with gr.Row():
        output_img = gr.Image(label="Live Feed", height=500)
        with gr.Column():
            predictions_box = gr.Textbox(label="Top 5 Predictions", lines=8, interactive=False)
            motion_status = gr.Textbox(label="Status", lines=1, interactive=False)
            reset_btn = gr.Button("Reset Reference Frame")

    reset_btn.click(fn=reset_reference, outputs=predictions_box)
    timer = gr.Timer(value=0.2)
    timer.tick(fn=get_all, outputs=[output_img, predictions_box, motion_status])

if __name__ == "__main__":
    try:
        demo.launch()
    finally:
        running = False
        cap.release()
