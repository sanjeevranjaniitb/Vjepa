import torch
import numpy as np

from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModelForVideoClassification

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and video preprocessor
hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"

model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(device)
processor = AutoVideoProcessor.from_pretrained(hf_repo)

# To load a video, sample the number of frames according to the model.
video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, model.config.frames_per_clip, 8)
video = vr.get_frames_at(indices=frame_idx).data  # frames x channels x height x width

# Preprocess and run inference
inputs = processor(video, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

print("Top 5 predicted class names:")
top5_indices = logits.topk(5).indices[0]
top5_probs = torch.softmax(logits, dim=-1).topk(5).values[0]
for idx, prob in zip(top5_indices, top5_probs):
    text_label = model.config.id2label[idx.item()]
    print(f" - {text_label}: {prob:.2f}")
