# VJepa2 for Production Incident Detection

## Table of Contents

1. [How VJepa2 Works for Incident Detection](#1-how-vjepa2-works-for-incident-detection)
2. [Approaches for Image Generation and Classification](#2-approaches-for-image-generation-and-classification)
3. [High-Level System Design](#3-high-level-system-design)
4. [Fine-Tuning Requirements and Compute Comparison](#4-fine-tuning-requirements-and-compute-comparison)
5. [Building a Production Autonomous Driving System](#5-building-a-production-autonomous-driving-system)

---

## 1. How VJepa2 Works for Incident Detection

### What is VJepa2?

VJepa2 is a video foundation model developed by Meta that learns to understand video by predicting what comes next. Think of it like a person who has watched millions of hours of video and developed an intuition for what "normal" looks like — when something unusual happens, it immediately notices.

### How does it understand video?

VJepa2 works in a fundamentally different way from traditional video classifiers:

- **Traditional approach**: Train a model with labeled examples — "this is a crash", "this is normal driving". The model only recognizes what it was explicitly taught.
- **VJepa2 approach**: The model learns a *world model* through Self Supervised Learning Approach — an internal representation of how the visual world works. It understands motion, physics, object interactions, and scene dynamics without being told what to look for.

### How does this apply to incident detection?

The core idea is simple:

1. **Encode**: Feed a short video clip (16 frames, ~0.5 seconds) into VJepa2. It produces a 1024-dimensional vector (an "embedding") that captures everything happening in that clip — vehicle motion, road context, relative speeds, scene dynamics.

2. **Baseline**: The first few seconds of any dashcam feed establish what "normal driving" looks like for that specific camera, road, and conditions. This becomes the reference embedding.

3. **Compare**: Every subsequent clip is compared to the baseline. Normal driving produces embeddings that are close to the baseline. An incident — sudden braking, a collision, a vehicle swerving — produces an embedding that is far from the baseline.

4. **Alert**: When the distance exceeds a threshold, the system flags it as an anomaly.

### Why is this powerful?

- **Zero-shot capability**: The model was never trained on crash data, yet it detects incidents because its world model understands what "normal" motion looks like. Anything that violates that understanding gets flagged.
- **Generalizes across conditions**: Unlike rule-based systems, it works in rain, night, snow, different road types — because it learned general motion understanding, not specific rules.
- **No labeled crash data needed to start**: Collecting and labeling crash footage is expensive and rare. VJepa2 gives you a working system from day one.

### Our POC validates this

In our proof-of-concept, VJepa2 successfully detected:
- Sudden impacts (risk score jumps from ~5 to 100 within 0.3 seconds)
- Intersection collisions (gradual risk buildup as the dangerous situation develops)
- Rear-end crashes (clear anomaly spike at the moment of impact)
- Normal driving correctly scored as low risk throughout

---

## 2. Approaches for Image Generation and Classification

For a production incident detection system, we need two capabilities:

### A. Classification — "What is happening?"

This is the core task: given a video clip, determine if an incident is occurring (or about to occur).

#### Approach 1: Zero-Shot Anomaly Detection (what our POC does)

- Use VJepa2 as a frozen feature extractor
- Compare embeddings against a "normal" baseline using cosine distance
- **Pros**: No training data needed, works immediately, adapts to any camera/road
- **Cons**: Cannot distinguish *types* of incidents, sensitivity tuning is manual

#### Approach 2: Fine-Tuned Binary Classifier

- Freeze VJepa2 backbone, train a lightweight classification head on top
- Binary output: "normal" vs "incident"
- Train on datasets < Eg. DoTA (4,677 videos, 13 anomaly types) or BDD100K >
- **Pros**: Much higher accuracy (expected 95%+ AP), learned threshold, fast inference
- **Cons**: Requires labeled training data, may not generalize to unseen incident types
- **Best for**: Production deployment with known incident categories

#### Approach 3: Multi-Class Incident Classifier

- Same as above but with multiple output classes
- Categories: rear-end collision, side-swipe, pedestrian incident, near-miss, road debris, aggressive driving, etc.
- **Pros**: Actionable output (not just "something happened" but "what happened")
- **Cons**: Needs more labeled data per category, class imbalance challenges
- **Best for**: Fleet management, insurance, detailed incident reporting

#### Approach 4: Temporal Prediction 

- Train the model to predict incidents *before* they happen
- Input: last N seconds of driving → Output: probability of incident in next 2-3 seconds
- Uses VJepa2's temporal understanding to detect pre-incident patterns (sudden deceleration of car ahead, pedestrian stepping off curb, vehicle running red light)
- **Pros**: Early warning capability (the key differentiator of BADAS)
- **Cons**: Most complex to train, needs precise temporal annotations
- **Best for**: Active safety systems, ADAS integration


### B. Image/Video Generation — "What could happen?"

Generation serves two purposes in this domain:

#### Purpose 1: Synthetic Training Data

The biggest bottleneck in incident detection is data — real crashes are rare. Generation solves this.

| Method | Description | Maturity |
|---|---|---|
| **Diffusion-based video generation** (Sora, Runway Gen-3, Stable Video Diffusion) | Generate synthetic crash scenarios from text prompts like "dashcam view of a car running a red light and T-boning another vehicle" | Medium — quality is improving rapidly but not yet photorealistic for training |
| **Simulation engines** (CARLA, NVIDIA DRIVE Sim, AirSim) | Physics-based driving simulators that can programmatically create crash scenarios | High — widely used in industry, controllable, but domain gap with real footage |
| **VJepa2 as a world model for generation** | VJepa2's predictor network internally generates future frame representations. This can be adapted to generate plausible future scenarios given current driving context | Early research — Meta's V-JEPA architecture was designed with this capability in mind but generation heads are not yet publicly released |
| **GAN-based augmentation** | Use GANs to modify real driving clips — add rain, change lighting, insert vehicles | Mature — effective for data augmentation |

#### Purpose 2: Scenario Visualization

- Given a detected pre-incident pattern, generate a visualization of "what could happen next" to explain the alert to drivers or fleet managers
- This is a future capability that becomes possible as video generation models mature

**Recommended path**: Use CARLA simulation for synthetic training data now. Monitor VJepa2's generation capabilities as Meta releases more of the architecture. Use diffusion models for data augmentation.

---

## 3. High-Level System Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        EDGE DEVICE (In-Vehicle)                 │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ Dashcam  │───▶│ Frame Buffer │───▶│ VJepa2 Encoder        │  │
│  │ 30 fps   │    │ 16-frame     │    │ (Quantized INT8/FP16) │  │
│  └──────────┘    │ sliding      │    │                       │  │
│                  │ window       │    │ ViT-L: 326M params    │  │
│                  └──────────────┘    │ Input: 16×3×256×256   │  │
│                                      │ Output: 1024-d embed  │  │
│                                      └───────────┬───────────┘  │
│                                                  │              │
│                                      ┌───────────▼───────────┐  │
│                                      │ Classification Head   │  │
│                                      │ (Fine-tuned MLP)      │  │
│                                      │                       │  │
│                                      │ Output:               │  │
│                                      │  - Incident prob      │  │
│                                      │  - Incident type      │  │
│                                      │  - Risk score 0-100   │  │
│                                      └───────────┬───────────┘  │
│                                                  │              │
│                              ┌───────────────────▼────────┐     │
│                              │ Decision Engine             │     │
│                              │  - Threshold logic          │     │
│                              │  - Temporal smoothing       │     │
│                              │  - Alert suppression        │     │
│                              └───────────┬────────────────┘     │
│                                          │                      │
│                    ┌─────────────────────┬┴──────────────┐      │
│                    ▼                     ▼               ▼      │
│              ┌──────────┐        ┌────────────┐  ┌───────────┐  │
│              │ Driver   │        │ Upload to  │  │ Local     │  │
│              │ Alert    │        │ Cloud      │  │ Event Log │  │
│              │ (buzzer/ │        │ (clip +    │  │           │  │
│              │  visual) │        │  metadata) │  │           │  │
│              └──────────┘        └─────┬──────┘  └───────────┘  │
│                                        │                        │
└────────────────────────────────────────┼────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CLOUD PLATFORM                           │
│                                                                 │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ Event Ingestion│  │ Analytics Engine  │  │ Model Training │  │
│  │ (Kinesis/Kafka)│─▶│                  │  │ Pipeline       │  │
│  │                │  │ - Fleet-wide     │  │                │  │
│  │ Receives:      │  │   risk scoring   │  │ - Continuous   │  │
│  │ - Video clips  │  │ - Pattern        │  │   fine-tuning  │  │
│  │ - Embeddings   │  │   detection      │  │ - A/B testing  │  │
│  │ - Risk scores  │  │ - Incident       │  │ - Model        │  │
│  │ - GPS/speed    │  │   correlation    │  │   versioning   │  │
│  └────────────────┘  └──────────────────┘  └────────────────┘  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Dashboard                                                  │ │
│  │  - Real-time fleet monitoring                              │ │
│  │  - Incident replay with risk timeline                      │ │
│  │  - Driver behavior scoring                                 │ │
│  │  - Insurance/compliance reporting                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Capture**: Dashcam streams at 30fps. A sliding window buffers the latest 16 frames.
2. **Encode**: Every 0.25s (8-frame stride), the buffer is fed through VJepa2 to produce a 1024-d embedding.
3. **Classify**: The embedding passes through a fine-tuned classification head that outputs incident probability and type.
4. **Decide**: A decision engine applies temporal smoothing (avoid false alarms from single-frame glitches), threshold logic, and alert suppression (don't alert 10 times for the same incident).
5. **Act**: If risk exceeds threshold — alert the driver, save the clip, upload metadata to cloud.
6. **Learn**: Cloud pipeline aggregates incidents across the fleet, retrains the model, pushes updates to edge devices.

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model size | ViT-L (326M) not ViT-G (1B) | ViT-L fits on edge devices (NVIDIA Orin, Qualcomm SA8650). ViT-G is for cloud reprocessing. |
| Frame count | 16 frames (fpc16) not 64 (fpc64) | 16 frames = 0.5s at 30fps. Fast enough for real-time. 64 frames adds latency without proportional accuracy gain for incident detection. |
| Inference precision | INT8 quantized | 2-4x speedup over FP16 on edge hardware. Minimal accuracy loss for classification tasks. |
| Backbone | Frozen VJepa2 + trainable head | Fine-tuning the full backbone is expensive and risks catastrophic forgetting. Frozen backbone + lightweight head is the standard approach. |

---

## 4. Fine-Tuning Requirements and Compute Comparison

### What Fine-Tuning is Needed

#### Phase 1: Classification Head Training (Minimum Viable)

- **What**: Freeze VJepa2 backbone. Train only a 2-layer MLP classification head (~2M parameters).
- **Data needed**: 5,000-10,000 labeled video clips (normal vs incident). Available from:
  - DoTA dataset: 4,677 videos with 13 anomaly types
  - BDD100K: 100K driving videos with annotations
  - Internal fleet data (if available)
- **Compute**: 1x A100 GPU, 2-4 hours
- **Expected accuracy**: 90-95% average precision

#### Phase 2: Backbone Fine-Tuning with LoRA (Recommended)

- **What**: Apply LoRA adapters to VJepa2's attention layers. Trains ~5% of parameters while keeping the rest frozen.
- **Data needed**: 20,000-50,000 labeled clips for robust generalization
- **Compute**: 4x A100 GPUs, 12-24 hours
- **Expected accuracy**: 95-98% average precision

#### Phase 3: Full Fine-Tuning (Maximum Performance)

- **What**: Fine-tune the entire VJepa2 backbone on driving data
- **Data needed**: 100,000+ clips, ideally 500K+ (BADAS was trained on 2M clips)
- **Compute**: 8-32x A100 GPUs, 3-7 days
- **Expected accuracy**: 98-99%+ average precision (BADAS-level)

### Compute Requirements Breakdown

| Phase | Parameters Trained | GPU Hours | Estimated AWS Cost | Accuracy |
|---|---|---|---|---|
| Head only | 2M (0.6%) | 4 hrs × 1 A100 | ~$16 | 90-95% AP |
| LoRA | 16M (5%) | 24 hrs × 4 A100 | ~$380 | 95-98% AP |
| Full fine-tune | 326M (100%) | 168 hrs × 8 A100 | ~$5,400 | 98-99% AP |

*AWS costs based on p4d.24xlarge on-demand pricing (~$4/hr per A100 equivalent)*

### Comparison with Other Approaches

| Approach | Model Size | Training Data | Training Cost | Inference Speed (edge) | Accuracy (AP) | Early Warning |
|---|---|---|---|---|---|---|
| **VJepa2 + head (ours, Phase 1)** | 326M | 5-10K clips | ~$16 | ~30ms (INT8, Orin) | 90-95% | Basic |
| **VJepa2 + LoRA (ours, Phase 2)** | 326M | 20-50K clips | ~$380 | ~30ms (INT8, Orin) | 95-98% | Good |
| **VJepa2 full fine-tune (Phase 3)** | 326M | 100K+ clips | ~$5,400 | ~30ms (INT8, Orin) | 98-99% | Strong |
| **Nexar BADAS 2.0** | ~350M (est.) | 2M clips | Proprietary | ~25ms (custom HW) | 99.4% | 91% recall |
| **NVIDIA COSMOS** | 12B+ | Massive | $100K+ | Not real-time | High | Research |
| **VideoMAE + classifier** | 87M (base) | 10-50K clips | ~$200 | ~15ms (INT8, Orin) | 85-92% | Limited |
| **TimeSformer + classifier** | 121M | 10-50K clips | ~$300 | ~20ms (INT8, Orin) | 87-93% | Limited |
| **3D CNN (SlowFast)** | 34M | 10-50K clips | ~$100 | ~10ms (INT8, Orin) | 82-88% | Limited |
| **Rule-based (optical flow + thresholds)** | N/A | None | $0 | <5ms | 60-70% | None |

### Why VJepa2 Over Alternatives

1. **vs VideoMAE/TimeSformer**: VJepa2 was trained with a Joint Embedding Predictive Architecture — it learns to predict in *embedding space*, not pixel space. This produces richer representations that transfer better to downstream tasks. VideoMAE reconstructs masked pixels, which biases it toward texture over semantics.

2. **vs 3D CNNs (SlowFast, X3D)**: CNNs have limited temporal receptive fields. VJepa2's transformer architecture with 16-64 frame windows captures longer-range temporal dependencies — critical for detecting the *buildup* to an incident, not just the impact.

3. **vs COSMOS**: COSMOS is 91x larger than BADAS and not designed for real-time edge deployment. VJepa2 ViT-L at 326M parameters is practical for edge devices.

4. **vs Rule-based**: Rules break in edge cases (unusual lighting, road types, weather). VJepa2's learned world model generalizes across conditions.

---

## 5. Building a Production Autonomous Driving System

### Possible Roadmap For Production

```
Phase 0 (DONE)          Phase 1                Phase 2               Phase 3
POC Demo                MVP Product            Scale                 ADAS Integration
─────────────          ─────────────          ─────────────         ─────────────
✅ Zero-shot            Binary classifier      Multi-class           Temporal prediction
   anomaly detection    on DoTA/BDD100K        incident types        (early warning)

✅ Pre-recorded         Real-time edge         Fleet-wide            Vehicle CAN bus
   video analysis       inference              deployment            integration

✅ Gradio UI            Mobile/web dashboard   Cloud analytics       OEM partnership
                                               pipeline

Timeline: Done          2-3 months             3-6 months            6-12 months
Cost:     ~$0           ~$50K                  ~$200K                ~$500K+
```



### AWS Architecture for Production

```
Dashcam Device
     │
     ▼
Amazon Kinesis Video Streams ──▶ AWS Lambda (event trigger)
     │                                    │
     ▼                                    ▼
Amazon S3                          Amazon SageMaker
(video clips)                      (model inference / retraining)
     │                                    │
     ▼                                    ▼
Amazon DynamoDB                    Amazon SNS
(event metadata,                   (real-time alerts)
 GPS, risk scores)                        │
     │                                    ▼
     ▼                             Fleet Manager
Amazon QuickSight                  Mobile App
(analytics dashboard)
```

### Key Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| False positives (alerting on non-incidents) | Driver alert fatigue, loss of trust | Temporal smoothing, confidence thresholds, human-in-the-loop review |
| False negatives (missing real incidents) | Safety failure | Conservative thresholds, ensemble with rule-based backup, continuous monitoring of recall metrics |
| Edge device compute constraints | Cannot run model in real-time | INT8 quantization, model distillation to smaller ViT-S, frame skipping |
| Domain shift (model trained on US roads, deployed in India) | Accuracy drops in new regions | Region-specific fine-tuning, diverse training data, continuous learning from fleet data |
| Data privacy (dashcam footage contains faces, plates) | Legal/compliance issues | On-device inference (video never leaves device for normal driving), face/plate blurring for uploaded clips, GDPR/CCPA compliance |
| Model drift over time | Accuracy degrades as driving patterns change | Monthly retraining, monitoring of prediction distributions, automated drift detection |

### Cost Estimate for Production

| Component | Monthly Cost (1,000 vehicles) | Monthly Cost (10,000 vehicles) |
|---|---|---|
| Edge hardware (amortized) | $5,000 | $40,000 |
| AWS Kinesis + S3 | $500 | $3,000 |
| SageMaker (inference) | $1,000 | $5,000 |
| SageMaker (retraining) | $500 | $2,000 |
| DynamoDB + Lambda | $200 | $1,500 |
| QuickSight | $300 | $1,000 |
| **Total** | **~$7,500/mo** | **~$52,500/mo** |

---

## Summary

VJepa2 provides a strong foundation for building a BADAS-like incident detection system. The key advantages are:

1. **World model understanding** — VJepa2 learns general video understanding, not just pattern matching. This gives it zero-shot anomaly detection capability and strong transfer to driving-specific tasks.

2. **Right-sized for edge** — At 326M parameters (ViT-L), it's large enough for high accuracy but small enough for edge deployment with quantization.

3. **Open source** — Unlike BADAS (proprietary), VJepa2 is fully open. You own the model, the weights, and the deployment.

4. **Clear upgrade path** — Start with zero-shot (our POC), add a classification head (Phase 1), fine-tune with LoRA (Phase 2), build temporal prediction (Phase 3). Each step is incremental.

