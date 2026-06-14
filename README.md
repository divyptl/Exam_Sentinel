# ExamSentinel 🛡️
**Agentic AI Exam Integrity System — FAR AWAY Hackathon 2026**
> Theme: Examinations × Agentic & Autonomous Systems

## The Problem
India runs the world's largest exams: 2.4M NEET candidates, 1.2M JEE aspirants.
Existing proctoring systems have 4 critical blind spots:

| Breach | Existing Tools | ExamSentinel |
|--------|---------------|--------------|
| Real-time deepfake face-swap | ❌ Login check only | ✅ Every 5s |
| Screen recapture / paper leak | ❌ Not addressed | ✅ Moiré detection |
| Autonomous triage at scale | ❌ Human reviewers | ✅ RL decision engine |
| Image forensics filters | ❌ Never applied | ✅ BayarConv + SRM |

## Novel Contributions
1. **First application of BayarConv + SRM to proctoring** — forensics residuals detect GAN artifacts
2. **Moiré-based recapture detection** — catches students photographing exam screens (paper leaks)
3. **RL autonomous triage** — 4-tier system acts without human-in-the-loop
4. **Unified pipeline** — deepfake + recapture + forgery + gaze in one inference loop

## Architecture
```
Webcam Frame
     │
     ▼
┌─────────────────────────────────┐
│  ForensicsFeatureExtractor      │
│  BayarConv + SRM → 64ch map     │
│  MoireDetector → FFT freq map   │
└──────────────┬──────────────────┘
               │
     ┌─────────▼──────────┐
     │  EfficientNet-B3   │ ← forensics injected at stage 1
     │  + StochasticPurifier│
     └─────────┬──────────┘
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
Deepfake   Recapture   Splicing/Forgery
Head       Head+Moiré  Heads
   └───────────┴───────────┘
               │
    AgenticDecisionEngine
    GREEN→YELLOW→ORANGE→RED
    (RL Q-table, no human needed)
```

## Setup
```bash
pip install -r requirements.txt
python scripts/download_weights.py
python scripts/run_demo.py --mode test
python scripts/run_demo.py --mode dashboard
```

## Demo Scenarios
```bash
python scripts/run_demo.py --mode cli --scenario deepfake_attack    --duration 30
python scripts/run_demo.py --mode cli --scenario recapture_attempt  --duration 30
python scripts/run_demo.py --mode cli --scenario gaze_cheat         --duration 20
```

## Stack
Python 3.10+ · PyTorch 2.x · EfficientNet-B3 · MediaPipe · Streamlit · Plotly

## Training (after downloading datasets)
```bash
python scripts/train.py --config configs/config.yaml
```
See `data/README.md` for dataset download instructions (FaceForensics++, CASIA v2).

---
*Built for FAR AWAY 2026 — India's Biggest International Hackathon*
