# ExamSentinel 🛡️

**Agentic AI Exam Integrity System — FAR AWAY Hackathon 2026**

> "The goal is not to write every line of code yourself. The goal is to build something meaningful."

ExamSentinel is a **fully autonomous**, multi-modal exam integrity system that combines image forensics, deepfake detection, recapture detection, and gaze tracking into a single pipeline — with an RL-inspired agentic decision engine that **acts** without human intervention.

---

## The Problem

India runs the world's largest competitive examinations:
- **2.4 million** NEET candidates (2024)
- **1.2 million** JEE aspirants annually
- ₹5,000 crore estimated loss from exam fraud in 2024 alone

### What current systems miss

| Breach | Existing Systems | ExamSentinel |
|--------|-----------------|--------------|
| Real-time deepfake face-swap | ❌ Only checks at login | ✅ Continuous every 5s |
| Recapture / paper leak detection | ❌ Not addressed anywhere | ✅ Moiré + forensics detection |
| Autonomous triage at scale | ❌ Human reviewers required | ✅ RL-based 4-tier decision engine |
| Image forensics (SRM/BayarConv) | ❌ Never applied to proctoring | ✅ Core detection layer |
| Copy-move / AI inpainting on ID | ❌ Simple face match only | ✅ Full forgery detection |

---

## Architecture

```
Webcam Feed
    │
    ▼
┌─────────────────────────────────────────────┐
│           ExamSentinel Core Pipeline        │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ BayarConv│  │  SRM     │  │ MediaPipe│   │
│  │ Filters  │  │ Filters  │  │  Gaze    │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │         │
│       └──────┬──────┘             │         │
│              ▼                    │         │
│  ┌─────────────────────┐          │         │
│  │  Multi-Head Detector│          │         │
│  │  ├ Deepfake Head    │          │         │
│  │  ├ Recapture Head   │◄─────────┘         │
│  │  ├ Splicing Head    │                    │
│  │  └ Forgery Head     │                    │
│  └──────────┬──────────┘                    │
│             │                               │
│             ▼                               │
│  ┌─────────────────────┐                    │
│  │  Agentic Decision   │                    │
│  │  Engine (RL-based)  │                    │
│  │  GREEN → YELLOW →   │                    │
│  │  ORANGE → RED       │                    │
│  └──────────┬──────────┘                    │
└─────────────┼───────────────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 Warn      Flag      Auto-Block
 Student   Session   + Alert
```

---

## Novel Contributions

1. **First application of BayarConv + SRM filters to exam proctoring** — borrowed from image forgery detection research, applied to webcam frame analysis
2. **Recapture detection for question paper leaks** — moiré pattern analysis to detect when a student photographs the screen
3. **RL-inspired autonomous triage** — 4-tier decision engine (GREEN/YELLOW/ORANGE/RED) that acts without human review
4. **Unified pipeline** — deepfake + recapture + forgery + gaze in one inference loop

---

## Stack

- Python 3.10+
- PyTorch 2.x (RTX 3060 6GB optimised)
- OpenCV 4.9
- MediaPipe 0.10
- Streamlit (dashboard)
- NumPy, SciPy, Pillow

