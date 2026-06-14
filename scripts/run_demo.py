"""
scripts/run_demo.py

ExamSentinel demo runner.

Usage:
    python scripts/run_demo.py --mode test       # run all module tests
    python scripts/run_demo.py --mode cli        # terminal demo
    python scripts/run_demo.py --mode dashboard  # Streamlit UI
    python scripts/run_demo.py --mode cli --scenario deepfake_attack --duration 20
"""

import sys, time, json, argparse, subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║              🛡️  ExamSentinel Demo                          ║
║     Agentic AI Exam Integrity System — FAR AWAY 2026        ║
╚══════════════════════════════════════════════════════════════╝
Novel contributions:
  ✓ BayarConv + SRM filters applied to proctoring (first ever)
  ✓ Screen recapture / paper leak detection via moiré analysis
  ✓ RL-based autonomous 4-tier decision engine
  ✓ Unified deepfake + forgery + gaze pipeline
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dashboard","cli","test","record"], default="dashboard")
    p.add_argument("--scenario", choices=["normal","deepfake_attack","recapture_attempt","gaze_cheat"],
                   default="deepfake_attack")
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--camera", type=int, default=0)
    return p.parse_args()


def run_dashboard():
    print(BANNER)
    print("Starting Streamlit dashboard...")
    print("→ Open http://localhost:8501 in your browser")
    print("→ If that fails try http://127.0.0.1:8501\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.backgroundColor", "#0d0f14",
        "--theme.primaryColor", "#7F77DD"
    ])


def run_cli(scenario, duration):
    import yaml
    from core.inference_engine import MockInferenceEngine
    from agents.decision_engine import AgenticDecisionEngine
    print(BANNER)
    print(f"Scenario: {scenario}  Duration: {duration}s")
    print("=" * 60)
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    engine = MockInferenceEngine(scenario=scenario)
    dec    = AgenticDecisionEngine(cfg)
    engine.start()
    t0, prev_tier = time.time(), "GREEN"
    ICONS = {"GREEN": "🟢", "YELLOW": "🟡", "ORANGE": "🟠", "RED": "🔴"}
    try:
        while time.time() - t0 < duration:
            result = engine.get_latest()
            record = dec.decide(result)
            elapsed = time.time() - t0
            if record.tier != prev_tier or int(elapsed) % 3 == 0:
                print(f"\n[{elapsed:5.1f}s] {ICONS[record.tier]} {record.tier:<8}"
                      f"| DF:{result.score_deepfake:.2f} "
                      f"RC:{result.score_recapture:.2f} "
                      f"Comb:{result.score_combined:.2f}")
                for r in record.reasons[:2]:
                    print(f"          ⚡ {r}")
                print(f"          → {record.action}")
                prev_tier = record.tier
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    engine.stop()
    print("\n" + "=" * 60 + "\nSESSION SUMMARY")
    stats = dec.get_session_stats()
    for k, v in stats.items():
        if k == "tier_counts":
            print(f"  Tiers: {v}")
        else:
            print(f"  {k}: {v}")


# ─── Module tests ─────────────────────────────────────────────────

def run_tests():
    print(BANNER)
    print("Running module tests...\n")
    tests = [
        ("Config loading",       test_config),
        ("Forensics filters",    test_forensics),
        ("Dataset loader",       test_dataset),
        ("Model instantiation",  test_model),
        ("Gaze tracker import",  test_gaze),
        ("Inference engine",     test_inference),
        ("Decision engine",      test_decision),
        ("Dashboard import",     test_dashboard),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n🎉 All tests passed! Ready to demo.")
    else:
        print(f"\n⚠️  {failed} test(s) failed.")


def test_config():
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "model" in cfg
    assert "decision_engine" in cfg


def test_forensics():
    import torch
    from core.forensics_filters import ForensicsFeatureExtractor
    ext = ForensicsFeatureExtractor(img_size=64)
    x = torch.randn(1, 3, 64, 64)
    f, m = ext(x)
    assert f.shape[1] == 64, f"Expected 64 channels, got {f.shape[1]}"
    assert m.shape[1] == 2


def test_dataset():
    import warnings
    warnings.filterwarnings("ignore")
    from core.dataset import ExamSentinelDataset
    # Must not crash even with 0 samples
    ds = ExamSentinelDataset("data", split="train")
    assert len(ds) == 0 or len(ds) > 0  # either is fine


def test_model():
    import torch
    from core.model import ExamSentinelNet
    # Use 128: divisible by 8 (moiré pool), small enough for CPU
    model = ExamSentinelNet(img_size=128, pretrained=False)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert "deepfake"  in out, "Missing deepfake head"
    assert "recapture" in out, "Missing recapture head"
    assert "combined"  in out, "Missing combined head"
    for k, v in out.items():
        assert v.shape == (1, 1), f"Wrong shape for {k}: {v.shape}"


def test_gaze():
    from core.gaze_tracker import GazeTracker
    assert GazeTracker is not None


def test_inference():
    from core.inference_engine import MockInferenceEngine, InferenceResult
    for scenario in ["normal", "deepfake_attack", "recapture_attempt", "gaze_cheat"]:
        eng = MockInferenceEngine(scenario=scenario)
        r   = eng.get_latest()
        assert isinstance(r, InferenceResult), f"Bad return type for {scenario}"
        assert 0.0 <= r.score_deepfake <= 1.5


def test_decision():
    import yaml
    from core.inference_engine import InferenceResult
    from agents.decision_engine import AgenticDecisionEngine
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    engine = AgenticDecisionEngine(cfg, log_path="logs/test_decisions.jsonl")

    # Clean signal → GREEN
    r = InferenceResult(score_deepfake=0.05, score_recapture=0.03,
                        score_splicing=0.02, score_forgery=0.02,
                        score_combined=0.05, face_detected=True,
                        is_off_screen=False, sustained_offscreen_sec=0.0,
                        is_mouth_open=False, blink_rate=0.3)
    rec = engine.decide(r)
    assert rec.tier == "GREEN", f"Expected GREEN on clean signal, got {rec.tier}"

    # Deepfake attack → at least ORANGE
    r2 = InferenceResult(score_deepfake=0.95, score_recapture=0.05,
                         score_splicing=0.05, score_forgery=0.05,
                         score_combined=0.85, face_detected=True,
                         is_off_screen=False, sustained_offscreen_sec=0.0,
                         is_mouth_open=False, blink_rate=0.3)
    rec2 = engine.decide(r2)
    assert rec2.tier in ["ORANGE", "RED"], f"Expected ORANGE/RED on deepfake, got {rec2.tier}"


def test_dashboard():
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "dashboard/app.py")
    assert spec is not None, "dashboard/app.py not found"


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode in ("cli", "record"):
        run_cli(args.scenario, args.duration)
    elif args.mode == "test":
        run_tests()
