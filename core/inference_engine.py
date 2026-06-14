"""
core/inference_engine.py  — Day 3

Real-time webcam inference pipeline.
Runs every N seconds (configurable) and publishes results
to a shared event queue consumed by the dashboard + decision engine.

Pipeline per cycle:
  1. Capture frame from webcam
  2. Run GazeTracker (MediaPipe, every frame)
  3. Every INFERENCE_INTERVAL seconds:
       a. Send frame to ExamSentinelNet (GPU)
       b. Get deepfake/recapture/splicing/forgery scores
  4. Publish InferenceResult to queue
"""

import time
import queue
import threading
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

from core.gaze_tracker import GazeTracker


@dataclass
class InferenceResult:
    """Single timestamped inference result from one analysis cycle."""
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    # Model scores (0.0 – 1.0 probabilities)
    score_deepfake: float = 0.0
    score_recapture: float = 0.0
    score_splicing: float = 0.0
    score_forgery: float = 0.0
    score_combined: float = 0.0

    # Gaze signals
    gaze_h_deg: float = 0.0
    gaze_v_deg: float = 0.0
    head_yaw_deg: float = 0.0
    is_off_screen: bool = False
    sustained_offscreen_sec: float = 0.0
    blink_rate: float = 0.0
    is_mouth_open: bool = False
    face_detected: bool = True

    # Decision engine output
    alert_tier: str = "GREEN"    # GREEN, YELLOW, ORANGE, RED
    alert_reason: str = ""
    action_taken: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)
        )
        return json.dumps(d)


class InferenceEngine:
    """
    Runs the full ExamSentinel pipeline in a background thread.
    Results are pushed to a thread-safe queue.

    Usage:
        engine = InferenceEngine(model, config)
        engine.start()
        ...
        result = engine.get_latest()
        ...
        engine.stop()
    """

    def __init__(self, model, config: Dict, log_path: str = "logs/events.jsonl"):
        self.model = model
        self.config = config
        self.log_path = log_path

        # Inference interval
        self.inference_interval = config.get("inference", {}).get(
            "interval_seconds", 5.0
        )
        self.device = next(model.parameters()).device

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Result queue (latest result always available)
        self._result_queue: queue.Queue = queue.Queue(maxsize=100)
        self._latest_result: Optional[InferenceResult] = None
        self._lock = threading.Lock()

        # Gaze tracker (runs every frame)
        decision_cfg = config.get("decision_engine", {})
        self.gaze_tracker = GazeTracker(
            off_screen_threshold_deg=decision_cfg.get("sustained_gaze_seconds", 30.0),
            sustained_limit_sec=decision_cfg.get("sustained_gaze_seconds", 3.5)
        )

        # Stats
        self.frame_count = 0
        self.inference_count = 0
        self.last_inference_time = 0.0

        # Log file
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        # Transforms (same as val transforms)
        from core.dataset import get_val_transforms
        self.transform = get_val_transforms(
            config.get("model", {}).get("img_size", 256)
        )

    def start(self, camera_index: int = 0):
        """Start inference in background thread."""
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._thread.start()
        print(f"[InferenceEngine] Started on camera {camera_index}")

    def stop(self):
        """Stop inference thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if hasattr(self, '_cap'):
            self._cap.release()
        self.gaze_tracker.release()
        print("[InferenceEngine] Stopped")

    def get_latest(self) -> Optional[InferenceResult]:
        """Get most recent inference result (non-blocking)."""
        with self._lock:
            return self._latest_result

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame for display."""
        with self._lock:
            return self._latest_frame.copy() if hasattr(self, '_latest_frame') and self._latest_frame is not None else None

    def _run_loop(self):
        """Main inference loop (runs in background thread)."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self.frame_count += 1

            # ── Gaze tracking (every frame, lightweight)
            gaze_result = self.gaze_tracker.process_frame(frame)

            # ── Model inference (every INTERVAL seconds)
            now = time.time()
            model_scores = {}
            if now - self.last_inference_time >= self.inference_interval:
                model_scores = self._run_model_inference(frame)
                self.last_inference_time = now
                self.inference_count += 1
            elif hasattr(self, '_last_model_scores'):
                model_scores = self._last_model_scores

            self._last_model_scores = model_scores

            # ── Build result
            result = InferenceResult(
                timestamp=now,
                frame_id=self.frame_count,
                score_deepfake=model_scores.get("deepfake", 0.0),
                score_recapture=model_scores.get("recapture", 0.0),
                score_splicing=model_scores.get("splicing", 0.0),
                score_forgery=model_scores.get("forgery", 0.0),
                score_combined=model_scores.get("combined", 0.0),
                gaze_h_deg=gaze_result["gaze_h_deg"],
                gaze_v_deg=gaze_result["gaze_v_deg"],
                head_yaw_deg=gaze_result["head_yaw_deg"],
                is_off_screen=gaze_result["is_off_screen"],
                sustained_offscreen_sec=gaze_result["sustained_offscreen_sec"],
                blink_rate=gaze_result["blink_rate"],
                is_mouth_open=gaze_result["is_mouth_open"],
                face_detected=gaze_result["face_detected"]
            )

            # ── Store frame for display
            with self._lock:
                self._latest_frame = gaze_result["annotated_frame"]
                self._latest_result = result

            # ── Push to queue (non-blocking)
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                    self._result_queue.put_nowait(result)
                except queue.Empty:
                    pass

        # End of loop
        print("[InferenceEngine] Loop ended")

    def _run_model_inference(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Run ExamSentinelNet on a single frame.
        Returns dict of probability scores.
        """
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=rgb)
            img_tensor = augmented["image"].unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                with autocast():
                    probs = self.model.get_probabilities(img_tensor)

            return {
                k: float(v.squeeze().item())
                for k, v in probs.items()
            }

        except Exception as e:
            print(f"[InferenceEngine] Model error: {e}")
            return {
                "deepfake": 0.0, "recapture": 0.0,
                "splicing": 0.0, "forgery": 0.0, "combined": 0.0
            }

    def log_event(self, result: InferenceResult):
        """Append result to JSONL event log."""
        with open(self.log_path, "a") as f:
            f.write(result.to_json() + "\n")

    @property
    def stats(self) -> Dict:
        return {
            "frames_processed": self.frame_count,
            "model_inferences": self.inference_count,
            "gaze_summary": self.gaze_tracker.get_summary()
        }


class MockInferenceEngine:
    """
    Mock engine for testing dashboard without a real model/webcam.
    Generates synthetic data that simulates various threat scenarios.
    """

    def __init__(self, scenario: str = "normal"):
        """
        scenario: 'normal' | 'deepfake_attack' | 'recapture_attempt' | 'gaze_cheat'
        """
        self.scenario = scenario
        self._frame_id = 0
        self._start_time = time.time()

    def get_latest(self) -> InferenceResult:
        self._frame_id += 1
        elapsed = time.time() - self._start_time

        # Simulate scenarios
        if self.scenario == "deepfake_attack":
            t = min(elapsed / 10.0, 1.0)  # Ramp up over 10s
            return InferenceResult(
                frame_id=self._frame_id,
                score_deepfake=0.3 + 0.65 * t + np.random.normal(0, 0.02),
                score_recapture=np.random.uniform(0.0, 0.1),
                score_combined=0.3 + 0.6 * t,
                face_detected=True,
                is_off_screen=False
            )
        elif self.scenario == "recapture_attempt":
            pulse = 0.5 + 0.4 * np.sin(elapsed * 0.5)
            return InferenceResult(
                frame_id=self._frame_id,
                score_recapture=pulse + np.random.normal(0, 0.03),
                score_deepfake=np.random.uniform(0, 0.08),
                score_combined=pulse * 0.8,
                face_detected=True
            )
        elif self.scenario == "gaze_cheat":
            sustained = min(elapsed, 8.0)
            return InferenceResult(
                frame_id=self._frame_id,
                score_combined=np.random.uniform(0, 0.15),
                is_off_screen=elapsed > 3.0,
                sustained_offscreen_sec=max(0, elapsed - 1.0),
                gaze_h_deg=45.0 if elapsed > 2.0 else 5.0,
                face_detected=True
            )
        else:
            # Normal — low scores, on-screen
            return InferenceResult(
                frame_id=self._frame_id,
                score_deepfake=np.random.uniform(0, 0.12),
                score_recapture=np.random.uniform(0, 0.08),
                score_combined=np.random.uniform(0, 0.1),
                face_detected=True,
                gaze_h_deg=np.random.normal(0, 5),
                is_off_screen=False
            )

    def start(self, camera_index: int = 0):
        print(f"[MockEngine] Started with scenario: {self.scenario}")

    def stop(self):
        print("[MockEngine] Stopped")

    def get_current_frame(self):
        return None


if __name__ == "__main__":
    print("InferenceEngine module loaded ✓")
    print("Run scripts/run_demo.py to start the full pipeline")
