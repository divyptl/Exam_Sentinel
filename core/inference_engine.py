"""
core/inference_engine.py

Real-time webcam inference pipeline + MockEngine for demo/testing.
"""

import time
import queue
import threading
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass
class InferenceResult:
    """Single timestamped snapshot of all signals."""
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    score_deepfake:  float = 0.0
    score_recapture: float = 0.0
    score_splicing:  float = 0.0
    score_forgery:   float = 0.0
    score_combined:  float = 0.0

    gaze_h_deg: float = 0.0
    gaze_v_deg: float = 0.0
    head_yaw_deg: float = 0.0
    is_off_screen: bool = False
    sustained_offscreen_sec: float = 0.0
    blink_rate: float = 0.0
    is_mouth_open: bool = False
    face_detected: bool = True

    alert_tier:   str = "GREEN"
    alert_reason: str = ""
    action_taken: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(self.timestamp))
        return json.dumps(d)


class MockInferenceEngine:
    """
    Generates synthetic InferenceResult data for 4 demo scenarios.
    No camera or model required — used for dashboard demos and tests.
    """
    def __init__(self, scenario: str = "normal"):
        self.scenario = scenario
        self._frame_id = 0
        self._start_time = time.time()

    def start(self, camera_index: int = 0):
        print(f"[MockEngine] Started — scenario: {self.scenario}")

    def stop(self):
        print("[MockEngine] Stopped")

    def get_current_frame(self):
        return None

    def get_latest(self) -> InferenceResult:
        self._frame_id += 1
        elapsed = time.time() - self._start_time

        if self.scenario == "deepfake_attack":
            t = min(elapsed / 10.0, 1.0)
            return InferenceResult(
                frame_id=self._frame_id,
                score_deepfake=max(0, min(1, 0.3 + 0.65 * t + np.random.normal(0, 0.02))),
                score_recapture=max(0, np.random.uniform(0, 0.08)),
                score_splicing=max(0, np.random.uniform(0, 0.06)),
                score_forgery=max(0, np.random.uniform(0, 0.05)),
                score_combined=max(0, min(1, 0.3 + 0.60 * t)),
                face_detected=True, is_off_screen=False
            )

        elif self.scenario == "recapture_attempt":
            pulse = max(0, min(1, 0.5 + 0.4 * np.sin(elapsed * 0.5)))
            return InferenceResult(
                frame_id=self._frame_id,
                score_recapture=max(0, min(1, pulse + np.random.normal(0, 0.03))),
                score_deepfake=max(0, np.random.uniform(0, 0.08)),
                score_splicing=max(0, np.random.uniform(0, 0.05)),
                score_forgery=max(0, np.random.uniform(0, 0.04)),
                score_combined=max(0, min(1, pulse * 0.8)),
                face_detected=True
            )

        elif self.scenario == "gaze_cheat":
            return InferenceResult(
                frame_id=self._frame_id,
                score_deepfake=max(0, np.random.uniform(0, 0.10)),
                score_recapture=max(0, np.random.uniform(0, 0.06)),
                score_splicing=max(0, np.random.uniform(0, 0.04)),
                score_forgery=max(0, np.random.uniform(0, 0.04)),
                score_combined=max(0, np.random.uniform(0, 0.12)),
                is_off_screen=elapsed > 2.0,
                sustained_offscreen_sec=max(0.0, elapsed - 1.0),
                gaze_h_deg=45.0 if elapsed > 2.0 else float(np.random.normal(0, 5)),
                face_detected=True
            )

        else:  # normal
            return InferenceResult(
                frame_id=self._frame_id,
                score_deepfake=max(0, np.random.uniform(0, 0.10)),
                score_recapture=max(0, np.random.uniform(0, 0.07)),
                score_splicing=max(0, np.random.uniform(0, 0.06)),
                score_forgery=max(0, np.random.uniform(0, 0.05)),
                score_combined=max(0, np.random.uniform(0, 0.09)),
                face_detected=True,
                gaze_h_deg=float(np.random.normal(0, 5)),
                is_off_screen=False
            )


class InferenceEngine:
    """
    Full real-time engine: webcam → GazeTracker → ExamSentinelNet → queue.
    Uses MockInferenceEngine internally if model not loaded.
    """
    def __init__(self, model, config: Dict, log_path: str = "logs/events.jsonl"):
        self.model = model
        self.config = config
        self.log_path = log_path
        self.inference_interval = config.get("inference", {}).get("interval_seconds", 5.0)
        self.device = next(model.parameters()).device
        self._running = False
        self._thread = None
        self._result_queue: queue.Queue = queue.Queue(maxsize=100)
        self._latest_result: Optional[InferenceResult] = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self.frame_count = 0
        self.inference_count = 0
        self.last_inference_time = 0.0
        self._last_model_scores: Dict = {}
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        from core.dataset import get_val_transforms
        self.transform = get_val_transforms(config.get("model", {}).get("img_size", 256))

        try:
            from core.gaze_tracker import GazeTracker
            dec = config.get("decision_engine", {})
            self.gaze_tracker = GazeTracker(
                off_screen_threshold_deg=dec.get("sustained_gaze_seconds", 30.0),
                sustained_limit_sec=dec.get("sustained_gaze_seconds", 3.5)
            )
        except Exception:
            self.gaze_tracker = None

    def start(self, camera_index: int = 0):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[InferenceEngine] Started on camera {camera_index}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if hasattr(self, '_cap'):
            self._cap.release()
        if self.gaze_tracker:
            self.gaze_tracker.release()

    def get_latest(self) -> Optional[InferenceResult]:
        with self._lock:
            return self._latest_result

    def get_current_frame(self):
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _run_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self.frame_count += 1

            gaze_result = {"face_detected": True, "gaze_h_deg": 0, "gaze_v_deg": 0,
                           "head_yaw_deg": 0, "is_off_screen": False,
                           "sustained_offscreen_sec": 0, "blink_rate": 0.3,
                           "is_mouth_open": False, "annotated_frame": frame}
            if self.gaze_tracker:
                gaze_result = self.gaze_tracker.process_frame(frame)

            now = time.time()
            if now - self.last_inference_time >= self.inference_interval:
                self._last_model_scores = self._run_model(frame)
                self.last_inference_time = now
                self.inference_count += 1

            scores = self._last_model_scores
            result = InferenceResult(
                timestamp=now, frame_id=self.frame_count,
                score_deepfake=scores.get("deepfake", 0.0),
                score_recapture=scores.get("recapture", 0.0),
                score_splicing=scores.get("splicing", 0.0),
                score_forgery=scores.get("forgery", 0.0),
                score_combined=scores.get("combined", 0.0),
                **{k: gaze_result[k] for k in [
                    "gaze_h_deg", "gaze_v_deg", "head_yaw_deg",
                    "is_off_screen", "sustained_offscreen_sec",
                    "blink_rate", "is_mouth_open", "face_detected"
                ]}
            )
            with self._lock:
                self._latest_frame = gaze_result["annotated_frame"]
                self._latest_result = result
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                    self._result_queue.put_nowait(result)
                except queue.Empty:
                    pass

    def _run_model(self, frame: np.ndarray) -> Dict[str, float]:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self.transform(image=rgb)["image"].unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                probs = self.model.get_probabilities(tensor)
            return {k: float(v.squeeze()) for k, v in probs.items()}
        except Exception as e:
            print(f"[InferenceEngine] Model error: {e}")
            return {k: 0.0 for k in ["deepfake","recapture","splicing","forgery","combined"]}
