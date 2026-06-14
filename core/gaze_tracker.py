"""
core/gaze_tracker.py  — Day 3

Real-time gaze tracking using MediaPipe FaceMesh.

Detects:
  1. Off-screen sustained gaze (looking at phone/notes)
  2. Abnormal blink rate (too low = reading from device, too high = stress)
  3. Head pose deviation (looking away)
  4. Mouth movement (whispering to earpiece)

All signals feed into the Agentic Decision Engine.
"""

import time
import math
import collections
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import mediapipe as mp


# MediaPipe landmark indices
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP = 1
CHIN = 152
LEFT_EAR = 234
RIGHT_EAR = 454
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


class GazeTracker:
    """
    Real-time gaze and behavioural signal tracker.

    Runs MediaPipe FaceMesh on each webcam frame and outputs:
      - gaze_direction: (horizontal_deg, vertical_deg) from screen center
      - is_off_screen: bool
      - blink_rate: blinks/second (rolling window)
      - head_yaw_deg: left-right head rotation
      - is_mouth_open: bool (mouth movement detection)
      - sustained_offscreen_sec: seconds continuously off-screen
    """

    def __init__(self,
                 off_screen_threshold_deg: float = 30.0,
                 sustained_limit_sec: float = 3.5,
                 blink_rate_min: float = 0.1,
                 history_frames: int = 60):
        self.off_screen_threshold = off_screen_threshold_deg
        self.sustained_limit = sustained_limit_sec
        self.blink_rate_min = blink_rate_min
        self.history_frames = history_frames

        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,   # enables iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # State
        self.gaze_history: collections.deque = collections.deque(
            maxlen=history_frames
        )
        self.blink_history: collections.deque = collections.deque(
            maxlen=history_frames
        )
        self.off_screen_start: Optional[float] = None
        self.last_blink_time: float = time.time()
        self.blink_count: int = 0
        self.frame_count: int = 0
        self.fps_times: collections.deque = collections.deque(maxlen=30)

        # Per-frame state
        self.prev_eye_open = True

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single BGR webcam frame.

        Returns dict with all gaze signals + annotated frame.
        """
        t_start = time.time()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = {
            "face_detected": False,
            "gaze_h_deg": 0.0,
            "gaze_v_deg": 0.0,
            "is_off_screen": False,
            "head_yaw_deg": 0.0,
            "head_pitch_deg": 0.0,
            "blink_rate": 0.0,
            "is_mouth_open": False,
            "sustained_offscreen_sec": 0.0,
            "ear_left": 0.0,
            "ear_right": 0.0,
            "annotated_frame": frame.copy()
        }

        # Run MediaPipe
        mp_result = self.face_mesh.process(rgb)
        if not mp_result.multi_face_landmarks:
            # No face detected
            result["annotated_frame"] = self._draw_no_face(frame)
            return result

        lm = mp_result.multi_face_landmarks[0].landmark
        result["face_detected"] = True

        # ── Iris-based gaze estimation
        gaze_h, gaze_v = self._estimate_gaze(lm, w, h)
        result["gaze_h_deg"] = gaze_h
        result["gaze_v_deg"] = gaze_v
        result["is_off_screen"] = (
            abs(gaze_h) > self.off_screen_threshold or
            abs(gaze_v) > self.off_screen_threshold
        )

        # ── Head pose (yaw + pitch from nose/ear/chin)
        yaw, pitch = self._estimate_head_pose(lm, w, h)
        result["head_yaw_deg"] = yaw
        result["head_pitch_deg"] = pitch

        # If head yaw > threshold, treat as off-screen regardless of gaze
        if abs(yaw) > self.off_screen_threshold:
            result["is_off_screen"] = True

        # ── Eye Aspect Ratio (blink detection)
        ear_l = self._eye_aspect_ratio(lm, LEFT_EYE_LANDMARKS, w, h)
        ear_r = self._eye_aspect_ratio(lm, RIGHT_EYE_LANDMARKS, w, h)
        ear_avg = (ear_l + ear_r) / 2.0
        result["ear_left"] = ear_l
        result["ear_right"] = ear_r

        # Blink: EAR drops below 0.2
        is_blink = ear_avg < 0.20
        if is_blink and self.prev_eye_open:
            self.blink_count += 1
            self.last_blink_time = time.time()
        self.prev_eye_open = not is_blink

        # Rolling blink rate (blinks/sec over last 30 frames)
        self.blink_history.append(1 if is_blink else 0)
        elapsed = len(self.blink_history) / max(
            self._estimate_fps(), 1
        )
        result["blink_rate"] = sum(self.blink_history) / max(elapsed, 1.0)

        # ── Mouth open detection
        mouth_top = lm[MOUTH_TOP]
        mouth_bot = lm[MOUTH_BOTTOM]
        mouth_gap = abs(mouth_bot.y - mouth_top.y) * h
        result["is_mouth_open"] = mouth_gap > 10  # pixels

        # ── Sustained off-screen tracking
        now = time.time()
        if result["is_off_screen"]:
            if self.off_screen_start is None:
                self.off_screen_start = now
            result["sustained_offscreen_sec"] = now - self.off_screen_start
        else:
            self.off_screen_start = None
            result["sustained_offscreen_sec"] = 0.0

        # ── Annotate frame
        result["annotated_frame"] = self._annotate(frame, result, lm, w, h)

        self.frame_count += 1
        self.fps_times.append(time.time())

        return result

    def _estimate_gaze(self, lm, w: int, h: int) -> Tuple[float, float]:
        """
        Estimate gaze direction from iris position relative to eye corners.
        Returns (horizontal_deg, vertical_deg) — positive = right/down.
        """
        def iris_ratio(eye_lm_ids, iris_lm_ids):
            # Eye corners
            left_corner = np.array([lm[eye_lm_ids[0]].x * w,
                                     lm[eye_lm_ids[0]].y * h])
            right_corner = np.array([lm[eye_lm_ids[3]].x * w,
                                      lm[eye_lm_ids[3]].y * h])
            top = np.array([lm[eye_lm_ids[1]].x * w,
                             lm[eye_lm_ids[1]].y * h])
            bottom = np.array([lm[eye_lm_ids[5]].x * w,
                                lm[eye_lm_ids[5]].y * h])

            # Iris center
            iris_x = np.mean([lm[i].x * w for i in iris_lm_ids])
            iris_y = np.mean([lm[i].y * h for i in iris_lm_ids])

            eye_width = np.linalg.norm(right_corner - left_corner) + 1e-6
            eye_height = np.linalg.norm(bottom - top) + 1e-6

            h_ratio = (iris_x - left_corner[0]) / eye_width  # 0=left, 1=right
            v_ratio = (iris_y - top[1]) / eye_height          # 0=top, 1=bottom

            return h_ratio - 0.5, v_ratio - 0.5  # center at 0

        # Average both eyes
        h_l, v_l = iris_ratio(LEFT_EYE_LANDMARKS, LEFT_IRIS)
        h_r, v_r = iris_ratio(RIGHT_EYE_LANDMARKS, RIGHT_IRIS)

        h_avg = (h_l + h_r) / 2.0
        v_avg = (v_l + v_r) / 2.0

        # Convert ratio to approximate degrees (calibration factor)
        gaze_h_deg = h_avg * 60.0  # ±30 deg range
        gaze_v_deg = v_avg * 40.0  # ±20 deg range

        return gaze_h_deg, gaze_v_deg

    def _estimate_head_pose(self, lm, w: int, h: int) -> Tuple[float, float]:
        """
        Estimate head yaw (left-right) from ear landmark distances.
        Positive yaw = looking right.
        """
        nose = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])
        left_ear = np.array([lm[LEFT_EAR].x * w, lm[LEFT_EAR].y * h])
        right_ear = np.array([lm[RIGHT_EAR].x * w, lm[RIGHT_EAR].y * h])
        chin = np.array([lm[CHIN].x * w, lm[CHIN].y * h])

        # Yaw from nose offset between ears
        face_center_x = (left_ear[0] + right_ear[0]) / 2.0
        face_width = abs(right_ear[0] - left_ear[0]) + 1e-6
        yaw_ratio = (nose[0] - face_center_x) / face_width
        yaw_deg = yaw_ratio * 90.0

        # Pitch from nose-chin vs nose-top
        face_center_y = (left_ear[1] + right_ear[1]) / 2.0
        pitch_ratio = (nose[1] - face_center_y) / (abs(chin[1] - face_center_y) + 1e-6)
        pitch_deg = (pitch_ratio - 0.5) * 60.0

        return yaw_deg, pitch_deg

    def _eye_aspect_ratio(self, lm, eye_lm_ids: List[int],
                           w: int, h: int) -> float:
        """
        Eye Aspect Ratio (EAR) — close to 0 when eye is closed.
        EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        """
        pts = [(lm[i].x * w, lm[i].y * h) for i in eye_lm_ids]

        def dist(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        vertical_1 = dist(pts[1], pts[5])
        vertical_2 = dist(pts[2], pts[4])
        horizontal = dist(pts[0], pts[3])

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
        return ear

    def _estimate_fps(self) -> float:
        if len(self.fps_times) < 2:
            return 30.0
        elapsed = self.fps_times[-1] - self.fps_times[0]
        return len(self.fps_times) / max(elapsed, 1e-6)

    def _draw_no_face(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        cv2.putText(out, "NO FACE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out

    def _annotate(self, frame: np.ndarray, result: Dict,
                  lm, w: int, h: int) -> np.ndarray:
        """Draw gaze direction + status overlays on frame."""
        out = frame.copy()

        # Status color
        if result["sustained_offscreen_sec"] > self.sustained_limit:
            color = (0, 0, 255)    # Red — threshold exceeded
            status = "ALERT: SUSTAINED GAZE"
        elif result["is_off_screen"]:
            color = (0, 165, 255)  # Orange — currently off
            status = "OFF-SCREEN GAZE"
        else:
            color = (0, 200, 0)    # Green — looking at screen
            status = "ON-SCREEN"

        # Status bar
        cv2.rectangle(out, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.putText(out, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Gaze info
        cv2.putText(out,
                    f"Gaze H:{result['gaze_h_deg']:+.0f} V:{result['gaze_v_deg']:+.0f}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out,
                    f"Yaw:{result['head_yaw_deg']:+.0f}  Offscr:{result['sustained_offscreen_sec']:.1f}s",
                    (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw gaze arrow from nose tip
        nose_x = int(lm[NOSE_TIP].x * w)
        nose_y = int(lm[NOSE_TIP].y * h)
        arrow_x = nose_x + int(result["gaze_h_deg"] * 2)
        arrow_y = nose_y + int(result["gaze_v_deg"] * 2)
        cv2.arrowedLine(out, (nose_x, nose_y), (arrow_x, arrow_y),
                        color, 2, tipLength=0.3)

        return out

    def get_summary(self) -> Dict:
        """Return current running statistics."""
        return {
            "total_frames": self.frame_count,
            "fps": self._estimate_fps(),
            "total_blinks": self.blink_count,
        }

    def release(self):
        self.face_mesh.close()


if __name__ == "__main__":
    print("Testing GazeTracker with webcam...")
    tracker = GazeTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No webcam found. Exiting.")
        exit()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = tracker.process_frame(frame)
        cv2.imshow("ExamSentinel — Gaze Tracker", result["annotated_frame"])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    print("✓ Gaze tracker test complete")
