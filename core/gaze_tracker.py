"""
core/gaze_tracker.py

Real-time gaze and behavioural signal tracker using MediaPipe FaceMesh.
Detects: off-screen gaze, head pose, blink rate, mouth movement.
"""

import time
import math
import collections
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

HAS_MEDIAPIPE_FACE_MESH = (
    mp is not None and hasattr(mp, "solutions") and
    hasattr(mp.solutions, "face_mesh")
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP   = 1
CHIN       = 152
LEFT_EAR   = 234
RIGHT_EAR  = 454
MOUTH_TOP  = 13
MOUTH_BOT  = 14


class GazeTracker:
    def __init__(self, off_screen_threshold_deg: float = 30.0,
                 sustained_limit_sec: float = 3.5, history_frames: int = 60):
        self.off_screen_threshold = off_screen_threshold_deg
        self.sustained_limit = sustained_limit_sec

        self.use_mediapipe = HAS_MEDIAPIPE_FACE_MESH
        self.face_mesh = None
        self.face_detector = None
        self.eye_detector = None

        if self.use_mediapipe:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        else:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.eye_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
            )
            if self.face_detector.empty() or self.eye_detector.empty():
                raise RuntimeError("OpenCV fallback classifiers are unavailable")

        self.off_screen_start: Optional[float] = None
        self.blink_history: collections.deque = collections.deque(maxlen=history_frames)
        self.fps_times: collections.deque = collections.deque(maxlen=30)
        self.prev_eye_open = True
        self.blink_count = 0
        self.frame_count = 0
        # For fallback eye-motion detection
        self._prev_eye_center: Optional[Tuple[float, float]] = None
        self._last_time = time.time()
        self._eye_movement_speed = 0.0

    def process_frame(self, frame: np.ndarray) -> Dict:
        if self.use_mediapipe:
            return self._process_mediapipe(frame)
        return self._process_fallback(frame)

    def _process_mediapipe(self, frame: np.ndarray) -> Dict:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = {
            "face_detected": False, "gaze_h_deg": 0.0, "gaze_v_deg": 0.0,
            "is_off_screen": False, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0,
            "blink_rate": 0.0, "is_mouth_open": False,
            "sustained_offscreen_sec": 0.0, "ear_left": 0.0, "ear_right": 0.0,
            "annotated_frame": frame.copy(), "lm": None
        }

        mp_result = self.face_mesh.process(rgb)
        if not mp_result.multi_face_landmarks:
            result["annotated_frame"] = self._draw_no_face(frame)
            return result

        lm = mp_result.multi_face_landmarks[0].landmark
        result["face_detected"] = True
        result["lm"] = lm

        gaze_h, gaze_v = self._estimate_gaze(lm, w, h)
        result["gaze_h_deg"] = gaze_h
        result["gaze_v_deg"] = gaze_v
        result["is_off_screen"] = (abs(gaze_h) > self.off_screen_threshold or
                                    abs(gaze_v) > self.off_screen_threshold)

        yaw, pitch = self._head_pose(lm, w, h)
        result["head_yaw_deg"] = yaw
        result["head_pitch_deg"] = pitch
        if abs(yaw) > self.off_screen_threshold:
            result["is_off_screen"] = True

        ear_l = self._ear(lm, LEFT_EYE, w, h)
        ear_r = self._ear(lm, RIGHT_EYE, w, h)
        result["ear_left"] = ear_l
        result["ear_right"] = ear_r
        is_blink = (ear_l + ear_r) / 2 < 0.20
        if is_blink and self.prev_eye_open:
            self.blink_count += 1
        self.prev_eye_open = not is_blink
        self.blink_history.append(1 if is_blink else 0)
        fps = self._fps()
        result["blink_rate"] = sum(self.blink_history) / max(len(self.blink_history) / fps, 1.0)

        mouth_gap = abs(lm[MOUTH_BOT].y - lm[MOUTH_TOP].y) * h
        result["is_mouth_open"] = mouth_gap > 10

        now = time.time()
        if result["is_off_screen"]:
            if self.off_screen_start is None:
                self.off_screen_start = now
            result["sustained_offscreen_sec"] = now - self.off_screen_start
        else:
            self.off_screen_start = None
            result["sustained_offscreen_sec"] = 0.0

        result["annotated_frame"] = self._annotate(frame, result, lm, w, h)
        self.frame_count += 1
        self.fps_times.append(time.time())
        return result

    def _process_fallback(self, frame: np.ndarray) -> Dict:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # More lenient cascade parameters for robustness
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=4, minSize=(80, 80)
        )
        result = {
            "face_detected": False, "gaze_h_deg": 0.0, "gaze_v_deg": 0.0,
            "is_off_screen": False, "head_yaw_deg": 0.0, "head_pitch_deg": 0.0,
            "blink_rate": 0.0, "is_mouth_open": False,
            "sustained_offscreen_sec": 0.0, "ear_left": 0.0, "ear_right": 0.0,
            "annotated_frame": frame.copy(), "lm": None
        }

        if len(faces) == 0:
            result["annotated_frame"] = self._draw_no_face(frame)
            return result

        x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
        face_cx = x + fw / 2
        face_cy = y + fh / 2
        frame_cx = w / 2
        frame_cy = h / 2

        roi_gray = gray[y:y+fh, x:x+fw]
        # More lenient eye detection
        eyes = self.eye_detector.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25)
        )

        if len(eyes) >= 2:
            eye_centers = []
            for ex, ey, ew, eh in eyes[:2]:
                eye_centers.append((x + ex + ew / 2, y + ey + eh / 2))
            avg_eye = np.mean(eye_centers, axis=0)
            gaze_h = ((avg_eye[0] - face_cx) / (fw + 1e-6)) * 90.0
            gaze_v = ((avg_eye[1] - face_cy) / (fh + 1e-6)) * 60.0
            curr_eye_center = (float(avg_eye[0]), float(avg_eye[1]))
        elif len(eyes) == 1:
            ex, ey, ew, eh = eyes[0]
            eye_x = x + ex + ew / 2
            eye_y = y + ey + eh / 2
            gaze_h = ((eye_x - face_cx) / (fw + 1e-6)) * 100.0
            gaze_v = ((eye_y - face_cy) / (fh + 1e-6)) * 75.0
            curr_eye_center = (float(eye_x), float(eye_y))
        else:
            gaze_h = ((face_cx - frame_cx) / (frame_cx + 1e-6)) * 25.0
            gaze_v = ((face_cy - frame_cy) / (frame_cy + 1e-6)) * 15.0
            curr_eye_center = None

        # Compute eye movement speed (pixels/sec normalized by face width)
        now = time.time()
        dt = max(1e-6, now - self._last_time)
        if curr_eye_center is not None and self._prev_eye_center is not None:
            dx = curr_eye_center[0] - self._prev_eye_center[0]
            dy = curr_eye_center[1] - self._prev_eye_center[1]
            dist = math.hypot(dx, dy)
            # Normalize by face width to make speed camera/resolution independent
            self._eye_movement_speed = (dist / (fw + 1e-6)) / dt
        else:
            self._eye_movement_speed = 0.0
        self._prev_eye_center = curr_eye_center
        self._last_time = now

        yaw = ((face_cx - frame_cx) / frame_cx) * 45.0
        result["face_detected"] = True
        result["gaze_h_deg"] = float(np.clip(gaze_h, -60.0, 60.0))
        result["gaze_v_deg"] = float(np.clip(gaze_v, -40.0, 40.0))
        result["head_yaw_deg"] = float(np.clip(yaw, -50.0, 50.0))
        result["is_off_screen"] = (
            abs(result["gaze_h_deg"]) > 28 or abs(result["gaze_v_deg"]) > 22 or
            abs(result["head_yaw_deg"]) > 28
        )

        eyes_visible = len(eyes) >= 1
        if not eyes_visible and self.prev_eye_open:
            self.blink_count += 1
        self.prev_eye_open = eyes_visible
        self.blink_history.append(0 if eyes_visible else 1)
        fps = self._fps()
        result["blink_rate"] = sum(self.blink_history) / max(len(self.blink_history) / fps, 1.0)

        now = time.time()
        if result["is_off_screen"]:
            if self.off_screen_start is None:
                self.off_screen_start = now
            result["sustained_offscreen_sec"] = now - self.off_screen_start
        else:
            self.off_screen_start = None
            result["sustained_offscreen_sec"] = 0.0

        result["annotated_frame"] = self._annotate_fallback(frame, result, x, y, fw, fh, eyes)
        result["eye_movement"] = float(self._eye_movement_speed)
        result["eye_center"] = self._prev_eye_center
        self.frame_count += 1
        self.fps_times.append(time.time())
        return result

    def _estimate_gaze(self, lm, w, h):
        def iris_ratio(eye_ids, iris_ids):
            lc = np.array([lm[eye_ids[0]].x * w, lm[eye_ids[0]].y * h])
            rc = np.array([lm[eye_ids[3]].x * w, lm[eye_ids[3]].y * h])
            tp = np.array([lm[eye_ids[1]].x * w, lm[eye_ids[1]].y * h])
            bt = np.array([lm[eye_ids[5]].x * w, lm[eye_ids[5]].y * h])
            ix = np.mean([lm[i].x * w for i in iris_ids])
            iy = np.mean([lm[i].y * h for i in iris_ids])
            ew = np.linalg.norm(rc - lc) + 1e-6
            eh = np.linalg.norm(bt - tp) + 1e-6
            return (ix - lc[0]) / ew - 0.5, (iy - tp[1]) / eh - 0.5
        hl, vl = iris_ratio(LEFT_EYE, LEFT_IRIS)
        hr, vr = iris_ratio(RIGHT_EYE, RIGHT_IRIS)
        return ((hl + hr) / 2) * 60.0, ((vl + vr) / 2) * 40.0

    def _head_pose(self, lm, w, h):
        nose  = np.array([lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h])
        le    = np.array([lm[LEFT_EAR].x  * w, lm[LEFT_EAR].y  * h])
        re    = np.array([lm[RIGHT_EAR].x * w, lm[RIGHT_EAR].y * h])
        chin  = np.array([lm[CHIN].x * w,      lm[CHIN].y * h])
        fc_x  = (le[0] + re[0]) / 2
        fw    = abs(re[0] - le[0]) + 1e-6
        yaw   = ((nose[0] - fc_x) / fw) * 90.0
        fc_y  = (le[1] + re[1]) / 2
        pitch = ((nose[1] - fc_y) / (abs(chin[1] - fc_y) + 1e-6) - 0.5) * 60.0
        return yaw, pitch

    def _ear(self, lm, ids, w, h):
        pts = [(lm[i].x * w, lm[i].y * h) for i in ids]
        def d(a, b): return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2 * d(pts[0], pts[3]) + 1e-6)

    def _fps(self):
        if len(self.fps_times) < 2:
            return 30.0
        return len(self.fps_times) / max(self.fps_times[-1] - self.fps_times[0], 1e-6)

    def _draw_no_face(self, frame):
        out = frame.copy()
        cv2.putText(out, "NO FACE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out

    def _annotate_fallback(self, frame, result, x, y, fw, fh, eyes):
        out = frame.copy()
        if result["sustained_offscreen_sec"] > self.sustained_limit:
            color, status = (0, 0, 255), "ALERT: SUSTAINED GAZE"
        elif result["is_off_screen"]:
            color, status = (0, 165, 255), "OFF-SCREEN GAZE"
        else:
            color, status = (0, 200, 0), "ON-SCREEN"
        cv2.rectangle(out, (0, 0), (out.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(out, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.rectangle(out, (x, y), (x + fw, y + fh), (180, 200, 70), 2)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(out, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (80, 180, 220), 1)
        cv2.putText(out, f"Gaze H:{result['gaze_h_deg']:+.0f} V:{result['gaze_v_deg']:+.0f}",
                    (10, out.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f"Head Yaw:{result['head_yaw_deg']:+.0f}",
                    (10, out.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Eye movement speed (normalized)
        em = result.get('eye_movement', 0.0)
        cv2.putText(out, f"Eye mv:{em:.2f}", (10, out.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,210), 1)
        # Draw last eye center if available
        if self._prev_eye_center is not None:
            try:
                cx, cy = self._prev_eye_center
                cv2.circle(out, (int(cx), int(cy)), 4, (200,220,60), -1)
            except Exception:
                pass
        return out

    def get_summary(self):
        return {"total_frames": self.frame_count, "fps": self._fps(),
                "total_blinks": self.blink_count}

    def release(self):
        if self.face_mesh is not None:
            self.face_mesh.close()


if __name__ == "__main__":
    print("Testing GazeTracker with webcam (press Q to quit)...")
    tracker = GazeTracker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = tracker.process_frame(frame)
            cv2.imshow("ExamSentinel — Gaze", result["annotated_frame"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    tracker.release()
    print("✓ Gaze tracker OK")
