"""
scripts/live_demo.py

ExamSentinel LIVE DEMO — for video recording.

Opens your actual webcam and runs the full pipeline in real-time.
No model weights needed — uses the forensics heuristics + gaze tracking
+ mock scores that ramp up when you press cheat keys.

HOW TO USE FOR VIDEO:
    python scripts/live_demo.py

CONTROLS (press while window is focused):
    1 — Simulate DEEPFAKE attack (scores ramp up, RED fires)
    2 — Simulate RECAPTURE / paper leak (hold phone to camera)
    3 — GAZE cheat mode (just look away from screen)
    R — RESET to normal / GREEN
    Q — Quit
    S — Save screenshot

WHAT TO DO ON CAMERA:
    Normal  → sit and look at screen (GREEN)
    Press 1 → system detects deepfake → ORANGE → RED
    Press R → resets
    Press 2 → hold your phone up to webcam → system detects recapture
    Press R → resets
    Press 3 → look away from screen → sustained gaze alert fires
"""

import sys
import cv2
import time
import math
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    import mediapipe as mp
    MEDIAPIPE_OK = (hasattr(mp, "solutions") and
                    hasattr(mp.solutions, "face_mesh"))
    if not MEDIAPIPE_OK:
        print("[!] mediapipe installed but does not expose solutions.face_mesh — using fallback gaze tracker")
except ImportError:
    mp = None
    MEDIAPIPE_OK = False
    print("[!] mediapipe not installed — gaze tracking disabled")
    print("    pip install mediapipe")


# ─── Colour palette ───────────────────────────────────────────────────────────

COLOURS = {
    "GREEN":  (0, 200, 80),
    "YELLOW": (0, 180, 220),
    "ORANGE": (0, 120, 220),
    "RED":    (0, 50, 230),
    "WHITE":  (255, 255, 255),
    "BLACK":  (0, 0, 0),
    "DARK":   (20, 20, 30),
    "PANEL":  (30, 32, 42),
}

TIER_COLORS = {
    "GREEN":  (0, 200, 80),
    "YELLOW": (0, 180, 220),
    "ORANGE": (30, 130, 230),
    "RED":    (40, 60, 230),
}


# ─── Score simulator ──────────────────────────────────────────────────────────

class ScoreSimulator:
    """
    Drives detection scores based on:
    1. Keyboard-triggered scenario (deepfake/recapture)
    2. Real gaze signals from MediaPipe
    3. Moiré heuristic from FFT of webcam frame
    """

    def __init__(self):
        self.scenario   = "normal"
        self.t_start    = time.time()
        self.noise      = lambda: float(np.random.normal(0, 0.015))

        self.scores = {
            "deepfake":  0.0,
            "recapture": 0.0,
            "splicing":  0.0,
            "forgery":   0.0,
            "combined":  0.0,
        }
        self._target = {k: 0.0 for k in self.scores}
        # mark as just-reset so initial frames don't immediately trigger recapture
        self.last_reset = time.time()

    def set_scenario(self, name: str):
        self.scenario = name
        self.t_start  = time.time()
        # record reset time so UI boosts can be briefly suppressed
        if name == "normal":
            self.last_reset = time.time()
        if name == "normal":
            self._target = {k: 0.0 for k in self.scores}
            # zero scores immediately on reset so overlay clears
            self.scores = {k: 0.0 for k in self.scores}

    def update(self, frame: np.ndarray, gaze_off: bool,
               sustained_sec: float) -> dict:
        elapsed = time.time() - self.t_start

        # ── Set targets based on scenario
        if self.scenario == "deepfake":
            ramp = min(elapsed / 8.0, 1.0)            # ramp over 8s
            self._target["deepfake"]  = 0.30 + 0.65 * ramp
            self._target["splicing"]  = 0.10 + 0.15 * ramp
            self._target["forgery"]   = 0.08 + 0.12 * ramp
            self._target["recapture"] = 0.03
            self._target["combined"]  = 0.25 + 0.65 * ramp

        elif self.scenario == "recapture":
            # Use actual FFT moiré score + simulated ramp
            moire_score = self._compute_moire(frame)
            ramp = min(elapsed / 6.0, 1.0)
            self._target["recapture"] = max(moire_score, 0.25 + 0.65 * ramp)
            self._target["deepfake"]  = 0.04
            self._target["splicing"]  = 0.06
            self._target["forgery"]   = 0.04
            self._target["combined"]  = self._target["recapture"] * 0.85

        elif self.scenario == "gaze":
            # Pure gaze-based — scores stay low but gaze signals trigger alert
            self._target["deepfake"]  = 0.06
            self._target["recapture"] = 0.04
            self._target["splicing"]  = 0.03
            self._target["forgery"]   = 0.03
            self._target["combined"]  = 0.05 + min(sustained_sec / 10.0, 0.30)

        else:  # normal
            self._target = {k: 0.0 for k in self.scores}

        # ── Smooth scores toward targets (exponential moving average)
        alpha = 0.15
        for k in self.scores:
            target = max(0.0, min(1.0, self._target[k] + self.noise()))
            self.scores[k] = alpha * target + (1 - alpha) * self.scores[k]

        return dict(self.scores)

    def _compute_moire(self, frame: np.ndarray) -> float:
        """
        Compute real moiré score from webcam frame via FFT.
        When you hold a phone/screen up to the webcam, the moiré
        patterns in the FFT become stronger — this detects it.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))
            f    = np.fft.fft2(gray.astype(np.float32))
            fs   = np.fft.fftshift(f)
            mag  = np.log1p(np.abs(fs))

            h, w = mag.shape
            cy, cx = h // 2, w // 2

            # Look at energy in mid-frequency band (where moiré lives)
            r_inner, r_outer = 12, 50
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            ring_mask   = (dist >= r_inner) & (dist <= r_outer)
            center_mask = dist < r_inner

            ring_energy   = mag[ring_mask].mean()
            center_energy = mag[center_mask].mean()

            # High ratio = moiré present (periodic pattern in mid-freq)
            ratio = ring_energy / (center_energy + 1e-6)

            # Normalise: more sensitive to moiré — lower threshold, steeper ramp
            # Normal: ~0.28-0.32, Phone held up: >0.40
            score = max(0.0, min(1.0, (ratio - 0.25) / 0.30))
            return score
        except Exception:
            return 0.0

    def compute_recapture_metric(self, frame: np.ndarray) -> float:
        """Combine moiré, high-frequency energy and specular highlights into one metric."""
        try:
            moire = self._compute_moire(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))
            # High-frequency energy (Sobel)
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(sx * sx + sy * sy)
            hf = float(np.mean(mag)) / (np.mean(gray) + 1e-6)

            # Specular highlights: proportion of very bright pixels
            bright = np.mean(gray > 240)

            # Combine with tuned weights
            combined = 0.6 * moire + 0.25 * np.clip(hf / 5.0, 0.0, 1.0) + 0.15 * float(np.clip(bright * 5.0, 0.0, 1.0))
            return float(np.clip(combined, 0.0, 1.0))
        except Exception:
            return 0.0


# ─── Gaze tracker (minimal, no MediaPipe dependency guard) ───────────────────

class SimpleGazeTracker:
    """Lightweight gaze tracker with a MediaPipe / OpenCV fallback."""

    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS  = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    def __init__(self):
        self.mp_ok = MEDIAPIPE_OK
        self.available = True
        self.off_start: float | None = None
        self.fm = None
        self.face_detector = None
        self.eye_detector = None
        self._prev_eye_center = None
        self._last_time = time.time()
        self._eye_movement_speed = 0.0

        if self.mp_ok:
            try:
                mp_fm = mp.solutions.face_mesh
                self.fm = mp_fm.FaceMesh(
                    static_image_mode=False, max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception:
                self.mp_ok = False

        if not self.mp_ok:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.eye_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
            )
            if self.face_detector.empty() or self.eye_detector.empty():
                self.available = False
                print("[!] OpenCV fallback gaze tracker unavailable")

    def process(self, frame: np.ndarray) -> dict:
        result = {
            "face": False, "gaze_h": 0.0, "gaze_v": 0.0,
            "yaw": 0.0, "off_screen": False, "sustained": 0.0,
            "lm": None
        }
        if not self.available:
            return result

        h, w = frame.shape[:2]
        curr_eye_center = None
        if self.mp_ok and self.fm is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = self.fm.process(rgb)
            if not out.multi_face_landmarks:
                return result

            lm = out.multi_face_landmarks[0].landmark
            result["face"] = True
            result["lm"]   = lm

            def iris_offset(eye_ids, iris_ids):
                lx = lm[eye_ids[0]].x * w
                rx = lm[eye_ids[3]].x * w
                ty = lm[eye_ids[1]].y * h
                by = lm[eye_ids[5]].y * h
                ix = np.mean([lm[i].x * w for i in iris_ids])
                iy = np.mean([lm[i].y * h for i in iris_ids])
                ew = rx - lx + 1e-6
                eh = by - ty + 1e-6
                return (ix - lx) / ew - 0.5, (iy - ty) / eh - 0.5

            hl, vl = iris_offset(self.LEFT_EYE,  self.LEFT_IRIS)
            hr, vr = iris_offset(self.RIGHT_EYE, self.RIGHT_IRIS)
            gh = ((hl + hr) / 2) * 60.0
            gv = ((vl + vr) / 2) * 40.0
            result["gaze_h"] = gh
            result["gaze_v"] = gv

            nose_x  = lm[1].x * w
            le_x    = lm[234].x * w
            re_x    = lm[454].x * w
            fc_x    = (le_x + re_x) / 2
            fw      = abs(re_x - le_x) + 1e-6
            result["yaw"] = ((nose_x - fc_x) / fw) * 90.0

            off = abs(gh) > 28 or abs(gv) > 22 or abs(result["yaw"]) > 28
            result["off_screen"] = off
            # compute eye center from iris landmarks for motion
            try:
                ix = np.mean([lm[i].x * w for i in self.LEFT_IRIS + self.RIGHT_IRIS])
                iy = np.mean([lm[i].y * h for i in self.LEFT_IRIS + self.RIGHT_IRIS])
                curr_eye_center = (float(ix), float(iy))
            except Exception:
                curr_eye_center = None

        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # More lenient face detection: lower minNeighbors for better sensitivity
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(80, 80)
            )
            if len(faces) == 0:
                return result

            x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
            face_cx = x + fw / 2
            face_cy = y + fh / 2
            frame_cx = w / 2
            frame_cy = h / 2

            roi_gray = gray[y:y+fh, x:x+fw]
            # More lenient eye detection: lower minNeighbors
            eyes = self.eye_detector.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25)
            )

            result["face"] = True
            if len(eyes) >= 2:
                # Two eyes detected: compute eye-based gaze
                eye_centers = []
                for ex, ey, ew, eh in eyes[:2]:
                    eye_centers.append((x + ex + ew / 2, y + ey + eh / 2))
                avg_eye = np.mean(eye_centers, axis=0)
                gh = ((avg_eye[0] - face_cx) / (fw + 1e-6)) * 90.0
                gv = ((avg_eye[1] - face_cy) / (fh + 1e-6)) * 60.0
                curr_eye_center = (float(avg_eye[0]), float(avg_eye[1]))
            elif len(eyes) == 1:
                # One eye: gaze toward it
                ex, ey, ew, eh = eyes[0]
                eye_x = x + ex + ew / 2
                eye_y = y + ey + eh / 2
                gh = ((eye_x - face_cx) / (fw + 1e-6)) * 100.0
                gv = ((eye_y - face_cy) / (fh + 1e-6)) * 75.0
            else:
                # No eyes: subtle gaze direction from face center vs frame center
                gh = ((face_cx - frame_cx) / (frame_cx + 1e-6)) * 25.0
                gv = ((face_cy - frame_cy) / (frame_cy + 1e-6)) * 15.0

            # Head yaw: how far left/right the face is from frame center
            yaw = ((face_cx - frame_cx) / (frame_cx + 1e-6)) * 40.0
            result["gaze_h"] = float(np.clip(gh, -60.0, 60.0))
            result["gaze_v"] = float(np.clip(gv, -40.0, 40.0))
            result["yaw"] = float(np.clip(yaw, -50.0, 50.0))
            # More sensitive thresholds for off-screen detection
            result["off_screen"] = (
                abs(result["gaze_h"]) > 24 or abs(result["gaze_v"]) > 18 or
                abs(result["yaw"]) > 24
            )

        now = time.time()
        off = result["off_screen"]
        if off:
            if self.off_start is None:
                self.off_start = now
            result["sustained"] = now - self.off_start
            # Ensure keys exist even when off-screen to avoid callers getting None
            result["eye_movement"] = float(self._eye_movement_speed)
            result["eye_center"] = self._prev_eye_center
            return result
        else:
            self.off_start = None
            result["sustained"] = 0.0

            # eye movement speed (normalized by face width)
            dt = max(1e-6, now - self._last_time)
            if curr_eye_center is not None and self._prev_eye_center is not None:
                dx = curr_eye_center[0] - self._prev_eye_center[0]
                dy = curr_eye_center[1] - self._prev_eye_center[1]
                dist = math.hypot(dx, dy)
                self._eye_movement_speed = (dist / (fw + 1e-6)) / dt
            else:
                self._eye_movement_speed = 0.0
            self._prev_eye_center = curr_eye_center
            self._last_time = now
            result["eye_movement"] = float(self._eye_movement_speed)
            result["eye_center"] = curr_eye_center

            return result

    def close(self):
        if self.fm is not None:
            self.fm.close()


# ─── Overlay renderer ─────────────────────────────────────────────────────────

def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=12, alpha=0.85):
    overlay = img.copy()
    pts = [(x1+radius, y1), (x2-radius, y1),
           (x2, y1+radius), (x2, y2-radius),
           (x2-radius, y2), (x1+radius, y2),
           (x1, y2-radius), (x1, y1+radius)]
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
    for cx, cy, a1, a2 in [
        (x1+radius, y1+radius, 180, 270),
        (x2-radius, y1+radius, 270, 360),
        (x2-radius, y2-radius, 0, 90),
        (x1+radius, y2-radius, 90, 180),
    ]:
        cv2.ellipse(overlay, (cx,cy), (radius,radius), 0, a1, a2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def draw_score_bar(img, x, y, w, h, score, color, label):
    """Draw a filled progress bar for a detection score."""
    # Background
    cv2.rectangle(img, (x, y), (x+w, y+h), (50, 52, 65), -1)
    # Fill
    fill = int(w * min(score, 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)
    # Border
    cv2.rectangle(img, (x, y), (x+w, y+h), (80, 82, 95), 1)
    # Label
    cv2.putText(img, f"{label}", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 190), 1)
    # Percentage
    cv2.putText(img, f"{score:.0%}", (x+w+5, y+h-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


def render_overlay(frame, scores, gaze, tier, reasons, scenario, fps):
    """
    Draw the full ExamSentinel overlay on the webcam frame.
    """
    H, W = frame.shape[:2]
    out  = frame.copy()
    tc   = TIER_COLORS[tier]

    # ── Top status bar
    draw_rounded_rect(out, 0, 0, W, 52, COLOURS["PANEL"], radius=0, alpha=0.92)

    # Logo
    cv2.putText(out, "ExamSentinel", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, tc, 2)

    # Tier badge (right side)
    badge_text = f"  {tier}  "
    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    bx = W - bw - 20
    draw_rounded_rect(out, bx-8, 10, bx+bw+8, 44, tc, radius=8, alpha=0.9)
    cv2.putText(out, tier, (bx, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # FPS
    cv2.putText(out, f"{fps:.0f} fps", (W//2 - 25, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 140), 1)

    # ── Left panel: detection scores
    PX, PY, PW = 10, 65, 200
    draw_rounded_rect(out, PX, PY, PX+PW+60, PY+165, COLOURS["PANEL"],
                      radius=10, alpha=0.88)

    cv2.putText(out, "DETECTION SCORES", (PX+8, PY+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 160), 1)

    bar_data = [
        ("Deepfake",  scores["deepfake"],  (60, 80, 230)),
        ("Recapture", scores["recapture"], (40, 130, 220)),
        ("Splicing",  scores["splicing"],  (30, 160, 180)),
        ("Forgery",   scores["forgery"],   (80, 100, 200)),
    ]
    for i, (label, score, color) in enumerate(bar_data):
        draw_score_bar(out, PX+8, PY+30+i*32, PW, 16, score, color, label)

    # Combined threat
    cv2.line(out, (PX+8, PY+162), (PX+PW+52, PY+162), (60,62,75), 1)
    comb = scores["combined"]
    cc   = TIER_COLORS.get(tier, (200,200,200))
    draw_score_bar(out, PX+8, PY+170, PW, 18, comb, cc, "COMBINED")

    # ── Right panel: gaze signals
    GX = W - 215
    GY = 65
    draw_rounded_rect(out, GX, GY, GX+205, GY+160, COLOURS["PANEL"],
                      radius=10, alpha=0.88)

    cv2.putText(out, "GAZE SIGNALS", (GX+8, GY+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140,140,160), 1)

    def sig_row(label, value_str, y, ok):
        color = (60, 200, 100) if ok else (60, 80, 230)
        icon  = "✓" if ok else "!"
        cv2.putText(out, f"{icon} {label}", (GX+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
        cv2.putText(out, value_str, (GX+125, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,210), 1)

    face_ok = gaze.get("face", False)
    off     = gaze.get("off_screen", False)
    sus     = gaze.get("sustained", 0.0)
    gh      = gaze.get("gaze_h", 0.0)
    gv      = gaze.get("gaze_v", 0.0)
    yaw     = gaze.get("yaw",    0.0)

    sig_row("Face detected",  "YES" if face_ok else "NO",  GY+38, face_ok)
    sig_row("Looking screen", "YES" if not off else "NO",  GY+62, not off)
    sig_row("Gaze H / V",  f"{gh:+.0f}° {gv:+.0f}°",     GY+86, abs(gh)<25 and abs(gv)<20)
    sig_row("Head yaw",       f"{yaw:+.0f}°",              GY+110, abs(yaw)<25)
    sig_row("Sustained",      f"{sus:.1f}s",               GY+134, sus < 3.5)
    sig_row("Eye mv",         f"{gaze.get('eye_movement',0.0):.2f}", GY+158, gaze.get('eye_movement',0.0) > 0.02)
    sig_row("Moiré",          f"{scores.get('moire',0.0):.2f}", GY+182, scores.get('moire',0.0) < 0.35)

    # Sustained gaze warning bar
    if sus > 0:
        bar_w = 185
        fill  = int(bar_w * min(sus / 5.0, 1.0))
        cv2.rectangle(out, (GX+10, GY+145), (GX+10+bar_w, GY+155), (50,52,65), -1)
        bar_col = (60,200,100) if sus < 2 else (30,130,220) if sus < 3.5 else (40,60,230)
        if fill > 0:
            cv2.rectangle(out, (GX+10, GY+145), (GX+10+fill, GY+155), bar_col, -1)

    # ── Alert reasons box (bottom)
    if reasons:
        RY = H - 55
        draw_rounded_rect(out, 10, RY, W-10, H-8, COLOURS["PANEL"],
                          radius=8, alpha=0.90)
        cv2.putText(out, "⚡ " + " | ".join(reasons[:3]), (18, RY+27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, tc, 1)

    # ── Alert flash (RED = pulsing border)
    if tier == "RED":
        pulse = abs(np.sin(time.time() * 4))
        thickness = max(1, int(pulse * 6))
        cv2.rectangle(out, (0,0), (W-1, H-1), (40,60,230), thickness)

    elif tier == "ORANGE":
        cv2.rectangle(out, (0,0), (W-1, H-1), (30,130,220), 3)

    # ── Scenario label (bottom right)
    scenario_labels = {
        "normal":    "SCENARIO: Normal",
        "deepfake":  "SCENARIO: Deepfake Attack",
        "recapture": "SCENARIO: Paper Leak",
        "gaze":      "SCENARIO: Gaze Cheating"
    }
    slabel = scenario_labels.get(scenario, "")
    cv2.putText(out, slabel, (W - 280, H - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,140), 1)

    # ── Key hints (small, bottom right corner)
    hints = ["1:Deepfake  2:Recapture  3:Gaze  R:Reset  S:Save  Q:Quit"]
    cv2.putText(out, hints[0], (10, H - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,82,95), 1)

    # ── Gaze arrow on face
    if face_ok and gaze.get("lm") is not None:
        lm  = gaze["lm"]
        nx  = int(lm[1].x * W)
        ny  = int(lm[1].y * H)
        ax  = nx + int(gh * 2.5)
        ay  = ny + int(gv * 2.5)
        cv2.arrowedLine(out, (nx,ny), (ax,ay), tc, 2, tipLength=0.35)
    # Draw eye center marker if available
    em = gaze.get('eye_movement', 0.0)
    if gaze.get('lm') is None and gaze.get('face') and hasattr(gaze, 'get'):
        # no lm available but face present: no-op (fallback draw handled elsewhere)
        pass
    if 'eye_center' in gaze and gaze['eye_center'] is not None:
        cx, cy = gaze['eye_center']
        cv2.circle(out, (int(cx), int(cy)), 4, (200,220,60), -1)

    return out


# ─── Decision logic ───────────────────────────────────────────────────────────

def compute_tier(scores, gaze_sustained, scenario):
    """Simplified decision logic for live demo."""
    reasons = []
    combined = scores["combined"]

    # Hard rules
    if scores["deepfake"] > 0.85:
        reasons.append(f"Deepfake detected ({scores['deepfake']:.0%})")
    if scores["recapture"] > 0.80:
        reasons.append(f"Screen recapture detected ({scores['recapture']:.0%})")

    # Gaze
    if gaze_sustained > 3.5:
        reasons.append(f"Sustained off-screen gaze: {gaze_sustained:.1f}s")

    # Tier
    if combined > 0.91 or (scores["deepfake"] > 0.91) or (scores["recapture"] > 0.91):
        return "RED", reasons
    if combined > 0.82 or scores["deepfake"] > 0.85 or scores["recapture"] > 0.80:
        return "ORANGE", reasons
    if combined > 0.72 or gaze_sustained > 3.5:
        return "YELLOW", reasons
    return "GREEN", reasons


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print(" ExamSentinel — Live Demo (Webcam)")
    print("="*60)
    print(" Controls:")
    print("   1 → Deepfake attack scenario")
    print("   2 → Paper leak / recapture scenario")
    print("   3 → Gaze cheating scenario")
    print("   R → Reset to normal (GREEN)")
    print("   S → Save screenshot")
    print("   Q → Quit")
    print("="*60 + "\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        print("  → Make sure no other app is using the camera.")
        print("  → Try: python scripts/live_demo.py  (from project root)")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    gaze_tracker = SimpleGazeTracker()
    simulator    = ScoreSimulator()

    Path("logs").mkdir(exist_ok=True)
    Path("screenshots").mkdir(exist_ok=True)

    scenario    = "normal"
    fps_times   = []
    frame_count = 0
    screenshot_n = 0

    print("Webcam running. Press keys to trigger scenarios.")
    print("Record your screen with OBS or Windows Game Bar (Win+G).\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Lost webcam feed.")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural feel
        frame_count += 1

        # FPS
        now = time.time()
        fps_times.append(now)
        fps_times = [t for t in fps_times if now - t < 2.0]
        fps = len(fps_times) / 2.0

        # Gaze
        gaze = gaze_tracker.process(frame)
        # Moiré diagnostic (raw score) and Scores
        moire = simulator._compute_moire(frame)
        recap_metric = simulator.compute_recapture_metric(frame)
        scores = simulator.update(
            frame,
            gaze.get("off_screen", False),
            gaze.get("sustained", 0.0)
        )
        
        # Grace period: suppress detections for 2s after startup or reset
        time_since_reset = time.time() - getattr(simulator, 'last_reset', 0.0)
        in_grace_period = time_since_reset < 2.0
        
        # Only boost recapture confidence if explicitly in recapture scenario (user pressed 2)
        # Don't boost in normal scenario, even if moiré is detected
        if (scenario == "recapture" and recap_metric > 0.05 and 
            time_since_reset > 0.5 and not in_grace_period):
            scores["recapture"] = max(scores.get("recapture", 0.0), float(np.clip(recap_metric * 1.25, 0.0, 1.0)))
        
        # During grace period, suppress sensitive detection fields
        if in_grace_period:
            scores["recapture"] = 0.0
            scores["combined"] = 0.0
            scores["moire"] = 0.0
        else:
            # Only set moire/recap diagnostics after grace period
            scores["moire"] = float(moire)
            scores["recapture_metric"] = float(recap_metric)

        # Tier
        tier, reasons = compute_tier(scores, gaze.get("sustained", 0.0), scenario)

        # Render
        display = render_overlay(frame, scores, gaze, tier, reasons, scenario, fps)

        cv2.imshow("ExamSentinel — Live Demo  [Press 1/2/3/R/S/Q]", display)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Exiting...")
            break
        elif key == ord('1'):
            scenario = "deepfake"
            simulator.set_scenario("deepfake")
            print("→ Deepfake attack scenario started")
        elif key == ord('2'):
            scenario = "recapture"
            simulator.set_scenario("recapture")
            print("→ Paper leak / recapture scenario started")
            print("  TIP: Hold your phone screen up to the webcam for extra effect!")
        elif key == ord('3'):
            scenario = "gaze"
            simulator.set_scenario("gaze")
            print("→ Gaze cheating scenario started — look away from the screen!")
        elif key == ord('r') or key == ord('R'):
            scenario = "normal"
            simulator.set_scenario("normal")
            print("→ Reset to normal")
        elif key == ord('s') or key == ord('S'):
            fname = f"screenshots/examsentinel_{int(time.time())}.jpg"
            cv2.imwrite(fname, display)
            screenshot_n += 1
            print(f"→ Screenshot saved: {fname}")

    cap.release()
    gaze_tracker.close()
    cv2.destroyAllWindows()
    print(f"\nDemo ended. {screenshot_n} screenshots saved to screenshots/")


if __name__ == "__main__":
    main()
