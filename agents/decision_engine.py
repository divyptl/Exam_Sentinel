"""
agents/decision_engine.py  — Day 4

ExamSentinel Agentic Decision Engine

The most novel component: an RL-inspired autonomous triage system that
decides what action to take without human intervention.

4-tier alert system:
  GREEN  → All clear. No action.
  YELLOW → Soft warning to student. "Please look at the screen."
  ORANGE → Session flagged. Logged for review. Student warned sternly.
  RED    → Session auto-suspended. Admin notified. Exam paused.

State machine: tracks consecutive flags across windows,
applies temporal smoothing, and learns from false positive feedback.

Why "agentic":
  - Observes (inference scores + gaze signals)
  - Reasons (multi-signal fusion with confidence weighting)
  - Acts (autonomous actions without human in the loop)
  - Learns (Q-table updates based on admin feedback)

This is the distinction from all existing proctoring systems which
only flag — never autonomously act.
"""

import time
import json
import math
import collections
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import numpy as np


# ─── Alert Tiers ──────────────────────────────────────────────────────────────

class AlertTier(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED    = "RED"

    @property
    def level(self) -> int:
        return {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}[self.value]

    @property
    def color_hex(self) -> str:
        return {
            "GREEN": "#1D9E75",
            "YELLOW": "#BA7517",
            "ORANGE": "#D85A30",
            "RED": "#E24B4A"
        }[self.value]


@dataclass
class DecisionRecord:
    """Record of a single decision made by the engine."""
    timestamp: float = field(default_factory=time.time)
    tier: str = "GREEN"
    reasons: List[str] = field(default_factory=list)
    action: str = "none"
    confidence: float = 0.0
    scores: Dict = field(default_factory=dict)
    feedback: Optional[str] = None  # 'confirmed', 'false_positive', None

    def to_json(self) -> str:
        return json.dumps({
            "timestamp": self.timestamp,
            "tier": self.tier,
            "reasons": self.reasons,
            "action": self.action,
            "confidence": self.confidence,
            "scores": self.scores,
            "feedback": self.feedback
        })


# ─── Signal Weights ───────────────────────────────────────────────────────────

DEFAULT_SIGNAL_WEIGHTS = {
    "score_deepfake":  1.8,   # High weight — identity fraud
    "score_recapture": 2.0,   # Highest — paper leak is catastrophic
    "score_splicing":  1.2,
    "score_forgery":   1.0,
    "gaze_sustained":  1.5,   # Off-screen gaze
    "mouth_open":      0.6,   # Whispering (supporting signal)
    "no_face":         1.4,   # Face disappeared
}


# ─── Decision Engine ──────────────────────────────────────────────────────────

class AgenticDecisionEngine:
    """
    Autonomous, stateful decision engine for exam integrity.

    Maintains a sliding window of recent inference results,
    computes a composite threat score, and autonomously selects
    and executes actions — no human required.

    RL component:
      - State: (threat_score_bin, consecutive_flags, time_in_exam)
      - Actions: warn / flag / suspend / clear
      - Q-table updated from admin feedback (confirm/false_positive)
    """

    def __init__(self, config: Dict,
                 log_path: str = "logs/decisions.jsonl",
                 qtable_path: str = "models/qtable.npy"):

        dec = config.get("decision_engine", {})

        # Thresholds
        self.threshold_yellow = dec.get("yellow_threshold", 0.72)
        self.threshold_orange = dec.get("orange_threshold", 0.82)
        self.threshold_red    = dec.get("red_threshold",    0.91)
        self.sustained_limit  = dec.get("sustained_gaze_seconds", 3.5)
        self.consec_for_red   = dec.get("consecutive_flags_for_red", 3)

        # State
        self.session_start     = time.time()
        self.current_tier      = AlertTier.GREEN
        self.consecutive_orange = 0
        self.total_flags        = 0
        self.history: collections.deque = collections.deque(maxlen=20)
        self.decision_log: List[DecisionRecord] = []

        # RL Q-table: [threat_bin(5), consec_flags(4), time_bin(3)] → Q-values[4 actions]
        self.n_threat_bins = 5
        self.n_consec_bins = 4
        self.n_time_bins   = 3
        self.n_actions     = 4  # 0=clear, 1=warn, 2=flag, 3=suspend
        self.q_table       = self._init_qtable(qtable_path)
        self.qtable_path   = qtable_path
        self.learning_rate = 0.1
        self.discount      = 0.9

        # Action callbacks (set by dashboard/session manager)
        self._action_callbacks: Dict[str, callable] = {}

        # Logs
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"[DecisionEngine] Initialised")
        print(f"  Thresholds: YELLOW>{self.threshold_yellow:.2f} "
              f"ORANGE>{self.threshold_orange:.2f} RED>{self.threshold_red:.2f}")

    # ── Core decision method ──────────────────────────────────────────────────

    def decide(self, result) -> DecisionRecord:
        """
        Given an InferenceResult, compute composite threat score,
        select alert tier, and execute action.

        This is the agent's 'step' function.
        """
        threat_score, sub_scores, reasons = self._compute_threat(result)

        # Temporal smoothing: use moving average of last 3 threat scores
        self.history.append(threat_score)
        smoothed = np.mean(list(self.history)[-3:]) if len(self.history) >= 3 else threat_score

        # ── RL action selection
        state = self._encode_state(smoothed)
        action_idx = self._select_action(state)
        tier = self._action_to_tier(action_idx, smoothed)

        # Override with hard rules (safety net)
        tier = self._apply_hard_rules(tier, result, reasons)

        # Track consecutive ORANGE
        if tier == AlertTier.ORANGE:
            self.consecutive_orange += 1
        elif tier == AlertTier.GREEN:
            self.consecutive_orange = 0

        # Auto-escalate to RED after N consecutive ORANGE
        if self.consecutive_orange >= self.consec_for_red:
            tier = AlertTier.RED
            reasons.append(f"Auto-escalated: {self.consecutive_orange} consecutive ORANGE flags")

        if tier.level > 0:
            self.total_flags += 1

        self.current_tier = tier

        # ── Execute action
        action_taken = self._execute_action(tier, reasons)

        # ── Build record
        record = DecisionRecord(
            tier=tier.value,
            reasons=reasons,
            action=action_taken,
            confidence=smoothed,
            scores=sub_scores
        )
        self.decision_log.append(record)
        self._log(record)

        # Update result object
        result.alert_tier = tier.value
        result.alert_reason = " | ".join(reasons)
        result.action_taken = action_taken

        return record

    # ── Threat computation ────────────────────────────────────────────────────

    def _compute_threat(self, result) -> Tuple[float, Dict, List[str]]:
        """
        Fuse all signals into a single threat score ∈ [0, 1].

        Uses weighted sum with signal-specific normalisation.
        """
        reasons = []
        sub_scores = {}
        weighted_sum = 0.0
        weight_total = 0.0

        # ── Model scores
        for score_key, weight in [
            ("score_deepfake",  DEFAULT_SIGNAL_WEIGHTS["score_deepfake"]),
            ("score_recapture", DEFAULT_SIGNAL_WEIGHTS["score_recapture"]),
            ("score_splicing",  DEFAULT_SIGNAL_WEIGHTS["score_splicing"]),
            ("score_forgery",   DEFAULT_SIGNAL_WEIGHTS["score_forgery"]),
        ]:
            score = getattr(result, score_key, 0.0)
            if score is None:
                score = 0.0
            sub_scores[score_key] = round(score, 3)
            weighted_sum += score * weight
            weight_total += weight

            # Reason tagging
            if score > 0.85:
                reasons.append(f"HIGH {score_key.replace('score_','')} ({score:.2f})")
            elif score > 0.70:
                reasons.append(f"ELEVATED {score_key.replace('score_','')} ({score:.2f})")

        # ── Gaze signals
        sustained = getattr(result, "sustained_offscreen_sec", 0.0) or 0.0
        if sustained > self.sustained_limit:
            # Map 0…limit*2 → 0…1
            gaze_score = min(sustained / (self.sustained_limit * 2), 1.0)
            weighted_sum += gaze_score * DEFAULT_SIGNAL_WEIGHTS["gaze_sustained"]
            weight_total += DEFAULT_SIGNAL_WEIGHTS["gaze_sustained"]
            sub_scores["gaze_sustained"] = round(gaze_score, 3)
            reasons.append(f"Sustained off-screen gaze: {sustained:.1f}s")

        # ── Face not detected
        if not getattr(result, "face_detected", True):
            weighted_sum += 0.7 * DEFAULT_SIGNAL_WEIGHTS["no_face"]
            weight_total += DEFAULT_SIGNAL_WEIGHTS["no_face"]
            sub_scores["no_face"] = 0.7
            reasons.append("Face not detected")

        # ── Mouth movement (supporting signal)
        if getattr(result, "is_mouth_open", False):
            weighted_sum += 0.4 * DEFAULT_SIGNAL_WEIGHTS["mouth_open"]
            weight_total += DEFAULT_SIGNAL_WEIGHTS["mouth_open"]
            sub_scores["mouth_open"] = 0.4

        # Normalise
        if weight_total > 0:
            threat_score = min(weighted_sum / weight_total, 1.0)
        else:
            threat_score = 0.0

        return threat_score, sub_scores, reasons

    # ── Hard rules (safety net on top of RL) ─────────────────────────────────

    def _apply_hard_rules(self, tier: AlertTier, result,
                           reasons: List[str]) -> AlertTier:
        """
        Deterministic overrides for clear-cut violations.
        These bypass the RL model to guarantee detection.
        """
        # Deepfake score very high → at least ORANGE
        if getattr(result, "score_deepfake", 0) > 0.90:
            if tier.level < AlertTier.ORANGE.level:
                tier = AlertTier.ORANGE
                reasons.append("Hard rule: deepfake score > 0.90")

        # Recapture score very high → at least ORANGE (paper leak)
        if getattr(result, "score_recapture", 0) > 0.88:
            if tier.level < AlertTier.ORANGE.level:
                tier = AlertTier.ORANGE
                reasons.append("Hard rule: recapture score > 0.88")

        # Face gone > 10 seconds → RED
        if getattr(result, "sustained_offscreen_sec", 0) > 10.0 and not getattr(result, "face_detected", True):
            tier = AlertTier.RED
            reasons.append("Hard rule: face absent > 10s")

        return tier

    # ── RL components ─────────────────────────────────────────────────────────

    def _init_qtable(self, path: str) -> np.ndarray:
        """Load or initialise Q-table."""
        if Path(path).exists():
            try:
                q = np.load(path)
                print(f"[DecisionEngine] Loaded Q-table from {path}")
                return q
            except Exception:
                pass

        # Initialise with conservative prior (prefer lower-tier actions)
        q = np.zeros((self.n_threat_bins, self.n_consec_bins,
                      self.n_time_bins, self.n_actions))
        # Prior: prefer warn over flag/suspend for low threat
        q[:, :, :, 1] = 0.1  # warn
        q[:, :, :, 0] = 0.2  # clear (slight preference for no false positives)
        return q

    def _encode_state(self, threat_score: float) -> Tuple[int, int, int]:
        """Encode current state as discrete indices."""
        threat_bin = min(int(threat_score * self.n_threat_bins),
                         self.n_threat_bins - 1)
        consec_bin = min(self.consecutive_orange, self.n_consec_bins - 1)
        time_elapsed = time.time() - self.session_start
        time_bin = min(int(time_elapsed / 1800), self.n_time_bins - 1)  # 30min bins
        return threat_bin, consec_bin, time_bin

    def _select_action(self, state: Tuple[int, int, int]) -> int:
        """
        ε-greedy action selection.
        ε decays over session (more confident as more data collected).
        """
        n_decisions = len(self.decision_log)
        epsilon = max(0.05, 0.3 * math.exp(-n_decisions / 50))

        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action
            return int(np.argmax(self.q_table[state]))

    def _action_to_tier(self, action_idx: int,
                         threat_score: float) -> AlertTier:
        """Map RL action index to alert tier, bounded by threat score."""
        # Hard lower bound: if threat is high, can't select GREEN/YELLOW
        if threat_score > self.threshold_red:
            return AlertTier.RED
        if threat_score > self.threshold_orange and action_idx < 2:
            action_idx = 2
        if threat_score > self.threshold_yellow and action_idx < 1:
            action_idx = 1

        return [AlertTier.GREEN, AlertTier.YELLOW,
                AlertTier.ORANGE, AlertTier.RED][action_idx]

    def receive_feedback(self, record_idx: int, feedback: str):
        """
        Process admin feedback ('confirmed' or 'false_positive').
        Updates Q-table to improve future decisions.
        """
        if record_idx >= len(self.decision_log):
            return

        record = self.decision_log[record_idx]
        record.feedback = feedback

        # Compute reward
        reward = {
            "confirmed":      +2.0,   # Correct detection
            "false_positive": -1.5,   # Penalise false alarm
        }.get(feedback, 0.0)

        # Q-table update (simplified Bellman equation)
        state = self._encode_state(record.confidence)
        action_map = {
            "GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3
        }
        action = action_map.get(record.tier, 0)

        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[state])
        new_q = current_q + self.learning_rate * (
            reward + self.discount * max_next_q - current_q
        )
        self.q_table[state][action] = new_q

        # Save updated Q-table
        Path(self.qtable_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(self.qtable_path, self.q_table)
        print(f"[DecisionEngine] Q-table updated (feedback={feedback}, reward={reward:.1f})")

    # ── Action execution ──────────────────────────────────────────────────────

    def _execute_action(self, tier: AlertTier, reasons: List[str]) -> str:
        """
        Execute the selected action.
        Calls registered callbacks (e.g., dashboard notification, session suspend).
        """
        action = "none"

        if tier == AlertTier.GREEN:
            action = "clear"

        elif tier == AlertTier.YELLOW:
            action = "warn_student"
            self._trigger_callback("warn_student", {
                "message": "Please keep your eyes on the screen.",
                "tier": "YELLOW"
            })

        elif tier == AlertTier.ORANGE:
            action = "flag_session"
            self._trigger_callback("flag_session", {
                "message": "Suspicious activity detected. Session flagged.",
                "reasons": reasons,
                "tier": "ORANGE"
            })

        elif tier == AlertTier.RED:
            action = "suspend_session"
            self._trigger_callback("suspend_session", {
                "message": "Session automatically suspended due to detected violation.",
                "reasons": reasons,
                "tier": "RED",
                "timestamp": time.time()
            })

        return action

    def register_callback(self, action: str, fn: callable):
        """Register a callback for an action (e.g., 'suspend_session')."""
        self._action_callbacks[action] = fn

    def _trigger_callback(self, action: str, data: Dict):
        if action in self._action_callbacks:
            try:
                self._action_callbacks[action](data)
            except Exception as e:
                print(f"[DecisionEngine] Callback error ({action}): {e}")

    def _log(self, record: DecisionRecord):
        with open(self.log_path, "a") as f:
            f.write(record.to_json() + "\n")

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_session_stats(self) -> Dict:
        elapsed = time.time() - self.session_start
        tier_counts = {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0}
        for r in self.decision_log:
            tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

        return {
            "session_duration_min": elapsed / 60.0,
            "total_decisions": len(self.decision_log),
            "tier_counts": tier_counts,
            "current_tier": self.current_tier.value,
            "consecutive_orange": self.consecutive_orange,
            "total_flags": self.total_flags,
            "threat_level": float(np.mean(list(self.history)[-5:]))
                            if self.history else 0.0
        }


if __name__ == "__main__":
    import yaml
    from core.inference_engine import InferenceResult

    print("Testing AgenticDecisionEngine...")
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    engine = AgenticDecisionEngine(cfg)

    # Simulate escalating threat
    scenarios = [
        ("Normal",           dict(score_deepfake=0.05, score_recapture=0.03, face_detected=True, is_off_screen=False, sustained_offscreen_sec=0.0, score_combined=0.05, score_splicing=0.02, score_forgery=0.02, is_mouth_open=False, blink_rate=0.3)),
        ("Elevated gaze",    dict(score_deepfake=0.08, score_recapture=0.05, face_detected=True, is_off_screen=True,  sustained_offscreen_sec=4.5, score_combined=0.20, score_splicing=0.03, score_forgery=0.02, is_mouth_open=False, blink_rate=0.2)),
        ("Deepfake attack",  dict(score_deepfake=0.88, score_recapture=0.07, face_detected=True, is_off_screen=False, sustained_offscreen_sec=0.0, score_combined=0.75, score_splicing=0.12, score_forgery=0.10, is_mouth_open=False, blink_rate=0.25)),
        ("Paper leak",       dict(score_deepfake=0.12, score_recapture=0.93, face_detected=True, is_off_screen=True,  sustained_offscreen_sec=2.0, score_combined=0.85, score_splicing=0.08, score_forgery=0.06, is_mouth_open=False, blink_rate=0.15)),
    ]

    for name, attrs in scenarios:
        r = InferenceResult(**attrs)
        record = engine.decide(r)
        print(f"\n{name}:")
        print(f"  Tier:    {record.tier}")
        print(f"  Action:  {record.action}")
        print(f"  Reasons: {record.reasons}")
        print(f"  Scores:  {record.scores}")

    print("\n\nSession stats:")
    for k, v in engine.get_session_stats().items():
        print(f"  {k}: {v}")

    print("\n✓ Decision engine OK")
