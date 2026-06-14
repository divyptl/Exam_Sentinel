"""
agents/decision_engine.py

AgenticDecisionEngine — autonomous 4-tier RL triage.
GREEN → YELLOW → ORANGE → RED without human intervention.
"""

import time, json, math, collections
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import numpy as np


class AlertTier(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED    = "RED"

    @property
    def level(self):
        return {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}[self.value]

    @property
    def color_hex(self):
        return {"GREEN":"#1D9E75","YELLOW":"#BA7517","ORANGE":"#D85A30","RED":"#E24B4A"}[self.value]


@dataclass
class DecisionRecord:
    timestamp: float = field(default_factory=time.time)
    tier:      str   = "GREEN"
    reasons:   List[str] = field(default_factory=list)
    action:    str   = "none"
    confidence: float = 0.0
    scores:    Dict  = field(default_factory=dict)
    feedback:  Optional[str] = None

    def to_json(self):
        return json.dumps({"timestamp": self.timestamp, "tier": self.tier,
                           "reasons": self.reasons, "action": self.action,
                           "confidence": self.confidence, "scores": self.scores,
                           "feedback": self.feedback})


SIGNAL_WEIGHTS = {
    "score_deepfake": 1.8, "score_recapture": 2.0,
    "score_splicing": 1.2, "score_forgery":   1.0,
    "gaze_sustained": 1.5, "no_face":          1.4,
    "mouth_open":     0.6,
}


class AgenticDecisionEngine:
    """
    Autonomous exam integrity decision engine.
    Observes inference signals → reasons → acts → learns from feedback.
    """
    def __init__(self, config: Dict, log_path: str = "logs/decisions.jsonl",
                 qtable_path: str = "models/qtable.npy"):
        dec = config.get("decision_engine", {})
        self.threshold_yellow = dec.get("yellow_threshold", 0.72)
        self.threshold_orange = dec.get("orange_threshold", 0.82)
        self.threshold_red    = dec.get("red_threshold",    0.91)
        self.sustained_limit  = dec.get("sustained_gaze_seconds", 3.5)
        self.consec_for_red   = dec.get("consecutive_flags_for_red", 3)

        self.session_start      = time.time()
        self.current_tier       = AlertTier.GREEN
        self.consecutive_orange = 0
        self.total_flags        = 0
        self.history: collections.deque = collections.deque(maxlen=20)
        self.decision_log: List[DecisionRecord] = []
        self._action_callbacks: Dict = {}

        self.q_table    = self._init_qtable(qtable_path)
        self.qtable_path = qtable_path
        self.lr, self.gamma = 0.1, 0.9

        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"[DecisionEngine] Initialised")
        print(f"  Thresholds: YELLOW>{self.threshold_yellow:.2f} "
              f"ORANGE>{self.threshold_orange:.2f} RED>{self.threshold_red:.2f}")

    def decide(self, result) -> DecisionRecord:
        threat, sub_scores, reasons = self._compute_threat(result)
        self.history.append(threat)
        smoothed = float(np.mean(list(self.history)[-3:])) if len(self.history) >= 3 else threat

        state     = self._encode_state(smoothed)
        action    = self._select_action(state)
        tier      = self._action_to_tier(action, smoothed)
        tier      = self._apply_hard_rules(tier, result, reasons)

        if tier == AlertTier.ORANGE:
            self.consecutive_orange += 1
        elif tier == AlertTier.GREEN:
            self.consecutive_orange = 0

        if self.consecutive_orange >= self.consec_for_red:
            tier = AlertTier.RED
            reasons.append(f"Auto-escalated: {self.consecutive_orange} consecutive ORANGE")

        if tier.level > 0:
            self.total_flags += 1
        self.current_tier = tier

        action_str = self._execute_action(tier, reasons)
        record = DecisionRecord(tier=tier.value, reasons=reasons, action=action_str,
                                confidence=smoothed, scores=sub_scores)
        self.decision_log.append(record)
        self._log(record)

        result.alert_tier   = tier.value
        result.alert_reason = " | ".join(reasons)
        result.action_taken = action_str
        return record

    def _compute_threat(self, result):
        reasons, sub_scores = [], {}
        weighted_sum, weight_total = 0.0, 0.0

        for key, weight in [("score_deepfake", 1.8), ("score_recapture", 2.0),
                             ("score_splicing", 1.2), ("score_forgery", 1.0)]:
            score = max(0.0, float(getattr(result, key, 0.0) or 0.0))
            sub_scores[key] = round(score, 3)
            weighted_sum  += score * weight
            weight_total  += weight
            if score > 0.85:
                reasons.append(f"HIGH {key.replace('score_','')} ({score:.2f})")
            elif score > 0.70:
                reasons.append(f"ELEVATED {key.replace('score_','')} ({score:.2f})")

        sustained = float(getattr(result, "sustained_offscreen_sec", 0.0) or 0.0)
        if sustained > self.sustained_limit:
            gs = min(sustained / (self.sustained_limit * 2), 1.0)
            weighted_sum += gs * 1.5
            weight_total += 1.5
            sub_scores["gaze_sustained"] = round(gs, 3)
            reasons.append(f"Sustained off-screen gaze: {sustained:.1f}s")

        if not getattr(result, "face_detected", True):
            weighted_sum += 0.7 * 1.4
            weight_total += 1.4
            sub_scores["no_face"] = 0.7
            reasons.append("Face not detected")

        if getattr(result, "is_mouth_open", False):
            weighted_sum += 0.4 * 0.6
            weight_total += 0.6

        threat = min(weighted_sum / weight_total, 1.0) if weight_total > 0 else 0.0
        return threat, sub_scores, reasons

    def _apply_hard_rules(self, tier, result, reasons):
        if getattr(result, "score_deepfake", 0) > 0.90 and tier.level < AlertTier.ORANGE.level:
            tier = AlertTier.ORANGE
            reasons.append("Hard rule: deepfake > 0.90")
        if getattr(result, "score_recapture", 0) > 0.88 and tier.level < AlertTier.ORANGE.level:
            tier = AlertTier.ORANGE
            reasons.append("Hard rule: recapture > 0.88")
        if (getattr(result, "sustained_offscreen_sec", 0) > 10.0 and
                not getattr(result, "face_detected", True)):
            tier = AlertTier.RED
            reasons.append("Hard rule: face absent > 10s")
        return tier

    def _init_qtable(self, path):
        if Path(path).exists():
            try:
                return np.load(path)
            except Exception:
                pass
        q = np.zeros((5, 4, 3, 4))
        q[:, :, :, 0] = 0.2
        q[:, :, :, 1] = 0.1
        return q

    def _encode_state(self, threat):
        tb = min(int(threat * 5), 4)
        cb = min(self.consecutive_orange, 3)
        elapsed = time.time() - self.session_start
        tmb = min(int(elapsed / 1800), 2)
        return tb, cb, tmb

    def _select_action(self, state):
        n = len(self.decision_log)
        eps = max(0.05, 0.3 * math.exp(-n / 50))
        if np.random.random() < eps:
            return np.random.randint(4)
        return int(np.argmax(self.q_table[state]))

    def _action_to_tier(self, action, threat):
        if threat > self.threshold_red:    return AlertTier.RED
        if threat > self.threshold_orange and action < 2: action = 2
        if threat > self.threshold_yellow and action < 1: action = 1
        return [AlertTier.GREEN, AlertTier.YELLOW, AlertTier.ORANGE, AlertTier.RED][action]

    def _execute_action(self, tier, reasons):
        actions = {
            AlertTier.GREEN:  "clear",
            AlertTier.YELLOW: "warn_student",
            AlertTier.ORANGE: "flag_session",
            AlertTier.RED:    "suspend_session"
        }
        action = actions[tier]
        self._trigger_callback(action, {"tier": tier.value, "reasons": reasons})
        return action

    def receive_feedback(self, record_idx, feedback):
        if record_idx >= len(self.decision_log):
            return
        record = self.decision_log[record_idx]
        record.feedback = feedback
        reward = {"confirmed": +2.0, "false_positive": -1.5}.get(feedback, 0.0)
        state  = self._encode_state(record.confidence)
        action = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}.get(record.tier, 0)
        cq = self.q_table[state][action]
        self.q_table[state][action] = cq + self.lr * (
            reward + self.gamma * np.max(self.q_table[state]) - cq
        )
        Path(self.qtable_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(self.qtable_path, self.q_table)
        print(f"[DecisionEngine] Q-table updated (feedback={feedback}, reward={reward})")

    def register_callback(self, action, fn):
        self._action_callbacks[action] = fn

    def _trigger_callback(self, action, data):
        if action in self._action_callbacks:
            try:
                self._action_callbacks[action](data)
            except Exception as e:
                print(f"[DecisionEngine] Callback error: {e}")

    def _log(self, record):
        with open(self.log_path, "a") as f:
            f.write(record.to_json() + "\n")

    def get_session_stats(self):
        elapsed = time.time() - self.session_start
        counts  = {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0}
        for r in self.decision_log:
            counts[r.tier] = counts.get(r.tier, 0) + 1
        return {
            "session_duration_min": elapsed / 60.0,
            "total_decisions": len(self.decision_log),
            "tier_counts": counts,
            "current_tier": self.current_tier.value,
            "consecutive_orange": self.consecutive_orange,
            "total_flags": self.total_flags,
            "threat_level": float(np.mean(list(self.history)[-5:])) if self.history else 0.0
        }


if __name__ == "__main__":
    import yaml
    from core.inference_engine import InferenceResult
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    engine = AgenticDecisionEngine(cfg)
    scenarios = [
        ("Normal",          dict(score_deepfake=0.05, score_recapture=0.03, score_splicing=0.02, score_forgery=0.02, score_combined=0.05, face_detected=True, is_off_screen=False, sustained_offscreen_sec=0.0, is_mouth_open=False, blink_rate=0.3)),
        ("Elevated gaze",   dict(score_deepfake=0.08, score_recapture=0.05, score_splicing=0.03, score_forgery=0.02, score_combined=0.20, face_detected=True, is_off_screen=True,  sustained_offscreen_sec=4.5, is_mouth_open=False, blink_rate=0.2)),
        ("Deepfake attack", dict(score_deepfake=0.92, score_recapture=0.07, score_splicing=0.12, score_forgery=0.10, score_combined=0.75, face_detected=True, is_off_screen=False, sustained_offscreen_sec=0.0, is_mouth_open=False, blink_rate=0.25)),
        ("Paper leak",      dict(score_deepfake=0.12, score_recapture=0.93, score_splicing=0.08, score_forgery=0.06, score_combined=0.85, face_detected=True, is_off_screen=True,  sustained_offscreen_sec=2.0, is_mouth_open=False, blink_rate=0.15)),
    ]
    for name, attrs in scenarios:
        r = InferenceResult(**attrs)
        rec = engine.decide(r)
        print(f"\n{name}: tier={rec.tier}  action={rec.action}")
        if rec.reasons:
            for reason in rec.reasons:
                print(f"  ⚡ {reason}")
    print("\nSession stats:")
    for k, v in engine.get_session_stats().items():
        print(f"  {k}: {v}")
    print("\n✓ Decision engine OK")
