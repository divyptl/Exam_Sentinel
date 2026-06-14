"""
dashboard/app.py  — Day 5

ExamSentinel Live Monitoring Dashboard (Streamlit)

Displays:
  - Live webcam feed with gaze overlays
  - Real-time threat scores per detection head
  - Alert tier indicator (GREEN/YELLOW/ORANGE/RED)
  - Event timeline
  - Session statistics
  - Admin feedback panel (feeds Q-table updates)

Run:
    streamlit run dashboard/app.py
    streamlit run dashboard/app.py -- --mock deepfake_attack
"""

import sys
import time
import json
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.inference_engine import MockInferenceEngine, InferenceResult
from agents.decision_engine import AgenticDecisionEngine, AlertTier
import yaml


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ExamSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background: #0d0f14; }

    .metric-card {
        background: #161a23;
        border: 1px solid #2a2f3d;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .metric-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.08em;
        color: #6b7280;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #f9fafb;
    }

    .tier-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 24px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .tier-GREEN  { background: #0d2e1f; color: #1D9E75; border: 1px solid #1D9E75; }
    .tier-YELLOW { background: #2a1e06; color: #BA7517; border: 1px solid #BA7517; }
    .tier-ORANGE { background: #2e1508; color: #D85A30; border: 1px solid #D85A30; }
    .tier-RED    { background: #2e0909; color: #E24B4A; border: 1px solid #E24B4A; }

    .event-row {
        background: #161a23;
        border-left: 3px solid #2a2f3d;
        border-radius: 0 8px 8px 0;
        padding: 8px 14px;
        margin-bottom: 6px;
        font-size: 12px;
        color: #d1d5db;
    }
    .event-row.YELLOW { border-left-color: #BA7517; }
    .event-row.ORANGE { border-left-color: #D85A30; }
    .event-row.RED    { border-left-color: #E24B4A; }

    .header-bar {
        background: linear-gradient(90deg, #0f1419 0%, #161a23 100%);
        border-bottom: 1px solid #2a2f3d;
        padding: 12px 24px;
        margin-bottom: 20px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session state initialisation ────────────────────────────────────────────

def init_session_state():
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "decision_engine" not in st.session_state:
        st.session_state.decision_engine = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "events" not in st.session_state:
        st.session_state.events = []
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    if "scenario" not in st.session_state:
        st.session_state.scenario = "normal"
    if "config" not in st.session_state:
        with open("configs/config.yaml") as f:
            st.session_state.config = yaml.safe_load(f)


# ─── Gauge chart helper ───────────────────────────────────────────────────────

def make_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 20, "color": "#f9fafb"}},
        title={"text": title, "font": {"size": 12, "color": "#9ca3af"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#374151",
                     "tickwidth": 1, "tickfont": {"color": "#6b7280", "size": 9}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1f2937",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 65], "color": "#111827"},
                {"range": [65, 80], "color": "#1c1a10"},
                {"range": [80, 100], "color": "#1c0f0f"}
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": value * 100
            }
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#161a23",
        plot_bgcolor="#161a23",
        font={"family": "Inter"}
    )
    return fig


# ─── Timeline chart ───────────────────────────────────────────────────────────

def make_timeline(history: List[Dict]) -> go.Figure:
    if not history:
        fig = go.Figure()
        fig.update_layout(
            height=200,
            paper_bgcolor="#161a23",
            plot_bgcolor="#161a23",
            title={"text": "Threat Score Timeline", "font": {"color": "#9ca3af"}}
        )
        return fig

    df = pd.DataFrame(history)
    fig = go.Figure()

    # Threat score line
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df.get("score_combined", [0] * len(df)),
        name="Combined Threat",
        line=dict(color="#7F77DD", width=2),
        fill="tozeroy",
        fillcolor="rgba(127,119,221,0.1)"
    ))

    # Individual head scores
    for col, color, name in [
        ("score_deepfake",  "#E24B4A", "Deepfake"),
        ("score_recapture", "#D85A30", "Recapture"),
        ("score_splicing",  "#BA7517", "Splicing"),
        ("score_forgery",   "#378ADD", "Forgery"),
    ]:
        if col in df:
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df[col],
                name=name,
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.7
            ))

    # Threshold lines
    thresholds = [
        (0.72, "#BA7517", "YELLOW"),
        (0.82, "#D85A30", "ORANGE"),
        (0.91, "#E24B4A", "RED"),
    ]
    for val, color, label in thresholds:
        fig.add_hline(y=val, line_dash="dash", line_color=color,
                      line_width=1, opacity=0.5,
                      annotation_text=label,
                      annotation_font_color=color,
                      annotation_font_size=10)

    fig.update_layout(
        height=240,
        paper_bgcolor="#161a23",
        plot_bgcolor="#161a23",
        font={"family": "Inter", "color": "#9ca3af"},
        legend=dict(
            bgcolor="#1f2937",
            bordercolor="#374151",
            font=dict(size=10, color="#d1d5db")
        ),
        xaxis=dict(showgrid=False, zeroline=False, color="#374151",
                   title="", tickfont=dict(color="#6b7280")),
        yaxis=dict(showgrid=True, gridcolor="#1f2937", zeroline=False,
                   range=[0, 1], color="#374151",
                   tickfont=dict(color="#6b7280")),
        margin=dict(t=20, b=30, l=40, r=20),
        title=dict(text="Live Threat Score History",
                   font=dict(color="#9ca3af", size=12))
    )
    return fig


# ─── Tier badge HTML ──────────────────────────────────────────────────────────

def tier_badge(tier: str) -> str:
    return f'<span class="tier-badge tier-{tier}">{tier}</span>'


# ─── Main dashboard ───────────────────────────────────────────────────────────

def main():
    init_session_state()

    # ── Header
    st.markdown("""
    <div class="header-bar">
        <span style="font-size:22px; font-weight:600; color:#f9fafb;">
            🛡️ ExamSentinel
        </span>
        <span style="font-size:13px; color:#6b7280; margin-left:12px;">
            Agentic AI Exam Integrity System
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Session Control")

        scenario = st.selectbox(
            "Demo Scenario",
            options=["normal", "deepfake_attack", "recapture_attempt", "gaze_cheat"],
            format_func=lambda x: {
                "normal": "🟢 Normal Exam",
                "deepfake_attack": "🔴 Deepfake Attack",
                "recapture_attempt": "🟠 Paper Leak Attempt",
                "gaze_cheat": "🟡 Gaze Cheating"
            }[x]
        )

        use_real_camera = st.checkbox("Use Real Camera", value=False)

        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶ Start", use_container_width=True,
                                  type="primary")
        with col2:
            stop_btn = st.button("⏹ Stop", use_container_width=True)

        if start_btn:
            st.session_state.scenario = scenario
            st.session_state.session_active = True
            st.session_state.history = []
            st.session_state.events = []
            st.session_state.engine = MockInferenceEngine(scenario=scenario)
            st.session_state.decision_engine = AgenticDecisionEngine(
                st.session_state.config
            )
            st.session_state.engine.start()
            st.rerun()

        if stop_btn:
            if st.session_state.engine:
                st.session_state.engine.stop()
            st.session_state.session_active = False
            st.rerun()

        st.divider()
        st.markdown("### 🎛️ Thresholds")
        cfg = st.session_state.config.get("decision_engine", {})
        st.markdown(f"""
        <div style="font-size:12px; color:#9ca3af;">
        🟡 YELLOW &nbsp; > {cfg.get('yellow_threshold', 0.72):.0%}<br>
        🟠 ORANGE &nbsp; > {cfg.get('orange_threshold', 0.82):.0%}<br>
        🔴 RED &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; > {cfg.get('red_threshold', 0.91):.0%}<br>
        👁️ Gaze limit &nbsp; {cfg.get('sustained_gaze_seconds', 3.5):.1f}s
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📊 Session Stats")
        if st.session_state.decision_engine:
            stats = st.session_state.decision_engine.get_session_stats()
            st.metric("Total Decisions", stats["total_decisions"])
            st.metric("Total Flags", stats["total_flags"])
            st.metric("Current Tier", stats["current_tier"])
        else:
            st.caption("No active session")

    # ── Main content (only if active)
    if not st.session_state.session_active:
        st.markdown("""
        <div style="text-align:center; padding: 80px 0;">
            <div style="font-size:64px;">🛡️</div>
            <div style="font-size:24px; font-weight:500; color:#f9fafb; margin-top:16px;">
                ExamSentinel
            </div>
            <div style="font-size:14px; color:#6b7280; margin-top:8px; max-width:500px; margin-left:auto; margin-right:auto;">
                First autonomous multi-modal exam integrity system combining
                image forensics (BayarConv/SRM), deepfake detection,
                recapture detection, and RL-based agentic triage.
            </div>
            <div style="margin-top:32px; font-size:13px; color:#4b5563;">
                Select a scenario in the sidebar and click ▶ Start
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Pull latest result
    engine = st.session_state.engine
    dec_engine = st.session_state.decision_engine

    result: Optional[InferenceResult] = engine.get_latest()
    if result is None:
        st.info("Waiting for first inference result...")
        time.sleep(0.5)
        st.rerun()
        return

    # Process decision
    record = dec_engine.decide(result)

    # Store history
    h_entry = {
        "score_deepfake":  result.score_deepfake,
        "score_recapture": result.score_recapture,
        "score_splicing":  result.score_splicing,
        "score_forgery":   result.score_forgery,
        "score_combined":  result.score_combined,
        "tier":            record.tier
    }
    st.session_state.history.append(h_entry)

    # Store non-GREEN events
    if record.tier != "GREEN":
        st.session_state.events.append({
            "time": time.strftime("%H:%M:%S"),
            "tier": record.tier,
            "action": record.action,
            "reasons": " | ".join(record.reasons[:2])
        })

    # ── Layout
    tier = record.tier

    # Alert banner for RED
    if tier == "RED":
        st.markdown("""
        <div style="background:#2e0909; border:2px solid #E24B4A; border-radius:12px;
                    padding:16px 24px; margin-bottom:16px; text-align:center;">
            <span style="font-size:18px; font-weight:600; color:#E24B4A;">
                🚨 SESSION SUSPENDED — Integrity Violation Detected
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif tier == "ORANGE":
        st.markdown("""
        <div style="background:#2e1508; border:1px solid #D85A30; border-radius:12px;
                    padding:12px 24px; margin-bottom:16px;">
            <span style="font-size:15px; font-weight:500; color:#D85A30;">
                ⚠️ Session Flagged — Suspicious Activity Detected
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 1: Alert status + gauges
    col_tier, col_g1, col_g2, col_g3, col_g4 = st.columns([1.5, 1, 1, 1, 1])

    with col_tier:
        st.markdown("**Alert Status**")
        st.markdown(tier_badge(tier), unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:11px; color:#6b7280; margin-top:8px;">
        Action: <span style="color:#d1d5db">{record.action}</span><br>
        Flags: <span style="color:#d1d5db">{dec_engine.total_flags}</span><br>
        Consec: <span style="color:#d1d5db">{dec_engine.consecutive_orange}</span>
        </div>
        """, unsafe_allow_html=True)

        if record.reasons:
            st.markdown("""
            <div style="font-size:11px; color:#BA7517; margin-top:8px;">
            """ + "<br>".join(f"• {r}" for r in record.reasons[:3]) + """
            </div>
            """, unsafe_allow_html=True)

    gauge_configs = [
        (result.score_deepfake,  "Deepfake",  "#E24B4A", col_g1),
        (result.score_recapture, "Recapture", "#D85A30", col_g2),
        (result.score_splicing,  "Splicing",  "#BA7517", col_g3),
        (result.score_forgery,   "Forgery",   "#378ADD", col_g4),
    ]
    for score, title, color, col in gauge_configs:
        with col:
            st.plotly_chart(
                make_gauge(score, title, color),
                use_container_width=True,
                config={"displayModeBar": False}
            )

    # ── Row 2: Timeline + Gaze signals
    col_timeline, col_gaze = st.columns([3, 1])

    with col_timeline:
        st.plotly_chart(
            make_timeline(st.session_state.history[-60:]),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with col_gaze:
        st.markdown("**Gaze Signals**")
        face_color = "#1D9E75" if result.face_detected else "#E24B4A"
        gaze_color = "#E24B4A" if result.is_off_screen else "#1D9E75"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Face Detected</div>
            <div style="font-size:18px; font-weight:600; color:{face_color};">
                {"✓ YES" if result.face_detected else "✗ NO"}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Off-Screen Gaze</div>
            <div style="font-size:18px; font-weight:600; color:{gaze_color};">
                {"⚠ YES" if result.is_off_screen else "✓ NO"}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sustained (sec)</div>
            <div class="metric-value">{result.sustained_offscreen_sec:.1f}s</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Gaze H / V</div>
            <div style="font-size:14px; font-weight:500; color:#d1d5db;">
                {result.gaze_h_deg:+.0f}° / {result.gaze_v_deg:+.0f}°
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 3: Event log + Admin feedback
    col_events, col_feedback = st.columns([2, 1])

    with col_events:
        st.markdown("**Event Log**")
        events = st.session_state.events[-10:][::-1]  # Most recent first
        if not events:
            st.markdown(
                '<div class="event-row">No suspicious events detected.</div>',
                unsafe_allow_html=True
            )
        for ev in events:
            st.markdown(f"""
            <div class="event-row {ev['tier']}">
                <strong>{ev['time']}</strong> &nbsp;
                <span class="tier-badge tier-{ev['tier']}"
                      style="font-size:10px; padding:2px 8px;">{ev['tier']}</span>
                &nbsp; {ev['reasons']} &nbsp;
                <span style="color:#6b7280;">→ {ev['action']}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_feedback:
        st.markdown("**Admin Feedback**")
        st.caption("Train the decision engine with feedback")
        if st.session_state.events:
            last_idx = len(dec_engine.decision_log) - 1
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✓ Confirm", key="confirm", use_container_width=True):
                    dec_engine.receive_feedback(last_idx, "confirmed")
                    st.success("Confirmed!")
            with col_b:
                if st.button("✗ False+", key="fp", use_container_width=True):
                    dec_engine.receive_feedback(last_idx, "false_positive")
                    st.warning("Noted!")
        else:
            st.caption("No flags to review yet.")

        st.divider()
        st.markdown("**Combined Score**")
        st.metric(
            label="",
            value=f"{result.score_combined:.1%}",
            delta=f"{result.score_combined - (st.session_state.history[-2].get('score_combined', result.score_combined) if len(st.session_state.history) > 1 else 0):.1%}"
        )

    # ── Auto-refresh
    time.sleep(0.8)
    st.rerun()


if __name__ == "__main__":
    main()
