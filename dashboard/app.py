"""
dashboard/app.py — ExamSentinel Live Monitoring Dashboard

Run: python scripts/run_demo.py --mode dashboard
  OR: python -m streamlit run dashboard/app.py
"""

import sys, time, json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.inference_engine import MockInferenceEngine, InferenceResult
from agents.decision_engine import AgenticDecisionEngine
import yaml

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ExamSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: #161a23; border: 1px solid #2a2f3d;
    border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
}
.metric-label { font-size:11px; font-weight:500; letter-spacing:0.08em;
    color:#6b7280; text-transform:uppercase; margin-bottom:4px; }
.metric-value { font-size:26px; font-weight:600; color:#f9fafb; }
.tier-badge { display:inline-block; padding:6px 18px; border-radius:24px;
    font-size:14px; font-weight:600; letter-spacing:0.05em; }
.tier-GREEN  { background:#0d2e1f; color:#1D9E75; border:1px solid #1D9E75; }
.tier-YELLOW { background:#2a1e06; color:#BA7517; border:1px solid #BA7517; }
.tier-ORANGE { background:#2e1508; color:#D85A30; border:1px solid #D85A30; }
.tier-RED    { background:#2e0909; color:#E24B4A; border:1px solid #E24B4A; }
.event-row   { background:#161a23; border-left:3px solid #2a2f3d;
    border-radius:0 8px 8px 0; padding:8px 14px; margin-bottom:6px;
    font-size:12px; color:#d1d5db; }
.event-row.YELLOW { border-left-color:#BA7517; }
.event-row.ORANGE { border-left-color:#D85A30; }
.event-row.RED    { border-left-color:#E24B4A; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────
def init_state():
    defaults = {
        "engine": None, "dec_engine": None, "history": [],
        "events": [], "active": False, "scenario": "normal", "config": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.config is None:
        with open("configs/config.yaml") as f:
            st.session_state.config = yaml.safe_load(f)


# ── Chart helpers ─────────────────────────────────────────────────
def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 18, "color": "#f9fafb"}},
        title={"text": title, "font": {"size": 11, "color": "#9ca3af"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#6b7280", "size": 8}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1f2937", "borderwidth": 0,
            "steps": [
                {"range": [0, 65],  "color": "#111827"},
                {"range": [65, 82], "color": "#1c1a10"},
                {"range": [82, 100],"color": "#1c0f0f"}
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": value * 100}
        }
    ))
    fig.update_layout(height=160, margin=dict(t=35, b=5, l=15, r=15),
                      paper_bgcolor="#161a23", plot_bgcolor="#161a23",
                      font={"family": "Inter"})
    return fig


def make_timeline(history):
    fig = go.Figure()
    if not history:
        fig.update_layout(height=220, paper_bgcolor="#161a23",
                          plot_bgcolor="#161a23",
                          title=dict(text="Threat Score Timeline",
                                     font=dict(color="#9ca3af", size=12)))
        return fig

    df  = pd.DataFrame(history)
    idx = list(range(len(df)))

    fig.add_trace(go.Scatter(
        x=idx, y=df.get("combined", [0]*len(df)),
        name="Combined", line=dict(color="#7F77DD", width=2),
        fill="tozeroy", fillcolor="rgba(127,119,221,0.1)"
    ))
    for col, color, name in [
        ("deepfake",  "#E24B4A", "Deepfake"),
        ("recapture", "#D85A30", "Recapture"),
        ("splicing",  "#BA7517", "Splicing"),
        ("forgery",   "#378ADD", "Forgery"),
    ]:
        if col in df:
            fig.add_trace(go.Scatter(
                x=idx, y=df[col], name=name,
                line=dict(color=color, width=1, dash="dot"), opacity=0.7
            ))

    for val, color, label in [(0.72,"#BA7517","YELLOW"),
                               (0.82,"#D85A30","ORANGE"),
                               (0.91,"#E24B4A","RED")]:
        fig.add_hline(y=val, line_dash="dash", line_color=color,
                      line_width=1, opacity=0.5,
                      annotation_text=label,
                      annotation_font_color=color,
                      annotation_font_size=9)

    fig.update_layout(
        height=220, paper_bgcolor="#161a23", plot_bgcolor="#161a23",
        font={"family":"Inter","color":"#9ca3af"},
        legend=dict(bgcolor="#1f2937", bordercolor="#374151",
                    font=dict(size=9, color="#d1d5db")),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#6b7280")),
        yaxis=dict(showgrid=True, gridcolor="#1f2937", zeroline=False,
                   range=[0,1], tickfont=dict(color="#6b7280")),
        margin=dict(t=25, b=25, l=35, r=15),
        title=dict(text="Live Threat Score History",
                   font=dict(color="#9ca3af", size=11))
    )
    return fig


def tier_badge(tier):
    return f'<span class="tier-badge tier-{tier}">{tier}</span>'


# ── Main ──────────────────────────────────────────────────────────
def main():
    init_state()

    # Header
    st.markdown("""
    <div style="padding:10px 0 16px;border-bottom:1px solid #2a2f3d;margin-bottom:16px">
      <span style="font-size:22px;font-weight:600;color:#f9fafb;">🛡️ ExamSentinel</span>
      <span style="font-size:12px;color:#6b7280;margin-left:10px;">
        Agentic AI Exam Integrity System — FAR AWAY 2026
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Session Control")
        scenario = st.selectbox("Demo Scenario", options=[
            "normal","deepfake_attack","recapture_attempt","gaze_cheat"
        ], format_func=lambda x: {
            "normal":             "🟢 Normal Exam",
            "deepfake_attack":    "🔴 Deepfake Attack",
            "recapture_attempt":  "🟠 Paper Leak Attempt",
            "gaze_cheat":         "🟡 Gaze Cheating"
        }[x])

        c1, c2 = st.columns(2)
        with c1:
            start = st.button("▶ Start", use_container_width=True, type="primary")
        with c2:
            stop  = st.button("⏹ Stop",  use_container_width=True)

        if start:
            if st.session_state.engine:
                st.session_state.engine.stop()
            st.session_state.scenario   = scenario
            st.session_state.history    = []
            st.session_state.events     = []
            st.session_state.engine     = MockInferenceEngine(scenario=scenario)
            st.session_state.dec_engine = AgenticDecisionEngine(
                st.session_state.config, log_path="logs/dashboard_events.jsonl"
            )
            st.session_state.engine.start()
            st.session_state.active = True
            st.rerun()

        if stop:
            if st.session_state.engine:
                st.session_state.engine.stop()
            st.session_state.active = False
            st.rerun()

        st.divider()
        st.markdown("### 📊 Session Stats")
        if st.session_state.dec_engine:
            stats = st.session_state.dec_engine.get_session_stats()
            st.metric("Decisions", stats["total_decisions"])
            st.metric("Flags",     stats["total_flags"])
            st.metric("Tier",      stats["current_tier"])
        else:
            st.caption("No active session.")

        st.divider()
        cfg = st.session_state.config.get("decision_engine", {})
        st.markdown(f"""
        **Thresholds**
        🟡 YELLOW > {cfg.get('yellow_threshold',0.72):.0%}
        🟠 ORANGE > {cfg.get('orange_threshold',0.82):.0%}
        🔴 RED > {cfg.get('red_threshold',0.91):.0%}
        👁️ Gaze limit: {cfg.get('sustained_gaze_seconds',3.5):.1f}s
        """)

    # Idle screen
    if not st.session_state.active:
        st.markdown("""
        <div style="text-align:center;padding:60px 0">
          <div style="font-size:56px">🛡️</div>
          <div style="font-size:22px;font-weight:500;color:#f9fafb;margin-top:12px">ExamSentinel</div>
          <div style="font-size:13px;color:#6b7280;margin-top:8px;max-width:480px;
                      margin-left:auto;margin-right:auto;line-height:1.6">
            First autonomous multi-modal exam integrity system.<br>
            BayarConv/SRM forensics · Deepfake detection ·
            Recapture/paper-leak detection · RL triage
          </div>
          <div style="margin-top:24px;font-size:13px;color:#4b5563">
            Select a scenario in the sidebar and click ▶ Start
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Pull result
    engine     = st.session_state.engine
    dec_engine = st.session_state.dec_engine
    result     = engine.get_latest()
    if result is None:
        st.info("Waiting for first result...")
        time.sleep(0.3)
        st.rerun()
        return

    record = dec_engine.decide(result)
    tier   = record.tier

    # Store history
    st.session_state.history.append({
        "deepfake":  result.score_deepfake,
        "recapture": result.score_recapture,
        "splicing":  result.score_splicing,
        "forgery":   result.score_forgery,
        "combined":  result.score_combined,
    })
    if tier != "GREEN":
        st.session_state.events.append({
            "time": time.strftime("%H:%M:%S"), "tier": tier,
            "action": record.action,
            "reasons": " | ".join(record.reasons[:2])
        })

    # Alert banners
    if tier == "RED":
        st.markdown("""
        <div style="background:#2e0909;border:2px solid #E24B4A;border-radius:10px;
                    padding:14px 20px;margin-bottom:14px;text-align:center">
          <span style="font-size:16px;font-weight:600;color:#E24B4A">
            🚨 SESSION SUSPENDED — Integrity Violation Detected
          </span>
        </div>""", unsafe_allow_html=True)
    elif tier == "ORANGE":
        st.markdown("""
        <div style="background:#2e1508;border:1px solid #D85A30;border-radius:10px;
                    padding:10px 20px;margin-bottom:14px">
          <span style="font-size:14px;font-weight:500;color:#D85A30">
            ⚠️ Session Flagged — Suspicious Activity Detected
          </span>
        </div>""", unsafe_allow_html=True)

    # Row 1: Status + Gauges
    c_tier, c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1, 1])
    with c_tier:
        st.markdown("**Alert Status**")
        st.markdown(tier_badge(tier), unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:11px;color:#6b7280;margin-top:8px">
          Action: <span style="color:#d1d5db">{record.action}</span><br>
          Flags: <span style="color:#d1d5db">{dec_engine.total_flags}</span><br>
          Consec Orange: <span style="color:#d1d5db">{dec_engine.consecutive_orange}</span>
        </div>""", unsafe_allow_html=True)
        if record.reasons:
            st.markdown(
                '<div style="font-size:11px;color:#BA7517;margin-top:6px">' +
                "".join(f"• {r}<br>" for r in record.reasons[:3]) +
                "</div>", unsafe_allow_html=True
            )

    for score, title, color, col in [
        (result.score_deepfake,  "Deepfake",  "#E24B4A", c1),
        (result.score_recapture, "Recapture", "#D85A30", c2),
        (result.score_splicing,  "Splicing",  "#BA7517", c3),
        (result.score_forgery,   "Forgery",   "#378ADD", c4),
    ]:
        with col:
            st.plotly_chart(make_gauge(score, title, color),
                            use_container_width=True,
                            config={"displayModeBar": False})

    # Row 2: Timeline + Gaze
    c_timeline, c_gaze = st.columns([3, 1])
    with c_timeline:
        st.plotly_chart(make_timeline(st.session_state.history[-60:]),
                        use_container_width=True,
                        config={"displayModeBar": False})
    with c_gaze:
        st.markdown("**Gaze Signals**")
        fc = "#1D9E75" if result.face_detected else "#E24B4A"
        gc = "#E24B4A" if result.is_off_screen  else "#1D9E75"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Face Detected</div>
          <div style="font-size:16px;font-weight:600;color:{fc}">
            {"✓ YES" if result.face_detected else "✗ NO"}
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Off-Screen</div>
          <div style="font-size:16px;font-weight:600;color:{gc}">
            {"⚠ YES" if result.is_off_screen else "✓ NO"}
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Sustained</div>
          <div class="metric-value">{result.sustained_offscreen_sec:.1f}s</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Gaze H / V</div>
          <div style="font-size:13px;font-weight:500;color:#d1d5db">
            {result.gaze_h_deg:+.0f}° / {result.gaze_v_deg:+.0f}°
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Row 3: Event log + Feedback
    c_ev, c_fb = st.columns([2, 1])
    with c_ev:
        st.markdown("**Event Log**")
        events = st.session_state.events[-10:][::-1]
        if not events:
            st.markdown('<div class="event-row">No alerts yet.</div>',
                        unsafe_allow_html=True)
        for ev in events:
            st.markdown(
                f'<div class="event-row {ev["tier"]}">'
                f'<strong>{ev["time"]}</strong> &nbsp;'
                f'<span class="tier-badge tier-{ev["tier"]}" '
                f'style="font-size:10px;padding:2px 7px">{ev["tier"]}</span>'
                f' &nbsp; {ev["reasons"]}'
                f' &nbsp; <span style="color:#6b7280">→ {ev["action"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    with c_fb:
        st.markdown("**Admin Feedback**")
        st.caption("Train the RL decision engine")
        if st.session_state.events:
            idx = len(dec_engine.decision_log) - 1
            fa, fb = st.columns(2)
            with fa:
                if st.button("✓ Confirm", key="cfm", use_container_width=True):
                    dec_engine.receive_feedback(idx, "confirmed")
                    st.success("Confirmed!")
            with fb:
                if st.button("✗ False+", key="fp", use_container_width=True):
                    dec_engine.receive_feedback(idx, "false_positive")
                    st.warning("Noted!")
        else:
            st.caption("No flags to review yet.")

        st.divider()
        prev = st.session_state.history[-2]["combined"] if len(st.session_state.history) > 1 else result.score_combined
        st.metric("Combined Score",
                  f"{result.score_combined:.1%}",
                  f"{result.score_combined - prev:+.1%}")

    # Auto-refresh
    time.sleep(0.8)
    st.rerun()


if __name__ == "__main__":
    main()
