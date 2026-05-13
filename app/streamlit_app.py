"""
VibeDiag v3 — Vibration Fault Diagnosis Dashboard.

Dark-mode Streamlit app with 12 panels covering:
   1.  How VibeDiag works (intro)
   2.  Status verdict (no fabricated maintenance advice)
   3.  Severity score / Confidence / Recording metadata
   4.  Main FFT spectrum with healthy baseline
   5.  Class-signature comparison
   6.  Per-axis FFTs
   7.  Spectrogram (model input)
   8.  Per-window prediction trace
   9.  Frequency comparison table
   10. Model probability breakdown
   11. Class fingerprint reference library
   12. Model performance (confusion matrix, training curves, per-class accuracy)

Run from project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st

from app import plots
from app import model_registry
from models.shared import config as shared_config


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VibeDiag — Vibration Fault Diagnosis",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Dark theme CSS (overrides Streamlit's default)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ======= Page background ======= */
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background: #181b22 !important;
    color: #d4d1c8 !important;
}
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1320px; }

/* Headers, paragraphs */
h1, h2, h3, h4, h5, h6 { color: #e8e6df !important; }
p, span, div, label { color: #d4d1c8; }
.stCaption, [data-testid="stCaptionContainer"] { color: #8a877f !important; }
.stMarkdown a { color: #5BA0E8 !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Inputs (radio, selectbox, file uploader) */
[data-testid="stRadio"] label, [data-testid="stSelectbox"] label,
[data-testid="stFileUploader"] label { color: #c8c5bd !important; }

[data-testid="stRadio"] [role="radiogroup"] label {
    background: #232730; border: 0.5px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 6px 10px; margin-bottom: 4px;
    color: #d4d1c8 !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #232730 !important; border-color: rgba(255,255,255,0.1) !important;
    color: #e8e6df !important;
}
[data-testid="stFileUploader"] section {
    background: #232730 !important;
    border: 1.5px dashed rgba(255,255,255,0.18) !important;
    color: #d4d1c8 !important;
}
[data-testid="stFileUploader"] button {
    background: #2c313c !important; color: #e8e6df !important;
    border-color: rgba(255,255,255,0.12) !important;
}

/* Streamlit alerts */
[data-testid="stAlert"] { background: #232730 !important;
    border: 0.5px solid rgba(255,255,255,0.1) !important; }

/* ======= Reusable card containers ======= */
.section-card {
    background: #21242c; border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
    color: #d4d1c8;
}
.section-title { font-size: 14px; font-weight: 500; color: #e8e6df; margin-bottom: 4px; }
.section-sub   { font-size: 12px; color: #9a978f; line-height: 1.55; margin-bottom: 12px; }

/* ======= Hero verdict (top status banner) ======= */
.hero {
    border-radius: 12px; padding: 18px 22px; margin: 8px 0 12px;
    border-width: 0.5px; border-style: solid; border-left-width: 4px;
}
.hero.healthy { background: #16352b; border-color: #28604d; border-left-color: #3FB58F; }
.hero.warning { background: #3a2f15; border-color: #6c5520; border-left-color: #D69042; }
.hero.danger  { background: #3a1f15; border-color: #6e3a26; border-left-color: #E07050; }

.hero-eyebrow {
    font-size: 12px; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 6px;
    display: inline-flex; align-items: center; gap: 8px;
}
.hero.healthy .hero-eyebrow { color: #6FCFA9; }
.hero.warning .hero-eyebrow { color: #E5A964; }
.hero.danger  .hero-eyebrow { color: #E89A82; }
.hero-eyebrow-dot { width: 8px; height: 8px; border-radius: 50%; }
.hero.healthy .hero-eyebrow-dot { background: #3FB58F; }
.hero.warning .hero-eyebrow-dot { background: #D69042; }
.hero.danger  .hero-eyebrow-dot { background: #E07050; }

.hero-title    { font-size: 22px; font-weight: 500; color: #f1efe6; margin-bottom: 4px; }
.hero-explanation { font-size: 13px; color: #c8c5bd; line-height: 1.55; }

/* ======= Inline verdict honesty box (replaces old action box) ======= */
.verdict-secondary {
    background: rgba(0,0,0,0.18); border: 0.5px solid rgba(255,255,255,0.06);
    border-radius: 8px; padding: 12px 14px;
}
.verdict-secondary-label {
    font-size: 10px; color: #9a978f; text-transform: uppercase;
    letter-spacing: 0.04em; font-weight: 500; margin-bottom: 6px;
}
.verdict-secondary-body { font-size: 12px; color: #c8c5bd; line-height: 1.55; }

/* ======= Metric cards ======= */
.metric-card {
    background: #21242c; border: 0.5px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px 16px; height: 100%;
}
.metric-label {
    font-size: 11px; color: #9a978f; text-transform: uppercase;
    letter-spacing: 0.04em; font-weight: 500; margin-bottom: 8px;
}
.metric-value {
    font-size: 22px; font-weight: 500;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; margin-bottom: 4px;
}
.metric-value.danger  { color: #E89A82; }
.metric-value.warning { color: #E5A964; }
.metric-value.healthy { color: #6FCFA9; }
.metric-value.neutral { color: #e8e6df; }
.metric-sub { font-size: 11px; color: #9a978f; line-height: 1.5; }

/* Severity gauge */
.gauge-track {
    display: flex; gap: 2px; height: 10px;
    border-radius: 5px; overflow: hidden; margin-top: 12px;
    position: relative;
}
.gauge-good       { background: #3FB58F; flex: 1.5; }
.gauge-acceptable { background: #D69042; flex: 1; }
.gauge-warning    { background: #E58060; flex: 1.5; }
.gauge-danger     { background: #E07050; flex: 2; }

.gauge-marker {
    position: absolute; top: -4px;
    width: 0; height: 0; transform: translateX(-50%);
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid #e8e6df;
}
.gauge-labels {
    display: flex; justify-content: space-between;
    font-size: 9px; color: #8a877f; margin-top: 6px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
}

/* Confidence pips */
.conf-pips { display: flex; gap: 2px; margin-top: 10px; }
.conf-pip { flex: 1; height: 6px; border-radius: 1px; background: #2c313c; }
.conf-pip.lit-high   { background: #3FB58F; }
.conf-pip.lit-medium { background: #D69042; }
.conf-pip.lit-low    { background: #E07050; }

/* Severity-gauge agreement note */
.agreement-note {
    margin-top: 12px; padding: 10px 12px;
    border-radius: 8px;
    font-size: 11px; line-height: 1.5;
}
.agreement-note.disagree { background: #3a2f15; border: 0.5px solid #6c5520; color: #ddc89a; }
.agreement-note.warn     { background: #3a1f15; border: 0.5px solid #6e3a26; color: #f0c0b3; }
.agreement-note .agreement-icon { margin-right: 6px; font-weight: 500; }
.agreement-note .agreement-label {
    font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.04em; font-size: 10px; margin-right: 6px;
}

/* ======= Plain-English fact cards ======= */
.fact-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    margin-top: 8px;
}
.fact-card {
    background: #1d2028; border: 0.5px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 14px 16px;
    display: flex; flex-direction: column; gap: 6px;
}
.fact-card-head {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: #9a978f; text-transform: uppercase;
    letter-spacing: 0.04em; font-weight: 500;
}
.fact-card-icon { font-size: 14px; }
.fact-card-body {
    font-size: 13px; color: #e8e6df; line-height: 1.55;
}
.fact-card-body .accent { color: #6FCFA9; font-weight: 500; }
.fact-card-body .accent-warn { color: #E5A964; font-weight: 500; }
.fact-card-body .accent-strong { color: #E89A82; font-weight: 500; }
.fact-card-body .muted { color: #9a978f; font-size: 11px; display: block; margin-top: 4px; }
.fact-card-body strong { color: #f1efe6; font-weight: 500; }

/* ======= Per-axis cards ======= */
.axis-card {
    background: #1d2028; border-radius: 8px; padding: 10px 12px;
    border: 0.5px solid rgba(255,255,255,0.05);
}
.axis-card-head {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 6px;
}
.axis-card-name { font-size: 11px; font-weight: 500; color: #e8e6df; }
.axis-card-stat {
    font-size: 10px; color: #E89A82;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
}

/* ======= Frequency comparison table ======= */
.freq-table {
    display: grid; grid-template-columns: 1.4fr 1fr 1fr 1fr;
    gap: 4px; font-size: 11px;
}
.freq-th {
    padding: 6px 10px; color: #9a978f; font-weight: 500;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em;
}
.freq-th.right { text-align: right; }
.freq-cell {
    padding: 6px 10px; background: #1d2028; border-radius: 6px;
    color: #c8c5bd;
}
.freq-cell.right {
    text-align: right;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; color: #d4d1c8;
}
.freq-cell.diagnostic         { background: #3a1f15; color: #f0d5cb; font-weight: 500; }
.freq-cell.diagnostic.right   { color: #f0d5cb; }
.freq-cell.diagnostic.ratio   { color: #E89A82; font-weight: 500; }

/* ======= Probability bars ======= */
.prob-section-label {
    font-size: 10px; color: #9a978f; text-transform: uppercase;
    letter-spacing: 0.04em; font-weight: 500; margin-bottom: 12px;
}
.prob-row { margin-bottom: 9px; }
.prob-row-head {
    display: flex; justify-content: space-between; font-size: 12px;
    margin-bottom: 4px; align-items: baseline;
}
.prob-row-name { color: #b0ada4; }
.prob-row-name.predicted { color: #f0d5cb; font-weight: 500; }
.prob-row-value {
    font-family: ui-monospace, "SF Mono", Menlo, monospace; color: #8a877f;
}
.prob-row-value.predicted { color: #E89A82; font-weight: 500; }
.prob-bar       { height: 5px; background: #1d2028; border-radius: 3px; overflow: hidden; }
.prob-bar-fill  { height: 100%; border-radius: 3px; }
.prob-bar-fill.muted     { background: #6a675f; }
.prob-bar-fill.predicted { background: #E07050; }
.prob-bar-fill.sev-low    { background: #3FB58F; }
.prob-bar-fill.sev-medium { background: #D69042; }
.prob-bar-fill.sev-high   { background: #E07050; }

/* ======= Intro panel (how it works) ======= */
.intro-panel {
    background: linear-gradient(135deg, #1f2832 0%, #1c2530 100%);
    border: 0.5px solid rgba(91,160,232,0.25);
    border-radius: 12px; padding: 16px 22px;
    margin: 8px 0 14px; display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px;
}
.intro-step {
    display: flex; flex-direction: column; align-items: flex-start; gap: 6px;
}
.intro-step-number {
    font-size: 10px; color: #5BA0E8; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.intro-step-title { font-size: 13px; font-weight: 500; color: #e8e6df; }
.intro-step-desc  { font-size: 11px; color: #9a978f; line-height: 1.5; }

/* ======= Class-comparison cards ======= */
.cc-card {
    background: #1d2028; border-radius: 8px; padding: 8px 10px;
    border: 0.5px solid rgba(255,255,255,0.05);
}
.cc-card.match-best {
    border: 1px solid #3FB58F; background: #16352b;
}
.cc-card-head {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 4px;
}
.cc-card-class { font-size: 11px; font-weight: 500; color: #e8e6df; }
.cc-card-prob {
    font-size: 10px; font-family: ui-monospace, monospace; color: #8a877f;
}
.cc-card.match-best .cc-card-prob { color: #6FCFA9; font-weight: 500; }

/* ======= Library card (#11) ======= */
.lib-card {
    background: #1d2028; border-radius: 8px; padding: 10px 12px;
    border: 0.5px solid rgba(255,255,255,0.05);
}
.lib-card-name { font-size: 12px; font-weight: 500; color: #e8e6df; margin-bottom: 4px; }
.lib-card-fact { font-size: 10px; color: #9a978f; line-height: 1.5; }

/* ======= Model performance section ======= */
.perf-stat {
    background: #1d2028; border: 0.5px solid rgba(255,255,255,0.05);
    border-radius: 8px; padding: 12px 14px; height: 100%;
}
.perf-stat-label {
    font-size: 10px; color: #9a978f; text-transform: uppercase;
    letter-spacing: 0.04em; font-weight: 500; margin-bottom: 6px;
}
.perf-stat-value {
    font-size: 22px; font-weight: 500;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    color: #6FCFA9; margin-bottom: 4px;
}
.perf-stat-sub { font-size: 11px; color: #9a978f; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_key: str):
    return model_registry.load_predictor(model_key)


@st.cache_resource(show_spinner=False)
def load_class_signatures() -> dict | None:
    """Load per-class average FFT signatures from the references file."""
    path = shared_config.HEALTHY_REFERENCES_PATH
    if not path.exists():
        return None
    try:
        npz = np.load(path)
    except Exception:
        return None
    out = {}
    for cls in shared_config.FAULT_CLASSES:
        fkey = f"class_{cls}_freqs"
        if fkey in npz.files:
            out[cls] = {
                "freqs": npz[fkey],
                "mag_x": npz[f"class_{cls}_mag_x"],
                "mag_y": npz[f"class_{cls}_mag_y"],
                "mag_z": npz[f"class_{cls}_mag_z"],
            }
    return out if out else None


@st.cache_resource(show_spinner=False)
def load_evaluation_results() -> dict | None:
    """Load test-set evaluation results."""
    path = shared_config.MODELS_DIR / "cnn2d" / "artifacts" / "evaluation_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_training_history() -> list | None:
    path = shared_config.MODELS_DIR / "cnn2d" / "artifacts" / "training_history.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _hero_class(verdict: dict, severity_zone: dict) -> str:
    if verdict["is_normal"]:
        return "healthy"
    if severity_zone.get("zone") in ("warning", "danger"):
        return "danger"
    return "warning"


def _confidence_pips(level: str) -> str:
    counts = {"High": 5, "Medium": 3, "Low": 2}.get(level, 1)
    klass = {"High": "lit-high", "Medium": "lit-medium",
             "Low": "lit-low"}.get(level, "lit-low")
    pips = []
    for i in range(5):
        pips.append(f"<div class='conf-pip {klass if i < counts else ''}'></div>")
    return "<div class='conf-pips'>" + "".join(pips) + "</div>"


def _zone_word(zone: str) -> str:
    return {"good": "Good", "acceptable": "Acceptable",
            "warning": "Warning", "danger": "Danger"}.get(zone, zone.title())


# ===========================================================================
# Header
# ===========================================================================
hdr_l, hdr_r = st.columns([3, 2])
with hdr_l:
    st.markdown("## VibeDiag")
    st.caption("Vibration fault diagnosis · Prototype rig")
with hdr_r:
    st.markdown(
        "<div style='text-align: right; padding-top: 1.5rem; "
        "font-size: 12px; color: #8a877f;'>"
        "Single-fault classification · 4 classes · 3 RPMs"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ===========================================================================
# SECTION 1 — Intro: how VibeDiag works
# ===========================================================================
st.markdown("""
<div class='intro-panel'>
  <div class='intro-step'>
    <span class='intro-step-number'>Step 1</span>
    <span class='intro-step-title'>📡 Sensors record</span>
    <span class='intro-step-desc'>3 accelerometers (X / Y / Z) capture vibration on a running machine at 945 Hz.</span>
  </div>
  <div class='intro-step'>
    <span class='intro-step-number'>Step 2</span>
    <span class='intro-step-title'>🌊 Signal → spectrum</span>
    <span class='intro-step-desc'>The signal is split into windows and converted into a frequency picture (FFT / spectrogram).</span>
  </div>
  <div class='intro-step'>
    <span class='intro-step-number'>Step 3</span>
    <span class='intro-step-title'>🧠 AI recognizes patterns</span>
    <span class='intro-step-desc'>A trained neural network compares your spectrum to thousands of healthy and faulty examples.</span>
  </div>
  <div class='intro-step'>
    <span class='intro-step-number'>Step 4</span>
    <span class='intro-step-title'>📋 Diagnosis</span>
    <span class='intro-step-desc'>You see the predicted fault, the model's confidence, and the evidence supporting it.</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# Input row (no model selector — all 3 models run in ensemble)
# ===========================================================================
in_b, in_c = st.columns([2.5, 1])

with in_b:
    st.markdown("**Recording**")
    uploaded = st.file_uploader(
        "Upload a .jsonl vibration file",
        type=["jsonl"], label_visibility="collapsed",
    )

with in_c:
    st.markdown("**RPM**")
    rpm_choice = st.selectbox(
        "Operating speed", [2000, 2500, 3000], index=1,
        format_func=lambda v: f"{v} RPM", label_visibility="collapsed",
    )
    st.caption("Auto-detected from filename when possible")


if uploaded is None:
    st.info(
        "**Upload a .jsonl recording to begin.** "
        "Files in `test_data/` are good starting examples. "
        "All 3 models (2D CNN, 1D CNN, MiniRocket) will run automatically and "
        "their predictions will be combined for a single best result."
    )
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
    tmp.write(uploaded.getvalue())
    tmp_path = Path(tmp.name)

try:
    from models.shared.data_utils import parse_filename
    rpm_choice = parse_filename(uploaded.name)["rpm"]
except Exception:
    pass

# Run all available models and ensemble their predictions
from app.ensemble import predict_with_ensemble
try:
    spinner_slot = st.empty()
    def _on_progress(label: str):
        spinner_slot.info(f"Running {label}...")
    spinner_slot.info("Loading models...")
    result = predict_with_ensemble(
        tmp_path, rpm=rpm_choice, progress_callback=_on_progress,
    )
    spinner_slot.empty()
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

verdict       = result["verdict"]
severity_zone = result["severity_zone"]
freq          = result["frequency"]
per_axis      = result["per_axis_peaks"]
freq_tab      = result["frequency_table"]
spec_data     = result["spectrogram"]
per_window    = result["per_window"]
sig_qual      = result["signal_quality"]
action        = result["action"]
meta          = result["meta"]


# ===========================================================================
# SECTION 2 — Status verdict
# ===========================================================================
hero_class = _hero_class(verdict, severity_zone)
eyebrow = "Machine healthy" if verdict["is_normal"] else "Fault detected"

st.markdown(f"""
<div class='hero {hero_class}'>
  <div style='display: grid; grid-template-columns: 1fr 320px; gap: 24px; align-items: center;'>
    <div>
      <div class='hero-eyebrow'>
        <span class='hero-eyebrow-dot'></span>{eyebrow}
      </div>
      <div class='hero-title'>{verdict["headline"]}</div>
      <div class='hero-explanation'>{verdict["plain_english"]}</div>
    </div>
    <div class='verdict-secondary'>
      <div class='verdict-secondary-label'>What this means</div>
      <div class='verdict-secondary-body'>{action["what_it_means"]}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# SECTION 3 — Severity / Confidence / Recording
# ===========================================================================
mc1, mc2, mc3, mc4 = st.columns([1.4, 1, 1, 1])

with mc1:
    # Predicted severity card — uses the model's own output (severity_probs)
    # rather than the heuristic peak-ratio gauge. Works for every file
    # regardless of whether a healthy baseline exists at this RPM.
    if verdict["is_normal"]:
        st.markdown("""
        <div class='metric-card'>
          <div class='metric-label'>Predicted severity</div>
          <div class='metric-value healthy' style='font-size: 22px;'>Not applicable</div>
          <div class='metric-sub'>The recording is classified as Normal.
            Severity is only predicted for faulty recordings.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pred_sev = verdict["predicted_severity"] or "—"
        sev_probs = verdict["severity_probs"]
        top_prob  = sev_probs.get(pred_sev, 0.0) if pred_sev != "—" else 0.0
        # Color class based on severity level
        sev_value_class = {"Low": "healthy", "Medium": "warning",
                            "High": "danger"}.get(pred_sev, "neutral")
        # Build inline probability bars
        bar_rows = []
        for level in ["Low", "Medium", "High"]:
            prob = sev_probs.get(level, 0.0)
            pct = max(prob, 0.0) * 100
            is_pred = (level == pred_sev)
            name_cls = "predicted" if is_pred else ""
            val_cls  = "predicted" if is_pred else ""
            fill_cls = f"sev-{level.lower()}" if is_pred else "muted"
            bar_rows.append(
                f"<div class='prob-row' style='margin-bottom:6px;'>"
                f"<div class='prob-row-head' style='font-size:11px;'>"
                f"<span class='prob-row-name {name_cls}'>{level}</span>"
                f"<span class='prob-row-value {val_cls}'>{prob:.2f}</span>"
                f"</div>"
                f"<div class='prob-bar' style='height:4px;'>"
                f"<div class='prob-bar-fill {fill_cls}' style='width:{pct:.1f}%'></div>"
                f"</div></div>"
            )
        bars_html = "".join(bar_rows)
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Predicted severity</div>
          <div class='metric-value {sev_value_class}'>{pred_sev}
            <span style='font-size:12px;color:#9a978f;font-weight:400;font-family:system-ui,sans-serif;'>
              · {top_prob*100:.0f}% confidence</span>
          </div>
          <div style='margin-top:12px;'>{bars_html}</div>
          <div class='metric-sub' style='margin-top:8px;color:#7a7770;'>
            From cascaded severity classifier ({verdict['predicted_fault']} model)
          </div>
        </div>
        """, unsafe_allow_html=True)

with mc2:
    cc = ("healthy" if verdict["confidence"] == "High"
          else "warning" if verdict["confidence"] == "Medium" else "danger")
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Confidence</div>
      <div class='metric-value {cc}'>{verdict["confidence"]}</div>
      <div class='metric-sub'>{verdict["confidence_basis"]}</div>
      {_confidence_pips(verdict["confidence"])}
    </div>
    """, unsafe_allow_html=True)

with mc3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Recording</div>
      <div class='metric-value neutral' style='font-size: 16px;'>
        {sig_qual['duration_seconds']:.1f} s · {int(sig_qual['sample_rate_hz'])} Hz
      </div>
      <div class='metric-sub'>{sig_qual['n_axes']} axes · {sig_qual['n_windows']} windows</div>
      <div class='metric-sub' style='color:{"#6FCFA9" if sig_qual["is_clean"] else "#E89A82"}; font-family: ui-monospace, monospace; font-size: 10px; margin-top: 6px;'>
        {sig_qual['quality_message']}
      </div>
    </div>
    """, unsafe_allow_html=True)

with mc4:
    # Ensemble agreement card — shows how many of the 3 models agreed
    ens = result.get("ensemble")
    if ens:
        n_agreed = ens["n_agreed"]
        n_total  = ens["n_total"]
        winning_fault = ens["winning_fault"]

        # Color based on agreement strength
        if n_agreed == n_total:
            agree_color = "healthy"
            agree_msg = "Unanimous"
        elif n_agreed >= 2:
            agree_color = "warning"
            agree_msg = "Majority"
        else:
            agree_color = "danger"
            agree_msg = "No consensus"

        # Build per-model badges
        badges = []
        for label, pred in ens["predictions"].items():
            is_winner = (pred == winning_fault)
            badge_color = "#6FCFA9" if is_winner else "#9a978f"
            badges.append(
                f"<div style='font-size:10px; color:{badge_color}; "
                f"font-family:ui-monospace,monospace; line-height:1.5;'>"
                f"{'✓' if is_winner else '·'} {label}: {pred}</div>"
            )
        # Mention failed models if any
        for label, _err in ens.get("failed", {}).items():
            badges.append(
                f"<div style='font-size:10px; color:#E89A82; "
                f"font-family:ui-monospace,monospace; line-height:1.5;'>✗ {label}: failed</div>"
            )
        badges_html = "".join(badges)

        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Models in agreement</div>
          <div class='metric-value {agree_color}' style='font-size: 22px;'>
            {n_agreed}/{n_total}
            <span style='font-size:11px;color:#9a978f;font-weight:400;font-family:system-ui,sans-serif;'>· {agree_msg}</span>
          </div>
          <div style='margin-top:8px;'>{badges_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='metric-card'>
          <div class='metric-label'>Models in agreement</div>
          <div class='metric-value neutral'>—</div>
          <div class='metric-sub'>Single model only</div>
        </div>
        """, unsafe_allow_html=True)



# ===========================================================================
# SECTION 9 — Plain-English summary (replaces the technical frequency table)
# ===========================================================================
# Build the cards from data + model output only — no domain knowledge,
# no fabricated business advice.
duration_min = meta["duration_seconds"] / 60.0
n_windows    = per_window["n_windows"]
n_agreed     = per_window["n_agreed"]
agreement_pct = (n_agreed / n_windows * 100) if n_windows else 0.0
predicted_fault    = verdict["predicted_fault"]
predicted_severity = verdict["predicted_severity"]
top_fault_prob     = verdict["fault_probs"].get(predicted_fault, 0.0)

# How many axes show the diagnostic peak above baseline (counts axes where
# peak_ratio > 1, i.e. measured peak above healthy)? Falls back to a generic
# "3 sensors" sentence if the ratios aren't available.
import math as _math
def _is_elevated(p):
    r = p.get("peak_ratio")
    return bool(r is not None and not _math.isnan(r) and r > 1.0)
elevated_axes = sum(1 for p in per_axis if _is_elevated(p))

# Pick accent class based on whether this is a fault
fault_class = "accent" if verdict["is_normal"] else "accent-strong"

# Build the cards (different content for Normal vs faulty)
if verdict["is_normal"]:
    cards = [
        ("🔬", "What the AI looked at",
         f"<strong>{duration_min:.1f} minutes</strong> of vibration data "
         f"from <strong>3 sensors</strong> at "
         f"<strong>{int(meta['rpm'])} RPM</strong>."),
        ("✓", "What the AI found",
         f"The recording is classified as <span class='accent'>Normal</span> "
         f"operation. No fault patterns detected."),
        ("📊", "How the analysis worked",
         f"The recording was split into <strong>{n_windows} short clips</strong> "
         f"and each clip was independently classified. "
         f"<strong>{n_agreed} of {n_windows}</strong> clips "
         f"({agreement_pct:.0f}%) agreed on Normal."),
        ("🎯", "Why we trust this",
         f"The AI's confidence is <strong>{verdict['confidence']}</strong>. "
         f"<span class='muted'>{verdict['confidence_basis']}</span>"),
    ]
else:
    sev_text = predicted_severity if predicted_severity else "(unknown)"
    cards = [
        ("🔬", "What the AI looked at",
         f"<strong>{duration_min:.1f} minutes</strong> of vibration data "
         f"from <strong>3 sensors</strong> at "
         f"<strong>{int(meta['rpm'])} RPM</strong>."),
        ("⚠", "What the AI found",
         f"A <span class='{fault_class}'>{predicted_fault}</span> fault "
         f"with <strong>{sev_text}</strong> severity. "
         f"<span class='muted'>Top probability: {top_fault_prob*100:.0f}%</span>"),
        ("📊", "How the analysis worked",
         f"The recording was split into <strong>{n_windows} short clips</strong>. "
         f"Each clip was first classified by a <strong>fault model</strong> "
         f"(4 possible classes). For clips identified as {predicted_fault}, "
         f"a second <strong>severity model</strong> trained specifically on "
         f"{predicted_fault} recordings predicted the severity level."),
        ("🎯", "Why we trust this diagnosis",
         f"<strong>{n_agreed} of {n_windows}</strong> clips "
         f"({agreement_pct:.0f}%) independently arrived at the same answer. "
         f"<span class='muted'>"
         f"Cross-axis check: peak elevation seen on "
         f"{elevated_axes} of 3 sensor axes."
         f"</span>"),
    ]

cards_html = ["<div class='section-card'>",
              "<div class='section-title'>③ Plain-English summary</div>",
              "<div class='section-sub'>What the AI did, what it found, and "
              "why we trust the answer — in everyday language.</div>",
              "<div class='fact-grid'>"]
for icon, label, body in cards:
    cards_html.append(
        f"<div class='fact-card'>"
        f"<div class='fact-card-head'>"
        f"<span class='fact-card-icon'>{icon}</span>{label}"
        f"</div>"
        f"<div class='fact-card-body'>{body}</div>"
        f"</div>"
    )
cards_html.append("</div></div>")
st.markdown("\n".join(cards_html), unsafe_allow_html=True)


# ===========================================================================
# SECTION 6 — Diagnostic spectrum (replaces pattern match panels)
# ===========================================================================
st.markdown("""
<div class='section-card' style='margin-top:12px;'>
  <div class='section-title'>④ Diagnostic spectrum</div>
  <div class='section-sub'>
    Frequency content of the recording with shaft harmonics marked at
    1× / 2× / 3× rotation speed. The diagnostic harmonic for the predicted
    fault is highlighted in accent color. KPI cards below quantify the
    energy in each diagnostic band.
  </div>
</div>
""", unsafe_allow_html=True)

st.plotly_chart(
    plots.annotated_diagnostic_spectrum(freq, verdict["predicted_fault"]),
    use_container_width=True, config={"displayModeBar": False},
)

# Per-band KPI cards
import numpy as _np
_freqs_arr = _np.array(freq["freqs_hz"])
_mag_arr   = _np.array(freq["magnitude_y"])
_rpm       = float(meta.get("rpm", 2500))
_f1        = _rpm / 60.0

def _peak_around(target_hz, window_hz=3.0):
    if target_hz is None or target_hz <= 0:
        return 0.0
    mask = _np.abs(_freqs_arr - target_hz) <= window_hz
    return float(_mag_arr[mask].max()) if mask.any() else 0.0

_amp_1x = _peak_around(_f1)
_amp_2x = _peak_around(2 * _f1)
_amp_3x = _peak_around(3 * _f1)
_high_mask = _freqs_arr >= 4 * _f1
_amp_broadband = float(_mag_arr[_high_mask].mean()) if _high_mask.any() else 0.0

_diag_fault = verdict["predicted_fault"]
_diag_label = {"Unbalance": "1×", "Misalignment": "2×", "Looseness": "3×"}.get(_diag_fault, "")

def _band_card(label_top, label_bot, value_g, accent: bool):
    bg    = "rgba(224,112,80,0.10)" if accent else "rgba(255,255,255,0.03)"
    bd    = "1px solid #E07050"     if accent else "1px solid rgba(255,255,255,0.05)"
    label_color = "#E07050"  if accent else "#9a978f"
    value_color = "#E07050"  if accent else "#e8e6df"
    return (
        f"<div style='background:{bg}; border:{bd}; border-radius:8px; padding:11px 13px;'>"
        f"  <div style='font-size:10px; color:{label_color}; text-transform:uppercase; letter-spacing:0.05em;{'font-weight:500;' if accent else ''}'>{label_top}</div>"
        f"  <div style='font-size:18px; font-family:ui-monospace,monospace; color:{value_color}; margin-top:3px;'>{value_g:.3f} g</div>"
        f"  <div style='font-size:10px; color:{label_color}; margin-top:1px;'>{label_bot}</div>"
        f"</div>"
    )

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(_band_card(
        f"1× ROT ({_f1:.1f} Hz)", "unbalance band",
        _amp_1x, accent=(_diag_label == "1×")
    ), unsafe_allow_html=True)
with k2:
    st.markdown(_band_card(
        f"2× ROT ({2*_f1:.1f} Hz)", "misalignment band",
        _amp_2x, accent=(_diag_label == "2×")
    ), unsafe_allow_html=True)
with k3:
    st.markdown(_band_card(
        f"3× ROT ({3*_f1:.1f} Hz)", "looseness band",
        _amp_3x, accent=(_diag_label == "3×")
    ), unsafe_allow_html=True)
with k4:
    st.markdown(_band_card(
        "BROADBAND", "noise floor",
        _amp_broadband, accent=False
    ), unsafe_allow_html=True)


# ===========================================================================
# SECTION 7 — Engineering Analysis (Bode + Nyquist-style + Spectrogram)
# ===========================================================================
# Three plots that satisfy both technical (Bode + Nyquist-style orbit) and
# industrial (spectrogram) audiences. All built from data already computed
# during prediction; raw signal is reloaded from the uploaded file for the
# orbit plot.
st.markdown("""
<div class='section-card' style='margin-top: 12px;'>
  <div class='section-title'>⑤ Engineering Analysis</div>
  <div class='section-sub'>
    Standard vibration-analysis plots used in industrial condition monitoring.
    The Bode-style spectrum shows the frequency response on log scale; the
    orbit plot reveals shaft motion in the radial plane.
  </div>
</div>
""", unsafe_allow_html=True)

# Bode-style spectrum
st.markdown("""
<div class='section-card' style='margin-top: 12px; margin-bottom: 0;'>
  <div class='section-title' style='font-size: 14px;'>Bode-style frequency response</div>
  <div class='section-sub'>
    Log-log magnitude vs frequency. Vertical dashed lines mark shaft harmonics;
    the diagnostic harmonic (predicted fault) is highlighted.
  </div>
</div>
""", unsafe_allow_html=True)
st.plotly_chart(
    plots.bode_style_spectrum(freq),
    use_container_width=True, config={"displayModeBar": False},
)

# Orbit plot (Nyquist-style)
st.markdown("""
<div class='section-card' style='margin-top: 12px; margin-bottom: 0;'>
  <div class='section-title' style='font-size: 14px;'>Shaft orbit (Nyquist-style)</div>
  <div class='section-sub'>
    X-axis vs Y-axis vibration. The shape of the orbit is diagnostic:
    circular = unbalance, elliptical / figure-8 = misalignment,
    chaotic = looseness.
  </div>
</div>
""", unsafe_allow_html=True)
try:
    from models.shared.data_utils import load_jsonl_signal
    _signal_for_orbit = load_jsonl_signal(tmp_path)
    st.plotly_chart(
        plots.orbit_plot(_signal_for_orbit),
        use_container_width=True, config={"displayModeBar": False},
    )
except Exception as _e:
    st.info(f"Orbit plot unavailable: {_e}")

# Spectrogram (frequency-time heatmap)
st.markdown("""
<div class='section-card' style='margin-top: 12px; margin-bottom: 0;'>
  <div class='section-title' style='font-size: 14px;'>Spectrogram — frequency over time</div>
  <div class='section-sub'>
    Frequency content as it evolves through the recording. Sustained
    horizontal bands at shaft harmonics are the fault signature.
  </div>
</div>
""", unsafe_allow_html=True)
if spec_data:
    st.plotly_chart(
        plots.spectrogram_panel(spec_data),
        use_container_width=True, config={"displayModeBar": False},
    )
else:
    st.info("Spectrogram data not available for this recording.")

# ===========================================================================
# Footer
# ===========================================================================
st.divider()

st.caption(
    f"File: {meta.get('filename', uploaded.name)} · "
    f"{meta['n_samples']:,} samples · {meta['n_windows']} windows · "
    f"{meta['duration_seconds']:.1f} s · {meta['rpm']} RPM · "
    f"model: {meta['model_name']}"
)