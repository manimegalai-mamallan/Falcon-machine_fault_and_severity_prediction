"""
Shared dashboard builder (v2).

Every model's predict.py runs its model on each window of the signal,
producing per-window probabilities. This module then takes those
probabilities and assembles the full dashboard dict that the Streamlit app
displays.

All FFT, severity-zone, and per-axis computations live here, so all three
models (1D CNN, 2D CNN, MiniRocket) produce identical-shaped dashboard
dicts. The Streamlit app code never needs to know which model is in use.

Top-level dict structure produced by build_full_dashboard_dict:

    verdict             - predicted fault, severity, plain English, confidence
    severity_zone       - peak ratio vs healthy baseline + zone classification
    frequency           - FFT data (per axis) + healthy reference + annotation
    per_axis_peaks      - peak-ratio per axis at the diagnostic harmonic
    frequency_table     - row per shaft harmonic, healthy vs measured
    signal_quality      - duration, sample rate, "is the recording clean?"
    action              - priority, recommended check, consequence
    meta                - bookkeeping (rpm, n_windows, model_name, ...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import detrend

from . import config
from .data_utils import make_windows
from .recommendations import get_action, get_plain_english


# ---------------------------------------------------------------------------
# Diagnostic harmonics: which shaft harmonic is most diagnostic of each fault.
# Used by the severity gauge and per-axis-peak panels.
# ---------------------------------------------------------------------------
DIAGNOSTIC_HARMONIC = {
    "Normal":       1,   # no real diagnostic; show 1x as a reference
    "Unbalance":    1,
    "Misalignment": 2,
    "Looseness":    3,
}

# Severity zones based on (peak at diagnostic harmonic) / (healthy baseline).
# Boundaries chosen to match what plant maintenance teams expect from
# ISO-style vibration-severity language while staying unit-free.
SEVERITY_ZONE_BOUNDARIES = [
    ("good",       1.5),   # ratio < 1.5  -> good (green)
    ("acceptable", 2.5),   # 1.5 <= ratio < 2.5  -> acceptable (amber)
    ("warning",    4.0),   # 2.5 <= ratio < 4.0  -> warning (orange)
    ("danger",     float("inf")),  # ratio >= 4.0 -> danger (red)
]


# ---------------------------------------------------------------------------
# Healthy reference loading (cached at module level)
# ---------------------------------------------------------------------------
_HEALTHY_REFS: Optional[dict] = None


def _load_healthy_references() -> Optional[dict]:
    """Lazy-load the healthy FFT references (built by build_references.py)."""
    global _HEALTHY_REFS
    if _HEALTHY_REFS is not None:
        return _HEALTHY_REFS
    path = config.HEALTHY_REFERENCES_PATH
    if not Path(path).exists():
        return None
    try:
        loaded = np.load(path)
        _HEALTHY_REFS = {k: loaded[k] for k in loaded.files}
        return _HEALTHY_REFS
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------
def build_full_dashboard_dict(
    signal: np.ndarray,
    rpm: int,
    fault_probs_per_window: np.ndarray,
    severity_probs_per_window: np.ndarray,
    model_name: str,
) -> dict:
    """Pack everything the v2 dashboard needs into one dict.

    Args:
        signal:                    (3, N) float32 array, raw 3-axis vibration
        rpm:                       integer RPM (2000 / 2500 / 3000)
        fault_probs_per_window:    (n_windows, 4) softmax probs over
                                   [Normal, Unbalance, Misalignment, Looseness]
        severity_probs_per_window: (n_windows, 3) probs over [Low, Medium, High]
        model_name:                friendly model name (e.g. "2D CNN")

    Returns:
        Dict with keys: verdict, severity_zone, frequency, per_axis_peaks,
        frequency_table, signal_quality, action, meta.
    """
    fault_probs_per_window = np.asarray(fault_probs_per_window, dtype=np.float32)
    severity_probs_per_window = np.asarray(severity_probs_per_window, dtype=np.float32)

    # ---- recording-level prediction --------------------------------------
    fault_probs = fault_probs_per_window.mean(axis=0)
    fault_idx   = int(np.argmax(fault_probs))
    fault_name  = config.FAULT_CLASSES[fault_idx]
    is_normal   = (fault_name == "Normal")

    sev_probs = severity_probs_per_window.mean(axis=0)
    severity_label = (None if is_normal
                      else config.SEVERITY_LEVELS[int(np.argmax(sev_probs))])

    n_windows = fault_probs_per_window.shape[0]
    n_samples = int(signal.shape[1])

    # ---- compute averaged per-axis FFT spectra ---------------------------
    freq_data = _compute_axis_spectra(signal)
    healthy_ref = _load_healthy_reference_for_rpm(rpm)

    # ---- per-axis peak ratios at the diagnostic harmonic -----------------
    diagnostic_harmonic = DIAGNOSTIC_HARMONIC[fault_name]
    diagnostic_freq_hz  = (rpm / 60.0) * diagnostic_harmonic
    per_axis_peaks = _compute_per_axis_peaks(
        freq_data, healthy_ref, diagnostic_freq_hz, diagnostic_harmonic,
    )

    # primary severity ratio = max of the three per-axis ratios
    primary_ratio = max(p["peak_ratio"] for p in per_axis_peaks)
    severity_zone_block = _build_severity_zone(
        primary_ratio, diagnostic_harmonic, diagnostic_freq_hz,
        fault_name=fault_name,
    )

    # ---- frequency comparison table (one row per shaft harmonic 1x..4x) --
    frequency_table = _build_frequency_table(
        freq_data, healthy_ref, rpm, diagnostic_harmonic,
    )

    # ---- assemble all blocks --------------------------------------------
    verdict = _build_verdict(
        fault_name, fault_probs, severity_label, sev_probs,
        fault_probs_per_window, fault_idx, rpm,
    )

    frequency = _build_frequency_block(
        freq_data, healthy_ref, rpm, fault_name,
        diagnostic_freq_hz, primary_ratio, diagnostic_harmonic,
    )

    signal_quality = _build_signal_quality(
        signal, n_samples, n_windows,
    )

    action = _build_action(
        fault_name, severity_label, fault_probs_per_window, fault_idx,
    )

    spectrogram_block = _build_spectrogram(signal)

    per_window_block = _build_per_window_predictions(
        fault_probs_per_window, fault_idx,
    )

    return {
        "verdict":         verdict,
        "severity_zone":   severity_zone_block,
        "frequency":       frequency,
        "per_axis_peaks":  per_axis_peaks,
        "frequency_table": frequency_table,
        "spectrogram":     spectrogram_block,
        "per_window":      per_window_block,
        "signal_quality":  signal_quality,
        "action":          action,
        "meta": {
            "rpm":              int(rpm),
            "n_windows":        n_windows,
            "n_samples":        n_samples,
            "duration_seconds": float(n_samples / config.SAMPLE_RATE_HZ),
            "sample_rate_hz":   float(config.SAMPLE_RATE_HZ),
            "model_name":       model_name,
        },
    }


# ===========================================================================
# Spectrum computation
# ===========================================================================
def _compute_axis_spectra(signal: np.ndarray) -> dict:
    """Average FFT magnitude per axis across all windows."""
    sr = config.SAMPLE_RATE_HZ
    n_window = config.WINDOW_SIZE
    windows = make_windows(signal)

    freqs = np.fft.rfftfreq(n_window, d=1.0 / sr)
    keep = freqs <= config.FFT_DISPLAY_MAX_HZ
    freqs_keep = freqs[keep].astype(np.float32)

    mag_per_axis = []
    for axis_idx in range(3):
        per_window_specs = []
        for w in windows:
            x = detrend(w[axis_idx], type="linear")
            m = np.abs(np.fft.rfft(x)) / n_window
            per_window_specs.append(m[keep])
        mag_per_axis.append(np.mean(per_window_specs, axis=0).astype(np.float32))

    return {
        "freqs":       freqs_keep,
        "magnitude_x": mag_per_axis[0],
        "magnitude_y": mag_per_axis[1],
        "magnitude_z": mag_per_axis[2],
    }


def _load_healthy_reference_for_rpm(rpm: int) -> Optional[dict]:
    refs = _load_healthy_references()
    if refs is None or f"{rpm}_freqs" not in refs:
        return None
    return {
        "freqs":       refs[f"{rpm}_freqs"].astype(np.float32),
        "magnitude_x": refs[f"{rpm}_mag_x"].astype(np.float32),
        "magnitude_y": refs[f"{rpm}_mag_y"].astype(np.float32),
        "magnitude_z": refs[f"{rpm}_mag_z"].astype(np.float32),
        "rpm":         int(rpm),
    }


def _peak_in_band(
    freqs: np.ndarray,
    mag:   np.ndarray,
    center_hz: float,
    bandwidth_hz: float = 2.0,
) -> float:
    """Return the maximum FFT magnitude within ±bandwidth around center_hz."""
    lo = center_hz - bandwidth_hz
    hi = center_hz + bandwidth_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    return float(mag[mask].max())


# ===========================================================================
# Per-axis peak ratios
# ===========================================================================
def _compute_per_axis_peaks(
    freq_data:   dict,
    healthy_ref: Optional[dict],
    center_hz:   float,
    harmonic:    int,
) -> list:
    """Per-axis peak ratio (measured / healthy) at the diagnostic frequency."""
    out = []
    freqs = freq_data["freqs"]
    for axis_label, mag_key in (("X", "magnitude_x"),
                                ("Y", "magnitude_y"),
                                ("Z", "magnitude_z")):
        meas_peak = _peak_in_band(freqs, freq_data[mag_key], center_hz)
        if healthy_ref is not None:
            healthy_peak = _peak_in_band(
                healthy_ref["freqs"], healthy_ref[mag_key], center_hz
            )
            ratio = (meas_peak / healthy_peak) if healthy_peak > 1e-9 else 1.0
        else:
            ratio = float("nan")
        out.append({
            "axis":           axis_label,
            "harmonic":       harmonic,
            "peak_freq_hz":   float(center_hz),
            "peak_magnitude": float(meas_peak),
            "peak_ratio":     float(ratio),
        })
    return out


# ===========================================================================
# Severity zone
# ===========================================================================
def _classify_zone(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "good"
    for name, upper in SEVERITY_ZONE_BOUNDARIES:
        if ratio < upper:
            return name
    return "danger"


def _zone_position(ratio: float) -> float:
    """Map a ratio to a [0, 1] position on the horizontal severity gauge.

    Zones are unequal width on the gauge to give visual emphasis to the
    danger zone:  good (0.0-0.25) | acceptable (0.25-0.42) |
                  warning (0.42-0.67) | danger (0.67-1.00).
    """
    if not np.isfinite(ratio):
        return 0.0
    r = max(ratio, 0.0)
    if r < 1.5:
        return min(r / 1.5, 1.0) * 0.25
    if r < 2.5:
        return 0.25 + ((r - 1.5) / 1.0) * 0.17
    if r < 4.0:
        return 0.42 + ((r - 2.5) / 1.5) * 0.25
    # ratio >= 4.0: spread the danger zone across r in [4, 8]
    return 0.67 + min((r - 4.0) / 4.0, 1.0) * 0.33


def _compute_zone_model_agreement(
    fault_name: str,
    zone:       str,
    is_baseline_known: bool,
) -> dict:
    """Compare the diagnostic-harmonic ratio with the model's verdict.

    The severity gauge looks at one specific frequency (the textbook
    diagnostic harmonic for the predicted fault). The model uses the entire
    spectrum, time-domain signal, and learned patterns. So the gauge ratio
    and the model verdict can disagree in legitimate ways:

      - Looseness shows broadband + subharmonics, not a clean 3× peak,
        so the gauge can read "good" while the model correctly says
        "Looseness".
      - A clipped or noisy recording can produce a phantom peak that
        elevates the ratio while the model still says "Normal".

    This function classifies which of these cases we're in so the UI
    can show an explanatory note when the two numbers disagree.
    """
    is_normal = (fault_name == "Normal")
    zone_says_problem = zone in ("warning", "danger")
    zone_says_healthy = zone in ("good", "acceptable")

    if not is_baseline_known:
        return {
            "state":       "no_baseline",
            "label":        None,
            "explanation": (
                "Healthy baseline not available for this RPM. The gauge "
                "cannot compute a ratio."
            ),
        }
    if is_normal and zone_says_healthy:
        return {
            "state":       "agree_healthy",
            "label":       "Both checks agree",
            "explanation":
                "Model says Normal and the spectrum has no elevated peaks.",
        }
    if is_normal and zone_says_problem:
        return {
            "state":       "disagree_normal_with_peak",
            "label":       "Verify recording",
            "explanation": (
                "Model classifies as Normal, but a peak is elevated at the "
                "reference harmonic. Check the recording for clipping or "
                "transient noise — a re-take may be needed."
            ),
        }
    if not is_normal and zone_says_problem:
        return {
            "state":       "agree_fault",
            "label":       "Both checks agree",
            "explanation": (
                f"Model detected {fault_name} and the spectrum confirms an "
                f"elevated peak at the diagnostic harmonic."
            ),
        }
    # not is_normal and zone_says_healthy
    return {
        "state":       "disagree_fault_no_peak",
        "label":       "Note on this gauge",
        "explanation": (
            f"This gauge checks a single frequency — the textbook signature "
            f"for {fault_name}. The model uses the entire spectrum, "
            f"time-domain patterns, and per-window agreement, so it can "
            f"detect {fault_name} from broader features even when the "
            f"diagnostic peak alone is not elevated. The model verdict above "
            f"is the primary diagnosis."
        ),
    }


def _build_severity_zone(
    ratio:      float,
    harmonic:   int,
    freq_hz:    float,
    fault_name: str,
) -> dict:
    zone = _classify_zone(ratio)
    is_baseline_known = bool(np.isfinite(ratio))
    agreement = _compute_zone_model_agreement(
        fault_name, zone, is_baseline_known,
    )
    return {
        "ratio":               float(ratio),
        "zone":                zone,
        "zone_label":          zone.title(),
        "zone_position":       _zone_position(ratio),
        "diagnostic_harmonic": int(harmonic),
        "diagnostic_freq_hz":  float(freq_hz),
        "harmonic_label":      f"{harmonic}× shaft",
        "is_baseline_known":   is_baseline_known,
        "agreement":           agreement,
    }


# ===========================================================================
# Frequency comparison table
# ===========================================================================
def _build_frequency_table(
    freq_data:           dict,
    healthy_ref:         Optional[dict],
    rpm:                 int,
    diagnostic_harmonic: int,
) -> list:
    """Row per shaft harmonic (1×, 2×, 3×, 4×) with healthy vs measured."""
    shaft_freq = rpm / 60.0
    rows = []
    for h in (1, 2, 3, 4):
        f_hz = shaft_freq * h
        meas_peak = _peak_in_band(
            freq_data["freqs"], freq_data["magnitude_y"], f_hz
        )
        if healthy_ref is not None:
            healthy_peak = _peak_in_band(
                healthy_ref["freqs"], healthy_ref["magnitude_y"], f_hz
            )
            ratio = (meas_peak / healthy_peak) if healthy_peak > 1e-9 else float("nan")
        else:
            healthy_peak = float("nan")
            ratio = float("nan")
        rows.append({
            "harmonic":       h,
            "label":          f"{h}× shaft ({f_hz:.0f} Hz)",
            "freq_hz":        float(f_hz),
            "healthy":        float(healthy_peak),
            "measured":       float(meas_peak),
            "ratio":          float(ratio),
            "is_diagnostic":  (h == diagnostic_harmonic),
        })
    return rows


# ===========================================================================
# Frequency block (FFT spectra + annotation for the main chart)
# ===========================================================================
def _build_frequency_block(
    freq_data:    dict,
    healthy_ref:  Optional[dict],
    rpm:          int,
    fault_name:   str,
    diag_freq_hz: float,
    primary_ratio: float,
    diag_harmonic: int,
) -> dict:
    shaft_freq = rpm / 60.0
    markers = {
        f"{h}x": {
            "freq_hz":          float(h * shaft_freq),
            "harmonic":         h,
            "label":             f"{h}× shaft",
            "associated_fault": ("Unbalance"    if h == 1 else
                                 "Misalignment" if h == 2 else
                                 "Looseness"),
            "is_diagnostic":     (h == diag_harmonic),
        }
        for h in (1, 2, 3, 4)
    }

    if np.isfinite(primary_ratio):
        annotation = {
            "freq_hz": float(diag_freq_hz),
            "label":   f"{diag_harmonic}× peak: {primary_ratio:.1f}× baseline",
            "sublabel": f"{fault_name.lower()} signature",
        }
    else:
        annotation = None

    return {
        "freqs_hz":          freq_data["freqs"],
        "magnitude_x":       freq_data["magnitude_x"],
        "magnitude_y":       freq_data["magnitude_y"],
        "magnitude_z":       freq_data["magnitude_z"],
        "primary_axis":      "y",
        "display_max_hz":    config.FFT_DISPLAY_MAX_HZ,
        "markers":           markers,
        "predicted_fault":   fault_name,
        "healthy_reference": healthy_ref,
        "annotation":        annotation,
    }


# ===========================================================================
# Verdict
# ===========================================================================
def _build_verdict(
    fault_name: str,
    fault_probs: np.ndarray,
    severity:    Optional[str],
    sev_probs:   np.ndarray,
    per_window:  np.ndarray,
    fault_idx:   int,
    rpm:         int,
) -> dict:
    sorted_probs = np.sort(fault_probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1])
    agree_frac = float((per_window.argmax(axis=-1) == fault_idx).mean())
    n_windows = per_window.shape[0]
    n_agree   = int(round(agree_frac * n_windows))

    if margin > 0.5 and agree_frac > 0.95:
        confidence_level = "High"
    elif margin > 0.25 and agree_frac > 0.8:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    plain = get_plain_english(fault_name, severity)
    if severity:
        headline = f"{fault_name} · {severity} severity at {rpm} RPM"
    else:
        headline = f"{fault_name} operation at {rpm} RPM"

    return {
        "predicted_fault":    fault_name,
        "predicted_severity": severity,
        "is_normal":          fault_name == "Normal",
        "headline":           headline,
        "plain_english":      plain,
        "fault_probs": {
            cls: float(p) for cls, p in zip(config.FAULT_CLASSES, fault_probs)
        },
        "severity_probs": {
            lvl: float(p) for lvl, p in zip(config.SEVERITY_LEVELS, sev_probs)
        },
        "confidence":        confidence_level,
        "confidence_basis":  f"{n_agree} of {n_windows} windows agreed on {fault_name}",
        "confidence_margin": margin,
        "window_agreement":  agree_frac,
    }


# ===========================================================================
# Signal quality
# ===========================================================================
def _build_signal_quality(
    signal:    np.ndarray,
    n_samples: int,
    n_windows: int,
) -> dict:
    duration = n_samples / config.SAMPLE_RATE_HZ
    max_abs = float(np.abs(signal).max()) if signal.size else 0.0
    is_clipped = max_abs > 9000.0  # the loader already filters >1e4
    is_clean = (n_windows >= 5) and (not is_clipped)
    if is_clean:
        message = "✓ steady · no clipping"
    elif is_clipped:
        message = "⚠ clipping detected"
    else:
        message = "⚠ short recording"
    return {
        "duration_seconds": float(duration),
        "sample_rate_hz":   float(config.SAMPLE_RATE_HZ),
        "n_windows":        int(n_windows),
        "n_axes":           3,
        "is_clean":         bool(is_clean),
        "quality_message":  message,
    }


# ===========================================================================
# Action — simplified, no fabricated maintenance instructions
# ===========================================================================
def _build_action(
    fault_name: str,
    severity:   Optional[str],
    per_window: np.ndarray,
    fault_idx:  int,
) -> dict:
    n_windows = per_window.shape[0]
    n_agree   = int((per_window.argmax(axis=-1) == fault_idx).sum())

    # Honest summary — no priority claims, no time windows, no maintenance
    # instructions we can't defend from the data.
    if fault_name == "Normal":
        diagnosis_line = "No fault detected."
        what_means = ("All analysis windows classify the recording as Normal "
                      "operation. No follow-up action recommended by this tool.")
    else:
        sev_text = severity if severity else "(unknown)"
        diagnosis_line = f"{fault_name} (severity: {sev_text})"
        what_means = (f"The model classifies this recording as {fault_name}. "
                      "Severity reflects the model's per-window output. Use "
                      "this output as ML evidence to inform a maintenance "
                      "decision — this dashboard does not prescribe specific "
                      "repair actions.")

    return {
        "diagnosis_line":    diagnosis_line,
        "what_it_means":     what_means,
        "confidence_basis":  f"{n_agree} of {n_windows} windows agreed on {fault_name}",
        "predicted_fault":   fault_name,
        "predicted_severity": severity,
    }


# ===========================================================================
# Spectrogram (the model's actual visual input, on the Y axis)
# ===========================================================================
def _build_spectrogram(signal: np.ndarray) -> dict:
    """Compute a Y-axis spectrogram for display."""
    from scipy.signal import detrend, spectrogram as scipy_spectrogram

    sr = config.SAMPLE_RATE_HZ
    y = detrend(signal[1], type="linear").astype(np.float32)
    f_spec, t_spec, Sxx = scipy_spectrogram(
        y, fs=sr, nperseg=512, noverlap=256, scaling="spectrum",
    )
    Sxx_log = np.log1p(Sxx).astype(np.float32)

    keep_f = f_spec <= config.FFT_DISPLAY_MAX_HZ
    Sxx_log = Sxx_log[keep_f]
    f_spec  = f_spec[keep_f].astype(np.float32)

    if Sxx_log.shape[1] > config.SPEC_DISPLAY_TIME_BINS:
        stride = max(1, Sxx_log.shape[1] // config.SPEC_DISPLAY_TIME_BINS)
        Sxx_log = Sxx_log[:, ::stride]
        t_spec  = t_spec[::stride]

    return {
        "frequencies_hz": f_spec.astype(np.float32),
        "times_seconds":  t_spec.astype(np.float32),
        "magnitude_log":  Sxx_log,
        "axis":           "y",
    }


# ===========================================================================
# Per-window predictions trace (model's vote per window)
# ===========================================================================
def _build_per_window_predictions(
    per_window: np.ndarray,
    fault_idx:  int,
) -> dict:
    """Pack per-window probabilities into one dict for plotting."""
    n_windows = per_window.shape[0]
    return {
        "indices":     np.arange(n_windows, dtype=np.int32),
        "fault_probs": {
            cls: per_window[:, i].astype(np.float32)
            for i, cls in enumerate(config.FAULT_CLASSES)
        },
        "predicted_class_per_window": [
            config.FAULT_CLASSES[i]
            for i in per_window.argmax(axis=-1).tolist()
        ],
        "predicted_recording_class": config.FAULT_CLASSES[fault_idx],
        "n_windows": int(n_windows),
        "n_agreed": int((per_window.argmax(axis=-1) == fault_idx).sum()),
    }