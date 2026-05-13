"""
The predictor contract (v2).

Every model in this project (1D CNN, 2D CNN, MiniRocket) exposes a class
named `FaultPredictor` in its `predict.py` that conforms to this interface:

    class FaultPredictor:
        @classmethod
        def is_available(cls) -> bool:
            "Check whether the model's artifacts (weights, config) are present."
            ...

        def __init__(self, ...):
            "Load the model into memory."
            ...

        def predict_file(self, path: str, rpm: int | None = None) -> dict:
            "Run inference on a .jsonl file and return a dashboard dict."
            ...

The dashboard dict has 8 top-level keys: verdict, severity_zone, frequency,
per_axis_peaks, frequency_table, signal_quality, action, meta.

To produce one, your predict.py should:

  1. Load the .jsonl file (use `models.shared.data_utils.load_jsonl_signal`)
  2. Window it (use `models.shared.data_utils.make_windows`)
  3. Run YOUR model on each window to produce:
     - fault_probs_per_window:    shape (n_windows, 4)
     - severity_probs_per_window: shape (n_windows, 3)
  4. Call `build_full_dashboard_dict(...)` from
     `models.shared.dashboard_builders` to assemble the full dict.

The shared builder handles all FFT, severity-zone, and per-axis computations
- so all 3 models produce identical dict shapes, and the Streamlit app code
is model-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BasePredictor(ABC):
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        ...

    @abstractmethod
    def predict_file(self, path: str, rpm: int | None = None) -> dict:
        ...


# Required keys for the v2 schema
REQUIRED_TOP_KEYS = {
    "verdict", "severity_zone", "frequency", "per_axis_peaks",
    "frequency_table", "spectrogram", "per_window",
    "signal_quality", "action", "meta",
}

REQUIRED_VERDICT_KEYS = {
    "predicted_fault", "predicted_severity", "is_normal",
    "headline", "plain_english",
    "fault_probs", "severity_probs",
    "confidence", "confidence_basis", "confidence_margin", "window_agreement",
}

REQUIRED_SEVERITY_ZONE_KEYS = {
    "ratio", "zone", "zone_label", "zone_position",
    "diagnostic_harmonic", "diagnostic_freq_hz", "harmonic_label",
    "is_baseline_known", "agreement",
}

REQUIRED_FREQUENCY_KEYS = {
    "freqs_hz", "magnitude_x", "magnitude_y", "magnitude_z",
    "primary_axis", "display_max_hz", "markers", "predicted_fault",
    "healthy_reference", "annotation",
}

REQUIRED_SPECTROGRAM_KEYS = {
    "frequencies_hz", "times_seconds", "magnitude_log", "axis",
}

REQUIRED_PER_WINDOW_KEYS = {
    "indices", "fault_probs", "predicted_class_per_window",
    "predicted_recording_class", "n_windows", "n_agreed",
}

REQUIRED_SIGNAL_QUALITY_KEYS = {
    "duration_seconds", "sample_rate_hz", "n_windows",
    "n_axes", "is_clean", "quality_message",
}

REQUIRED_ACTION_KEYS = {
    "diagnosis_line", "what_it_means", "confidence_basis",
    "predicted_fault", "predicted_severity",
}

REQUIRED_META_KEYS = {
    "rpm", "n_windows", "n_samples", "duration_seconds",
    "sample_rate_hz", "model_name",
}


def validate_dashboard_dict(d: dict) -> list:
    """Return a list of error messages (empty list = valid)."""
    errors = []

    if not REQUIRED_TOP_KEYS <= set(d.keys()):
        errors.append(
            f"Missing top keys: {REQUIRED_TOP_KEYS - set(d.keys())}"
        )
        return errors

    for sub_key, required in [
        ("verdict",        REQUIRED_VERDICT_KEYS),
        ("severity_zone",  REQUIRED_SEVERITY_ZONE_KEYS),
        ("frequency",      REQUIRED_FREQUENCY_KEYS),
        ("spectrogram",    REQUIRED_SPECTROGRAM_KEYS),
        ("per_window",     REQUIRED_PER_WINDOW_KEYS),
        ("signal_quality", REQUIRED_SIGNAL_QUALITY_KEYS),
        ("action",         REQUIRED_ACTION_KEYS),
        ("meta",           REQUIRED_META_KEYS),
    ]:
        sub = d[sub_key]
        if not isinstance(sub, dict):
            errors.append(f"Section '{sub_key}' must be a dict")
            continue
        missing = required - set(sub.keys())
        if missing:
            errors.append(f"Section '{sub_key}' missing keys: {missing}")

    # per_axis_peaks should be a list of 3 dicts (X, Y, Z)
    pap = d.get("per_axis_peaks")
    if not isinstance(pap, list) or len(pap) != 3:
        errors.append("per_axis_peaks must be a list of 3 dicts (X, Y, Z)")

    # frequency_table should be a list of 4 dicts (1x..4x harmonics)
    ft = d.get("frequency_table")
    if not isinstance(ft, list) or len(ft) != 4:
        errors.append("frequency_table must be a list of 4 dicts (1x..4x harmonics)")

    return errors