"""
MiniRocket inference adapter.

Wraps the friend's sklearn-style models (RidgeClassifierCV on top of
MiniRocket convolutional kernels) so they return the same dashboard
dict structure as `cnn2d/predict.py`.

Architecture differences from the 2D CNN:
  * Input: raw waveforms (1024, 3) — NO FFT, NO scaling
            (she trained MINIROCKET on raw time-domain signals)
  * Fault head: MultilabelMiniRocket -> 3 sigmoid-like outputs
                in HER_FAULT_ORDER. Normal is implicit.
  * Severity: MulticlassMiniRocket -> 3-class softmax-like outputs
              in [Low, Medium, High].

The pickled models depend on `src.models.minirocket` being importable
because joblib stores fully-qualified class names. The user's project
includes that module in `src/models/minirocket.py` for this purpose.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

# Make `src.models.minirocket` importable so joblib can unpickle the
# friend's MiniRocket models. The src/ folder lives at the project root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.shared import config as shared_config
from models.shared.dashboard_builders import build_full_dashboard_dict
from models.shared.data_utils import parse_filename
from models.shared.friend_loader import (
    merge_triplets_from_jsonl,
    waveforms_to_signal,
)
from . import config


MODEL_DISPLAY_NAME = "MiniRocket"

# Order of her fault model's output columns — same as 1D CNN
HER_FAULT_ORDER = ("Mechanical", "Misalignment", "Unbalanced")

HER_TO_OURS = {
    "Mechanical":   "Looseness",
    "Misalignment": "Misalignment",
    "Unbalanced":   "Unbalance",
}


class FaultPredictor:
    """MiniRocket cascaded predictor (friend's sklearn models)."""

    @classmethod
    def is_available(cls) -> bool:
        return config.FAULT_MODEL_PATH.exists()

    def __init__(self):
        if not self.is_available():
            raise FileNotFoundError(
                "MiniRocket artifacts not found. Drop these files into "
                f"{config.ARTIFACT_DIR}:\n"
                "  fault_minirocket.joblib\n"
                "  severity_mechanical_minirocket.joblib\n"
                "  severity_misalignment_minirocket.joblib\n"
                "  severity_unbalanced_minirocket.joblib\n"
                "  metadata.joblib  (optional)"
            )

        # Fault model
        self.fault_model = joblib.load(config.FAULT_MODEL_PATH)

        # Severity models (one per fault, keyed by OUR display name)
        self.severity_models: dict = {}
        for our_name, path in config.SEVERITY_MODEL_PATHS.items():
            if path.exists():
                self.severity_models[our_name] = joblib.load(path)

    # ------------------------------------------------------------------
    def predict_file(self, path: str | Path, rpm: Optional[int] = None) -> dict:
        path = Path(path)
        if rpm is None:
            try:
                rpm = parse_filename(path)["rpm"]
            except Exception:
                raise ValueError(
                    "Could not infer RPM from filename. Pass rpm=... explicitly."
                )

        # 1. Load .jsonl, merge X/Y/Z triplets -> (N, 1024, 3) waveforms
        waveforms = merge_triplets_from_jsonl(path)
        n_windows = waveforms.shape[0]

        if n_windows == 0:
            raise ValueError(f"No valid triplets in {path.name}")

        # 2. Run fault model on raw waveforms -> (N, 3) probs in HER_FAULT_ORDER
        her_fault_probs = self.fault_model.predict_proba(waveforms)
        her_fault_probs = np.asarray(her_fault_probs, dtype=np.float32)
        # Defensive: predict_proba on multi-output classifiers can sometimes
        # return a list of (N, 2) matrices instead of (N, n_outputs). Detect
        # and reshape so we always end up with shape (N, 3).
        if her_fault_probs.ndim == 3 and her_fault_probs.shape[2] == 2:
            her_fault_probs = her_fault_probs[..., 1]  # take "positive" class

        # 3. Convert to our 4-class single-label format
        fault_probs_per_window = self._to_our_fault_probs(her_fault_probs)

        # 4. Recording-level fault prediction
        recording_fault_probs = fault_probs_per_window.mean(axis=0)
        fault_idx  = int(np.argmax(recording_fault_probs))
        fault_name = shared_config.FAULT_CLASSES[fault_idx]
        is_normal  = (fault_name == "Normal")

        # 5. Severity prediction
        n_severity = len(shared_config.SEVERITY_LEVELS)
        sev_probs_per_window = np.zeros((n_windows, n_severity), dtype=np.float32)

        if not is_normal and fault_name in self.severity_models:
            sev_model = self.severity_models[fault_name]
            sev_raw = np.asarray(
                sev_model.predict_proba(waveforms), dtype=np.float32
            )
            # Her MulticlassMiniRocket exposes classes_ in [Low, Medium, High]
            # (canonical order). Reorder if the model's classes_ disagree.
            sev_probs_per_window = self._reorder_severity_probs(sev_raw, sev_model)
        elif is_normal:
            sev_probs_per_window[:, 0] = 1.0
        else:
            sev_probs_per_window[:, :] = 1.0 / n_severity

        # 6. Hand off to the shared dashboard builder
        signal = waveforms_to_signal(waveforms)
        result = build_full_dashboard_dict(
            signal=signal,
            rpm=int(rpm),
            fault_probs_per_window=fault_probs_per_window.astype(np.float32),
            severity_probs_per_window=sev_probs_per_window.astype(np.float32),
            model_name=MODEL_DISPLAY_NAME,
        )
        result["meta"]["filename"]   = path.name
        result["meta"]["file_bytes"] = path.stat().st_size
        return result

    # ------------------------------------------------------------------
    def _to_our_fault_probs(self, her_probs: np.ndarray) -> np.ndarray:
        """Same conversion as the 1D-CNN predictor (multi-label sigmoid -> 4-class)."""
        N = her_probs.shape[0]
        ours = np.zeros((N, len(shared_config.FAULT_CLASSES)), dtype=np.float32)
        for j, her_name in enumerate(HER_FAULT_ORDER):
            our_name = HER_TO_OURS[her_name]
            ours[:, shared_config.FAULT_CLASSES.index(our_name)] = her_probs[:, j]
        max_fault = ours[:, 1:].max(axis=1)
        ours[:, 0] = np.clip(1.0 - max_fault, 0.0, 1.0)
        row_sum = ours.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return ours / row_sum

    def _reorder_severity_probs(self, sev_raw: np.ndarray, model) -> np.ndarray:
        """Ensure severity columns are in [Low, Medium, High] order regardless
        of how the underlying RidgeClassifierCV stored them.

        Handles all common label encodings the friend's training pipeline
        might have used:
            * Title-case strings: "Low", "Medium", "High"        (our format)
            * Lower-case strings: "low", "medium", "high"
            * Single-char codes:  "L", "M", "H"
            * Integer codes:      0, 1, 2  (or  1, 2, 3)
            * Numpy int variants: np.int64(0), np.int32(0), ...

        If none of these patterns match, falls back to numeric sort order
        (assuming the model's labels can be cast to float and are ordered
        Low → Medium → High by magnitude).
        """
        target = list(shared_config.SEVERITY_LEVELS)  # ["Low", "Medium", "High"]

        # Sanity check on shape — if we don't get a 2D probability matrix,
        # bail out and let downstream code see all-zeros (existing behavior).
        if sev_raw.ndim != 2:
            return sev_raw

        if not hasattr(model, "classes_") or model.classes_ is None:
            # No class info — assume canonical order
            return sev_raw if sev_raw.shape[1] == len(target) else \
                   np.zeros((sev_raw.shape[0], len(target)), dtype=np.float32)

        model_classes = list(model.classes_)
        n_target      = len(target)

        # ---- Strategy 1: case-insensitive string match -------------------
        def _norm(x):
            return str(x).strip().lower()

        norm_model  = [_norm(c) for c in model_classes]
        # Direct names + short codes
        name_map = {
            "low": "Low",     "medium": "Medium",  "high":   "High",
            "l":   "Low",     "m":      "Medium",  "h":      "High",
            "mild":"Low",     "moderate":"Medium", "severe": "High",
        }
        mapped = [name_map.get(c) for c in norm_model]

        if any(m is not None for m in mapped):
            out = np.zeros((sev_raw.shape[0], n_target), dtype=np.float32)
            for j, our_label in enumerate(mapped):
                if our_label is None:
                    continue
                i = target.index(our_label)
                out[:, i] = sev_raw[:, j]
            return out

        # ---- Strategy 2: numeric — sort ascending and assume L<M<H -------
        try:
            numeric_vals = [float(c) for c in model_classes]
            sorted_idx = list(np.argsort(numeric_vals))
            if len(sorted_idx) >= n_target:
                out = np.zeros((sev_raw.shape[0], n_target), dtype=np.float32)
                for i in range(n_target):
                    out[:, i] = sev_raw[:, sorted_idx[i]]
                return out
        except (ValueError, TypeError):
            pass

        # ---- Strategy 3: shape matches, assume same order ----------------
        if sev_raw.shape[1] == n_target:
            return sev_raw

        # ---- Last resort: return zeros (visible bug, not silent failure) -
        return np.zeros((sev_raw.shape[0], n_target), dtype=np.float32)