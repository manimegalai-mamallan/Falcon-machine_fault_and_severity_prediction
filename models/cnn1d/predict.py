"""
1D CNN inference adapter.

Wraps the friend's keras models (trained on log-FFT spectra + RPM) so they
return the same dashboard dict structure as `cnn2d/predict.py`.

Architecture differences from the 2D CNN:
  * Input: log-FFT magnitude (513, 3) + RPM scalar (vs. spectrogram in 2D)
  * Fault head: multi-label sigmoid over [Mechanical, Misalignment, Unbalanced]
                — Normal is implicit (all probs < threshold)
  * Severity: same cascaded design as 2D CNN — 3 separate softmax models

Class-name mapping (her convention -> our dashboard convention):
    Mechanical     -> Looseness
    Misalignment   -> Misalignment
    Unbalanced     -> Unbalance
    (none above)   -> Normal
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Quiet TensorFlow's startup chatter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np

from models.shared import config as shared_config
from models.shared.dashboard_builders import build_full_dashboard_dict
from models.shared.data_utils import parse_filename
from models.shared.friend_loader import (
    compute_fft_features,
    merge_triplets_from_jsonl,
    waveforms_to_signal,
)
from . import config


MODEL_DISPLAY_NAME = "1D CNN"

# Order of her fault model's output columns
HER_FAULT_ORDER = ("Mechanical", "Misalignment", "Unbalanced")

# Map her fault names -> our dashboard fault names
HER_TO_OURS = {
    "Mechanical":   "Looseness",
    "Misalignment": "Misalignment",
    "Unbalanced":   "Unbalance",
}


class FaultPredictor:
    """1D CNN cascaded predictor (friend's models)."""

    @classmethod
    def is_available(cls) -> bool:
        required = [
            config.FAULT_MODEL_PATH,
            config.SPECTRA_SCALER_PATH,
            config.SPEED_SCALER_PATH,
            config.METADATA_PATH,
        ]
        return all(p.exists() for p in required)

    def __init__(self, device=None):
        if not self.is_available():
            raise FileNotFoundError(
                "1D CNN artifacts not found. Drop these files into "
                f"{config.ARTIFACT_DIR}:\n"
                "  fault_cnn1d.keras\n"
                "  severity_mechanical_cnn1d.keras\n"
                "  severity_misalignment_cnn1d.keras\n"
                "  severity_unbalanced_cnn1d.keras\n"
                "  metadata.joblib\n"
                "  spectra_scaler.joblib\n"
                "  speed_scaler.joblib"
            )

        # Lazy import — TensorFlow is heavy; only pay if user actually picks 1D CNN
        from tensorflow.keras.models import load_model

        # Preprocessing scalers
        self.spectra_scaler = joblib.load(config.SPECTRA_SCALER_PATH)
        self.speed_scaler   = joblib.load(config.SPEED_SCALER_PATH)
        self.metadata       = joblib.load(config.METADATA_PATH)

        # Fault model (3-output multi-label sigmoid)
        self.fault_model = load_model(config.FAULT_MODEL_PATH, compile=False)

        # Severity models (one per fault, keyed by OUR display name)
        self.severity_models: dict = {}
        for our_name, path in config.SEVERITY_MODEL_PATHS.items():
            if path.exists():
                self.severity_models[our_name] = load_model(path, compile=False)

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

        # 2. Preprocess: log-FFT, channel-wise standardize
        spectra_raw = compute_fft_features(waveforms)            # (N, 513, 3)
        spectra = self._scale_spectra(spectra_raw)
        speed_scaled = self._scale_speed(rpm, n_windows)         # (N, 1)

        # 3. Run fault model -> (N, 3) sigmoid probs in HER_FAULT_ORDER
        her_fault_probs = self.fault_model.predict(
            {"spectra": spectra, "speed": speed_scaled}, verbose=0
        )

        # 4. Convert her multi-label output to our 4-class single-label format
        fault_probs_per_window = self._to_our_fault_probs(her_fault_probs)

        # 5. Recording-level fault prediction (mean over windows, then argmax)
        recording_fault_probs = fault_probs_per_window.mean(axis=0)
        fault_idx  = int(np.argmax(recording_fault_probs))
        fault_name = shared_config.FAULT_CLASSES[fault_idx]
        is_normal  = (fault_name == "Normal")

        # 6. Severity (same cascaded approach as 2D CNN)
        n_severity = len(shared_config.SEVERITY_LEVELS)
        sev_probs_per_window = np.zeros((n_windows, n_severity), dtype=np.float32)

        if not is_normal and fault_name in self.severity_models:
            sev_model = self.severity_models[fault_name]
            sev_probs_per_window = sev_model.predict(
                {"spectra": spectra, "speed": speed_scaled}, verbose=0
            ).astype(np.float32)
        elif is_normal:
            sev_probs_per_window[:, 0] = 1.0
        else:
            sev_probs_per_window[:, :] = 1.0 / n_severity

        # 7. Hand off to the shared dashboard builder
        signal = waveforms_to_signal(waveforms)  # (3, N*1024) view of triplets
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
    # Preprocessing helpers
    # ------------------------------------------------------------------
    def _scale_spectra(self, spectra: np.ndarray) -> np.ndarray:
        n_channels = spectra.shape[2]
        flat = self.spectra_scaler.transform(spectra.reshape(-1, n_channels))
        return flat.reshape(spectra.shape).astype(np.float32)

    def _scale_speed(self, rpm: int, n_windows: int) -> np.ndarray:
        speeds = np.full((n_windows, 1), float(rpm), dtype=np.float32)
        return self.speed_scaler.transform(speeds).astype(np.float32)

    def _to_our_fault_probs(self, her_probs: np.ndarray) -> np.ndarray:
        """Convert (N, 3) sigmoid probs in HER_FAULT_ORDER to (N, 4) probs in
        our dashboard's class order [Normal, Unbalance, Misalignment, Looseness].

        Strategy:
          * Map each fault output to its corresponding "ours" column.
          * Normal probability = clip(1 - max(her probs), 0, 1).
            (When all her probs are low, Normal is high. When any is high,
            Normal is suppressed.)
          * Renormalize each row to sum to 1 so argmax behaves predictably.
        """
        N = her_probs.shape[0]
        ours = np.zeros((N, len(shared_config.FAULT_CLASSES)), dtype=np.float32)

        # Map her columns -> ours columns
        for j, her_name in enumerate(HER_FAULT_ORDER):
            our_name = HER_TO_OURS[her_name]
            ours[:, shared_config.FAULT_CLASSES.index(our_name)] = her_probs[:, j]

        # Normal = 1 - max(others)
        max_fault = ours[:, 1:].max(axis=1)  # max over Unbalance/Misalignment/Looseness
        ours[:, 0] = np.clip(1.0 - max_fault, 0.0, 1.0)

        # Renormalize rows to sum to 1
        row_sum = ours.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return ours / row_sum
