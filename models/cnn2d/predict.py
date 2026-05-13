"""
2D CNN cascaded inference.

At prediction time:
  1. Run the fault classifier to get per-window fault probabilities.
  2. Find the recording-level fault class.
  3. If "Normal", set severity to None.
     Else load the severity model trained on that specific fault and
     run it on every window to get per-window severity probabilities.
  4. Hand off to the shared dashboard builder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.shared.dashboard_builders import build_full_dashboard_dict
from models.shared.data_utils import (
    load_jsonl_signal, make_windows, parse_filename,
)
from . import config
from .model import FaultClassifier, SeverityClassifier
from .stft_utils import window_to_spectrogram


MODEL_DISPLAY_NAME = "2D CNN"


class FaultPredictor:
    """Inference wrapper for the cascaded 2D-CNN classifiers."""

    @classmethod
    def is_available(cls) -> bool:
        """Fault model + at least one severity model required."""
        if not (config.MODEL_PATH.exists() and config.PREPROC_PATH.exists()
                and config.LABEL_MAP_PATH.exists()):
            return False
        # We need at least the fault model. Severity models are nice-to-have;
        # if missing, severity defaults to None.
        return True

    def __init__(self, device: Optional[str] = None):
        if not self.is_available():
            raise FileNotFoundError(
                f"2D CNN artifacts not found in {config.ARTIFACT_DIR}. "
                "Train first: python -m models.cnn2d.train"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        with open(config.PREPROC_PATH) as f:
            self.preproc = json.load(f)
        with open(config.LABEL_MAP_PATH) as f:
            self.label_map = json.load(f)

        self.window_size = int(self.preproc["window_size"])
        self.hop_size    = int(self.preproc["hop_size"])
        self.rpm_min     = int(self.preproc["rpm_min"])
        self.rpm_max     = int(self.preproc["rpm_max"])
        self.fault_classes   = self.label_map["fault_classes"]
        self.severity_levels = self.label_map["severity_levels"]

        # Load the fault classifier
        self.fault_model = FaultClassifier(
            n_faults=len(self.fault_classes),
            in_channels=int(self.preproc["n_channels"]),
        ).to(self.device)
        self.fault_model.load_state_dict(
            torch.load(config.MODEL_PATH, map_location=self.device)
        )
        self.fault_model.eval()

        # Load each severity classifier (if present)
        self.severity_models: dict = {}
        for fault_name, path in config.SEVERITY_MODEL_PATHS.items():
            if path.exists():
                m = SeverityClassifier(
                    n_severity=len(self.severity_levels),
                    in_channels=int(self.preproc["n_channels"]),
                ).to(self.device)
                m.load_state_dict(torch.load(path, map_location=self.device))
                m.eval()
                self.severity_models[fault_name] = m

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
        signal = load_jsonl_signal(path)
        result = self._predict(signal, int(rpm))
        result["meta"]["filename"]   = path.name
        result["meta"]["file_bytes"] = path.stat().st_size
        return result

    # ------------------------------------------------------------------
    def _predict(self, signal: np.ndarray, rpm: int) -> dict:
        signal = signal.astype(np.float32)
        if signal.ndim != 2 or signal.shape[0] != 3:
            raise ValueError(f"Expected (3, N) signal, got {signal.shape}")
        if signal.shape[1] < self.window_size:
            raise ValueError(
                f"Signal has {signal.shape[1]} samples; need >= {self.window_size}."
            )

        windows = make_windows(signal, self.window_size, self.hop_size)

        # ----- Run fault classifier on every window -----
        specs = np.stack([window_to_spectrogram(w) for w in windows], axis=0)
        x_spec = torch.from_numpy(specs).to(self.device)
        rpm_norm = (float(rpm) - self.rpm_min) / max(self.rpm_max - self.rpm_min, 1)
        rpm_t = torch.full(
            (x_spec.size(0), 1), rpm_norm, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            fault_logits = self.fault_model(x_spec, rpm_t)
            fault_probs_per_window = F.softmax(fault_logits, dim=-1).cpu().numpy()

        # ----- Recording-level fault prediction -----
        recording_fault_probs = fault_probs_per_window.mean(axis=0)
        fault_idx = int(np.argmax(recording_fault_probs))
        fault_name = self.fault_classes[fault_idx]
        is_normal = (fault_name == "Normal")

        # ----- Run severity classifier (only if predicted fault is non-Normal) -----
        n_windows = fault_probs_per_window.shape[0]
        n_severity = len(self.severity_levels)
        sev_probs_per_window = np.zeros((n_windows, n_severity), dtype=np.float32)

        if not is_normal and fault_name in self.severity_models:
            sev_model = self.severity_models[fault_name]
            with torch.no_grad():
                sev_logits = sev_model(x_spec, rpm_t)
                sev_probs_per_window = F.softmax(sev_logits, dim=-1).cpu().numpy()
        elif is_normal:
            # Severity not applicable -- pad with uniform low so the dashboard
            # has shape-compatible data. (build_full_dashboard_dict will set
            # predicted_severity to None when fault is Normal.)
            sev_probs_per_window[:, 0] = 1.0
        else:
            # Predicted fault is non-Normal but no severity model is loaded
            # (e.g. severity model artifacts missing). Default to uniform.
            sev_probs_per_window[:, :] = 1.0 / n_severity

        # ----- Hand off to shared dashboard builder -----
        return build_full_dashboard_dict(
            signal=signal,
            rpm=rpm,
            fault_probs_per_window=fault_probs_per_window,
            severity_probs_per_window=sev_probs_per_window,
            model_name=MODEL_DISPLAY_NAME,
        )