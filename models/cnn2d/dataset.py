"""
PyTorch Dataset wrapper for the 2D CNN.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from models.shared.data_utils import normalize_rpm
from . import config
from .augmentation import (
    add_gaussian_noise, random_amplitude_scale, random_time_shift,
)
from .stft_utils import SpectrogramCache, window_to_spectrogram


class SpectrogramDataset(Dataset):
    def __init__(self, data: dict, cache: SpectrogramCache | None = None,
                 train: bool = False):
        self.windows  = data["windows"]
        self.rpm      = data["rpm"]
        self.fault    = data["fault"]
        self.severity = data["severity"]
        self.source   = data["source"]
        self.cache = cache
        self.train = train

    def __len__(self) -> int:
        return len(self.windows)

    def _augment(self, w: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            w = random_time_shift(w)
        if np.random.rand() < 0.5:
            w = random_amplitude_scale(w)
        if np.random.rand() < 0.3:
            w = add_gaussian_noise(w, snr_db=float(np.random.uniform(20, 30)))
        return w

    def __getitem__(self, idx: int) -> dict:
        if self.train:
            win = self._augment(self.windows[idx].copy())
            spec = window_to_spectrogram(win)
        elif self.cache is not None:
            spec = self.cache.get_or_compute(
                self.source[idx], idx, self.windows[idx]
            )
        else:
            spec = window_to_spectrogram(self.windows[idx])

        rpm_norm = normalize_rpm(int(self.rpm[idx]))
        sev = int(self.severity[idx])
        sev_mask = 0.0 if sev < 0 else 1.0
        sev_target = 0 if sev < 0 else int(sev)

        return {
            "x_spec":        torch.from_numpy(spec),
            "rpm":           torch.tensor([rpm_norm], dtype=torch.float32),
            "fault":         torch.tensor(int(self.fault[idx]), dtype=torch.long),
            "severity":      torch.tensor(sev_target, dtype=torch.long),
            "severity_mask": torch.tensor(sev_mask, dtype=torch.float32),
        }
