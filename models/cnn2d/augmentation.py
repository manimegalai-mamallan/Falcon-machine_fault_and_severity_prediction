"""Conservative augmentation that preserves the fault signature."""
from __future__ import annotations

import numpy as np


def add_gaussian_noise(window: np.ndarray, snr_db: float = 25.0) -> np.ndarray:
    sig_power   = np.mean(window ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=window.shape)
    return (window + noise).astype(np.float32)


def random_amplitude_scale(window: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    scale = np.random.uniform(lo, hi, size=(window.shape[0], 1)).astype(np.float32)
    return window * scale


def random_time_shift(window: np.ndarray, max_shift: int = 128) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(window, shift, axis=-1)
