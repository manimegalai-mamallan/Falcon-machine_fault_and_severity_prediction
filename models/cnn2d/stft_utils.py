"""
STFT preprocessing for the 2D CNN model.

Each (3, T) raw signal window becomes a (3, F, T_frames) log-magnitude
spectrogram that the CNN consumes as a 3-channel image.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import detrend, stft

from . import config


def window_to_spectrogram(window: np.ndarray) -> np.ndarray:
    """Convert one (3, T) window into a (3, F, T_frames) log-mag spectrogram."""
    if window.ndim != 2 or window.shape[0] != 3:
        raise ValueError(f"Expected (3, T) window, got {window.shape}")

    x = detrend(window, axis=-1, type="linear").astype(np.float32)

    specs = []
    for c in range(3):
        _, _, Z = stft(
            x[c],
            nperseg=config.STFT_WINDOW_SAMPLES,
            noverlap=config.STFT_WINDOW_SAMPLES - config.STFT_HOP_SAMPLES,
            boundary=None, padded=False,
        )
        mag = np.abs(Z).astype(np.float32)
        mag = mag[: config.STFT_FREQ_KEEP_BINS, :]
        specs.append(mag)

    spec = np.stack(specs, axis=0)
    spec = np.log1p(spec)
    mean = spec.mean(axis=(1, 2), keepdims=True)
    std  = spec.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((spec - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Disk cache (training only)
# ---------------------------------------------------------------------------
def _cache_key(source: str, idx: int) -> str:
    h = hashlib.md5(f"{source}::{idx}".encode()).hexdigest()[:16]
    return f"{Path(source).stem}_{idx:05d}_{h}.npy"


class SpectrogramCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "_manifest.json"

        params = {
            "stft_window":    config.STFT_WINDOW_SAMPLES,
            "stft_hop":       config.STFT_HOP_SAMPLES,
            "freq_keep_bins": config.STFT_FREQ_KEEP_BINS,
            "window_size":    config.WINDOW_SIZE,
        }
        if self.manifest_path.exists():
            try:
                old = json.loads(self.manifest_path.read_text())
                if old != params:
                    print("[cache] STFT params changed -- invalidating cache.")
                    for f in self.cache_dir.glob("*.npy"):
                        f.unlink()
            except json.JSONDecodeError:
                pass
        self.manifest_path.write_text(json.dumps(params, indent=2))

    def get_or_compute(self, source: str, idx: int, raw_window: np.ndarray) -> np.ndarray:
        path = self.cache_dir / _cache_key(source, idx)
        if path.exists():
            return np.load(path)
        spec = window_to_spectrogram(raw_window)
        np.save(path, spec)
        return spec
