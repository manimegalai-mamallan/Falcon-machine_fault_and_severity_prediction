"""
Triplet-based loader for the friend's data format.

In her dataset, each .jsonl file contains rows like:
    {"axis": "X", "data": [...1024 values...], "time": 0.0}
    {"axis": "Y", "data": [...1024 values...], "time": 0.0}
    {"axis": "Z", "data": [...1024 values...], "time": 0.0}
    {"axis": "X", "data": [...1024 values...], "time": 1.0}
    ...

A "triplet" is one X + one Y + one Z reading. Each triplet becomes one
(1024, 3) window — directly fed to her model with no further windowing.

This module is used by `models/cnn1d/predict.py` and
`models/minirocket/predict.py` to replicate her training-time pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def merge_triplets_from_jsonl(path: str | Path,
                               max_gap_factor: float = 5.0) -> np.ndarray:
    """Load a .jsonl file and merge consecutive X/Y/Z rows into triplets.

    Returns:
        waveforms: (N, 1024, 3) float32 array  --  N triplets, channel-last
                   (matches the friend's `extract_waveforms` layout).

    Raises:
        ValueError if no triplets can be assembled.
    """
    path = Path(path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        raise ValueError(f"No valid JSON rows in {path}")

    # Sort by time
    rows.sort(key=lambda r: r.get("time", 0))

    # Compute median time gap to detect discontinuities
    times = [r.get("time", 0) for r in rows]
    if len(times) >= 2:
        diffs = np.diff(times)
        diffs = diffs[diffs > 0]
        median_gap = float(np.median(diffs)) if len(diffs) else 0.0
    else:
        median_gap = 0.0
    gap_threshold = (median_gap * max_gap_factor) if median_gap > 0 else float("inf")

    # Walk through rows, accumulating one reading per axis. Emit a triplet
    # as soon as all three axes are seen, then reset the buffer.
    triplets: list[dict] = []
    buffer: dict = {}
    last_t = None
    for r in rows:
        axis = r.get("axis")
        data = r.get("data")
        t    = r.get("time", 0)

        if axis not in ("X", "Y", "Z") or not isinstance(data, list):
            continue

        # Time discontinuity -> reset buffer
        if last_t is not None and (t - last_t) > gap_threshold:
            buffer = {}
        last_t = t

        # Duplicate axis before triplet completes -> drop partial buffer
        if axis in buffer:
            buffer = {axis: data}
            continue

        buffer[axis] = data

        if len(buffer) == 3:
            triplets.append({"X": buffer["X"], "Y": buffer["Y"], "Z": buffer["Z"]})
            buffer = {}

    if not triplets:
        raise ValueError(f"No complete X/Y/Z triplets found in {path}")

    # Stack into (N, seq_len, 3). Truncate any abnormal lengths to match the
    # most common one (defensive — should be 1024 throughout).
    seq_len = len(triplets[0]["X"])
    valid = [t for t in triplets
             if len(t["X"]) == seq_len and len(t["Y"]) == seq_len and len(t["Z"]) == seq_len]

    X = np.array([t["X"] for t in valid], dtype=np.float32)
    Y = np.array([t["Y"] for t in valid], dtype=np.float32)
    Z = np.array([t["Z"] for t in valid], dtype=np.float32)
    return np.stack([X, Y, Z], axis=-1)  # (N, seq_len, 3)


def compute_fft_features(waveforms: np.ndarray) -> np.ndarray:
    """Hann window -> rfft -> magnitude -> log1p (matches friend's features.py).

    Args:
        waveforms: (N, seq_len, 3) float array

    Returns:
        (N, seq_len // 2 + 1, 3) log-magnitude spectra, float32.
    """
    seq_len = waveforms.shape[1]
    window = np.hanning(seq_len).astype(np.float32)
    windowed = waveforms * window[None, :, None]
    spectra = np.abs(np.fft.rfft(windowed, axis=1))
    return np.log1p(spectra).astype(np.float32)


def waveforms_to_signal(waveforms: np.ndarray) -> np.ndarray:
    """Convert (N, seq_len, 3) triplet stack into a flat (3, N*seq_len) signal.

    The dashboard builders expect the (3, N) shape format used by the 2D-CNN
    pipeline. This function provides that view by concatenating triplets
    end-to-end. Per-axis content is preserved.
    """
    N, seq_len, C = waveforms.shape
    return waveforms.transpose(2, 0, 1).reshape(C, N * seq_len)  # (3, N*seq_len)
