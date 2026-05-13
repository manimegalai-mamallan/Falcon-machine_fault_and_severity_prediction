"""
Shared data utilities — used by all three models.

Loads .jsonl recordings (per-axis-block format), parses filenames into
labels, and slices signals into windows.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
_RPM_RE = re.compile(r"(\d{4})")


def parse_filename(path: str | Path) -> dict:
    """Extract fault, severity, and RPM from a single-fault filename.

    Returns:
        {"fault": "Unbalance", "severity": "High", "rpm": 2500}
        {"fault": "Normal", "severity": None, "rpm": 2500}
    """
    name = Path(path).stem.lower()
    tokens = re.split(r"[_\-]+", name)

    fault = None
    severity = None
    rpm = None

    for tok in tokens:
        if tok in config.FILENAME_FAULT_TOKENS:
            fault = config.FILENAME_FAULT_TOKENS[tok]
        if tok in config.FILENAME_SEVERITY_TOKENS:
            severity = config.FILENAME_SEVERITY_TOKENS[tok]
        if rpm is None:
            m = _RPM_RE.fullmatch(tok)
            if m:
                rpm_candidate = int(m.group(1))
                if rpm_candidate in config.RPM_VALUES:
                    rpm = rpm_candidate

    if rpm is None:
        for m in _RPM_RE.finditer(name):
            rpm_candidate = int(m.group(1))
            if rpm_candidate in config.RPM_VALUES:
                rpm = rpm_candidate
                break

    if fault is None:
        raise ValueError(f"Could not determine fault class from: {path}")
    if rpm is None:
        raise ValueError(f"Could not determine RPM from: {path}")

    if fault == "Normal":
        severity = None
    elif severity is None:
        raise ValueError(f"Missing severity in filename: {path}")

    return {"fault": fault, "severity": severity, "rpm": rpm}


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------
def _block_is_healthy(arr: np.ndarray, max_abs_threshold: float = 1e4) -> bool:
    """Reject obviously corrupted blocks (huge DC offsets / spikes)."""
    return float(np.max(np.abs(arr))) < max_abs_threshold


def load_jsonl_signal(path: str | Path) -> np.ndarray:
    """Load a .jsonl vibration recording and return a (3, N) float32 array."""
    path = Path(path)
    per_axis: dict[str, list] = {"X": [], "Y": [], "Z": []}
    n_skipped = {"X": 0, "Y": 0, "Z": 0}

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ax = str(obj.get("axis", "")).upper()
            if ax not in per_axis:
                raise ValueError(
                    f"Unexpected axis {obj.get('axis')!r} in {path}:{line_no}"
                )
            data = obj["data"]
            if not isinstance(data, (list, tuple)):
                raise ValueError(f"Expected 'data' list in {path}:{line_no}")
            block = np.asarray(data, dtype=np.float32)
            if not _block_is_healthy(block):
                n_skipped[ax] += 1
                continue
            per_axis[ax].append(block)

    if any(len(b) == 0 for b in per_axis.values()):
        missing = [a for a, b in per_axis.items() if not b]
        raise ValueError(f"No usable data for axes {missing} in {path}")

    if sum(n_skipped.values()) > 0:
        print(
            f"  [warn] dropped corrupted blocks in {path.name}: "
            f"X={n_skipped['X']} Y={n_skipped['Y']} Z={n_skipped['Z']}"
        )

    x_arr = np.concatenate(per_axis["X"]).astype(np.float32)
    y_arr = np.concatenate(per_axis["Y"]).astype(np.float32)
    z_arr = np.concatenate(per_axis["Z"]).astype(np.float32)

    n = min(len(x_arr), len(y_arr), len(z_arr))
    return np.stack([x_arr[:n], y_arr[:n], z_arr[:n]], axis=0)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def make_windows(
    signal: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    hop_size: int = config.HOP_SIZE,
) -> np.ndarray:
    """Slide a window over a (C, N) signal -> (num_windows, C, window_size)."""
    if signal.ndim != 2:
        raise ValueError(f"Expected (C, N) signal, got shape {signal.shape}")
    C, N = signal.shape
    if N < window_size:
        raise ValueError(f"Signal too short ({N}) for window {window_size}")
    starts = np.arange(0, N - window_size + 1, hop_size)
    return np.stack(
        [signal[:, s : s + window_size] for s in starts], axis=0
    ).astype(np.float32)


def normalize_rpm(rpm: float | int) -> float:
    """Map RPM into [0, 1] over the operating envelope."""
    lo, hi = min(config.RPM_VALUES), max(config.RPM_VALUES)
    return float((rpm - lo) / (hi - lo))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def discover_files(data_dir: Path) -> list:
    files = sorted(Path(data_dir).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
    return files
