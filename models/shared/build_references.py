"""
Build reference FFT data from your training dataset.

Two separate things are computed and saved as one .npz file:

  1. Healthy references — average FFT of Normal recordings, per RPM.
     Used by the dashboard's "healthy vs measured" overlay.

  2. Class signatures — average FFT of every fault class, per RPM.
     Used by the dashboard's "fault pattern comparison" panel and
     the "fault signature library" reference.

Both are computed identically — Y-axis FFT averaged across all windows
of all training files in the (class, RPM) bucket. No model required.

Usage (from project root):
    python -m models.shared.build_references
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import detrend

from . import config
from .data_utils import (
    load_jsonl_signal,
    make_windows,
    parse_filename,
)


DISPLAY_FREQ_MAX_HZ = config.FFT_DISPLAY_MAX_HZ


def _fft_per_axis(window: np.ndarray, sample_rate: float) -> tuple:
    """Return (freqs, mag_x, mag_y, mag_z) for one (3, T) window."""
    x = detrend(window, axis=-1, type="linear").astype(np.float32)
    n = window.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mags = []
    for c in range(3):
        spec = np.abs(np.fft.rfft(x[c])) / n
        mags.append(spec.astype(np.float32))
    return freqs, mags[0], mags[1], mags[2]


def _file_avg_fft(path: Path, sample_rate: float) -> dict:
    """Average FFT across all windows of one file."""
    sig = load_jsonl_signal(path)
    windows = make_windows(sig)
    if len(windows) == 0:
        raise ValueError(f"No windows extracted from {path}")
    freqs = None
    sums = [None, None, None]
    for w in windows:
        f, mx, my, mz = _fft_per_axis(w, sample_rate)
        if freqs is None:
            freqs = f
            sums[0] = mx.copy(); sums[1] = my.copy(); sums[2] = mz.copy()
        else:
            sums[0] += mx; sums[1] += my; sums[2] += mz
    n = len(windows)
    return {
        "freqs": freqs.astype(np.float32),
        "mag_x": (sums[0] / n).astype(np.float32),
        "mag_y": (sums[1] / n).astype(np.float32),
        "mag_z": (sums[2] / n).astype(np.float32),
    }


def main():
    data_dir = config.DATA_DIR
    test_dir = config.TEST_DIR
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}. "
            "Place the folder at the project root and run setup_data first."
        )

    # Reference spectra are NOT training data for the model.
    # We compute them from every available recording so the dashboard
    # has a healthy baseline at every RPM, regardless of train/test split.
    train_files = list(data_dir.glob("*.jsonl"))
    test_files  = list(test_dir.glob("*.jsonl")) if test_dir.exists() else []
    files = sorted(train_files + test_files)
    print(f"[info] scanning {len(train_files)} training + "
          f"{len(test_files)} test = {len(files)} total files for references")

    # Group files by (fault, rpm)
    by_class_rpm = defaultdict(list)
    by_class     = defaultdict(list)
    by_rpm_normal = {}          # for healthy reference

    for fp in files:
        try:
            meta = parse_filename(fp)
        except Exception:
            continue
        fault = meta["fault"]
        rpm   = meta["rpm"]
        by_class_rpm[(fault, rpm)].append(fp)
        by_class[fault].append(fp)
        if fault == "Normal" and rpm not in by_rpm_normal:
            by_rpm_normal[rpm] = fp

    if not by_rpm_normal:
        raise RuntimeError(
            f"No Normal recordings in {data_dir.name}/. Need at least one "
            "Normal_<RPM>_*.jsonl per RPM."
        )

    sample_rate = config.SAMPLE_RATE_HZ
    out_dict = {}      # what we'll save into the npz
    metadata = {       # human-readable summary
        "rpms": [],
        "classes": [],
        "healthy_sources": {},
        "class_file_counts": {},
        "class_rpm_file_counts": {},
        "freq_max_hz": DISPLAY_FREQ_MAX_HZ,
    }

    # ----- Healthy reference (per RPM) -----
    print("\n[1] Healthy reference (per RPM)")
    for rpm, fp in sorted(by_rpm_normal.items()):
        print(f"    {rpm} RPM  <- {fp.name}")
        avg = _file_avg_fft(fp, sample_rate)
        keep = avg["freqs"] <= DISPLAY_FREQ_MAX_HZ
        out_dict[f"healthy_{rpm}_freqs"] = avg["freqs"][keep].astype(np.float32)
        out_dict[f"healthy_{rpm}_mag_x"] = avg["mag_x"][keep].astype(np.float32)
        out_dict[f"healthy_{rpm}_mag_y"] = avg["mag_y"][keep].astype(np.float32)
        out_dict[f"healthy_{rpm}_mag_z"] = avg["mag_z"][keep].astype(np.float32)
        metadata["healthy_sources"][str(rpm)] = fp.name
        if rpm not in metadata["rpms"]:
            metadata["rpms"].append(rpm)

    # ----- Per-class signature (averaged across RPMs) -----
    print("\n[2] Per-class signature (averaged across all RPMs)")
    for fault, fps in sorted(by_class.items()):
        print(f"    {fault}: averaging {len(fps)} files")
        sums = None; n_total = 0; freqs = None
        for fp in fps:
            avg = _file_avg_fft(fp, sample_rate)
            if freqs is None:
                freqs = avg["freqs"]
                sums = {"x": np.zeros_like(avg["mag_x"]),
                        "y": np.zeros_like(avg["mag_y"]),
                        "z": np.zeros_like(avg["mag_z"])}
            sums["x"] += avg["mag_x"]
            sums["y"] += avg["mag_y"]
            sums["z"] += avg["mag_z"]
            n_total  += 1
        keep = freqs <= DISPLAY_FREQ_MAX_HZ
        out_dict[f"class_{fault}_freqs"] = freqs[keep].astype(np.float32)
        out_dict[f"class_{fault}_mag_x"] = (sums["x"][keep] / n_total).astype(np.float32)
        out_dict[f"class_{fault}_mag_y"] = (sums["y"][keep] / n_total).astype(np.float32)
        out_dict[f"class_{fault}_mag_z"] = (sums["z"][keep] / n_total).astype(np.float32)
        metadata["class_file_counts"][fault] = n_total
        if fault not in metadata["classes"]:
            metadata["classes"].append(fault)

    # ----- Per-(class, rpm) signature (for RPM-aware comparison) -----
    print("\n[3] Per-(class, RPM) signature")
    for (fault, rpm), fps in sorted(by_class_rpm.items()):
        print(f"    {fault:13s} @ {rpm} RPM  ({len(fps)} files)")
        sums = None; n_total = 0; freqs = None
        for fp in fps:
            avg = _file_avg_fft(fp, sample_rate)
            if freqs is None:
                freqs = avg["freqs"]
                sums = {"x": np.zeros_like(avg["mag_x"]),
                        "y": np.zeros_like(avg["mag_y"]),
                        "z": np.zeros_like(avg["mag_z"])}
            sums["x"] += avg["mag_x"]
            sums["y"] += avg["mag_y"]
            sums["z"] += avg["mag_z"]
            n_total += 1
        keep = freqs <= DISPLAY_FREQ_MAX_HZ
        out_dict[f"classrpm_{fault}_{rpm}_freqs"] = freqs[keep].astype(np.float32)
        out_dict[f"classrpm_{fault}_{rpm}_mag_x"] = (sums["x"][keep] / n_total).astype(np.float32)
        out_dict[f"classrpm_{fault}_{rpm}_mag_y"] = (sums["y"][keep] / n_total).astype(np.float32)
        out_dict[f"classrpm_{fault}_{rpm}_mag_z"] = (sums["z"][keep] / n_total).astype(np.float32)
        metadata["class_rpm_file_counts"][f"{fault}_{rpm}"] = n_total

    # ----- Save -----
    out_path = config.HEALTHY_REFERENCES_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict)
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    print(f"\n[done] wrote {out_path}")
    print(f"       wrote {out_path.with_suffix('.json')}")
    print(f"       contents: {len(out_dict)} arrays "
          f"(healthy + class + class-rpm signatures)")


if __name__ == "__main__":
    main()