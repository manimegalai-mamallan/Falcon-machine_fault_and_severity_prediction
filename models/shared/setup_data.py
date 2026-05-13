"""
Setup script: split the dataset folder into MMS_Data/ (train) and test_data/.

Performs a stratified train/test split:
  - Groups files by fault class (Normal / Unbalance / Misalignment / Looseness)
  - From each group, holds out ~20% of files for testing
  - Ensures every fault class is represented in both training and test sets
  - Uses a fixed random seed so the split is reproducible

The split is also idempotent. Running this script multiple times always
produces the same train/test partition.

Usage (from the project root):
    python -m models.shared.setup_data

After this, the dashboard's accuracy can be evaluated on test_data/, while
training only sees MMS_Data/.
"""
from __future__ import annotations

import random
import shutil

from . import config
from .data_utils import parse_filename


TEST_FRACTION = 0.20   # fraction of files held out for testing
SPLIT_SEED    = 42     # fixed seed -> deterministic, reproducible split


def main():
    data_dir = config.DATA_DIR
    test_dir = config.TEST_DIR
    test_dir.mkdir(parents=True, exist_ok=True)

    # Combine files from both folders so the split is consistent across re-runs
    in_train = {f.name: f for f in data_dir.glob("*.jsonl")} if data_dir.exists() else {}
    in_test  = {f.name: f for f in test_dir.glob("*.jsonl")}
    all_names = sorted(set(in_train) | set(in_test))

    if not all_names:
        raise FileNotFoundError(
            f"No .jsonl files found in {data_dir} or {test_dir}.\n"
            f"Place the {data_dir.name}/ folder (containing all .jsonl files) "
            "in the project root before running setup."
        )

    # Group by (fault, severity) — Normal has severity=None
    by_bucket: dict[tuple, list[str]] = {}
    skipped: list[str] = []
    for name in all_names:
        try:
            meta = parse_filename(name)
            key = (meta["fault"], meta.get("severity"))
            by_bucket.setdefault(key, []).append(name)
        except Exception as e:
            skipped.append(f"{name}  ({e})")

    if skipped:
        print(f"[warn] could not parse {len(skipped)} filenames "
              "(they will stay in the training folder):")
        for line in skipped:
            print(f"   - {line}")

    # ----- Per-(fault, severity) split -----
    # For each bucket with >= 2 files, hold out exactly 1 file for testing.
    # This guarantees: every (fault, severity) combination has at least 1
    # training example AND 1 test example. This is critical for severity
    # prediction, since each severity model needs multiple examples per
    # severity level to learn meaningful boundaries.
    rng = random.Random(SPLIT_SEED)
    test_set: set[str] = set()
    print()
    print("Per-(fault, severity) train / test split:")
    print("-" * 60)
    for key in sorted(by_bucket, key=lambda k: (k[0], str(k[1]))):
        names = sorted(by_bucket[key])
        fault, severity = key
        sev_label = severity or "—"
        if len(names) >= 2:
            sample = [rng.choice(names)]
            test_set.update(sample)
            n_test = 1
        else:
            n_test = 0
        n_train = len(names) - n_test
        print(f"  {fault:13s}  {sev_label:6s}  {len(names):2d} files  "
              f"->  {n_train} train  /  {n_test} test")
    print("-" * 60)

    # Move each file to its correct folder
    n_moved = 0
    for name in all_names:
        target_dir = test_dir if name in test_set else data_dir
        current = in_test.get(name) or in_train.get(name)
        if current.parent == target_dir:
            continue  # already in the right place
        target_path = target_dir / name
        if target_path.exists():
            current.unlink()
        else:
            shutil.move(str(current), str(target_path))
        n_moved += 1

    train_files = sorted(data_dir.glob("*.jsonl")) if data_dir.exists() else []
    test_files  = sorted(test_dir.glob("*.jsonl"))

    print()
    print(f"  moved {n_moved} files into place")
    print(f"  {data_dir.name}/   : {len(train_files)} files (training)")
    print(f"  {test_dir.name}/   : {len(test_files)} files (held-out test)")
    print()
    if test_files:
        print("Test files (kept aside, never seen during training):")
        for f in test_files:
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()