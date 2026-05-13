"""
Train the cascaded 2D-CNN classifiers.

Trains 4 separate models in sequence:
  1. Fault classifier   (4-way: Normal / Unbalance / Misalignment / Looseness)
  2. Severity given Unbalance
  3. Severity given Misalignment
  4. Severity given Looseness

Each model uses a file-level train/val split for honest validation
(no per-recording leakage), warmup + cosine LR schedule, and
SpecAugment-style frequency/time masking on the spectrogram during
training.

Usage from the project root:
    python -m models.cnn2d.train                  # full default training
    python -m models.cnn2d.train --epochs 30      # shorter run
    python -m models.cnn2d.train --seed 7         # try a different seed
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.shared import config as shared_config
from models.shared.data_utils import (
    discover_files, load_jsonl_signal, make_windows, parse_filename,
)
from . import config
from .dataset import SpectrogramDataset
from .model import FaultClassifier, SeverityClassifier
from .stft_utils import SpectrogramCache


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_dataset(files: list, label_kind: str, label_class_lookup: dict) -> dict:
    """Discover, load, and window every file. Produce labels.

    label_kind: "fault" or "severity"
    label_class_lookup: dict mapping class name -> int label
    """
    all_windows, all_rpm, all_label, all_src = [], [], [], []
    for fp in files:
        meta = parse_filename(fp)
        if label_kind == "fault":
            label = label_class_lookup[meta["fault"]]
        elif label_kind == "severity":
            if meta["severity"] is None:
                continue   # skip Normal recordings for severity training
            label = label_class_lookup[meta["severity"]]
        else:
            raise ValueError(f"Unknown label_kind={label_kind}")
        sig = load_jsonl_signal(fp)
        wins = make_windows(sig)
        n = len(wins)
        all_windows.append(wins)
        all_rpm.append(np.full(n, meta["rpm"], dtype=np.int32))
        all_label.append(np.full(n, label, dtype=np.int64))
        all_src.extend([fp.name] * n)
    if not all_windows:
        return {"windows": np.zeros((0, 3, config.WINDOW_SIZE), dtype=np.float32),
                "rpm":      np.zeros((0,), dtype=np.int32),
                "fault":    np.zeros((0,), dtype=np.int64),
                "severity": np.zeros((0,), dtype=np.int64),
                "source":   []}
    out = {
        "windows": np.concatenate(all_windows, axis=0),
        "rpm":     np.concatenate(all_rpm, axis=0),
        "source":  all_src,
    }
    labels = np.concatenate(all_label, axis=0)
    if label_kind == "fault":
        out["fault"]    = labels
        out["severity"] = np.full_like(labels, -1)
    else:
        out["severity"] = labels
        out["fault"]    = np.zeros_like(labels)
    return out


# ---------------------------------------------------------------------------
# Training loop (per-task, cross-entropy only)
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, label_key: str, train: bool):
    model.train() if train else model.eval()
    total_loss, total_n, total_correct = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x_spec = batch["x_spec"].to(device)
            rpm    = batch["rpm"].to(device)
            target = batch[label_key].to(device)
            logits = model(x_spec, rpm)
            loss = torch.nn.functional.cross_entropy(logits, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            B = x_spec.size(0)
            total_loss += loss.item() * B
            total_n += B
            total_correct += int((logits.argmax(dim=-1) == target).sum())
    n = max(total_n, 1)
    return {"loss": total_loss / n, "acc": total_correct / n}


# ---------------------------------------------------------------------------
# File-level train/val split
# ---------------------------------------------------------------------------
def file_level_split(files: list, n_val: int, seed: int,
                     bucket_fn=None) -> tuple:
    """Hold out n_val entire files for validation, balanced across buckets."""
    rng = np.random.RandomState(seed)
    if bucket_fn is None:
        bucket_fn = lambda fp: parse_filename(fp).get("fault", "?")

    by_bucket: dict = {}
    for fp in files:
        try:
            key = bucket_fn(fp)
        except Exception:
            continue
        by_bucket.setdefault(key, []).append(fp)

    val: list = []
    bucket_keys = list(by_bucket.keys())
    rng.shuffle(bucket_keys)
    for key in bucket_keys:
        files_in = list(by_bucket[key])
        if len(files_in) >= 2 and len(val) < n_val:
            rng.shuffle(files_in)
            val.append(files_in[0])
    val = val[:n_val] if len(val) >= n_val else val
    train = [f for f in files if f not in set(val)]
    return train, val


# ---------------------------------------------------------------------------
# Train a single classifier
# ---------------------------------------------------------------------------
def train_classifier(
    *,
    name: str,
    model: torch.nn.Module,
    train_files: list,
    val_files: list,
    label_kind: str,
    label_class_lookup: dict,
    label_key: str,
    save_path: Path,
    history_path: Path,
    args,
    device,
    cache,
):
    """Train one classifier (fault or severity) end-to-end."""
    print()
    print("=" * 70)
    print(f"  Training: {name}")
    print("=" * 70)
    print(f"  train files: {len(train_files)}  |  val files: {len(val_files)}")

    train_data = build_dataset(train_files, label_kind, label_class_lookup)
    val_data   = build_dataset(val_files,   label_kind, label_class_lookup)
    if len(train_data["windows"]) == 0:
        print(f"  [skip] no training windows for {name}")
        return None

    train_ds = SpectrogramDataset(train_data, cache=cache, train=True)
    val_ds   = SpectrogramDataset(val_data,   cache=cache, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                   num_workers=0, pin_memory=(device.type == "cuda"))
        if len(val_ds) > 0 else None
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model parameters: {n_params:,}  (dropout={config.DROPOUT})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY,
    )

    warmup_epochs = min(config.WARMUP_EPOCHS, max(1, args.epochs // 6))
    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / max(1, warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, label_key, train=True)
        if val_loader is not None:
            va = run_epoch(model, val_loader, optimizer, device, label_key, train=False)
        else:
            va = {"loss": float("nan"), "acc": float("nan")}
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  [epoch {epoch:3d}/{args.epochs}] lr={cur_lr:.1e} | "
              f"train acc {tr['acc']:.3f} | val acc {va['acc']:.3f}")
        history.append({"epoch": epoch, "train": tr, "val": va, "lr": cur_lr})

        # Save by val accuracy (or by train acc if no val set)
        cur_metric = va["acc"] if val_loader is not None else tr["acc"]
        if cur_metric > best_val_acc:
            best_val_acc = cur_metric
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    print(f"  [best] epoch {best_epoch} acc {best_val_acc:.3f}")
    history_path.write_text(json.dumps(history, indent=2))
    return {"best_epoch": best_epoch, "best_val_acc": best_val_acc, "history": history}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=shared_config.DATA_DIR)
    parser.add_argument("--epochs",   type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch",    type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr",       type=float, default=config.LEARNING_RATE)
    parser.add_argument("--seed",     type=int, default=config.SEED)
    parser.add_argument("--device",   type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--n-val-files", type=int, default=config.N_VAL_FILES)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[info] device = {device}")
    print(f"[info] seed = {args.seed}")

    all_files = discover_files(args.data_dir)
    print(f"[info] {len(all_files)} files in {args.data_dir.name}")

    cache = None if args.no_cache else SpectrogramCache(config.CACHE_DIR)

    # ===================================================================
    # Stage 1: Train the fault classifier on ALL files
    # ===================================================================
    fault_class_lookup = {c: i for i, c in enumerate(config.FAULT_CLASSES)}
    train_files, val_files = file_level_split(
        all_files, args.n_val_files, args.seed,
        bucket_fn=lambda fp: parse_filename(fp).get("fault", "?"),
    )
    print(f"\n[fault] file-level split: {len(train_files)} train, {len(val_files)} val")
    for fp in val_files:
        print(f"          val: {fp.name}")

    fault_model = FaultClassifier().to(device)
    fault_summary = train_classifier(
        name="Fault classifier (4-way)",
        model=fault_model,
        train_files=train_files,
        val_files=val_files,
        label_kind="fault",
        label_class_lookup=fault_class_lookup,
        label_key="fault",
        save_path=config.MODEL_PATH,
        history_path=config.HISTORY_PATH,
        args=args, device=device, cache=cache,
    )

    # ===================================================================
    # Stage 2: Train one severity classifier per non-Normal fault
    # ===================================================================
    severity_class_lookup = {s: i for i, s in enumerate(config.SEVERITY_LEVELS)}
    severity_summaries = {}

    for fault_name in ["Unbalance", "Misalignment", "Looseness"]:
        # Filter to files of this fault type only
        files_for_fault = [
            fp for fp in all_files
            if (lambda m: m["fault"] == fault_name and m["severity"] is not None)(parse_filename(fp))
        ]
        if not files_for_fault:
            print(f"\n[severity:{fault_name}] no files found — skipping")
            continue

        # File-level split bucketed by severity
        try:
            tr_files, va_files = file_level_split(
                files_for_fault, n_val=2, seed=args.seed,
                bucket_fn=lambda fp: parse_filename(fp).get("severity", "?"),
            )
        except Exception:
            tr_files, va_files = files_for_fault, []

        print(f"\n[severity:{fault_name}] {len(files_for_fault)} files: "
              f"{len(tr_files)} train, {len(va_files)} val")
        for fp in va_files:
            print(f"          val: {fp.name}")

        sev_model = SeverityClassifier().to(device)
        summary = train_classifier(
            name=f"Severity given {fault_name}",
            model=sev_model,
            train_files=tr_files,
            val_files=va_files,
            label_kind="severity",
            label_class_lookup=severity_class_lookup,
            label_key="severity",
            save_path=config.SEVERITY_MODEL_PATHS[fault_name],
            history_path=config.SEVERITY_HISTORY_PATHS[fault_name],
            args=args, device=device, cache=cache,
        )
        severity_summaries[fault_name] = summary

    # ===================================================================
    # Save shared artifacts
    # ===================================================================
    preproc = {
        "model_kind":           "2d_cnn_stft_cascaded",
        "window_size":          config.WINDOW_SIZE,
        "hop_size":             config.HOP_SIZE,
        "stft_window_samples":  config.STFT_WINDOW_SAMPLES,
        "stft_hop_samples":     config.STFT_HOP_SAMPLES,
        "stft_freq_keep_bins":  config.STFT_FREQ_KEEP_BINS,
        "n_channels":           config.N_CHANNELS,
        "rpm_min":              min(config.RPM_VALUES),
        "rpm_max":              max(config.RPM_VALUES),
    }
    config.PREPROC_PATH.write_text(json.dumps(preproc, indent=2))
    label_map = {
        "fault_classes":   config.FAULT_CLASSES,
        "severity_levels": config.SEVERITY_LEVELS,
    }
    config.LABEL_MAP_PATH.write_text(json.dumps(label_map, indent=2))

    # Summary output
    print()
    print("=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    if fault_summary:
        print(f"  Fault model:         best val acc {fault_summary['best_val_acc']:.3f}")
    for fault_name, summary in severity_summaries.items():
        if summary:
            print(f"  Severity {fault_name:13s}: best val acc {summary['best_val_acc']:.3f}")
    print()
    print(f"  Artifacts in: {config.ARTIFACT_DIR}")


if __name__ == "__main__":
    main()