"""
Evaluate the trained 2D CNN on the held-out test set.

Runs the model on every file in test_data/ and saves:
  - Per-file predictions and ground truth
  - Overall confusion matrix
  - Per-class accuracy

Output: models/cnn2d/artifacts/evaluation_results.json

Run after training (and after setup_data has populated test_data/):
    python -m models.cnn2d.evaluate
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from models.shared import config as shared_config
from models.shared.data_utils import discover_files, parse_filename
from . import config
from .predict import FaultPredictor


def main():
    test_dir = shared_config.TEST_DIR
    if not test_dir.exists() or not list(test_dir.glob("*.jsonl")):
        raise FileNotFoundError(
            f"No files in {test_dir}. Run setup_data first."
        )
    if not FaultPredictor.is_available():
        raise RuntimeError(
            "Trained model not found. Train the model first: "
            "python -m models.cnn2d.train"
        )

    files = discover_files(test_dir)
    print(f"[info] evaluating on {len(files)} test files")

    classes = list(config.FAULT_CLASSES)
    severities = list(config.SEVERITY_LEVELS)
    n_cls = len(classes)
    n_sev = len(severities)

    fault_cm = np.zeros((n_cls, n_cls), dtype=np.int32)
    sev_cm   = np.zeros((n_sev, n_sev), dtype=np.int32)

    per_file = []
    fault_correct = 0
    sev_correct   = 0
    sev_evaluated = 0

    predictor = FaultPredictor()
    for fp in files:
        true = parse_filename(fp)
        true_fault = true["fault"]
        true_sev   = true["severity"]

        result = predictor.predict_file(fp)
        v = result["verdict"]
        pred_fault = v["predicted_fault"]
        pred_sev   = v["predicted_severity"]

        ti = classes.index(true_fault)
        pi = classes.index(pred_fault)
        fault_cm[ti, pi] += 1
        if true_fault == pred_fault:
            fault_correct += 1

        # Severity confusion only when both true and predicted have severity
        if true_sev is not None and pred_sev is not None:
            sti = severities.index(true_sev)
            spi = severities.index(pred_sev)
            sev_cm[sti, spi] += 1
            sev_evaluated += 1
            if true_sev == pred_sev:
                sev_correct += 1

        per_file.append({
            "file":               fp.name,
            "rpm":                int(true["rpm"]),
            "true_fault":         true_fault,
            "predicted_fault":    pred_fault,
            "true_severity":      true_sev,
            "predicted_severity": pred_sev,
            "fault_correct":      bool(true_fault == pred_fault),
            "fault_probs":        v["fault_probs"],
            "severity_probs":     v["severity_probs"],
            "confidence":         v["confidence"],
        })
        mark = "✓" if true_fault == pred_fault else "✗"
        print(f"  {mark} {fp.name:50s} true={true_fault:13s} "
              f"pred={pred_fault:13s} ({v['fault_probs'][pred_fault]:.2f})")

    # Per-class accuracy from the confusion matrix
    fault_per_class = {}
    for i, cls in enumerate(classes):
        n_total = int(fault_cm[i].sum())
        n_correct = int(fault_cm[i, i])
        fault_per_class[cls] = {
            "total":    n_total,
            "correct":  n_correct,
            "accuracy": (n_correct / n_total) if n_total else None,
        }

    sev_per_class = {}
    for i, level in enumerate(severities):
        n_total = int(sev_cm[i].sum())
        n_correct = int(sev_cm[i, i])
        sev_per_class[level] = {
            "total":    n_total,
            "correct":  n_correct,
            "accuracy": (n_correct / n_total) if n_total else None,
        }

    out = {
        "n_files":            len(files),
        "fault_classes":      classes,
        "severity_levels":    severities,
        "fault_confusion":    fault_cm.tolist(),
        "severity_confusion": sev_cm.tolist(),
        "overall_fault_accuracy":    fault_correct / max(len(files), 1),
        "overall_severity_accuracy": (sev_correct / sev_evaluated) if sev_evaluated else None,
        "fault_per_class":    fault_per_class,
        "severity_per_class": sev_per_class,
        "per_file":           per_file,
        "model_name":         "2D CNN",
    }

    out_path = config.ARTIFACT_DIR / "evaluation_results.json"
    out_path.write_text(json.dumps(out, indent=2))

    print()
    print("=" * 60)
    print(f"  Overall fault accuracy:    {out['overall_fault_accuracy']:.1%}")
    if out["overall_severity_accuracy"] is not None:
        print(f"  Overall severity accuracy: {out['overall_severity_accuracy']:.1%}")
    print("=" * 60)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
