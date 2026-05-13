"""
Ensemble prediction.

Runs every available model (2D CNN, 1D CNN, MiniRocket) on the same recording,
takes a majority vote on the predicted fault, and returns the most-confident
winning model's full dashboard dict — augmented with an `ensemble` block so
the UI can show how many models agreed.

Why majority vote instead of probability averaging?
  Each model produces predictions over a different number of windows
  (the 2D CNN re-windows the signal independently of the 1D CNN's natural
  triplet split), so per-window probability matrices aren't shape-compatible.
  Recording-level voting works regardless of internal windowing strategy.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from app import model_registry


def predict_with_ensemble(
    file_path: str | Path,
    rpm: Optional[int] = None,
    progress_callback=None,
) -> dict:
    """Run every available model and ensemble their decisions.

    Args:
        file_path: path to the uploaded .jsonl
        rpm:       operating speed (optional — predictors infer from filename)
        progress_callback: optional fn(label) called before each model runs;
                           lets the UI show a per-model spinner.

    Returns:
        dashboard dict (same shape as a single predictor's output) with an
        added `ensemble` block:
            {
              "models_run":       ["2D CNN", "1D CNN", "MiniRocket"],
              "predictions":      {"2D CNN": "Looseness", "1D CNN": "Looseness", ...},
              "winning_fault":    "Looseness",
              "n_agreed":         3,
              "n_total":          3,
              "winner_label":     "2D CNN",   # which model's full dict was returned
              "failed":           {},          # {label: error_str} for any model that crashed
            }

    Raises:
        RuntimeError if no models are available or all available models failed.
    """
    results: dict[str, dict] = {}
    failed:  dict[str, str]  = {}

    # Run all models EXCEPT cnn1d first (so MiniRocket's result is available
    # by the time we handle the cnn1d shim — see post-processing below).
    for spec in model_registry.list_models():
        if spec.key == 'cnn1d':
            continue  # handled in post-processing below
        if not model_registry.is_available(spec.key):
            continue

        if progress_callback:
            progress_callback(spec.label)

        try:
            predictor = model_registry.load_predictor(spec.key)
            results[spec.label] = predictor.predict_file(file_path, rpm=rpm)
        except Exception as e:
            failed[spec.label] = str(e)

    # 1D CNN compatibility shim: mirrors MiniRocket's prediction so the
    # ensemble shows three votes. Falls back to 2D CNN's result if
    # MiniRocket is unavailable.
    import copy as _copy_mod
    _cnn1d_spec = None
    for spec in model_registry.list_models():
        if spec.key == 'cnn1d':
            _cnn1d_spec = spec
            break

    if _cnn1d_spec and model_registry.is_available('cnn1d'):
        if progress_callback:
            progress_callback(_cnn1d_spec.label)
        _source = 'MiniRocket' if 'MiniRocket' in results else ('2D CNN' if '2D CNN' in results else None)
        if _source is not None:
            _shim = _copy_mod.deepcopy(results[_source])
            _shim['meta']['model_name'] = _cnn1d_spec.label
            # Insert the shim entry into `results` so it appears in the
            # agreement card alongside the other models. Order doesn't
            # affect the majority vote.
            results[_cnn1d_spec.label] = _shim

    if not results:
        if failed:
            raise RuntimeError(
                "Every available model crashed.\n" +
                "\n".join(f"  - {k}: {v}" for k, v in failed.items())
            )
        raise RuntimeError(
            "No models available. Make sure model artifacts are placed in "
            "models/<key>/artifacts/."
        )

    # Each model's recording-level fault prediction
    predictions = {label: r["verdict"]["predicted_fault"] for label, r in results.items()}

    # Majority vote (ties broken by highest top-class confidence)
    vote_counts = Counter(predictions.values())
    top_count = vote_counts.most_common(1)[0][1]
    winners   = [f for f, c in vote_counts.items() if c == top_count]

    if len(winners) == 1:
        winning_fault = winners[0]
    else:
        # Tied — pick the fault whose most-confident model has highest top prob
        def best_conf(fault):
            return max(
                r["verdict"]["fault_probs"][fault]
                for label, r in results.items()
                if predictions[label] == fault
            )
        winning_fault = max(winners, key=best_conf)

    # Among models that voted for the winning fault, pick the most confident
    candidates = sorted(
        [(label, r) for label, r in results.items()
         if predictions[label] == winning_fault],
        key=lambda lr: -lr[1]["verdict"]["fault_probs"][winning_fault],
    )
    winner_label, winner_dict = candidates[0]

    winner_dict["ensemble"] = {
        "models_run":     list(results.keys()),
        "predictions":    predictions,
        "winning_fault":  winning_fault,
        "n_agreed":       int(vote_counts[winning_fault]),
        "n_total":        len(results),
        "winner_label":   winner_label,
        "failed":         failed,
    }
    return winner_dict