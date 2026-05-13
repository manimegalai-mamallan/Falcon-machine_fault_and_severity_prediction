"""Paths to the friend's MiniRocket models and preprocessing artifacts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "models" / "minirocket" / "artifacts"

# MiniRocket models (sklearn/joblib format)
FAULT_MODEL_PATH = ARTIFACT_DIR / "fault_minirocket.joblib"
SEVERITY_MODEL_PATHS = {
    "Looseness":    ARTIFACT_DIR / "severity_mechanical_minirocket.joblib",
    "Misalignment": ARTIFACT_DIR / "severity_misalignment_minirocket.joblib",
    "Unbalance":    ARTIFACT_DIR / "severity_unbalanced_minirocket.joblib",
}

# Preprocessing artifacts (only metadata is consulted; MiniRocket operates
# on raw waveforms, so spectra/speed scalers aren't applied here)
METADATA_PATH = ARTIFACT_DIR / "metadata.joblib"

# Threshold above which a fault sigmoid output counts as "detected"
FAULT_THRESHOLD = 0.5
