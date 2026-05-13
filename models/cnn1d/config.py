"""Paths to the friend's 1D CNN keras models and preprocessing artifacts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "models" / "cnn1d" / "artifacts"

# 1D CNN models (keras format)
FAULT_MODEL_PATH = ARTIFACT_DIR / "fault_cnn1d.keras"
SEVERITY_MODEL_PATHS = {
    # Map MY display name -> her file name (note: "Looseness" is "mechanical")
    "Looseness":    ARTIFACT_DIR / "severity_mechanical_cnn1d.keras",
    "Misalignment": ARTIFACT_DIR / "severity_misalignment_cnn1d.keras",
    "Unbalance":    ARTIFACT_DIR / "severity_unbalanced_cnn1d.keras",
}

# Preprocessing artifacts (joblib)
SPECTRA_SCALER_PATH = ARTIFACT_DIR / "spectra_scaler.joblib"
SPEED_SCALER_PATH   = ARTIFACT_DIR / "speed_scaler.joblib"
METADATA_PATH       = ARTIFACT_DIR / "metadata.joblib"

# Threshold above which a fault sigmoid output counts as "detected" (matches
# her training default).
FAULT_THRESHOLD = 0.5
