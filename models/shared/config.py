"""
Shared configuration used by all three models and the Streamlit app.

Single source of truth for:
  - paths (project root, data/, test_data/, model artifact dirs)
  - class definitions (FAULT_CLASSES, SEVERITY_LEVELS, RPM_VALUES)
  - filename-parsing tokens (handles dataset typos)
  - signal-processing parameters (sample rate, window size, hop)
  - dashboard-display constants (FFT cutoff frequency, etc.)

Hyperparameters specific to one model (e.g. learning rate, model channels)
live inside that model's own folder.
"""
from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at  models/shared/config.py
# So PROJECT_ROOT = ../../ from this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR  = PROJECT_ROOT / "MMS_Data"
TEST_DIR  = PROJECT_ROOT / "test_data"
MODELS_DIR = PROJECT_ROOT / "models"

# Per-model directories
CNN2D_DIR       = MODELS_DIR / "cnn2d"
CNN1D_DIR       = MODELS_DIR / "cnn1d"
MINIROCKET_DIR  = MODELS_DIR / "minirocket"

CNN2D_ARTIFACTS      = CNN2D_DIR / "artifacts"
CNN1D_ARTIFACTS      = CNN1D_DIR / "artifacts"
MINIROCKET_ARTIFACTS = MINIROCKET_DIR / "artifacts"

# Healthy reference FFTs live with the cnn2d model since cnn2d builds them,
# but they are reusable across all models.
HEALTHY_REFERENCES_PATH = CNN2D_ARTIFACTS / "healthy_references.npz"


# ---------------------------------------------------------------------------
# Classes and labels
# ---------------------------------------------------------------------------
FAULT_CLASSES   = ["Normal", "Unbalance", "Misalignment", "Looseness"]
SEVERITY_LEVELS = ["Low", "Medium", "High"]
RPM_VALUES      = [2000, 2500, 3000]

# Filename token -> fault class. Multiple tokens map to the same class to
# handle typos and synonyms in the dataset.
FILENAME_FAULT_TOKENS = {
    "normal":       "Normal",
    "unbalanced":   "Unbalance",
    "unbalance":    "Unbalance",
    "misalignment": "Misalignment",
    "misaligment":  "Misalignment",   # handles the typo in the dataset
    "misaligned":   "Misalignment",
    "mechanical":   "Looseness",       # "Mechanical" in filenames = Looseness
    "looseness":    "Looseness",
    "loose":        "Looseness",
}
FILENAME_SEVERITY_TOKENS = {"low": "Low", "medium": "Medium", "high": "High"}


# ---------------------------------------------------------------------------
# Train/test split is performed programmatically in setup_data.py.
# It groups files by fault class and holds out ~20% of each class for testing,
# using a fixed random seed so the split is deterministic.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Signal processing (shared across all models)
# ---------------------------------------------------------------------------
N_CHANNELS     = 3                    # X, Y, Z accelerometer axes
SAMPLE_RATE_HZ = 945                  # samples per second

WINDOW_SIZE = 4096                    # samples per window (~4.3 s at 945 Hz)
HOP_SIZE    = 2048                    # 50% overlap


# ---------------------------------------------------------------------------
# Dashboard display constants
# ---------------------------------------------------------------------------
# Frequency range to show on FFT panels. Covers 1x..4x for all RPMs in use.
FFT_DISPLAY_MAX_HZ = 200.0

# Time-domain trace downsampling target (so plotly doesn't choke on 850k pts).
TIME_TRACE_TARGET_POINTS = 1500

# Spectrogram display size.
SPEC_DISPLAY_TIME_BINS = 96