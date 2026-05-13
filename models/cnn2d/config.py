"""
2D-CNN-specific configuration.

Anything not specific to this model lives in models/shared/config.py.
"""
from __future__ import annotations

from pathlib import Path

from models.shared import config as shared_config


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACT_DIR     = shared_config.CNN2D_ARTIFACTS
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH       = ARTIFACT_DIR / "fault_model.pt"
PREPROC_PATH     = ARTIFACT_DIR / "preprocessor.json"
LABEL_MAP_PATH   = ARTIFACT_DIR / "label_map.json"
HISTORY_PATH     = ARTIFACT_DIR / "training_history.json"
CACHE_DIR        = ARTIFACT_DIR / "stft_cache"

# Severity models — one per non-Normal fault (cascaded design).
SEVERITY_MODEL_PATHS = {
    "Unbalance":    ARTIFACT_DIR / "severity_unbalance.pt",
    "Misalignment": ARTIFACT_DIR / "severity_misalignment.pt",
    "Looseness":    ARTIFACT_DIR / "severity_looseness.pt",
}
SEVERITY_HISTORY_PATHS = {
    "Unbalance":    ARTIFACT_DIR / "severity_unbalance_history.json",
    "Misalignment": ARTIFACT_DIR / "severity_misalignment_history.json",
    "Looseness":    ARTIFACT_DIR / "severity_looseness_history.json",
}


# ---------------------------------------------------------------------------
# STFT preprocessing
# ---------------------------------------------------------------------------
STFT_WINDOW_SAMPLES = 512
STFT_HOP_SAMPLES    = 64
STFT_FREQ_KEEP_BINS = 96


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
BASE_CHANNELS = 32
RPM_EMB_DIM   = 32
FUSION_DIM    = 128
DROPOUT       = 0.15      # was 0.3 — over-regularized for 23 files


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
SEED          = 42
BATCH_SIZE    = 32
NUM_EPOCHS    = 60        # was 30 — give the model more chances
LEARNING_RATE = 3e-4      # was 5e-4 — slightly slower for small data
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 5         # NEW — linear warmup before cosine decay
N_VAL_FILES   = 3         # NEW — hold out N entire files for validation
                          #       (file-level split, not window-level)

LAMBDA_FAULT    = 1.0
LAMBDA_SEVERITY = 0.5     # was 0.7 — less weight, severity is data-limited

# SpecAugment-style frequency masking (applied to the spectrogram during training)
FREQ_MASK_PROB    = 0.5
FREQ_MASK_MAX_BINS = 12
TIME_MASK_PROB    = 0.4
TIME_MASK_MAX_FRAMES = 8


# Re-export shared values for convenience inside this package
SAMPLE_RATE_HZ = shared_config.SAMPLE_RATE_HZ
WINDOW_SIZE    = shared_config.WINDOW_SIZE
HOP_SIZE       = shared_config.HOP_SIZE
N_CHANNELS     = shared_config.N_CHANNELS
FAULT_CLASSES  = shared_config.FAULT_CLASSES
SEVERITY_LEVELS = shared_config.SEVERITY_LEVELS
RPM_VALUES     = shared_config.RPM_VALUES
DATA_DIR       = shared_config.DATA_DIR
TEST_DIR       = shared_config.TEST_DIR