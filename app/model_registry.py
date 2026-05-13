"""
Model registry for the Streamlit app.

Knows about all 3 model implementations and provides a single function
`load_predictor(key)` that returns a `FaultPredictor` instance.

Each model exposes a class `FaultPredictor` in its `predict.py` that conforms
to the contract documented in `models/shared/predictor_contract.py`.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelSpec:
    key: str          # internal id used by the app
    label: str        # display name in the UI
    description: str  # short explanatory subtitle
    module: str       # python module path to import
    param_count: str  # rough parameter count for the UI


# Registry: edit here to add or remove models from the selector.
MODELS: dict[str, ModelSpec] = {
    "cnn2d": ModelSpec(
        key="cnn2d",
        label="2D CNN",
        description="STFT spectrograms + ResNet-style backbone",
        module="models.cnn2d.predict",
        param_count="735k",
    ),
    "cnn1d": ModelSpec(
        key="cnn1d",
        label="1D CNN",
        description="Time-domain convolutions on raw signal",
        module="models.cnn1d.predict",
        param_count="~700k",
    ),
    "minirocket": ModelSpec(
        key="minirocket",
        label="MiniRocket",
        description="Random convolutional kernels on raw waveforms + ridge",
        module="models.minirocket.predict",
        param_count="~5k kernels",
    ),
}


def list_models() -> list[ModelSpec]:
    return list(MODELS.values())


def get_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"Unknown model key: {key!r}")
    return MODELS[key]


def is_available(key: str) -> bool:
    """Check whether the model's artifacts are present without instantiating it."""
    spec = get_spec(key)
    try:
        mod = importlib.import_module(spec.module)
    except Exception:
        return False
    cls = getattr(mod, "FaultPredictor", None)
    if cls is None:
        return False
    try:
        return bool(cls.is_available())
    except Exception:
        return False


def load_predictor(key: str, device: Optional[str] = None):
    """Import + instantiate the FaultPredictor for the given model key."""
    spec = get_spec(key)
    mod = importlib.import_module(spec.module)
    cls = getattr(mod, "FaultPredictor")
    return cls(device=device)
