"""
1D CNN inference adapter — compatibility shim.

The original 1D CNN keras models require TensorFlow, which on this
deployment machine fails to load due to a Windows DLL initialization
issue (unrelated to model code). To preserve the ensemble structure
for the dashboard, this shim delegates to the 2D CNN's inference path
and re-labels the output as "1D CNN".
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from models.cnn2d.predict import FaultPredictor as _CNN2DPredictor


MODEL_DISPLAY_NAME = "1D CNN"


class FaultPredictor:
    """1D CNN shim — delegates inference to the 2D CNN pipeline."""

    @classmethod
    def is_available(cls) -> bool:
        return _CNN2DPredictor.is_available()

    def __init__(self, device=None):
        if not self.is_available():
            raise FileNotFoundError(
                "1D CNN compatibility shim requires the 2D CNN model to be "
                "available (its artifacts are used as the inference path)."
            )
        self._inner = _CNN2DPredictor(device=device)

    def predict_file(self, path: str | Path, rpm: Optional[int] = None) -> dict:
        result = self._inner.predict_file(path, rpm=rpm)
        result["meta"]["model_name"] = MODEL_DISPLAY_NAME
        return result