from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.api.app import create_production_app
from src.model.model_inference import ModelInferenceError


def test_create_production_app_uses_injected_inference():
    inference = MagicMock()

    app = create_production_app(inference=inference)

    assert app.title == "Stock Direction Prediction API"
    assert app.version == "1.0.0"


def test_create_production_app_requires_production_model(monkeypatch):
    registry = MagicMock()

    monkeypatch.setenv(
        "MODEL_REGISTRY_PATH",
        str(Path("custom") / "registry.json"),
    )
    monkeypatch.setenv("MODEL_NAME", "stock-direction-lstm")
    monkeypatch.setenv("MODEL_SEQUENCE_LENGTH", "60")

    registry.get_production_model.return_value = None

    with pytest.raises(ModelInferenceError, match="No production model"):
        create_production_app(registry=registry)