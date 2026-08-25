from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.model.model_inference import (
    ModelInferenceError,
)
from src.model.production_inference import (
    ProductionModelInference,
)


class FakeModel:

    def predict(
        self,
        X,
        verbose=0,
    ):
        assert X.shape == (
            1,
            60,
            3,
        )

        return np.array(
            [[0.80]]
        )


class FakeScaler:

    def transform(self, X):
        return X


def make_registry(tmp_path: Path):
    from src.model.model_registry import (
        ModelRegistry,
    )

    registry = ModelRegistry(
        registry_path=tmp_path / "registry.json"
    )

    registry.register_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        run_id="20260823T233026Z",
        artifact_path=(
            "artifacts/models/refactored-v7"
        ),
        metrics={
            "accuracy": 0.5222,
            "precision": 0.5245,
            "recall": 0.6956,
            "f1": 0.5981,
            "roc_auc": 0.5128,
        },
    )

    validation = type(
        "ValidationResult",
        (),
        {
            "passed": True,
            "checks": {
                "accuracy_vs_baseline": True,
                "minimum_roc_auc": True,
            },
            "reasons": [],
            "to_dict": lambda self: {
                "passed": self.passed,
                "checks": self.checks,
                "reasons": self.reasons,
            },
        },
    )()

    registry.record_validation(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        validation=validation,
    )

    registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        target_status="production",
    )

    return registry


def test_production_inference_loads_production_model(
    tmp_path: Path,
):

    registry = make_registry(
        tmp_path
    )

    with patch(
        "src.model.production_inference.load_model",
        return_value=FakeModel(),
    ), patch(
        "src.model.production_inference.load_scaler",
        return_value=FakeScaler(),
    ):

        inference = ProductionModelInference(
            registry=registry,
            model_name="stock-direction-lstm",
            sequence_length=60,
            feature_columns=[
                "Return",
                "Return_5D",
                "Return_10D",
            ],
        )

        result = inference.predict(
            np.ones(
                (60, 3)
            )
        )

    assert result["direction"] == "UP"
    assert result["probability"] == pytest.approx(
        0.80
    )
    assert (
        result["model_name"]
        == "stock-direction-lstm"
    )
    assert (
        result["model_version"]
        == "refactored-v7"
    )


def test_production_inference_requires_production_model(
    tmp_path: Path,
):

    from src.model.model_registry import (
        ModelRegistry,
    )

    registry = ModelRegistry(
        registry_path=tmp_path / "registry.json"
    )

    registry.register_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        run_id="run-001",
        artifact_path=(
            "artifacts/models/refactored-v7"
        ),
        metrics={},
    )

    with pytest.raises(
        ModelInferenceError
    ):

        ProductionModelInference(
            registry=registry,
            model_name="stock-direction-lstm",
            sequence_length=60,
            feature_columns=[
                "Return",
                "Return_5D",
                "Return_10D",
            ],
        )


def test_production_inference_exposes_model_metadata(
    tmp_path: Path,
):

    registry = make_registry(
        tmp_path
    )

    with patch(
        "src.model.production_inference.load_model",
        return_value=FakeModel(),
    ), patch(
        "src.model.production_inference.load_scaler",
        return_value=FakeScaler(),
    ):

        inference = ProductionModelInference(
            registry=registry,
            model_name="stock-direction-lstm",
            sequence_length=60,
            feature_columns=[
                "Return",
                "Return_5D",
                "Return_10D",
            ],
        )

    metadata = inference.metadata

    assert (
        metadata["model_name"]
        == "stock-direction-lstm"
    )

    assert (
        metadata["model_version"]
        == "refactored-v7"
    )

    assert (
        metadata["status"]
        == "production"
    )
