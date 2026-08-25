from pathlib import Path

import pytest

from src.model.model_registry import (
    InvalidPromotionError,
    ModelRegistry,
)


def make_registry(
    tmp_path: Path,
) -> ModelRegistry:
    return ModelRegistry(
        registry_path=(
            tmp_path / "registry.json"
        )
    )


def register_and_validate(
    registry: ModelRegistry,
    version: str,
):
    registry.register_model(
        model_name="stock-direction-lstm",
        model_version=version,
        run_id=f"run-{version}",
        artifact_path=(
            f"artifacts/models/{version}"
        ),
        metrics={
            "accuracy": 0.55,
            "precision": 0.55,
            "recall": 0.60,
            "f1": 0.57,
            "roc_auc": 0.60,
            "confusion_matrix": [
                [40, 20],
                [15, 45],
            ],
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
        model_version=version,
        validation=validation,
    )


def promote(
    registry: ModelRegistry,
    version: str,
):
    return registry.promote_model(
        model_name="stock-direction-lstm",
        model_version=version,
        target_status="production",
    )


def test_production_model_can_be_rolled_back(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_and_validate(
        registry,
        "v1",
    )

    promote(
        registry,
        "v1",
    )

    register_and_validate(
        registry,
        "v2",
    )

    promote(
        registry,
        "v2",
    )

    rollback = registry.rollback_model(
        model_name="stock-direction-lstm",
        model_version="v1",
    )

    assert (
        rollback["status"]
        == "production"
    )

    v1 = registry.get_model(
        model_name="stock-direction-lstm",
        model_version="v1",
    )

    v2 = registry.get_model(
        model_name="stock-direction-lstm",
        model_version="v2",
    )

    assert v1["status"] == "production"
    assert v2["status"] == "validated"


def test_rollback_requires_validated_target(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_and_validate(
        registry,
        "v1",
    )

    promote(
        registry,
        "v1",
    )

    registry.register_model(
        model_name="stock-direction-lstm",
        model_version="candidate-v2",
        run_id="run-candidate-v2",
        artifact_path=(
            "artifacts/models/candidate-v2"
        ),
        metrics={},
    )

    with pytest.raises(
        InvalidPromotionError
    ):
        registry.rollback_model(
            model_name="stock-direction-lstm",
            model_version="candidate-v2",
        )


def test_rollback_cannot_target_rejected_model(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_and_validate(
        registry,
        "v1",
    )

    promote(
        registry,
        "v1",
    )

    registry.register_model(
        model_name="stock-direction-lstm",
        model_version="rejected-v2",
        run_id="run-rejected-v2",
        artifact_path=(
            "artifacts/models/rejected-v2"
        ),
        metrics={},
    )

    validation = type(
        "ValidationResult",
        (),
        {
            "passed": False,
            "checks": {
                "accuracy_vs_baseline": False,
                "minimum_roc_auc": False,
            },
            "reasons": [
                "Model failed validation."
            ],
            "to_dict": lambda self: {
                "passed": self.passed,
                "checks": self.checks,
                "reasons": self.reasons,
            },
        },
    )()

    registry.record_validation(
        model_name="stock-direction-lstm",
        model_version="rejected-v2",
        validation=validation,
    )

    with pytest.raises(
        InvalidPromotionError
    ):
        registry.rollback_model(
            model_name="stock-direction-lstm",
            model_version="rejected-v2",
        )


def test_rollback_persists_across_registry_reload(
    tmp_path: Path,
):
    registry_path = (
        tmp_path / "registry.json"
    )

    registry = ModelRegistry(
        registry_path=registry_path
    )

    register_and_validate(
        registry,
        "v1",
    )
    promote(
        registry,
        "v1",
    )

    register_and_validate(
        registry,
        "v2",
    )
    promote(
        registry,
        "v2",
    )

    registry.rollback_model(
        model_name="stock-direction-lstm",
        model_version="v1",
    )

    reloaded = ModelRegistry(
        registry_path=registry_path
    )

    v1 = reloaded.get_model(
        model_name="stock-direction-lstm",
        model_version="v1",
    )

    v2 = reloaded.get_model(
        model_name="stock-direction-lstm",
        model_version="v2",
    )

    assert v1["status"] == "production"
    assert v2["status"] == "validated"
