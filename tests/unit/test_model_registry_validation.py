from pathlib import Path

import pytest

from src.model.model_registry import (
    InvalidPromotionError,
    ModelRegistry,
)
from src.model.model_validation import (
    ValidationResult,
)


def make_registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(
        registry_path=tmp_path / "registry.json"
    )


def register_candidate(
    registry: ModelRegistry,
    version: str = "refactored-v7",
):
    return registry.register_model(
        model_name="stock-direction-lstm",
        model_version=version,
        run_id="20260823T233026Z",
        artifact_path="artifacts/models/refactored-v7",
        metrics={
            "accuracy": 0.60,
            "precision": 0.60,
            "recall": 0.65,
            "f1": 0.62,
            "roc_auc": 0.65,
            "confusion_matrix": [
                [50, 40],
                [30, 60],
            ],
        },
    )


def test_candidate_can_store_validation_result(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    record = register_candidate(
        registry
    )

    validation = ValidationResult(
        passed=True,
        checks={
            "accuracy_vs_baseline": True,
            "minimum_roc_auc": True,
        },
        reasons=[],
    )

    updated = registry.record_validation(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        validation=validation,
    )

    assert updated["status"] == "validated"

    assert updated["validation"]["passed"] is True

    assert updated["validation"]["checks"][
        "accuracy_vs_baseline"
    ] is True

    assert updated["validation"]["reasons"] == []


def test_failed_validation_rejects_candidate(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_candidate(
        registry
    )

    validation = ValidationResult(
        passed=False,
        checks={
            "accuracy_vs_baseline": True,
            "minimum_roc_auc": False,
        },
        reasons=[
            "ROC-AUC is below the minimum "
            "required threshold."
        ],
    )

    updated = registry.record_validation(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        validation=validation,
    )

    assert updated["status"] == "rejected"

    assert updated["validation"]["passed"] is False

    assert updated["validation"]["checks"][
        "minimum_roc_auc"
    ] is False

    assert len(
        updated["validation"]["reasons"]
    ) == 1


def test_validated_model_can_be_promoted(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_candidate(
        registry
    )

    validation = ValidationResult(
        passed=True,
        checks={
            "accuracy_vs_baseline": True,
            "minimum_roc_auc": True,
        },
        reasons=[],
    )

    registry.record_validation(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        validation=validation,
    )

    promoted = registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        target_status="production",
    )

    assert promoted["status"] == "production"


def test_rejected_model_cannot_be_promoted(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_candidate(
        registry
    )

    validation = ValidationResult(
        passed=False,
        checks={
            "accuracy_vs_baseline": False,
            "minimum_roc_auc": False,
        },
        reasons=[
            "Accuracy does not exceed "
            "the majority baseline.",
            "ROC-AUC is below the minimum "
            "required threshold.",
        ],
    )

    registry.record_validation(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        validation=validation,
    )

    with pytest.raises(
        InvalidPromotionError
    ):
        registry.promote_model(
            model_name="stock-direction-lstm",
            model_version="refactored-v7",
            target_status="production",
        )
