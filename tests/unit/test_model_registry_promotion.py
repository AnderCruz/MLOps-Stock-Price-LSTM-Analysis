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


def register_model(
    registry: ModelRegistry,
    version: str,
):
    return registry.register_model(
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


def validate_model(
    registry: ModelRegistry,
    version: str,
):
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

    return registry.record_validation(
        model_name="stock-direction-lstm",
        model_version=version,
        validation=validation,
    )


def test_candidate_cannot_be_promoted_to_production(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_model(
        registry,
        "v1",
    )

    with pytest.raises(
        InvalidPromotionError
    ):
        registry.promote_model(
            model_name="stock-direction-lstm",
            model_version="v1",
            target_status="production",
        )


def test_validated_model_can_be_promoted_to_production(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_model(
        registry,
        "v1",
    )

    validate_model(
        registry,
        "v1",
    )

    promoted = registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="v1",
        target_status="production",
    )

    assert promoted["status"] == "production"


def test_rejected_model_cannot_be_promoted_to_production(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_model(
        registry,
        "v1",
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
        model_version="v1",
        validation=validation,
    )

    with pytest.raises(
        InvalidPromotionError
    ):
        registry.promote_model(
            model_name="stock-direction-lstm",
            model_version="v1",
            target_status="production",
        )


def test_promoting_new_version_demotes_previous_production(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    register_model(
        registry,
        "v1",
    )

    validate_model(
        registry,
        "v1",
    )

    registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="v1",
        target_status="production",
    )

    register_model(
        registry,
        "v2",
    )

    validate_model(
        registry,
        "v2",
    )

    promoted = registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="v2",
        target_status="production",
    )

    assert promoted["status"] == "production"

    v1 = registry.get_model(
        model_name="stock-direction-lstm",
        model_version="v1",
    )

    v2 = registry.get_model(
        model_name="stock-direction-lstm",
        model_version="v2",
    )

    assert v1["status"] == "validated"
    assert v2["status"] == "production"


def test_only_one_production_model_exists(
    tmp_path: Path,
):
    registry = make_registry(tmp_path)

    for version in (
        "v1",
        "v2",
        "v3",
    ):
        register_model(
            registry,
            version,
        )

        validate_model(
            registry,
            version,
        )

    for version in (
        "v1",
        "v2",
        "v3",
    ):
        registry.promote_model(
            model_name="stock-direction-lstm",
            model_version=version,
            target_status="production",
        )

    models = registry.list_models(
        model_name="stock-direction-lstm"
    )

    production_models = [
        model
        for model in models
        if model["status"] == "production"
    ]

    assert len(
        production_models
    ) == 1

    assert (
        production_models[0][
            "model_version"
        ]
        == "v3"
    )
