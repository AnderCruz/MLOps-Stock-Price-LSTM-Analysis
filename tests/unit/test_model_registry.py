from pathlib import Path

import pytest

from src.model.model_registry import (
    ModelAlreadyRegisteredError,
    ModelNotFoundError,
    InvalidPromotionError,
    ModelRegistry,
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
            "accuracy": 0.5222222222,
            "precision": 0.5245901639,
            "recall": 0.6956521739,
            "f1": 0.5981308411,
            "roc_auc": 0.5128458498,
        },
    )


def test_register_model_creates_candidate(tmp_path):
    registry = make_registry(tmp_path)

    record = register_candidate(registry)

    assert record["model_name"] == (
        "stock-direction-lstm"
    )

    assert record["model_version"] == (
        "refactored-v7"
    )

    assert record["run_id"] == (
        "20260823T233026Z"
    )

    assert record["status"] == "candidate"


def test_register_model_persists_registry(tmp_path):
    registry = make_registry(tmp_path)

    register_candidate(registry)

    assert registry.registry_path.exists()

    reloaded = ModelRegistry(
        registry_path=registry.registry_path
    )

    record = reloaded.get_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
    )

    assert record["status"] == "candidate"


def test_duplicate_model_version_is_rejected(tmp_path):
    registry = make_registry(tmp_path)

    register_candidate(registry)

    with pytest.raises(
        ModelAlreadyRegisteredError
    ):
        register_candidate(registry)


def test_get_model_returns_registered_version(tmp_path):
    registry = make_registry(tmp_path)

    register_candidate(registry)

    record = registry.get_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
    )

    assert record["model_version"] == (
        "refactored-v7"
    )

    assert record["metrics"]["accuracy"] == pytest.approx(
        0.5222222222
    )


def test_missing_model_raises_error(tmp_path):
    registry = make_registry(tmp_path)

    with pytest.raises(ModelNotFoundError):
        registry.get_model(
            model_name="stock-direction-lstm",
            model_version="refactored-v999",
        )


def test_list_models_returns_all_versions(tmp_path):
    registry = make_registry(tmp_path)

    register_candidate(
        registry,
        version="refactored-v6",
    )

    register_candidate(
        registry,
        version="refactored-v7",
    )

    models = registry.list_models(
        model_name="stock-direction-lstm"
    )

    assert len(models) == 2

    versions = {
        model["model_version"]
        for model in models
    }

    assert versions == {
        "refactored-v6",
        "refactored-v7",
    }


def test_candidate_can_be_validated(tmp_path):
    registry = make_registry(tmp_path)

    register_candidate(registry)

    record = registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        target_status="validated",
    )

    assert record["status"] == "validated"


def test_candidate_cannot_skip_validation(tmp_path):
    registry = make_registry(tmp_path)

    register_candidate(registry)

    with pytest.raises(InvalidPromotionError):
        registry.promote_model(
            model_name="stock-direction-lstm",
            model_version="refactored-v7",
            target_status="production",
        )


def test_validated_model_can_be_promoted_to_production(
    tmp_path,
):
    registry = make_registry(tmp_path)

    register_candidate(registry)

    registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        target_status="validated",
    )

    record = registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        target_status="production",
    )

    assert record["status"] == "production"


def test_only_one_production_version_is_allowed(
    tmp_path,
):
    registry = make_registry(tmp_path)

    register_candidate(
        registry,
        version="refactored-v6",
    )

    register_candidate(
        registry,
        version="refactored-v7",
    )

    registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v6",
        target_status="validated",
    )

    registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v6",
        target_status="production",
    )

    registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        target_status="validated",
    )

    record = registry.promote_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        target_status="production",
    )

    assert record["status"] == "production"

    v6 = registry.get_model(
        model_name="stock-direction-lstm",
        model_version="refactored-v6",
    )

    assert v6["status"] != "production"


def test_promotion_of_unknown_model_is_rejected(
    tmp_path,
):
    registry = make_registry(tmp_path)

    with pytest.raises(ModelNotFoundError):
        registry.promote_model(
            model_name="stock-direction-lstm",
            model_version="refactored-v7",
            target_status="validated",
        )


def test_invalid_status_is_rejected(tmp_path):
    registry = make_registry(tmp_path)

    register_candidate(registry)

    with pytest.raises(InvalidPromotionError):
        registry.promote_model(
            model_name="stock-direction-lstm",
            model_version="refactored-v7",
            target_status="unknown",
        )
