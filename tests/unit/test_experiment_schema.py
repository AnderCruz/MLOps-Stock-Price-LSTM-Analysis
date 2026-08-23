import pytest

from src.pipeline.experiment_schema import (
    ExperimentRecord,
)


def test_experiment_record_contains_required_identity_fields():
    record = ExperimentRecord(
        run_id="20260823T170000Z",
        experiment_name="stock-direction-lstm",
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
    )

    data = record.to_dict()

    assert data["run_id"] == "20260823T170000Z"
    assert data["experiment_name"] == "stock-direction-lstm"
    assert data["model_name"] == "stock-direction-lstm"
    assert data["model_version"] == "refactored-v7"


def test_experiment_record_contains_data_configuration():
    record = ExperimentRecord(
        run_id="run-001",
        experiment_name="stock-direction-lstm",
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        ticker="TSLA",
        features=[
            "Return",
            "Return_5D",
            "Return_10D",
        ],
        target="Direction",
        sequence_length=60,
        train_size=1000,
        test_size=200,
    )

    data = record.to_dict()

    assert data["ticker"] == "TSLA"
    assert data["features"] == [
        "Return",
        "Return_5D",
        "Return_10D",
    ]
    assert data["target"] == "Direction"
    assert data["sequence_length"] == 60
    assert data["train_size"] == 1000
    assert data["test_size"] == 200


def test_experiment_record_contains_training_configuration():
    record = ExperimentRecord(
        run_id="run-001",
        experiment_name="stock-direction-lstm",
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        training={
            "epochs_requested": 60,
            "epochs_completed": 42,
            "batch_size": 32,
            "validation_split": 0.10,
            "patience": 10,
        },
    )

    data = record.to_dict()

    assert data["training"]["epochs_requested"] == 60
    assert data["training"]["epochs_completed"] == 42
    assert data["training"]["batch_size"] == 32
    assert data["training"]["validation_split"] == 0.10
    assert data["training"]["patience"] == 10


def test_experiment_record_defaults_status_to_created():
    record = ExperimentRecord(
        run_id="run-001",
        experiment_name="stock-direction-lstm",
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
    )

    assert record.to_dict()["status"] == "created"


def test_experiment_record_can_store_evaluation_metrics():
    record = ExperimentRecord(
        run_id="run-001",
        experiment_name="stock-direction-lstm",
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        metrics={
            "accuracy": 0.5389,
            "f1": 0.6407,
            "roc_auc": 0.5200,
            "p_value": 0.251,
        },
    )

    data = record.to_dict()

    assert data["metrics"]["accuracy"] == pytest.approx(0.5389)
    assert data["metrics"]["f1"] == pytest.approx(0.6407)
    assert data["metrics"]["roc_auc"] == pytest.approx(0.5200)
    assert data["metrics"]["p_value"] == pytest.approx(0.251)


def test_experiment_record_rejects_missing_run_id():
    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        ExperimentRecord(
            run_id="",
            experiment_name="stock-direction-lstm",
            model_name="stock-direction-lstm",
            model_version="refactored-v7",
        )


def test_experiment_record_rejects_invalid_status():
    with pytest.raises(
        ValueError,
        match="status",
    ):
        ExperimentRecord(
            run_id="run-001",
            experiment_name="stock-direction-lstm",
            model_name="stock-direction-lstm",
            model_version="refactored-v7",
            status="invalid",
        )


def test_experiment_record_serialisation_is_independent():
    features = [
        "Return",
        "Return_5D",
        "Return_10D",
    ]

    record = ExperimentRecord(
        run_id="run-001",
        experiment_name="stock-direction-lstm",
        model_name="stock-direction-lstm",
        model_version="refactored-v7",
        features=features,
    )

    data = record.to_dict()

    data["features"].append("UnexpectedFeature")

    assert "UnexpectedFeature" not in record.features
