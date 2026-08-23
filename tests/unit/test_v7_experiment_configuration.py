from src.pipeline.experiment_schema import (
    ExperimentRecord,
)


def test_v7_experiment_configuration_is_recorded():
    record = ExperimentRecord(
        run_id="20260823T170000Z",
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
        training={
            "epochs_requested": 60,
            "batch_size": 32,
            "validation_split": 0.10,
            "patience": 10,
        },
    )

    data = record.to_dict()

    assert data["experiment_name"] == (
        "stock-direction-lstm"
    )

    assert data["model_name"] == (
        "stock-direction-lstm"
    )

    assert data["model_version"] == (
        "refactored-v7"
    )

    assert data["ticker"] == "TSLA"

    assert data["features"] == [
        "Return",
        "Return_5D",
        "Return_10D",
    ]

    assert data["target"] == "Direction"

    assert data["sequence_length"] == 60

    assert data["training"][
        "epochs_requested"
    ] == 60

    assert data["training"][
        "batch_size"
    ] == 32

    assert data["training"][
        "validation_split"
    ] == 0.10

    assert data["training"][
        "patience"
    ] == 10


def test_v7_experiment_configuration_has_no_evaluation_metrics_before_training():
    record = ExperimentRecord(
        run_id="20260823T170000Z",
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
        training={
            "epochs_requested": 60,
            "batch_size": 32,
            "validation_split": 0.10,
            "patience": 10,
        },
    )

    data = record.to_dict()

    assert data["metrics"] == {}
    assert data["status"] == "created"
