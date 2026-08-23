import numpy as np

from src.pipeline.experiment_schema import (
    ExperimentRecord,
)


def make_experiment():
    return ExperimentRecord(
        run_id="v7-test-run-001",
        experiment_name="v7-directional-lstm",
        model_version="refactored-v7",
        model_name="stock-direction-lstm",
        ticker="TSLA",
        target="Direction",
        sequence_length=60,
        features=[
            "Return",
            "Return_5D",
            "Return_10D",
        ],
        training={
            "epochs_requested": 60,
            "batch_size": 32,
            "validation_split": 0.10,
            "patience": 10,
            "epochs_completed": None,
        },
    )


def test_experiment_can_store_evaluation_metrics():
    experiment = make_experiment()

    experiment.metrics = {
        "accuracy": 0.5389,
        "precision": 0.5714,
        "recall": 0.6154,
        "f1": 0.5926,
        "roc_auc": 0.5200,
    }

    assert experiment.metrics["accuracy"] == 0.5389
    assert experiment.metrics["precision"] == 0.5714
    assert experiment.metrics["recall"] == 0.6154
    assert experiment.metrics["f1"] == 0.5926
    assert experiment.metrics["roc_auc"] == 0.5200


def test_evaluation_metrics_are_serializable():
    experiment = make_experiment()

    experiment.metrics = {
        "accuracy": np.float64(0.5389),
        "precision": np.float64(0.5714),
        "recall": np.float64(0.6154),
        "f1": np.float64(0.5926),
        "roc_auc": np.float64(0.5200),
    }

    data = experiment.to_dict()

    assert isinstance(data["metrics"], dict)
    assert data["metrics"]["accuracy"] == 0.5389
    assert data["metrics"]["roc_auc"] == 0.5200


def test_experiment_starts_without_evaluation_metrics():
    experiment = make_experiment()

    data = experiment.to_dict()

    assert data["metrics"] == {}


def test_evaluation_metrics_can_be_updated_after_training():
    experiment = make_experiment()

    assert experiment.metrics == {}

    experiment.metrics = {
        "accuracy": 0.5389,
        "precision": 0.5714,
        "recall": 0.6154,
        "f1": 0.5926,
        "roc_auc": 0.5200,
    }

    assert len(experiment.metrics) == 5
