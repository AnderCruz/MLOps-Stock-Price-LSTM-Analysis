import json

import pytest

from src.pipeline.experiment_schema import (
    ExperimentRecord,
)
from src.pipeline.run_artifacts import (
    save_experiment_record,
)


def make_experiment() -> ExperimentRecord:
    return ExperimentRecord(
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
        train_size=1000,
        test_size=200,
        training={
            "epochs_requested": 60,
            "batch_size": 32,
        },
        metrics={
            "accuracy": 0.5389,
            "roc_auc": 0.5200,
        },
        status="completed",
    )


def test_experiment_record_is_persisted_as_json(tmp_path):
    experiment = make_experiment()

    run_dir = tmp_path / "runs" / experiment.run_id

    result = save_experiment_record(
        experiment=experiment,
        run_dir=run_dir,
    )

    assert result == (
        run_dir / "experiment.json"
    )

    assert result.exists()


def test_persisted_experiment_contains_expected_data(
    tmp_path,
):
    experiment = make_experiment()

    run_dir = tmp_path / "runs" / experiment.run_id

    path = save_experiment_record(
        experiment=experiment,
        run_dir=run_dir,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert data["run_id"] == (
        "20260823T170000Z"
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

    assert data["training"][
        "epochs_requested"
    ] == 60

    assert data["metrics"][
        "accuracy"
    ] == pytest.approx(0.5389)

    assert data["status"] == "completed"


def test_persistence_creates_run_directory(
    tmp_path,
):
    experiment = make_experiment()

    run_dir = (
        tmp_path
        / "new"
        / "run"
        / experiment.run_id
    )

    save_experiment_record(
        experiment=experiment,
        run_dir=run_dir,
    )

    assert run_dir.exists()


def test_persistence_rejects_invalid_experiment(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        experiment = ExperimentRecord(
            run_id="",
            experiment_name=(
                "stock-direction-lstm"
            ),
            model_name=(
                "stock-direction-lstm"
            ),
            model_version="refactored-v7",
        )

        save_experiment_record(
            experiment=experiment,
            run_dir=tmp_path,
        )
