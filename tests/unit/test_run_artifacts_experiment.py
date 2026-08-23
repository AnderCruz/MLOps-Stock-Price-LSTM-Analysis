import json

from src.pipeline.experiment_schema import (
    ExperimentRecord,
)
from src.pipeline.run_artifacts import (
    save_pipeline_run,
)


def make_v7_experiment() -> ExperimentRecord:
    return ExperimentRecord(
        run_id="v7-run-001",
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


def test_v7_experiment_is_persisted_with_pipeline_run(
    tmp_path,
    monkeypatch,
):
    experiment = make_v7_experiment()

    monkeypatch.setattr(
        "src.pipeline.run_artifacts.ARTIFACT_ROOT",
        tmp_path / "runs",
    )

    run_dir = save_pipeline_run(
        run_id=experiment.run_id,
        metadata={
            "model_version": "refactored-v7",
        },
        experiment=experiment,
    )

    experiment_path = (
        run_dir / "experiment.json"
    )

    metadata_path = (
        run_dir / "metadata.json"
    )

    assert experiment_path.exists()
    assert metadata_path.exists()

    with experiment_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        experiment_data = json.load(file)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert experiment_data["run_id"] == (
        metadata["run_id"]
    )

    assert experiment_data["run_id"] == (
        "v7-run-001"
    )

    assert experiment_data["model_version"] == (
        "refactored-v7"
    )

    assert experiment_data["model_name"] == (
        "stock-direction-lstm"
    )

    assert experiment_data["ticker"] == "TSLA"

    assert experiment_data["target"] == (
        "Direction"
    )

    assert experiment_data[
        "sequence_length"
    ] == 60


def test_pipeline_run_remains_compatible_without_experiment(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.pipeline.run_artifacts.ARTIFACT_ROOT",
        tmp_path / "runs",
    )

    run_dir = save_pipeline_run(
        run_id="run-without-experiment",
        metadata={
            "model_version": "refactored-v7",
        },
    )

    assert run_dir.exists()

    metadata_path = (
        run_dir / "metadata.json"
    )

    assert metadata_path.exists()

    experiment_path = (
        run_dir / "experiment.json"
    )

    assert not experiment_path.exists()
