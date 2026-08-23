from unittest.mock import patch

import numpy as np
import pandas as pd

from src.pipeline.experiment_schema import (
    ExperimentRecord,
)
from src.pipeline.train_direction_v7_pipeline import (
    run_training_pipeline,
)


def make_pipeline_result():
    class PipelineResult:
        run_id = "v7-test-run-001"
        ticker = "TSLA"

        features = pd.DataFrame(
            {
                "TSLA_Close": np.arange(
                    100.0,
                    180.0,
                ),
            },
            index=pd.date_range(
                "2026-01-01",
                periods=80,
                freq="D",
            ),
        )

        def metadata(self):
            return {
                "ticker": self.ticker,
            }

    return PipelineResult()


def test_training_pipeline_creates_experiment_record():
    experiment_snapshots = []

    def capture_save_pipeline_run(
        *,
        run_id,
        metadata,
        feature_data=None,
        train_data=None,
        test_data=None,
        metrics=None,
        predictions=None,
        experiment=None,
    ):
        if experiment is not None:
            experiment_snapshots.append(
                experiment.to_dict()
            )

        return "artifacts/runs/test"

    with patch(
        "src.pipeline.train_direction_v7_pipeline.run_market_pipeline",
        return_value=make_pipeline_result(),
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.save_pipeline_run",
        side_effect=capture_save_pipeline_run,
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.add_return_features",
        side_effect=lambda df, price_column: pd.DataFrame(
            {
                "Return": np.linspace(
                    0.01,
                    0.02,
                    len(df),
                ),
                "Return_5D": np.linspace(
                    0.01,
                    0.02,
                    len(df),
                ),
                "Return_10D": np.linspace(
                    0.01,
                    0.02,
                    len(df),
                ),
            },
            index=df.index,
        ),
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.add_direction_target",
        side_effect=lambda df, return_column: (
            df.assign(
                Direction=(
                    np.arange(len(df)) % 2
                )
            )
        ),
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.prepare_direction_dataset",
    ) as prepare_dataset, patch(
        "src.pipeline.train_direction_v7_pipeline.build_directional_lstm_model",
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.train_lstm_model",
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.save_model_artifact",
    ):

        prepare_dataset.return_value = type(
            "Prepared",
            (),
            {
                "train_size": 10,
                "test_size": 5,
                "sequence_length": 60,
                "n_features": 3,
                "X_train": np.zeros(
                    (10, 60, 3)
                ),
                "y_train": np.zeros(10),
                "X_test": np.zeros(
                    (5, 60, 3)
                ),
                "y_test": np.zeros(5),
                "train_data": pd.DataFrame(
                    {"Direction": [0, 1]}
                ),
                "test_data": pd.DataFrame(
                    {"Direction": [0, 1]}
                ),
                "scaler": object(),
            },
        )()

        train_result = type(
            "TrainingResult",
            (),
            {
                "model": object(),
                "epochs_completed": 42,
            },
        )()

        from src.pipeline import (
            train_direction_v7_pipeline,
        )

        train_direction_v7_pipeline.train_lstm_model.return_value = (
            train_result
        )

        run_training_pipeline(
            ticker="TSLA",
            sentiment_enabled=False,
        )

    assert len(experiment_snapshots) == 2

    first = experiment_snapshots[0]
    second = experiment_snapshots[1]

    # --------------------------------------------------
    # Initial experiment state
    # --------------------------------------------------

    assert first["run_id"] == (
        "v7-test-run-001"
    )

    assert first["model_version"] == (
        "refactored-v7"
    )

    assert first["model_name"] == (
        "stock-direction-lstm"
    )

    assert first["ticker"] == "TSLA"

    assert first["target"] == "Direction"

    assert first["sequence_length"] == 60

    assert first["features"] == [
        "Return",
        "Return_5D",
        "Return_10D",
    ]

    assert first["training"][
        "epochs_requested"
    ] == 60

    assert first["training"][
        "batch_size"
    ] == 32

    assert first["training"][
        "validation_split"
    ] == 0.10

    assert first["training"][
        "patience"
    ] == 10

    assert (
        "epochs_completed"
        not in first["training"]
    )

    assert first["status"] == "started"

    # --------------------------------------------------
    # Final experiment state
    # --------------------------------------------------

    assert second["run_id"] == (
        "v7-test-run-001"
    )

    assert second["model_version"] == (
        "refactored-v7"
    )

    assert second["model_name"] == (
        "stock-direction-lstm"
    )

    assert second["ticker"] == "TSLA"

    assert second["target"] == "Direction"

    assert second["sequence_length"] == 60

    assert second["training"][
        "epochs_requested"
    ] == 60

    assert second["training"][
        "epochs_completed"
    ] == 42

    assert second["status"] == "completed"


def test_training_pipeline_uses_same_run_id_for_all_experiment_states():
    experiment_snapshots = []

    def capture_save_pipeline_run(
        *,
        run_id,
        metadata,
        feature_data=None,
        train_data=None,
        test_data=None,
        metrics=None,
        predictions=None,
        experiment=None,
    ):
        if experiment is not None:
            experiment_snapshots.append(
                {
                    "run_id": run_id,
                    "experiment": experiment.to_dict(),
                }
            )

        return "artifacts/runs/test"

    with patch(
        "src.pipeline.train_direction_v7_pipeline.run_market_pipeline",
        return_value=make_pipeline_result(),
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.save_pipeline_run",
        side_effect=capture_save_pipeline_run,
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.add_return_features",
        side_effect=lambda df, price_column: pd.DataFrame(
            {
                "Return": np.linspace(
                    0.01,
                    0.02,
                    len(df),
                ),
                "Return_5D": np.linspace(
                    0.01,
                    0.02,
                    len(df),
                ),
                "Return_10D": np.linspace(
                    0.01,
                    0.02,
                    len(df),
                ),
            },
            index=df.index,
        ),
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.add_direction_target",
        side_effect=lambda df, return_column: (
            df.assign(
                Direction=(
                    np.arange(len(df)) % 2
                )
            )
        ),
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.prepare_direction_dataset",
    ) as prepare_dataset, patch(
        "src.pipeline.train_direction_v7_pipeline.build_directional_lstm_model",
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.train_lstm_model",
    ) as train_model, patch(
        "src.pipeline.train_direction_v7_pipeline.save_model_artifact",
    ):

        prepare_dataset.return_value = type(
            "Prepared",
            (),
            {
                "train_size": 10,
                "test_size": 5,
                "sequence_length": 60,
                "n_features": 3,
                "X_train": np.zeros(
                    (10, 60, 3)
                ),
                "y_train": np.zeros(10),
                "X_test": np.zeros(
                    (5, 60, 3)
                ),
                "y_test": np.zeros(5),
                "train_data": pd.DataFrame(
                    {"Direction": [0, 1]}
                ),
                "test_data": pd.DataFrame(
                    {"Direction": [0, 1]}
                ),
                "scaler": object(),
            },
        )()

        train_model.return_value = type(
            "TrainingResult",
            (),
            {
                "model": object(),
                "epochs_completed": 42,
            },
        )()

        run_training_pipeline(
            ticker="TSLA",
            sentiment_enabled=False,
        )

    assert len(experiment_snapshots) == 2

    assert all(
        snapshot["run_id"]
        == "v7-test-run-001"
        for snapshot in experiment_snapshots
    )

    assert all(
        snapshot["experiment"]["run_id"]
        == "v7-test-run-001"
        for snapshot in experiment_snapshots
    )
