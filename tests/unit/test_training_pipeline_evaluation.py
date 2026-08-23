from unittest.mock import patch

import numpy as np
import pandas as pd

from src.pipeline.train_direction_v7_pipeline import (
    run_training_pipeline,
)


def make_pipeline_result():
    class PipelineResult:
        run_id = "v7-evaluation-test-001"
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


class FakeModel:
    def predict(
        self,
        X,
        verbose=0,
    ):
        return np.array(
            [
                [0.10],
                [0.80],
                [0.90],
                [0.85],
                [0.70],
            ][: len(X)]
        )


def test_training_pipeline_persists_real_evaluation_metrics():
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
    ) as train_model, patch(
        "src.pipeline.train_direction_v7_pipeline.save_model_artifact",
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.evaluate_binary_classifier",
    ) as evaluate:

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
                "y_test": np.array(
                    [0, 1, 0, 1, 1]
                ),
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
                "model": FakeModel(),
                "epochs_completed": 42,
            },
        )()

        evaluate.return_value = {
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 1.00,
            "f1": 0.8571428571,
            "roc_auc": 0.9166666667,
            "confusion_matrix": [
                [1, 1],
                [0, 3],
            ],
            "predictions": [
                0,
                1,
                1,
                1,
                1,
            ],
        }

        run_training_pipeline(
            ticker="TSLA",
            sentiment_enabled=False,
        )

    assert len(experiment_snapshots) == 2

    first = experiment_snapshots[0]
    second = experiment_snapshots[1]

    assert first["metrics"] == {}

    assert second["metrics"]["accuracy"] == 0.80

    assert (
        second["metrics"]["precision"]
        == 0.75
    )

    assert (
        second["metrics"]["recall"]
        == 1.00
    )

    assert (
        second["metrics"]["f1"]
        == 0.8571428571
    )

    assert (
        second["metrics"]["roc_auc"]
        == 0.9166666667
    )

    assert (
        second["metrics"]["confusion_matrix"]
        == [
            [1, 1],
            [0, 3],
        ]
    )

    assert second["status"] == "completed"

    assert (
        second["training"]["epochs_completed"]
        == 42
    )


def test_training_pipeline_passes_test_data_to_evaluation():
    captured = {}

    def capture_evaluation(
        *,
        y_true,
        probabilities,
    ):
        captured["y_true"] = np.asarray(
            y_true
        )

        captured["probabilities"] = np.asarray(
            probabilities
        )

        return {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "roc_auc": 0.5,
            "confusion_matrix": [
                [1, 1],
                [1, 1],
            ],
            "predictions": [
                0,
                1,
                0,
                1,
            ],
        }

    with patch(
        "src.pipeline.train_direction_v7_pipeline.run_market_pipeline",
        return_value=make_pipeline_result(),
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.save_pipeline_run",
        return_value="artifacts/runs/test",
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
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.evaluate_binary_classifier",
        side_effect=capture_evaluation,
    ):

        prepare_dataset.return_value = type(
            "Prepared",
            (),
            {
                "train_size": 10,
                "test_size": 4,
                "sequence_length": 60,
                "n_features": 3,
                "X_train": np.zeros(
                    (10, 60, 3)
                ),
                "y_train": np.zeros(10),
                "X_test": np.zeros(
                    (4, 60, 3)
                ),
                "y_test": np.array(
                    [0, 1, 0, 1]
                ),
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
                "model": FakeModel(),
                "epochs_completed": 10,
            },
        )()

        run_training_pipeline(
            ticker="TSLA",
            sentiment_enabled=False,
        )

    np.testing.assert_array_equal(
        captured["y_true"],
        np.array([0, 1, 0, 1]),
    )

    np.testing.assert_array_equal(
        captured["probabilities"],
        np.array(
            [
                0.10,
                0.80,
                0.90,
                0.85,
            ]
        ),
    )
