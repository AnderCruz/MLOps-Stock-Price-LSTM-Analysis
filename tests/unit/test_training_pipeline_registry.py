from unittest.mock import patch

import numpy as np
import pandas as pd

from src.pipeline.train_direction_v7_pipeline import (
    run_training_pipeline,
)


def make_pipeline_result():
    class PipelineResult:
        run_id = "v7-registry-test-001"
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


def test_training_pipeline_registers_model_candidate():

    registered_models = []

    def capture_register_model(
        *,
        model_name,
        model_version,
        run_id,
        artifact_path,
        metrics,
    ):
        registered_models.append(
            {
                "model_name": model_name,
                "model_version": model_version,
                "run_id": run_id,
                "artifact_path": artifact_path,
                "metrics": metrics,
            }
        )

        return {
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id,
            "artifact_path": artifact_path,
            "metrics": metrics,
            "status": "candidate",
        }

    with patch(
        "src.pipeline.train_direction_v7_pipeline.run_market_pipeline",
        return_value=make_pipeline_result(),
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
        return_value="artifacts/models/refactored-v7",
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.evaluate_binary_classifier",
        return_value={
            "accuracy": 0.52,
            "precision": 0.52,
            "recall": 0.69,
            "f1": 0.59,
            "roc_auc": 0.51,
            "confusion_matrix": [
                [30, 58],
                [28, 64],
            ],
            "predictions": [
                0,
                1,
                1,
                1,
                1,
            ],
        },
    ), patch(
        "src.pipeline.train_direction_v7_pipeline.ModelRegistry",
    ) as registry_class:

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
                "model": FakeModel(),
                "epochs_completed": 11,
            },
        )()

        registry_instance = (
            registry_class.return_value
        )

        registry_instance.register_model.side_effect = (
            capture_register_model
        )

        run_training_pipeline(
            ticker="TSLA",
            sentiment_enabled=False,
        )

    assert len(registered_models) == 1

    record = registered_models[0]

    assert record["model_name"] == (
        "stock-direction-lstm"
    )

    assert record["model_version"] == (
        "refactored-v7"
    )

    assert record["run_id"] == (
        "v7-registry-test-001"
    )

    assert record["artifact_path"] == (
        "artifacts/models/refactored-v7"
    )

    assert record["metrics"]["accuracy"] == 0.52

    assert record["metrics"]["f1"] == 0.59

    assert record["metrics"]["roc_auc"] == 0.51

    registry_instance.register_model.assert_called_once()
