from unittest.mock import patch
import importlib

import numpy as np
import pandas as pd


training_pipeline = importlib.import_module(
    "src.pipeline.train_direction_v7_pipeline"
)


def make_pipeline_result():
    class PipelineResult:
        run_id = "v7-validation-test-001"
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


def test_training_pipeline_records_validation_result():

    captured_validation = []

    class FakeRegistry:

        def register_model(
            self,
            *,
            model_name,
            model_version,
            run_id,
            artifact_path,
            metrics,
        ):
            return {
                "model_name": model_name,
                "model_version": model_version,
                "run_id": run_id,
                "artifact_path": artifact_path,
                "metrics": metrics,
                "status": "candidate",
            }

        def record_validation(
            self,
            *,
            model_name,
            model_version,
            validation,
        ):
            captured_validation.append(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "validation": validation,
                }
            )

            return {
                "model_name": model_name,
                "model_version": model_version,
                "status": (
                    "rejected"
                    if not validation.passed
                    else "validated"
                ),
                "validation": (
                    validation.to_dict()
                ),
            }

    expected_validation = {
        "passed": False,
        "checks": {
            "accuracy_vs_baseline": True,
            "minimum_roc_auc": False,
        },
        "reasons": [
            "ROC-AUC is below the minimum "
            "required threshold."
        ],
    }

    with patch.object(
        training_pipeline,
        "run_market_pipeline",
        return_value=make_pipeline_result(),
    ), patch.object(
        training_pipeline,
        "ModelRegistry",
        return_value=FakeRegistry(),
    ), patch.object(
        training_pipeline,
        "validate_model_candidate",
    ) as validate_candidate, patch.object(
        training_pipeline,
        "save_pipeline_run",
        return_value="artifacts/runs/test",
    ), patch.object(
        training_pipeline,
        "save_model_artifact",
        return_value="artifacts/models/refactored-v7",
    ), patch.object(
        training_pipeline,
        "add_return_features",
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
    ), patch.object(
        training_pipeline,
        "add_direction_target",
        side_effect=lambda df, return_column: df.assign(
            Direction=np.arange(len(df)) % 2
        ),
    ), patch.object(
        training_pipeline,
        "prepare_direction_dataset",
    ) as prepare_dataset, patch.object(
        training_pipeline,
        "build_directional_lstm_model",
    ), patch.object(
        training_pipeline,
        "train_lstm_model",
    ) as train_model, patch.object(
        training_pipeline,
        "evaluate_binary_classifier",
        return_value={
            "accuracy": 0.40,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.51,
            "confusion_matrix": [
                [1, 2],
                [1, 1],
            ],
        },
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
                "y_test": np.array(
                    [0, 1, 0, 1, 0]
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

        class FakeModel:

            def predict(
                self,
                X,
                verbose=0,
            ):
                return np.array(
                    [
                        [0.20],
                        [0.80],
                        [0.70],
                        [0.30],
                        [0.60],
                    ]
                )

        train_model.return_value = type(
            "TrainingResult",
            (),
            {
                "model": FakeModel(),
                "epochs_completed": 1,
            },
        )()

        validate_candidate.return_value = type(
            "ValidationResult",
            (),
            {
                "passed": False,
                "to_dict": lambda self: expected_validation,
            },
        )()

        run_training_pipeline = (
            training_pipeline.run_training_pipeline
        )

        run_training_pipeline(
            ticker="TSLA",
            sentiment_enabled=False,
        )

    validate_candidate.assert_called_once()

    assert len(
        captured_validation
    ) == 1

    assert (
        captured_validation[0][
            "validation"
        ].passed
        is False
    )

    assert (
        captured_validation[0][
            "model_name"
        ]
        == "stock-direction-lstm"
    )

    assert (
        captured_validation[0][
            "model_version"
        ]
        == "refactored-v7"
    )
