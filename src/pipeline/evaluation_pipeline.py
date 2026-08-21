from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.model.predictor import ModelPredictor
from src.model.sequence_builder import build_sequences
from src.pipeline.run_artifacts import save_pipeline_run


def run_evaluation_pipeline(
    model_dir: str | Path,
    run_dir: str | Path,
    target_column: str,
    sequence_length: int = 60,
) -> dict:
    """
    Evaluate a persisted LSTM model using the exact
    train/test boundary created during training.

    Evaluation flow:

        train_data.csv
              |
              | last 60 rows
              v
        historical context
              +
        test_data.csv
              |
              v
        test sequences
              |
              v
        persisted scaler
              |
              v
        persisted model
              |
              v
        predictions
              |
              v
        evaluation metrics

    No scaler is fitted during evaluation.
    """

    model_dir = Path(model_dir)
    run_dir = Path(run_dir)

    run_id = run_dir.name

    print("\n" + "=" * 70)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 70)

    print("Run ID:", run_id)
    print("Model:", model_dir)
    print("Run directory:", run_dir)

    # ==================================================
    # 1. VALIDATE ARTIFACTS
    # ==================================================

    train_path = (
        run_dir / "train_data.csv"
    )

    test_path = (
        run_dir / "test_data.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    # ==================================================
    # 2. LOAD TRAIN DATA
    # ==================================================

    train_data = pd.read_csv(
        train_path,
        index_col=0,
        parse_dates=True,
    )

    if train_data.empty:
        raise ValueError(
            "Persisted train dataset is empty."
        )

    print("\nTRAIN DATA")
    print("-" * 70)

    print(
        "Rows:",
        len(train_data),
    )

    print(
        "Columns:",
        list(train_data.columns),
    )

    # ==================================================
    # 3. LOAD TEST DATA
    # ==================================================

    test_data = pd.read_csv(
        test_path,
        index_col=0,
        parse_dates=True,
    )

    if test_data.empty:
        raise ValueError(
            "Persisted test dataset is empty."
        )

    print("\nTEST DATA")
    print("-" * 70)

    print(
        "Rows:",
        len(test_data),
    )

    print(
        "Columns:",
        list(test_data.columns),
    )

    # ==================================================
    # 4. VALIDATE TARGET
    # ==================================================

    if target_column not in train_data.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found in train data."
        )

    if target_column not in test_data.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found in test data."
        )

    # ==================================================
    # 5. LOAD PERSISTED MODEL + SCALER
    # ==================================================

    predictor = ModelPredictor(
        model_dir
    )

    print("\nMODEL")
    print("-" * 70)

    print(
        "Model:",
        predictor.model.name,
    )

    print(
        "Input shape:",
        predictor.model.input_shape,
    )

    print(
        "Scaler features:",
        predictor.scaler.n_features_in_,
    )

    # ==================================================
    # 6. GET HISTORICAL CONTEXT
    # ==================================================

    if len(train_data) < sequence_length:
        raise ValueError(
            "Training dataset does not contain enough "
            "history for the requested sequence length."
        )

    historical_context = (
        train_data.tail(
            sequence_length
        )
    )

    print("\nSEQUENCE CONTEXT")
    print("-" * 70)

    print(
        "Sequence length:",
        sequence_length,
    )

    print(
        "Historical rows:",
        len(historical_context),
    )

    print(
        "Test rows:",
        len(test_data),
    )

    # ==================================================
    # 7. COMBINE TRAIN CONTEXT + TEST
    # ==================================================

    evaluation_data = pd.concat(
        [
            historical_context,
            test_data,
        ]
    )

    # --------------------------------------------------
    # Ensure chronological order
    # --------------------------------------------------

    evaluation_data = (
        evaluation_data
        .sort_index()
    )

    # --------------------------------------------------
    # Prevent duplicated dates
    # --------------------------------------------------

    if evaluation_data.index.has_duplicates:
        raise ValueError(
            "Evaluation dataset contains duplicate dates."
        )

    print(
        "Combined rows:",
        len(evaluation_data),
    )

    # ==================================================
    # 8. APPLY PERSISTED SCALER
    # ==================================================

    expected_features = (
        predictor.scaler.n_features_in_
    )

    if len(evaluation_data.columns) != (
        expected_features
    ):
        raise ValueError(
            "Feature count mismatch between "
            "evaluation data and persisted scaler. "
            f"Expected {expected_features}, "
            f"received {len(evaluation_data.columns)}."
        )

    print("\nSCALING")
    print("-" * 70)

    print(
        "Using persisted scaler: PASS"
    )

    scaled = predictor.scaler.transform(
        evaluation_data
    )

    scaled_data = pd.DataFrame(
        scaled,
        index=evaluation_data.index,
        columns=evaluation_data.columns,
    )

    # ==================================================
    # 9. BUILD TEST SEQUENCES
    # ==================================================

    X_all, y_all = build_sequences(
        df=scaled_data,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    # --------------------------------------------------
    # IMPORTANT
    #
    # Because evaluation_data contains:
    #
    #     last 60 train rows
    #     +
    #     complete test data
    #
    # build_sequences() produces exactly one
    # sequence for each test observation.
    # --------------------------------------------------

    expected_samples = len(
        test_data
    )

    if len(X_all) != expected_samples:
        raise RuntimeError(
            "Unexpected number of evaluation sequences. "
            f"Expected {expected_samples}, "
            f"received {len(X_all)}."
        )

    if len(y_all) != expected_samples:
        raise RuntimeError(
            "Unexpected number of evaluation targets. "
            f"Expected {expected_samples}, "
            f"received {len(y_all)}."
        )

    print("\nSEQUENCES")
    print("-" * 70)

    print(
        "X shape:",
        X_all.shape,
    )

    print(
        "y shape:",
        y_all.shape,
    )

    print(
        "Expected samples:",
        expected_samples,
    )

    print(
        "Sequence reconstruction: PASS"
    )

    # ==================================================
    # 10. MODEL PREDICTION
    # ==================================================

    print("\nPREDICTION")
    print("-" * 70)

    predictions_scaled = (
        predictor.model.predict(
            X_all,
            verbose=0,
        )
        .reshape(-1)
    )

    # ==================================================
    # 11. INVERSE TRANSFORM TARGET
    # ==================================================

    def inverse_target(
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Convert scaled target values back to
        original target scale using the persisted scaler.
        """

        values = np.asarray(
            values
        ).reshape(-1)

        reconstructed = np.zeros(
            (
                len(values),
                expected_features,
            )
        )

        reconstructed[:, 0] = values

        inverse = (
            predictor.scaler.inverse_transform(
                reconstructed
            )
        )

        return inverse[:, 0]

    actual = inverse_target(
        y_all
    )

    predictions = inverse_target(
        predictions_scaled
    )

    # ==================================================
    # 12. EVALUATION
    # ==================================================

    print("\nMETRICS")
    print("-" * 70)

    mae = float(
        np.mean(
            np.abs(
                actual - predictions
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    actual - predictions
                )
            )
        )
    )

    non_zero_mask = (
        actual != 0
    )

    if non_zero_mask.any():

        mape_percent = float(
            np.mean(
                np.abs(
                    (
                        actual[non_zero_mask]
                        - predictions[non_zero_mask]
                    )
                    / actual[non_zero_mask]
                )
            )
            * 100
        )

    else:

        mape_percent = 0.0

    # ==================================================
    # 13. DIRECTIONAL ACCURACY
    # ==================================================

    if len(actual) > 1:

        actual_direction = np.sign(
            np.diff(actual)
        )

        predicted_direction = np.sign(
            np.diff(predictions)
        )

        direction_accuracy_percent = float(
            np.mean(
                actual_direction
                == predicted_direction
            )
            * 100
        )

    else:

        direction_accuracy_percent = 0.0

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape_percent,
        "direction_accuracy_percent": (
            direction_accuracy_percent
        ),
        "n_samples": int(
            len(actual)
        ),
    }

    print(
        f"MAE: {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"MAPE: {mape_percent:.2f}%"
    )

    print(
        "Direction accuracy: "
        f"{direction_accuracy_percent:.2f}%"
    )

    print(
        "Samples:",
        len(actual),
    )

    # ==================================================
    # 14. PREDICTION ARTIFACT
    # ==================================================

    prediction_frame = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predictions,
        },
        index=test_data.index,
    )

    # ==================================================
    # 15. SAVE EVALUATION ARTIFACTS
    # ==================================================

    save_pipeline_run(
        run_id=run_id,
        metadata={
            "status": "evaluation_completed",
            "evaluation": {
                "model_directory": str(
                    model_dir
                ),
                "sequence_length": (
                    sequence_length
                ),
                "target_column": (
                    target_column
                ),
                "train_context_rows": (
                    len(historical_context)
                ),
                "test_rows": (
                    len(test_data)
                ),
            },
        },
        metrics=metrics,
        predictions=prediction_frame,
    )

    # ==================================================
    # 16. FINAL VALIDATION
    # ==================================================

    metrics_path = (
        run_dir / "metrics.json"
    )

    predictions_path = (
        run_dir / "predictions.csv"
    )

    assert metrics_path.exists()
    assert predictions_path.exists()

    print("\n" + "=" * 70)
    print("EVALUATION PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "Run ID:",
        run_id,
    )

    print(
        "Metrics:",
        metrics_path,
    )

    print(
        "Predictions:",
        predictions_path,
    )

    print(
        "\nEvaluation artifact persistence: PASS"
    )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "predictions": prediction_frame,
    }
