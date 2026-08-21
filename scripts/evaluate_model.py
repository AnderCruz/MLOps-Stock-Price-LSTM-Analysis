from __future__ import annotations

from pathlib import Path
import sys

# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# IMPORTS
# ==================================================

import numpy as np

from src.pipeline.market_pipeline import run_market_pipeline
from src.model.dataset import prepare_model_dataset
from src.model.predictor import ModelPredictor
from src.model.evaluator import evaluate_predictions


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "refactored-v1"
)

TICKER = "TSLA"

SEQUENCE_LENGTH = 60

TEST_RATIO = 0.20


# ==================================================
# MAIN
# ==================================================

def main() -> None:

    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    # ==================================================
    # 1. LOAD PERSISTED MODEL
    # ==================================================

    print("\n" + "=" * 70)
    print("1. LOADING PERSISTED MODEL")
    print("=" * 70)

    predictor = ModelPredictor(
        str(MODEL_DIR)
    )

    print("Model:", predictor.model.name)
    print("Input:", predictor.model.input_shape)
    print(
        "Scaler features:",
        predictor.scaler.n_features_in_,
    )

    # ==================================================
    # 2. RECREATE FEATURE DATASET
    # ==================================================

    print("\n" + "=" * 70)
    print("2. RECREATING FEATURE DATASET")
    print("=" * 70)

    result = run_market_pipeline(
        ticker=TICKER,
        sentiment_enabled=True,
    )

    print(
        "\nRecords:",
        result.n_records,
    )

    print(
        "Features:",
        result.feature_columns,
    )

    # ==================================================
    # 3. PREPARE TEST DATASET
    # ==================================================

    print("\n" + "=" * 70)
    print("3. PREPARING TEST DATASET")
    print("=" * 70)

    prepared = prepare_model_dataset(
        df=result.features,
        target_column=f"{TICKER}_Close",
        sequence_length=SEQUENCE_LENGTH,
        test_ratio=TEST_RATIO,
    )

    print(
        "Train rows:",
        prepared.train_size,
    )

    print(
        "Test rows:",
        prepared.test_size,
    )

    print(
        "X_test:",
        prepared.X_test.shape,
    )

    print(
        "y_test:",
        prepared.y_test.shape,
    )

    # ==================================================
    # 4. GENERATE PREDICTIONS
    # ==================================================

    print("\n" + "=" * 70)
    print("4. GENERATING PREDICTIONS")
    print("=" * 70)

    predictions_scaled = predictor.model.predict(
        prepared.X_test,
        verbose=0,
    )

    predictions_scaled = np.asarray(
        predictions_scaled
    ).reshape(-1)

    y_test_scaled = np.asarray(
        prepared.y_test
    ).reshape(-1)

    print(
        "Prediction samples:",
        len(predictions_scaled),
    )

    print(
        "Scaled prediction min:",
        predictions_scaled.min(),
    )

    print(
        "Scaled prediction max:",
        predictions_scaled.max(),
    )

    # ==================================================
    # 5. INVERSE TRANSFORM
    # ==================================================

    print("\n" + "=" * 70)
    print("5. CONVERTING TO REAL PRICE SCALE")
    print("=" * 70)

    actual = prepared.inverse_transform_target(
        y_test_scaled
    )

    predictions = prepared.inverse_transform_target(
        predictions_scaled
    )

    print(
        "Actual min:",
        actual.min(),
    )

    print(
        "Actual max:",
        actual.max(),
    )

    print(
        "Prediction min:",
        predictions.min(),
    )

    print(
        "Prediction max:",
        predictions.max(),
    )

    # ==================================================
    # 6. EVALUATE
    # ==================================================

    print("\n" + "=" * 70)
    print("6. EVALUATING TEST SET")
    print("=" * 70)

    metrics = evaluate_predictions(
        y_true=actual,
        y_pred=predictions,
    )

    # ==================================================
    # 7. TEST SET METRICS
    # ==================================================

    print("\n" + "=" * 70)
    print("TEST SET METRICS")
    print("=" * 70)

    print(
        f"MAE:  ${metrics.mae:.2f}"
    )

    print(
        f"RMSE: ${metrics.rmse:.2f}"
    )

    print(
        f"MAPE: {metrics.mape_percent:.2f}%"
    )

    print(
        f"Direction Accuracy: "
        f"{metrics.direction_accuracy_percent:.2f}%"
    )

    print(
        "Samples:",
        metrics.n_samples,
    )

    # ==================================================
    # 8. ADDITIONAL DIAGNOSTICS
    # ==================================================

    mean_error = np.mean(
        predictions - actual
    )

    max_absolute_error = np.max(
        np.abs(
            predictions - actual
        )
    )

    print("\n" + "=" * 70)
    print("ADDITIONAL DIAGNOSTICS")
    print("=" * 70)

    print(
        "Actual mean:",
        f"${actual.mean():.2f}",
    )

    print(
        "Predicted mean:",
        f"${predictions.mean():.2f}",
    )

    print(
        "Mean error:",
        f"${mean_error:.2f}",
    )

    print(
        "Max absolute error:",
        f"${max_absolute_error:.2f}",
    )

    # ==================================================
    # 9. STRUCTURAL VALIDATION
    # ==================================================

    print("\n" + "=" * 70)
    print("9. VALIDATION")
    print("=" * 70)

    assert len(actual) == len(predictions)

    assert len(actual) == prepared.test_size

    assert len(actual) == metrics.n_samples

    assert np.isfinite(actual).all()

    assert np.isfinite(predictions).all()

    assert metrics.mae >= 0

    assert metrics.rmse >= 0

    assert metrics.mape_percent >= 0

    assert (
        0 <= metrics.direction_accuracy_percent <= 100
    )

    print("Prediction count: PASS")
    print("Finite values: PASS")
    print("Metrics validity: PASS")

    # ==================================================
    # FINAL
    # ==================================================

    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION: PASS")
    print("=" * 70)

    print("Model:", predictor.model.name)
    print("Ticker:", TICKER)
    print("Model version:", "refactored-v1")
    print("Test samples:", metrics.n_samples)

    print(
        f"MAE:  ${metrics.mae:.2f}"
    )

    print(
        f"RMSE: ${metrics.rmse:.2f}"
    )

    print(
        f"MAPE: {metrics.mape_percent:.2f}%"
    )

    print(
        f"Direction Accuracy: "
        f"{metrics.direction_accuracy_percent:.2f}%"
    )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()