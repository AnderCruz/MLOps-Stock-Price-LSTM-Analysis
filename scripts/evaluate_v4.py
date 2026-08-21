from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# IMPORTS
# ==================================================

from src.pipeline.market_pipeline import (
    run_market_pipeline,
)

from src.model.dataset import (
    prepare_model_dataset,
)

from src.model.predictor import (
    ModelPredictor,
)

from src.model.return_target import (
    add_return_target,
)

from src.model.evaluator import (
    evaluate_predictions,
)


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "refactored-v4"
)

TICKER = "TSLA"

PRICE_COLUMN = f"{TICKER}_Close"

VOLUME_COLUMN = f"{TICKER}_Volume"

TARGET_COLUMN = "Return"

SEQUENCE_LENGTH = 60

TEST_RATIO = 0.20


# ==================================================
# MAIN
# ==================================================

def main() -> None:

    print("=" * 70)
    print("RETURN MODEL EVALUATION")
    print("=" * 70)

    # ==================================================
    # 1. LOAD MODEL
    # ==================================================

    print("\n" + "=" * 70)
    print("1. LOADING PERSISTED RETURN MODEL")
    print("=" * 70)

    predictor = ModelPredictor(
        str(MODEL_DIR)
    )

    print(
        "Model:",
        predictor.model.name,
    )

    print(
        "Input:",
        predictor.model.input_shape,
    )

    print(
        "Scaler features:",
        predictor.scaler.n_features_in_,
    )

    # ==================================================
    # 2. MARKET DATA
    # ==================================================

    print("\n" + "=" * 70)
    print("2. RECREATING MARKET DATA")
    print("=" * 70)

    result = run_market_pipeline(
        ticker=TICKER,
        sentiment_enabled=False,
    )

    print(
        "Records:",
        result.n_records,
    )

    print(
        "Market features:",
        result.feature_columns,
    )

    # ==================================================
    # 3. BUILD RETURN DATASET
    # ==================================================

    print("\n" + "=" * 70)
    print("3. BUILDING RETURN DATASET")
    print("=" * 70)

    return_data = add_return_target(
        df=result.features,
        price_column=PRICE_COLUMN,
    )

    feature_columns = [
        TARGET_COLUMN,
        PRICE_COLUMN,
        VOLUME_COLUMN,
    ]

    return_data = return_data[
        feature_columns
    ]

    print(
        "Features:",
        list(return_data.columns),
    )

    print(
        "Records:",
        len(return_data),
    )

    print(
        "Target:",
        TARGET_COLUMN,
    )

    # ==================================================
    # 4. PREPARE DATASET
    # ==================================================

    print("\n" + "=" * 70)
    print("4. PREPARING RETURN DATASET")
    print("=" * 70)

    prepared = prepare_model_dataset(
        df=return_data,
        target_column=TARGET_COLUMN,
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
    # 5. STRUCTURAL VALIDATION
    # ==================================================

    expected_features = (
        predictor.scaler.n_features_in_
    )

    actual_features = (
        prepared.n_features
    )

    if actual_features != expected_features:
        raise RuntimeError(
            "Feature count mismatch: "
            f"model expects {expected_features}, "
            f"dataset contains {actual_features}."
        )

    if prepared.X_test.shape[2] != expected_features:
        raise RuntimeError(
            "X_test feature dimension does not "
            "match persisted model."
        )

    print(
        "Feature compatibility: PASS"
    )

    # ==================================================
    # 6. GENERATE RETURN PREDICTIONS
    # ==================================================

    print("\n" + "=" * 70)
    print("5. GENERATING RETURN PREDICTIONS")
    print("=" * 70)

    predictions_scaled = (
        predictor.predict_scaled(
            prepared.X_test
        )
    )

    y_test_scaled = (
        np.asarray(
            prepared.y_test
        ).reshape(-1)
    )

    print(
        "Prediction samples:",
        len(predictions_scaled),
    )

    # ==================================================
    # 7. INVERSE TRANSFORM RETURNS
    # ==================================================

    print("\n" + "=" * 70)
    print("6. CONVERTING RETURNS TO ORIGINAL SCALE")
    print("=" * 70)

    actual_returns = (
        prepared.inverse_transform_target(
            y_test_scaled
        )
    )

    predicted_returns = (
        prepared.inverse_transform_target(
            predictions_scaled
        )
    )

    print(
        "Actual return min:",
        f"{actual_returns.min() * 100:.4f}%",
    )

    print(
        "Actual return max:",
        f"{actual_returns.max() * 100:.4f}%",
    )

    print(
        "Predicted return min:",
        f"{predicted_returns.min() * 100:.4f}%",
    )

    print(
        "Predicted return max:",
        f"{predicted_returns.max() * 100:.4f}%",
    )

    # ==================================================
    # 8. RETURN METRICS
    # ==================================================

    print("\n" + "=" * 70)
    print("7. RETURN MODEL METRICS")
    print("=" * 70)

    metrics = evaluate_predictions(
        y_true=actual_returns,
        y_pred=predicted_returns,
    )

    print(
        "Return MAE:",
        f"{metrics.mae * 100:.4f}%",
    )

    print(
        "Return RMSE:",
        f"{metrics.rmse * 100:.4f}%",
    )

    print(
        "Direction Accuracy:",
        f"{metrics.direction_accuracy_percent:.2f}%",
    )

    print(
        "Samples:",
        metrics.n_samples,
    )

    # ==================================================
    # 9. NAIVE RETURN BASELINE
    # ==================================================

    print("\n" + "=" * 70)
    print("8. NAIVE RETURN BASELINE")
    print("=" * 70)

    # Naive price prediction:
    # tomorrow's return = 0
    #
    # This corresponds to predicting that tomorrow's
    # price will equal today's price.

    naive_returns = np.zeros_like(
        actual_returns
    )

    naive_metrics = evaluate_predictions(
        y_true=actual_returns,
        y_pred=naive_returns,
    )

    print(
        "Naive Return MAE:",
        f"{naive_metrics.mae * 100:.4f}%",
    )

    print(
        "Naive Return RMSE:",
        f"{naive_metrics.rmse * 100:.4f}%",
    )

    print(
        "Naive Direction Accuracy:",
        f"{naive_metrics.direction_accuracy_percent:.2f}%",
    )

    # ==================================================
    # 10. RETURN MODEL VS BASELINE
    # ==================================================

    print("\n" + "=" * 70)
    print("9. MODEL VS BASELINE")
    print("=" * 70)

    mae_improvement = (
        1
        - metrics.mae
        / naive_metrics.mae
    ) * 100

    rmse_improvement = (
        1
        - metrics.rmse
        / naive_metrics.rmse
    ) * 100

    print(
        "MAE improvement:",
        f"{mae_improvement:.2f}%",
    )

    print(
        "RMSE improvement:",
        f"{rmse_improvement:.2f}%",
    )

    # ==================================================
    # 11. RECONSTRUCT PRICE
    # ==================================================

    print("\n" + "=" * 70)
    print("10. RECONSTRUCTING PRICE PREDICTIONS")
    print("=" * 70)

    #
    # The test set begins at prepared.test_data.index[0].
    #
    # For each test observation:
    #
    # predicted_price[t]
    # =
    # previous_close[t]
    # *
    # (1 + predicted_return[t])
    #

    test_index = (
        prepared.test_data.index
    )

    market_prices = (
        result.features.loc[
            test_index,
            PRICE_COLUMN,
        ]
    )

    previous_prices = (
        result.features[
            PRICE_COLUMN
        ]
        .shift(1)
        .loc[test_index]
        .to_numpy()
    )

    actual_prices = (
        market_prices
        .to_numpy()
    )

    predicted_prices = (
        previous_prices
        * (
            1
            + predicted_returns
        )
    )

    # ==================================================
    # 12. PRICE METRICS
    # ==================================================

    print("\n" + "=" * 70)
    print("11. RECONSTRUCTED PRICE METRICS")
    print("=" * 70)

    price_metrics = evaluate_predictions(
        y_true=actual_prices,
        y_pred=predicted_prices,
    )

    print(
        "LSTM Price MAE:",
        f"${price_metrics.mae:.2f}",
    )

    print(
        "LSTM Price RMSE:",
        f"${price_metrics.rmse:.2f}",
    )

    print(
        "LSTM Price MAPE:",
        f"{price_metrics.mape_percent:.2f}%",
    )

    print(
        "Price Direction Accuracy:",
        f"{price_metrics.direction_accuracy_percent:.2f}%",
    )

    # ==================================================
    # 13. PRICE NAIVE BASELINE
    # ==================================================

    naive_prices = (
        previous_prices
    )

    naive_price_metrics = (
        evaluate_predictions(
            y_true=actual_prices,
            y_pred=naive_prices,
        )
    )

    print("\n" + "=" * 70)
    print("12. PRICE BASELINE")
    print("=" * 70)

    print(
        "Naive Price MAE:",
        f"${naive_price_metrics.mae:.2f}",
    )

    print(
        "Naive Price RMSE:",
        f"${naive_price_metrics.rmse:.2f}",
    )

    print(
        "Naive Price MAPE:",
        f"{naive_price_metrics.mape_percent:.2f}%",
    )

    print(
        "Naive Direction Accuracy:",
        f"{naive_price_metrics.direction_accuracy_percent:.2f}%",
    )

    # ==================================================
    # 14. FINAL VALIDATION
    # ==================================================

    print("\n" + "=" * 70)
    print("13. VALIDATION")
    print("=" * 70)

    assert len(actual_returns) == len(
        predicted_returns
    )

    assert len(actual_prices) == len(
        predicted_prices
    )

    assert np.isfinite(
        actual_returns
    ).all()

    assert np.isfinite(
        predicted_returns
    ).all()

    assert np.isfinite(
        actual_prices
    ).all()

    assert np.isfinite(
        predicted_prices
    ).all()

    assert metrics.mae >= 0

    assert metrics.rmse >= 0

    assert price_metrics.mae >= 0

    assert price_metrics.rmse >= 0

    print(
        "Feature compatibility: PASS"
    )

    print(
        "Return predictions: PASS"
    )

    print(
        "Price reconstruction: PASS"
    )

    print(
        "Metrics validity: PASS"
    )

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 70)
    print("FINAL RETURN MODEL EVALUATION: PASS")
    print("=" * 70)

    print(
        "Model:",
        predictor.model.name,
    )

    print(
        "Model version:",
        "refactored-v4",
    )

    print(
        "Target:",
        TARGET_COLUMN,
    )

    print(
        "Test samples:",
        metrics.n_samples,
    )

    print(
        "Return MAE:",
        f"{metrics.mae * 100:.4f}%",
    )

    print(
        "Return RMSE:",
        f"{metrics.rmse * 100:.4f}%",
    )

    print(
        "Direction Accuracy:",
        f"{metrics.direction_accuracy_percent:.2f}%",
    )

    print(
        "Price MAE:",
        f"${price_metrics.mae:.2f}",
    )

    print(
        "Price RMSE:",
        f"${price_metrics.rmse:.2f}",
    )


if __name__ == "__main__":
    main()