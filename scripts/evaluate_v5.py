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

from src.model.return_preprocessing import (
    prepare_return_dataset,
)

from src.model.return_target import (
    add_return_target,
)

from src.model.predictor import (
    ModelPredictor,
)

from src.model.evaluator import (
    evaluate_predictions,
)


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_VERSION = "refactored-v5"

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / MODEL_VERSION
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
    print("RETURN MODEL V5 EVALUATION")
    print("=" * 70)

    # ==================================================
    # 1. LOAD PERSISTED MODEL
    # ==================================================

    print("\n" + "=" * 70)
    print("1. LOADING PERSISTED RETURN MODEL")
    print("=" * 70)

    predictor = ModelPredictor(
        str(MODEL_DIR)
    )

    print("Model:", predictor.model.name)
    print("Input:", predictor.model.input_shape)
    print(
        "Scaler:",
        type(predictor.scaler).__name__,
    )
    print(
        "Scaler features:",
        predictor.scaler.n_features_in_,
    )

    # ==================================================
    # 2. RECREATE MARKET DATA
    # ==================================================

    print("\n" + "=" * 70)
    print("2. RECREATING MARKET DATA")
    print("=" * 70)

    result = run_market_pipeline(
        ticker=TICKER,
        sentiment_enabled=False,
    )

    print("Records:", result.n_records)
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

    return_data = return_data[
        [
            TARGET_COLUMN,
            PRICE_COLUMN,
            VOLUME_COLUMN,
        ]
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
    # 4. PREPARE RETURN DATASET
    # ==================================================

    print("\n" + "=" * 70)
    print("4. PREPARING RETURN DATASET")
    print("=" * 70)

    prepared = prepare_return_dataset(
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
        "X_train:",
        prepared.X_train.shape,
    )

    print(
        "X_test:",
        prepared.X_test.shape,
    )

    # ==================================================
    # 5. STRUCTURAL VALIDATION
    # ==================================================

    print("\n" + "=" * 70)
    print("5. STRUCTURAL VALIDATION")
    print("=" * 70)

    expected_features = (
        predictor.scaler.n_features_in_
    )

    if prepared.n_features != expected_features:
        raise RuntimeError(
            "Feature count mismatch: "
            f"model expects {expected_features}, "
            f"dataset contains {prepared.n_features}."
        )

    if prepared.X_test.shape[1] != SEQUENCE_LENGTH:
        raise RuntimeError(
            "Sequence length mismatch."
        )

    if prepared.X_test.shape[2] != expected_features:
        raise RuntimeError(
            "X_test feature dimension mismatch."
        )

    if len(prepared.X_test) != prepared.test_size:
        raise RuntimeError(
            "Unexpected test sequence count."
        )

    print("Feature count: PASS")
    print("Sequence length: PASS")
    print("Test sequence count: PASS")

    # ==================================================
    # 6. GENERATE PREDICTIONS
    # ==================================================

    print("\n" + "=" * 70)
    print("6. GENERATING RETURN PREDICTIONS")
    print("=" * 70)

    predictions_scaled = (
        predictor.predict_scaled(
            prepared.X_test
        )
    )

    y_test_scaled = np.asarray(
        prepared.y_test
    ).reshape(-1)

    print(
        "Prediction samples:",
        len(predictions_scaled),
    )

    print(
        "Scaled prediction min:",
        f"{predictions_scaled.min():.6f}",
    )

    print(
        "Scaled prediction max:",
        f"{predictions_scaled.max():.6f}",
    )

    # ==================================================
    # 7. INVERSE TRANSFORM
    # ==================================================

    print("\n" + "=" * 70)
    print("7. CONVERTING RETURNS TO ORIGINAL SCALE")
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
        "Actual return mean:",
        f"{actual_returns.mean() * 100:.4f}%",
    )

    print(
        "Actual return std:",
        f"{actual_returns.std() * 100:.4f}%",
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
        "Predicted return mean:",
        f"{predicted_returns.mean() * 100:.4f}%",
    )

    print(
        "Predicted return std:",
        f"{predicted_returns.std() * 100:.4f}%",
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
    # 8. RETURN MODEL METRICS
    # ==================================================

    print("\n" + "=" * 70)
    print("8. RETURN MODEL METRICS")
    print("=" * 70)

    return_metrics = evaluate_predictions(
        y_true=actual_returns,
        y_pred=predicted_returns,
    )

    print(
        "Return MAE:",
        f"{return_metrics.mae * 100:.4f}%",
    )

    print(
        "Return RMSE:",
        f"{return_metrics.rmse * 100:.4f}%",
    )

    print(
        "Direction Accuracy:",
        f"{return_metrics.direction_accuracy_percent:.2f}%",
    )

    print(
        "Samples:",
        return_metrics.n_samples,
    )

    # ==================================================
    # 9. NAIVE RETURN BASELINE
    # ==================================================

    print("\n" + "=" * 70)
    print("9. NAIVE RETURN BASELINE")
    print("=" * 70)

    # Predict zero return:
    #
    # predicted_return[t] = 0
    #
    # Equivalent to predicting:
    #
    # price[t] = price[t-1]

    naive_returns = np.zeros_like(
        actual_returns
    )

    naive_return_metrics = (
        evaluate_predictions(
            y_true=actual_returns,
            y_pred=naive_returns,
        )
    )

    print(
        "Naive Return MAE:",
        f"{naive_return_metrics.mae * 100:.4f}%",
    )

    print(
        "Naive Return RMSE:",
        f"{naive_return_metrics.rmse * 100:.4f}%",
    )

    # ==================================================
    # 10. MODEL VS BASELINE
    # ==================================================

    print("\n" + "=" * 70)
    print("10. RETURN MODEL VS BASELINE")
    print("=" * 70)

    mae_improvement = (
        1
        - return_metrics.mae
        / naive_return_metrics.mae
    ) * 100

    rmse_improvement = (
        1
        - return_metrics.rmse
        / naive_return_metrics.rmse
    ) * 100

    print(
        "MAE improvement:",
        f"{mae_improvement:.2f}%",
    )

    print(
        "RMSE improvement:",
        f"{rmse_improvement:.2f}%",
    )

    if mae_improvement > 0:
        print(
            "MAE verdict: MODEL BETTER"
        )
    else:
        print(
            "MAE verdict: BASELINE BETTER"
        )

    if rmse_improvement > 0:
        print(
            "RMSE verdict: MODEL BETTER"
        )
    else:
        print(
            "RMSE verdict: BASELINE BETTER"
        )

    # ==================================================
    # 11. RECONSTRUCT PRICE
    # ==================================================

    print("\n" + "=" * 70)
    print("11. RECONSTRUCTING PRICE PREDICTIONS")
    print("=" * 70)

    test_index = prepared.test_data.index

    # The return at t is:
    #
    # Return[t] =
    #     Price[t] / Price[t-1] - 1
    #
    # Therefore:
    #
    # PredictedPrice[t] =
    #     ActualPrice[t-1] *
    #     (1 + PredictedReturn[t])
    #
    # For a one-step-ahead evaluation this uses only
    # information available immediately before t.

    prices = result.features[
        PRICE_COLUMN
    ].loc[test_index]

    previous_prices = (
        result.features[
            PRICE_COLUMN
        ]
        .shift(1)
        .loc[test_index]
        .to_numpy()
    )

    actual_prices = (
        prices.to_numpy()
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
    print("12. RECONSTRUCTED PRICE METRICS")
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
    # 13. NAIVE PRICE BASELINE
    # ==================================================

    print("\n" + "=" * 70)
    print("13. PRICE BASELINE")
    print("=" * 70)

    naive_prices = previous_prices

    naive_price_metrics = evaluate_predictions(
        y_true=actual_prices,
        y_pred=naive_prices,
    )

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
    # 14. PRICE MODEL VS BASELINE
    # ==================================================

    print("\n" + "=" * 70)
    print("14. PRICE MODEL VS BASELINE")
    print("=" * 70)

    price_mae_improvement = (
        1
        - price_metrics.mae
        / naive_price_metrics.mae
    ) * 100

    price_rmse_improvement = (
        1
        - price_metrics.rmse
        / naive_price_metrics.rmse
    ) * 100

    print(
        "Price MAE improvement:",
        f"{price_mae_improvement:.2f}%",
    )

    print(
        "Price RMSE improvement:",
        f"{price_rmse_improvement:.2f}%",
    )

    # ==================================================
    # 15. VALIDATION
    # ==================================================

    print("\n" + "=" * 70)
    print("15. VALIDATION")
    print("=" * 70)

    assert len(actual_returns) == len(
        predicted_returns
    )

    assert len(actual_prices) == len(
        predicted_prices
    )

    assert len(actual_returns) == (
        prepared.test_size
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

    assert return_metrics.mae >= 0
    assert return_metrics.rmse >= 0

    assert price_metrics.mae >= 0
    assert price_metrics.rmse >= 0

    assert (
        0
        <= return_metrics.direction_accuracy_percent
        <= 100
    )

    assert (
        0
        <= price_metrics.direction_accuracy_percent
        <= 100
    )

    print("Feature compatibility: PASS")
    print("Return predictions: PASS")
    print("Return inverse transform: PASS")
    print("Price reconstruction: PASS")
    print("Metrics validity: PASS")

    # ==================================================
    # 16. FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 70)
    print("FINAL RETURN MODEL V5 EVALUATION: PASS")
    print("=" * 70)

    print("Model:", predictor.model.name)
    print("Model version:", MODEL_VERSION)
    print("Ticker:", TICKER)
    print("Target:", TARGET_COLUMN)
    print(
        "Test samples:",
        return_metrics.n_samples,
    )

    print()
    print(
        f"Return MAE: "
        f"{return_metrics.mae * 100:.4f}%"
    )

    print(
        f"Return RMSE: "
        f"{return_metrics.rmse * 100:.4f}%"
    )

    print(
        f"Return Direction Accuracy: "
        f"{return_metrics.direction_accuracy_percent:.2f}%"
    )

    print()
    print(
        f"Price MAE: "
        f"${price_metrics.mae:.2f}"
    )

    print(
        f"Price RMSE: "
        f"${price_metrics.rmse:.2f}"
    )

    print(
        f"Price MAPE: "
        f"{price_metrics.mape_percent:.2f}%"
    )

    print(
        f"Price Direction Accuracy: "
        f"{price_metrics.direction_accuracy_percent:.2f}%"
    )

    print()
    print(
        f"Price MAE vs Naive: "
        f"{price_mae_improvement:.2f}%"
    )

    print(
        f"Price RMSE vs Naive: "
        f"{price_rmse_improvement:.2f}%"
    )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
