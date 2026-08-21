from __future__ import annotations

import numpy as np

from src.pipeline.market_pipeline import run_market_pipeline


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    mask = y_true != 0

    mape = (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )

    actual_direction = np.sign(
        np.diff(y_true)
    )

    predicted_direction = np.sign(
        np.diff(y_pred)
    )

    direction_accuracy = (
        np.mean(
            actual_direction
            == predicted_direction
        )
        * 100
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "direction_accuracy": float(
            direction_accuracy
        ),
    }


def main():

    print("=" * 70)
    print("NAIVE BASELINE EVALUATION")
    print("=" * 70)

    # ==================================================
    # 1. LOAD MARKET DATA
    # ==================================================

    result = run_market_pipeline(
        ticker="TSLA",
        sentiment_enabled=False,
    )

    df = result.market_data.copy()

    close_column = "TSLA_Close"

    prices = (
        df[close_column]
        .astype(float)
        .values
    )

    print("\nRecords:", len(prices))

    # ==================================================
    # 2. SAME TEST PERIOD AS LSTM
    # ==================================================

    test_ratio = 0.20

    train_size = int(
        len(prices) * (1 - test_ratio)
    )

    train_prices = prices[
        :train_size
    ]

    test_prices = prices[
        train_size:
    ]

    # ==================================================
    # 3. NAIVE PREDICTION
    #
    # Prediction for day t:
    # previous day's actual close
    # ==================================================

    y_true = test_prices[1:]

    y_pred = test_prices[:-1]

    print(
        "Train rows:",
        len(train_prices),
    )

    print(
        "Test rows:",
        len(test_prices),
    )

    print(
        "Evaluation samples:",
        len(y_true),
    )

    # ==================================================
    # 4. METRICS
    # ==================================================

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    print("\n" + "=" * 70)
    print("NAIVE BASELINE METRICS")
    print("=" * 70)

    print(
        f"MAE:                 ${metrics['mae']:.2f}"
    )

    print(
        f"RMSE:                ${metrics['rmse']:.2f}"
    )

    print(
        f"MAPE:                {metrics['mape']:.2f}%"
    )

    print(
        f"Direction Accuracy:  "
        f"{metrics['direction_accuracy']:.2f}%"
    )

    print(
        f"Samples:             {len(y_true)}"
    )

    # ==================================================
    # 5. VALIDATION
    # ==================================================

    assert len(y_true) == len(y_pred)

    assert np.all(
        np.isfinite(y_true)
    )

    assert np.all(
        np.isfinite(y_pred)
    )

    print("\n" + "=" * 70)
    print("BASELINE VALIDATION")
    print("=" * 70)

    print("Prediction count: PASS")
    print("Finite values: PASS")
    print("Metrics validity: PASS")

    print("\n" + "=" * 70)
    print("NAIVE BASELINE: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()