import numpy as np
import pytest

from src.model.model_inference import (
    ModelInference,
    ModelInferenceError,
)


class FakeModel:

    def predict(
        self,
        X,
        verbose=0,
    ):
        assert X.shape == (
            1,
            60,
            3,
        )

        return np.array(
            [[0.80]]
        )


class FakeScaler:

    def transform(self, X):
        return X


def test_inference_returns_direction_and_probability():

    inference = ModelInference(
        model=FakeModel(),
        scaler=FakeScaler(),
        sequence_length=60,
        feature_columns=[
            "Return",
            "Return_5D",
            "Return_10D",
        ],
    )

    features = np.ones(
        (60, 3)
    )

    result = inference.predict(
        features
    )

    assert result["probability"] == pytest.approx(
        0.80
    )

    assert result["direction"] == "UP"


def test_probability_below_threshold_returns_down():

    class DownModel:

        def predict(
            self,
            X,
            verbose=0,
        ):
            return np.array(
                [[0.20]]
            )

    inference = ModelInference(
        model=DownModel(),
        scaler=FakeScaler(),
        sequence_length=60,
        feature_columns=[
            "Return",
            "Return_5D",
            "Return_10D",
        ],
    )

    features = np.ones(
        (60, 3)
    )

    result = inference.predict(
        features
    )

    assert result["probability"] == pytest.approx(
        0.20
    )

    assert result["direction"] == "DOWN"


def test_inference_requires_exact_sequence_length():

    inference = ModelInference(
        model=FakeModel(),
        scaler=FakeScaler(),
        sequence_length=60,
        feature_columns=[
            "Return",
            "Return_5D",
            "Return_10D",
        ],
    )

    features = np.ones(
        (59, 3)
    )

    with pytest.raises(
        ModelInferenceError
    ):
        inference.predict(
            features
        )


def test_inference_requires_expected_number_of_features():

    inference = ModelInference(
        model=FakeModel(),
        scaler=FakeScaler(),
        sequence_length=60,
        feature_columns=[
            "Return",
            "Return_5D",
            "Return_10D",
        ],
    )

    features = np.ones(
        (60, 2)
    )

    with pytest.raises(
        ModelInferenceError
    ):
        inference.predict(
            features
        )


def test_inference_rejects_invalid_probability():

    class InvalidModel:

        def predict(
            self,
            X,
            verbose=0,
        ):
            return np.array(
                [[1.5]]
            )

    inference = ModelInference(
        model=InvalidModel(),
        scaler=FakeScaler(),
        sequence_length=60,
        feature_columns=[
            "Return",
            "Return_5D",
            "Return_10D",
        ],
    )

    features = np.ones(
        (60, 3)
    )

    with pytest.raises(
        ModelInferenceError
    ):
        inference.predict(
            features
        )
