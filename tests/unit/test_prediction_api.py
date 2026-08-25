from unittest.mock import MagicMock

import numpy as np
import pytest

from src.api.prediction_api import (
    create_prediction_app,
)


def make_inference():

    inference = MagicMock()

    inference.predict.return_value = {
        "probability": 0.80,
        "direction": "UP",
        "model_name": "stock-direction-lstm",
        "model_version": "refactored-v7",
    }

    inference.metadata = {
        "model_name": "stock-direction-lstm",
        "model_version": "refactored-v7",
        "status": "production",
    }

    return inference


def test_health_endpoint():

    from fastapi.testclient import TestClient

    app = create_prediction_app(
        inference=make_inference()
    )

    client = TestClient(app)

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok"
    }


def test_prediction_endpoint_returns_prediction():

    from fastapi.testclient import TestClient

    inference = make_inference()

    app = create_prediction_app(
        inference=inference
    )

    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "features": np.ones(
                (60, 3)
            ).tolist()
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["probability"] == pytest.approx(
        0.80
    )

    assert data["direction"] == "UP"

    assert (
        data["model_name"]
        == "stock-direction-lstm"
    )

    assert (
        data["model_version"]
        == "refactored-v7"
    )


def test_prediction_endpoint_passes_features_to_inference():

    from fastapi.testclient import TestClient

    inference = make_inference()

    app = create_prediction_app(
        inference=inference
    )

    client = TestClient(app)

    features = np.ones(
        (60, 3)
    ).tolist()

    response = client.post(
        "/predict",
        json={
            "features": features
        },
    )

    assert response.status_code == 200

    inference.predict.assert_called_once()

    received_features = (
        inference.predict.call_args.args[0]
    )

    assert received_features.shape == (
        60,
        3,
    )


def test_prediction_endpoint_rejects_invalid_shape():

    from fastapi.testclient import TestClient

    app = create_prediction_app(
        inference=make_inference()
    )

    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "features": np.ones(
                (59, 3)
            ).tolist()
        },
    )

    assert response.status_code == 400


def test_prediction_endpoint_handles_inference_error():

    from fastapi.testclient import TestClient

    inference = MagicMock()

    inference.predict.side_effect = (
        ValueError(
            "Inference failed."
        )
    )

    app = create_prediction_app(
        inference=inference
    )

    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "features": np.ones(
                (60, 3)
            ).tolist()
        },
    )

    assert response.status_code == 400
