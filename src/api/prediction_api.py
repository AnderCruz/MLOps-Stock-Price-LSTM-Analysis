from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model.model_inference import (
    ModelInferenceError,
)


class PredictionRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description=(
            "Feature sequence with shape "
            "(60, 3)."
        ),
    )


class PredictionResponse(BaseModel):
    probability: float
    direction: str
    model_name: str
    model_version: str


def create_prediction_app(
    *,
    inference: Any,
) -> FastAPI:
    """
    Create the stock-direction prediction API.

    The API is intentionally thin. It handles HTTP
    validation and delegates model inference to the
    supplied inference service.
    """

    app = FastAPI(
        title="Stock Direction Prediction API",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok"
        }

    @app.post(
        "/predict",
        response_model=PredictionResponse,
    )
    def predict(
        request: PredictionRequest,
    ) -> dict[str, Any]:

        try:
            features = np.asarray(
                request.features,
                dtype=float,
            )

            if features.ndim != 2:
                raise ModelInferenceError(
                    "Features must be a "
                    "2-dimensional array."
                )

            if features.shape != (
                60,
                3,
            ):
                raise ModelInferenceError(
                    "Features must have shape "
                    "(60, 3)."
                )

            result = inference.predict(
                features
            )

            return result

        except ModelInferenceError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except (
            ValueError,
            TypeError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    return app
