from copy import deepcopy

from .config_loader import load_model_config


def build_runtime_config(
    ticker: str | None = None,
    sentiment_enabled: bool | None = None,
    multivariate_enabled: bool | None = None,
) -> dict:
    """
    Build a runtime configuration from the base model configuration.

    Runtime parameters override the values defined in configs/model.yaml.
    The original configuration is never modified.
    """

    config = deepcopy(load_model_config())

    if ticker is not None:
        ticker = ticker.strip().upper()

        if not ticker:
            raise ValueError("Ticker cannot be empty.")

        config["data"]["ticker"] = ticker

    if sentiment_enabled is not None:
        config["features"]["sentiment"]["enabled"] = bool(
            sentiment_enabled
        )

    if multivariate_enabled is not None:
        config["features"]["multivariate"]["enabled"] = bool(
            multivariate_enabled
        )

    return config