from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


def load_model_config(path: Path = CONFIG_PATH) -> dict:
    """
    Load and validate the model configuration from YAML.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Model configuration not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a YAML mapping.")

    validate_model_config(config)

    return config


def validate_model_config(config: dict) -> None:
    """
    Validate the minimum required model configuration.
    """

    required_sections = [
        "model",
        "data",
        "features",
        "sequence",
        "forecast",
        "training",
        "architecture",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: '{section}'"
            )

    if not config["data"].get("ticker"):
        raise ValueError("Ticker must be defined.")

    if config["sequence"].get("length", 0) <= 0:
        raise ValueError("Sequence length must be greater than zero.")

    if config["forecast"].get("horizon", 0) <= 0:
        raise ValueError("Forecast horizon must be greater than zero.")

    if config["training"].get("learning_rate", 0) <= 0:
        raise ValueError("Learning rate must be greater than zero.")

    if config["training"].get("batch_size", 0) <= 0:
        raise ValueError("Batch size must be greater than zero.")

    if config["architecture"]["lstm_1"]["units"] <= 0:
        raise ValueError("LSTM 1 units must be greater than zero.")

    if config["architecture"]["lstm_2"]["units"] <= 0:
        raise ValueError("LSTM 2 units must be greater than zero.")