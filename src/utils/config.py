"""Configuration loading and validation module."""
import os
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Relative or absolute path to the configuration YAML file.

    Returns:
        Dictionary containing configuration parameters.
    """
    if not os.path.exists(config_path):
        # Check relative to repo root if current working dir is different
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alt_path = os.path.join(root_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    return config
