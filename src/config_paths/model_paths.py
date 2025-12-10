from pathlib import Path
from src.config_paths.env import CONFIG_ROOT


def inference_config_root() -> Path:
    return CONFIG_ROOT / "inference"

def get_inference_config_path(config_name) -> Path:
    return inference_config_root() / f"{config_name}.yaml"