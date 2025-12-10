# src/pipeline/detector_builder.py
from __future__ import annotations

from pathlib import Path
import yaml

from src.config_paths.env import CONFIG_ROOT
from src.detector_config.base_detector_config import DetectorConfigFactory
from src.detector_model_loader import create_detector_model_loader
from src.detector_config import BaseDetectorConfig


def load_global_inference_config(path: str | Path) -> dict:
    path = Path(path)
    return yaml.safe_load(path.read_text())


def build_detector(global_cfg: dict):
    """
    Framework-agnostic detector builder.

    Expects:
      global_cfg['detector'] = {
        'framework': 'ultralytics',
        'model_name': 'yolov12n',
      }
    """
    det_block = global_cfg["detector"]
    framework = det_block["framework"]     # e.g. "ultralytics"
    model_name = det_block["model_name"]   # e.g. "yolov12n"

    # 1) get the config class for that framework
    cfg_cls = DetectorConfigFactory.get_config_class(framework)

    # 2) require that each config class implements from_model_name
    if not hasattr(cfg_cls, "from_model_name"):
        raise TypeError(
            f"Detector config class for framework '{framework}' "
            f"must implement from_model_name(model_name, config_root)."
        )

    cfg: BaseDetectorConfig = cfg_cls.from_model_name(model_name, CONFIG_ROOT)

    # 3) use your existing model loader factory
    det_loader = create_detector_model_loader(framework, cfg)
    detector = det_loader.load_model()

    return detector, cfg