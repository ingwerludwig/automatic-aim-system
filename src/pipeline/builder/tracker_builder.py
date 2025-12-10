from __future__ import annotations

from pathlib import Path
import yaml

from src.config_paths.env import CONFIG_ROOT
from src.tracker_config.base_tracker_config import TrackerConfigFactory, BaseTrackerConfig
from src.tracker_model_loader import create_tracker_model_loader


def load_global_inference_config(path: str | Path) -> dict:
    path = Path(path)
    return yaml.safe_load(path.read_text())


def _normalize_tracker_model_name(model_name: str) -> str:
    """
    Turn 'bytetracker' -> 'bytetrack'
    Keep 'botsort' as 'botsort', etc.

    This just needs to match how you registered TrackerConfigFactory keys:
      'ultralytics_bytetrack'
      'ultralytics_botsort'
    """
    name = model_name.lower()
    if name == "bytetracker":
        return "bytetrack"
    return name  # e.g. botsort stays botsort


def build_tracker(global_cfg: dict):
    """
    Framework-agnostic tracker builder.

    Expects:
      global_cfg['tracker'] = {
        'framework': 'ultralytics',
        'model_name': 'bytetracker'  # or 'botsort'
      }
    """
    trk_block = global_cfg["tracker"]
    framework = trk_block["framework"]      # e.g. "ultralytics"
    model_name = trk_block["model_name"]    # e.g. "bytetracker"

    # 1) derive tracker_type key used in TrackerConfigFactory
    normalized = _normalize_tracker_model_name(model_name)  # 'bytetracker' -> 'bytetrack'
    tracker_type = f"{framework}_{normalized}"              # 'ultralytics_bytetrack'

    # 2) get the config class for that tracker_type
    cfg_cls = TrackerConfigFactory.get_config_class(tracker_type)

    # 3) instantiate config from model_name + CONFIG_ROOT
    cfg: BaseTrackerConfig = cfg_cls.from_model_name(model_name, CONFIG_ROOT)

    # 4) create framework-specific tracker model loader
    trk_loader = create_tracker_model_loader(framework, cfg)
    trk_loader.load_model()  # Initialize the internal tracker

    return trk_loader, cfg  # Return the loader, not the raw tracker