"""Factory and registry access for tracker model loaders."""

import logging
from typing import List, Type

from src.tracker_config import BaseTrackerConfig
from src.tracker_model_loader.base_tracker_model_loader import BaseTrackerModelLoader
from src.tracker_model_loader.tracker_model_loader_registry import tracker_model_loader_registry

# Existing Ultralytics loader
from src.tracker_model_loader.ultralytics.loader import UltralyticsTrackerModelLoader  # noqa: F401

from src.tracker_model_loader.torch.loader import TorchDeepsortTrackerModelLoader  # noqa: F401

logger = logging.getLogger(__name__)


def get_available_tracker_model_loaders() -> List[str]:
    """
    Get list of all registered tracker model loader names.

    Returns:
        List of registered loader names (e.g., ["ultralytics", "deepsort"]).
    """
    return tracker_model_loader_registry.list_names()


def create_tracker_model_loader(
    loader_name: str,
    config: BaseTrackerConfig,
) -> BaseTrackerModelLoader:
    """
    Factory function to create a tracker model loader instance.

    Args:
        loader_name: Registry name of the loader implementation,
                    e.g. "ultralytics" or "deepsort".
        config: Tracker configuration (e.g., UltralyticsBytetrackConfig, DeepSortRealtimeConfig).

    Returns:
        An instance of BaseTrackerModelLoader subclass.
    """
    # Validate loader exists
    available = get_available_tracker_model_loaders()
    if loader_name not in available:
        raise ValueError(
            f"Unknown tracker model loader '{loader_name}'. Available: {available}"
        )

    # Get the loader class from registry
    loader_cls: Type[BaseTrackerModelLoader] = tracker_model_loader_registry.get(loader_name)

    # Create instance and log
    loader_instance = loader_cls(config)
    logger.info(f"Created tracker model loader: {loader_instance.get_name()}")

    return loader_instance