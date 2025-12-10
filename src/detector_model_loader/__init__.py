"""Factory and registry access for detector model loaders."""

import logging
from typing import List, Type

from src.detector_config import BaseDetectorConfig
from src.detector_model_loader.base_detector_model_loader import BaseDetectorModelLoader
from src.detector_model_loader.detector_model_loader_registry import detector_model_loader_registry

# Import concrete implementations so they register themselves
from src.detector_model_loader.ultralytics.loader import UltralyticsDetectorModelLoader  # noqa: F401
from src.detector_model_loader.torch.loader import TorchDetectorModelLoader  # noqa: F401

logger = logging.getLogger(__name__)


def get_available_detector_model_loaders() -> List[str]:
    """
    Get list of all registered detector model loader names.

    Returns:
        List of registered loader names (e.g., ["ultralytics"]).
    """
    return detector_model_loader_registry.list_names()


def create_detector_model_loader(
    loader_name: str,
    config: BaseDetectorConfig,
) -> BaseDetectorModelLoader:
    """
    Factory function to create a detector model loader instance.

    Args:
        loader_name: Registry name of the loader implementation, e.g. "ultralytics".
        config: Detector configuration (e.g., UltralyticsDetectorConfig).

    Returns:
        An instance of BaseDetectorModelLoader subclass.
    """
    # Validate loader exists
    available = get_available_detector_model_loaders()
    if loader_name not in available:
        raise ValueError(
            f"Unknown detector model loader '{loader_name}'. Available: {available}"
        )

    # Get the loader class from registry
    loader_cls: Type[BaseDetectorModelLoader] = detector_model_loader_registry.get(loader_name)

    # Create instance and log
    loader_instance = loader_cls(config)
    logger.info(f"Created detector model loader: {loader_instance.get_name()}")

    return loader_instance
