"""Base configuration classes for detector factory pattern."""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any
from pathlib import Path


class BaseDetectorConfig(ABC):
    """Abstract base class for all detector configurations."""

    @abstractmethod
    def get_init_params(self) -> Dict[str, Any]:
        """Return parameters for detector initialization."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseDetectorConfig':
        """Create config from dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_model_name(cls, model_name: str, config_root: Path) -> "BaseDetectorConfig":
        """
        Create config from a model_name and a config root directory.

        Each framework (ultralytics, mmdet, etc.) can decide how model_name
        maps to a specific YAML under config_root.
        """
        pass


class DetectorConfigFactory:
    """Factory class to create detector configurations based on type."""

    _registry: Dict[str, Type[BaseDetectorConfig]] = {}

    @classmethod
    def register(cls, detector_type: str):
        """Decorator to register detector config classes."""

        def wrapper(config_class: Type[BaseDetectorConfig]):
            cls._registry[detector_type] = config_class
            return config_class

        return wrapper

    @classmethod
    def create(cls, detector_type: str, config_dict: Dict[str, Any]) -> BaseDetectorConfig:
        """Create detector configuration instance."""
        if detector_type not in cls._registry:
            raise ValueError(f"Unknown detector type: {detector_type}. Available: {list(cls._registry.keys())}")

        config_class = cls._registry[detector_type]
        return config_class.from_dict(config_dict)

    @classmethod
    def get_available_detectors(cls) -> list:
        """Get list of available detector types."""
        return list(cls._registry.keys())

    @classmethod
    def get_config_class(cls, detector_type: str) -> Type[BaseDetectorConfig]:
        if detector_type not in cls._registry:
            raise ValueError(
                f"Unknown detector type: {detector_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[detector_type]