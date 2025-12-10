"""Base configuration classes for tracker factory pattern."""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any
from pathlib import Path


class BaseTrackerConfig(ABC):
    """Abstract base class for all tracker configurations."""

    @abstractmethod
    def get_init_params(self) -> Dict[str, Any]:
        """Return parameters for tracker initialization."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseTrackerConfig':
        """Create config from dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_model_name(cls, model_name: str, config_root: Path) -> "BaseTrackerConfig":
        """
        Create config from a model_name and a config root directory.

        Each framework (ultralytics, etc.) can decide how model_name
        maps to a specific YAML under config_root.
        """
        pass


class TrackerConfigFactory:
    """Factory class to create tracker configurations based on type."""

    _registry: Dict[str, Type[BaseTrackerConfig]] = {}

    @classmethod
    def register(cls, tracker_type: str):
        """Decorator to register tracker config classes."""

        def wrapper(config_class: Type[BaseTrackerConfig]):
            cls._registry[tracker_type] = config_class
            return config_class

        return wrapper

    @classmethod
    def create(cls, tracker_type: str, config_dict: Dict[str, Any]) -> BaseTrackerConfig:
        """Create tracker configuration instance."""
        if tracker_type not in cls._registry:
            raise ValueError(f"Unknown tracker type: {tracker_type}. Available: {list(cls._registry.keys())}")

        config_class = cls._registry[tracker_type]
        return config_class.from_dict(config_dict)

    @classmethod
    def get_available_trackers(cls) -> list:
        """Get list of available tracker types."""
        return list(cls._registry.keys())

    @classmethod
    def get_config_class(cls, tracker_type: str) -> Type[BaseTrackerConfig]:
        if tracker_type not in cls._registry:
            raise ValueError(
                f"Unknown tracker type: {tracker_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[tracker_type]