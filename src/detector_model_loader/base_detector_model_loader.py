"""
Abstract base class for model loaders.
Provides interface for different model loading implementations with different technique (PyTorch Ultralytics, PyTorch MMDetection, Onnx.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


from src.detector_config import BaseDetectorConfig


class BaseDetectorModelLoader(ABC):
    """
    Abstract base class for model loaders.

    Different implementations (PyTorch Ultralytics, PyTorch MMDetection, Onnx, etc.)
    should inherit from this class and implement the required methods.
    """

    def __init__(self, config: BaseDetectorConfig):
        """
        Initialize model loader with configuration.

        Args:
            config: ModelLoaderConfig with model settings
        """
        self.config = config

    @abstractmethod
    def load_model(self) -> Any:
        """
        Load model.

        Returns:
            Any of model
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Return the name of this loader.

        Returns:
            String identifier for this loader type.
        """
        pass
