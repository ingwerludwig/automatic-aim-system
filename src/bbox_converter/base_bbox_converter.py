# src/bbox_converter/base_bbox_converter.py
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseBboxConverter(ABC):
    """
    Abstract base class for converting bbox format.

    Each converter implements format-specific logic to convert
    bbox format into another format.
    """

    @abstractmethod
    def convert(self, bbox_data: np.ndarray) -> Any:
        """
        Convert certain bbox data into another bbox format.

        Args:
            bbox_data: Bbox data, typically shape [N, 4] or [N, >=4] as np.ndarray

        Returns:
            Converted bbox data in another format (usually np.ndarray).
        """
        pass