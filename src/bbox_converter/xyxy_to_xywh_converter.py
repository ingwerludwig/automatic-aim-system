"""Converter for xyxy format bbox data into xywh format."""
from typing import Any

import numpy as np
from numpy import ndarray, dtype

from src.bbox_converter.base_bbox_converter import BaseBboxConverter
from src.bbox_converter.bbox_converter_registry import bbox_converter_registry


@bbox_converter_registry.register(
    "xyxy_to_xywh_converter",
    description="Convert xyxy format into xywh format"
)
class XyxyToXywhConverter(BaseBboxConverter):
    """
    Converter for bounding boxes from [x1, y1, x2, y2] format
    to [x_center, y_center, width, height] format.
    """

    def convert(self, xyxy_bbox_data: np.ndarray) -> ndarray[Any, dtype[Any]]:
        """
        Convert xyxy to xywh format.

        Args:
            xyxy_bbox_data: np.ndarray of shape [N, 4] (x1, y1, x2, y2)

        Returns:
            np.ndarray of shape [N, 4] (x_center, y_center, width, height)
        """
        x1, y1, x2, y2 = xyxy_bbox_data.T
        width = x2 - x1
        height = y1 * 0 + (y2 - y1)  # or simply y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        return np.column_stack([x_center, y_center, width, height])