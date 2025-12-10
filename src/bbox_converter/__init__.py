# src/bbox_converter/__init__.py
"""Factory and registry access for bbox converters."""

from typing import Any, Type

from src.bbox_converter.base_bbox_converter import BaseBboxConverter
from src.bbox_converter.bbox_converter_registry import bbox_converter_registry

from src.bbox_converter.xyxy_to_xywh_converter import XyxyToXywhConverter  # noqa: F401


def create_bbox_converter(name: str) -> BaseBboxConverter:
    """
    Factory function to create a bbox converter instance.

    Args:
        name: Registry name of the converter, e.g. "xyxy_to_xywh_converter".

    Returns:
        An instance of BaseBboxConverter subclass.
    """
    converter_cls: Type[BaseBboxConverter] = bbox_converter_registry.get(name)

    if converter_cls is None:
        available = (
            bbox_converter_registry.list()
            if hasattr(bbox_converter_registry, "list")
            else "unknown"
        )
        raise ValueError(
            f"Unknown bbox converter '{name}'. Available: {available}"
        )

    return converter_cls()


def get_available_bbox_converters() -> Any:
    """
    Return the list of available bbox converters from the registry.
    """
    if hasattr(bbox_converter_registry, "list"):
        return bbox_converter_registry.list()
    if hasattr(bbox_converter_registry, "keys"):
        return bbox_converter_registry.keys()
    return []