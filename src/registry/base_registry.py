"""
Unified registry for framework-based components.
This replaces hardcoded dictionaries in all factory functions.
"""

from typing import Dict, Type, Optional, Any, List
from abc import ABC


class BaseRegistry:
    """Generic registry for framework-based components with two-level hierarchy."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Type]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self, level1: str, level2: str, **metadata):
        """
        Decorator for registering implementations.

        Args:
            level1: First level key (e.g., "framework" like "ultralytics")
            level2: Second level key (e.g., "model" like "yolo")
            **metadata: Optional metadata about the implementation

        Returns:
            Decorator function

        Example:
            @converter_registry.register("ultralytics", "yolo", description="Ultralytics YOLO model")
            class UltralyticsYoloModelLoader(BaseModelLoader):
                pass
        """
        def decorator(cls: Type) -> Type:
            if level1 not in self._registry:
                self._registry[level1] = {}
            self._registry[level1][level2] = cls

            # Store metadata if provided
            if metadata:
                key = f"{level1}:{level2}"
                self._metadata[key] = metadata

            return cls
        return decorator

    def get(self, level1: str, level2: str) -> Type:
        """
        Retrieve registered class.

        Args:
            level1: First level key
            level2: Second level key

        Returns:
            Registered class

        Raises:
            ValueError: If no implementation found
        """
        try:
            return self._registry[level1][level2]
        except KeyError:
            available = self._get_available_keys()
            raise ValueError(
                f"No implementation found for {level1}/{level2}.\n"
                f"Available: {available}"
            )

    def list_registered(self) -> Dict[str, Dict[str, Type]]:
        """Return all registered implementations."""
        return self._registry.copy()

    def get_metadata(self, level1: str, level2: str) -> Dict[str, Any]:
        """Get metadata for a registered implementation."""
        key = f"{level1}:{level2}"
        return self._metadata.get(key, {})

    def _get_available_keys(self) -> str:
        """Format available keys for error messages."""
        items = []
        for l1, l2_dict in self._registry.items():
            for l2 in l2_dict.keys():
                items.append(f"{l1}/{l2}")
        return ", ".join(items) if items else "None registered"


class SingleLevelRegistry:
    """Simple registry for single-level component registration."""

    def __init__(self):
        self._registry: Dict[str, Type] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, **metadata):
        """
        Decorator for registering implementations.

        Args:
            name: Component name (e.g., "ultralytics", "mmdetection")
            **metadata: Optional metadata about the implementation

        Returns:
            Decorator function

        Example:
            @generator_registry.register("ultralytics", description="Ultralytics Serve Model")
            class UltralyticsModelServe(BaseModelServe):
                pass
        """
        def decorator(cls: Type) -> Type:
            self._registry[name] = cls

            # Store metadata if provided
            if metadata:
                self._metadata[name] = metadata

            return cls
        return decorator

    def get(self, name: str) -> Type:
        """
        Retrieve registered class.

        Args:
            name: Component name

        Returns:
            Registered class

        Raises:
            ValueError: If no implementation found
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ValueError(
                f"Unknown component '{name}'.\n"
                f"Available: {available if available else 'None registered'}"
            )
        return self._registry[name]

    def list_registered(self) -> Dict[str, Type]:
        """Return all registered implementations."""
        return self._registry.copy()

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered implementation."""
        return self._metadata.get(name, {})

    def list_names(self) -> List[str]:
        """Return list of registered component names."""
        return list(self._registry.keys())
