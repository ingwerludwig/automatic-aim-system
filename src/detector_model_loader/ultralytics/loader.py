"""Ultralytics-specific detector model loader implementation."""

from typing import Any

from ultralytics import YOLO

from src.detector_config.ultralytics.config import UltralyticsDetectorConfig
from src.detector_model_loader.base_detector_model_loader import BaseDetectorModelLoader
from src.detector_model_loader.detector_model_loader_registry import detector_model_loader_registry


@detector_model_loader_registry.register(
    "ultralytics",
    description="Detector model loader for Ultralytics (YOLO, RT-DETR, etc.)"
)
class UltralyticsDetectorModelLoader(BaseDetectorModelLoader):
    """
    Ultralytics implementation of detector model loader.

    Given an UltralyticsDetectorConfig, this loader instantiates the
    appropriate Ultralytics model (YOLO, RT-DETR, etc.).
    """

    def load_model(self) -> Any:
        """
        Instantiate and return an Ultralytics detector model.

        Returns:
            An instance of ultralytics.YOLO.
        """
        cfg = self.config

        if not isinstance(cfg, UltralyticsDetectorConfig):
            raise TypeError(
                f"UltralyticsDetectorModelLoader received unsupported config type: {type(cfg)}"
            )

        params = cfg.get_init_params()

        model_path = params["model"]
        model = YOLO(model_path)

        # Device - set via overrides for Ultralytics models
        device = params.get("device")
        if device:
            model.overrides['device'] = device

        # Fuse Conv+BN for speed
        if params.get("fuse", False):
            model.fuse()

        # FP16 - set via overrides
        if params.get("half", False):
            model.overrides['half'] = True

        # Optionally attach defaults to the model instance
        model._default_imgsz = params.get("default_imgsz", 640)
        model._default_conf = params.get("default_conf", 0.25)
        model._default_iou = params.get("default_iou", 0.45)
        model._default_classes = params.get("classes", None)
        model._default_device = device
        model._task = params.get("task", None)

        return model

    @classmethod
    def get_name(cls) -> str:
        """Return the name of this loader."""
        return "ultralytics"
