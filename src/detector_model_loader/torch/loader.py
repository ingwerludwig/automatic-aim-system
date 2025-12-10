"""Torch / torchvision-specific detector model loader implementation."""

from typing import Any

import torch
import torchvision

from src.detector_config.torch.config import TorchDetectorConfig
from src.detector_model_loader.base_detector_model_loader import BaseDetectorModelLoader
from src.detector_model_loader.detector_model_loader_registry import detector_model_loader_registry


@detector_model_loader_registry.register(
    "torch",
    description="Detector model loader for torchvision-based models (e.g. Faster R-CNN ResNet50 FPN).",
)
class TorchDetectorModelLoader(BaseDetectorModelLoader):
    """
    Torch / torchvision implementation of detector model loader.

    Given a TorchDetectorConfig, this loader instantiates a torchvision
    detection model, e.g.:

        - fasterrcnn_resnet50_fpn
        - fasterrcnn_resnet50_fpn_v2
        - fasterrcnn_mobilenet_v3_large_320_fpn
        etc.
    """

    def load_model(self) -> Any:
        """
        Instantiate and return a torchvision detector model.

        Returns:
            An instance of a torchvision.models.detection.* model.
        """
        cfg = self.config

        if not isinstance(cfg, TorchDetectorConfig):
            raise TypeError(
                f"TorchDetectorModelLoader received unsupported config type: {type(cfg)}"
            )

        params = cfg.get_init_params()

        model_fn_name = params["model_fn"]
        weights = params["weights"]
        weights_path = params["weights_path"]
        device_str = params.get("device") or "cpu"

        # --------- Resolve the torchvision function ----------
        if not hasattr(torchvision.models.detection, model_fn_name):
            raise ValueError(
                f"torchvision.models.detection has no attribute '{model_fn_name}'. "
                f"Check your 'model_init_name' in config.yaml."
            )

        model_fn = getattr(torchvision.models.detection, model_fn_name)

        # --------- Collect Faster R-CNN kwargs (only non-None) ----------
        detector_kwargs = {}

        # These kwargs are forwarded into torchvision's FasterRCNN constructor
        # via the model_fn(**detector_kwargs) call.
        for key in [
            "box_score_thresh",
            "box_nms_thresh",
            "box_detections_per_img",
            "rpn_nms_thresh",
            "rpn_pre_nms_top_n_train",
            "rpn_pre_nms_top_n_test",
            "rpn_post_nms_top_n_train",
            "rpn_post_nms_top_n_test",
        ]:
            value = params.get(key)
            if value is not None:
                detector_kwargs[key] = value

        # Optionally handle num_classes (if you later fine-tune)
        if params.get("num_classes") is not None:
            detector_kwargs["num_classes"] = params["num_classes"]

        # --------- Build model with either built-in or custom weights ----------
        if weights_path is not None:
            # Custom weights from a .pth file
            model = model_fn(weights=None, **detector_kwargs)
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            # Use torchvision's built-in weights enum / "DEFAULT"
            # e.g. weights="DEFAULT" or a specific weights enum
            model = model_fn(weights=weights, **detector_kwargs)

        # --------- Send to device & eval mode ----------
        device = torch.device(device_str)
        model.to(device)
        model.eval()

        # --------- Attach useful inference-related attributes ----------
        # These are used later in your inference / tracking pipeline
        model._score_threshold = params.get("score_threshold", 0.3)
        model._person_only = params.get("person_only", True)
        model._device_str = device_str

        return model

    @classmethod
    def get_name(cls) -> str:
        """Return the name of this loader."""
        return "torch"