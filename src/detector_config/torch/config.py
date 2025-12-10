from __future__ import annotations

import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

from src.detector_config.base_detector_config import BaseDetectorConfig, DetectorConfigFactory


@DetectorConfigFactory.register("torch")
@dataclass
class TorchDetectorConfig(BaseDetectorConfig):
    """
    Configuration for torchvision-based detectors
    (e.g. Faster R-CNN ResNet50 FPN).

    All important hyperparameters that affect detection / tracking behavior
    are exposed here and can be set via config.yaml.
    """

    # ---- Model definition ----
    model_fn: str = "fasterrcnn_resnet50_fpn"   # torchvision.models.detection.<model_fn>
    weights: Optional[str] = None               # e.g. "DEFAULT" or None
    weights_path: Optional[str] = None          # path to .pth file (if using custom weights)
    num_classes: Optional[int] = None           # only relevant if you fine-tune / replace head

    # ---- Inference / runtime behaviour ----
    device: str = "cuda"                        # "cuda", "cuda:0", "cpu"
    score_threshold: float = 0.3                # post-filter in our adapter
    person_only: bool = True                    # if True, keep only COCO 'person' (id=1)

    # ---- Faster R-CNN internal hyperparameters ----
    # These are forwarded to the torchvision FasterRCNN constructor via **kwargs.
    # NOTE: Not all models will use all of these; they're primarily for Faster R-CNN.
    box_score_thresh: Optional[float] = None        # default inside torchvision is 0.05
    box_nms_thresh: Optional[float] = None          # default 0.5
    box_detections_per_img: Optional[int] = None    # default 100

    rpn_nms_thresh: Optional[float] = None          # default 0.7
    rpn_pre_nms_top_n_train: Optional[int] = None   # default 2000
    rpn_pre_nms_top_n_test: Optional[int] = None    # default 1000
    rpn_post_nms_top_n_train: Optional[int] = None  # default 2000
    rpn_post_nms_top_n_test: Optional[int] = None   # default 1000

    def get_init_params(self) -> Dict[str, Any]:
        """
        Return parameters needed by the Torch detector model loader and
        downstream inference code.
        """
        return {
            "model_fn": self.model_fn,
            "weights": self.weights,
            "weights_path": self.weights_path,
            "num_classes": self.num_classes,

            "device": self.device,
            "score_threshold": self.score_threshold,
            "person_only": self.person_only,

            "box_score_thresh": self.box_score_thresh,
            "box_nms_thresh": self.box_nms_thresh,
            "box_detections_per_img": self.box_detections_per_img,

            "rpn_nms_thresh": self.rpn_nms_thresh,
            "rpn_pre_nms_top_n_train": self.rpn_pre_nms_top_n_train,
            "rpn_pre_nms_top_n_test": self.rpn_pre_nms_top_n_test,
            "rpn_post_nms_top_n_train": self.rpn_post_nms_top_n_train,
            "rpn_post_nms_top_n_test": self.rpn_post_nms_top_n_test,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TorchDetectorConfig":
        return cls(**config_dict)

    @classmethod
    def from_torch_model_yaml(cls, path: str | Path) -> "TorchDetectorConfig":
        """
        Load from a YAML file with structure:

        config:
          model_init_name: fasterrcnn_resnet50_fpn
          weights: null
          weights_path: /path/to/fasterrcnn_resnet50_fpn_coco.pth

          inference_cfg:
            device: cuda
            score_threshold: 0.3
            person_only: true

            # Optional internal FRCNN hyperparameters:
            # box_score_thresh: 0.05
            # box_nms_thresh: 0.5
            # box_detections_per_img: 100
            # rpn_nms_thresh: 0.7
            # rpn_pre_nms_top_n_train: 2000
            # rpn_pre_nms_top_n_test: 1000
            # rpn_post_nms_top_n_train: 2000
            # rpn_post_nms_top_n_test: 1000
        """
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)

        if "config" not in raw:
            raise ValueError(f"Expected top-level 'config' key in {path}")

        cfg_block = raw["config"]

        model_init_name = cfg_block.get("model_init_name", "fasterrcnn_resnet50_fpn")
        weights = cfg_block.get("weights", None)
        weights_path = cfg_block.get("weights_path", None)

        inf = cfg_block.get("inference_cfg", {})

        return cls(
            model_fn=model_init_name,
            weights=weights,
            weights_path=weights_path,
            num_classes=inf.get("num_classes"),

            device=inf.get("device", "cuda"),
            score_threshold=inf.get("score_threshold", 0.3),
            person_only=inf.get("person_only", True),

            box_score_thresh=inf.get("box_score_thresh"),
            box_nms_thresh=inf.get("box_nms_thresh"),
            box_detections_per_img=inf.get("box_detections_per_img"),

            rpn_nms_thresh=inf.get("rpn_nms_thresh"),
            rpn_pre_nms_top_n_train=inf.get("rpn_pre_nms_top_n_train"),
            rpn_pre_nms_top_n_test=inf.get("rpn_pre_nms_top_n_test"),
            rpn_post_nms_top_n_train=inf.get("rpn_post_nms_top_n_train"),
            rpn_post_nms_top_n_test=inf.get("rpn_post_nms_top_n_test"),
        )

    @classmethod
    def from_model_name(cls, model_name: str, config_root: Path) -> "TorchDetectorConfig":
        """
        Given a model_name (e.g. 'fasterrcnn_resnet50_fpn') and a config_root,
        look for:

            <config_root> / "torch" / "detector" / <model_name> / "config.yaml"
        """
        yaml_path = (
            config_root
            / "torch"
            / "detector"
            / model_name
            / "config.yaml"
        )
        return cls.from_torch_model_yaml(yaml_path)