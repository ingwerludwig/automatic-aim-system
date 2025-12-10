from __future__ import annotations

import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.detector_config.base_detector_config import BaseDetectorConfig, DetectorConfigFactory


@DetectorConfigFactory.register("ultralytics")
@dataclass
class UltralyticsDetectorConfig(BaseDetectorConfig):
    """
    Configuration for Ultralytics models (YOLO, RT-DETR, etc.).

    `model` can be:
      - yolov8n.pt, yolov8s.pt, ...
      - rtdetr-l.pt, rtdetr-x.pt, ...
      - a custom checkpoint: runs/detect/train/weights/best.pt
      - an architecture yaml: yolov8n.yaml, rtdetr-l.yaml, ...
    """
    model: str = "yolov8n.pt"

    # Device / optimization
    device: Optional[str] = "cuda:0"   # "cuda:0", "cpu", "mps", or None
    half: bool = False                # FP16 after loading (CUDA only)
    fuse: bool = False                # fuse Conv+BN for speed

    # Optional meta / default inference settings (used later when calling model(...))
    default_imgsz: int = 640
    default_conf: float = 0.25
    default_iou: float = 0.45
    classes: Optional[List[int]] = None  # e.g. [0] for person only

    task: Optional[str] = None  # "detect", "segment", "pose", "classify" (optional)

    def get_init_params(self) -> Dict[str, Any]:
        """
        Return parameters for detector initialization / post-processing.
        Note: Ultralytics.YOLO __init__ itself only uses `model`.
        Device, half, fuse, etc. are applied after creation.
        """
        return {
            "model": self.model,
            "device": self.device,
            "half": self.half,
            "fuse": self.fuse,
            "default_imgsz": self.default_imgsz,
            "default_conf": self.default_conf,
            "default_iou": self.default_iou,
            "classes": self.classes,
            "task": self.task,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UltralyticsDetectorConfig":
        return cls(**config_dict)

    @classmethod
    def from_ultralytics_model_yaml(cls, path: str | Path) -> "UltralyticsDetectorConfig":
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)

        if "config" not in raw:
            raise ValueError(f"Expected top-level 'config' key in {path}")

        cfg_block = raw["config"]
        model_init_name = cfg_block["model_init_name"]
        inf = cfg_block.get("inference_cfg", {})

        return cls(
            model=model_init_name,
            device=inf.get("device", "cuda"),
            half=inf.get("fp16", False),
            fuse=inf.get("fuse", False),
            default_imgsz=inf.get("imgsz", 640),
            default_conf=inf.get("conf", 0.25),
            default_iou=inf.get("iou", 0.45),
            classes=inf.get("classes"),
            task=inf.get("task", "detect"),
        )

    @classmethod
    def from_model_name(cls, model_name: str, config_root: Path) -> "UltralyticsDetectorConfig":
        """
        Generic factory: given a model_name (e.g. 'yolov12n') and config_root,
        choose the right YAML and build a config instance.
        """

        yaml_path = (
                config_root
                / "ultralytics"
                / "detector"
                / model_name
                / "config.yaml"
        )
        return cls.from_ultralytics_model_yaml(yaml_path)