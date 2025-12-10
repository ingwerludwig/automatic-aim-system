"""DeepSORT-specific tracker configuration (relative-path version)."""

import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

from src.tracker_config.base_tracker_config import (
    BaseTrackerConfig,
    TrackerConfigFactory,
)


@TrackerConfigFactory.register("torch_deepsort")
@dataclass
class TorchDeepsortConfig(BaseTrackerConfig):
    """
    Configuration for DeepSORT tracker (deep-sort-realtime).

    Automatically resolves CLIP embedder paths relative to project root.
    """

    # Core DeepSORT parameters
    max_age: int
    n_init: int
    max_iou_distance: float
    max_cosine_distance: float
    nms_max_overlap: float

    # Embedder / device
    embedder: str
    embedder_gpu: bool
    half: bool
    bgr: bool

    # Optional extras
    max_age_id: Optional[int]
    polygon: bool
    embedder_wts: Optional[str]

    def get_init_params(self) -> Dict[str, Any]:
        """Return kwargs for DeepSort(...) initialization."""
        params: Dict[str, Any] = {
            "max_age": self.max_age,
            "n_init": self.n_init,
            "max_iou_distance": self.max_iou_distance,
            "max_cosine_distance": self.max_cosine_distance,
            "nms_max_overlap": self.nms_max_overlap,
            "embedder": self.embedder,
            "embedder_gpu": self.embedder_gpu,
            "half": self.half,
            "bgr": self.bgr,
            "polygon": self.polygon,
        }

        if self.max_age_id is not None:
            params["max_age_id"] = self.max_age_id

        # Only pass embedder_wts if we actually resolved something
        if self.embedder_wts:
            params["embedder_wts"] = self.embedder_wts

        return params

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TorchDeepsortConfig":
        return cls(**config_dict)

    @classmethod
    def from_model_name(cls, model_name: str, config_root: Path) -> "TorchDeepsortConfig":
        """
        Resolve config and automatically locate embedder checkpoints:

            PROJECT_ROOT = config_root.parent
            CHECKPOINT_DIR = PROJECT_ROOT / "src" / "checkpoint" / "tracker"

        For model_name="deepsort", load:

            config_root/torch/tracker/deepsort/config.yaml
        """
        yaml_path = config_root / "torch" / "tracker" / "deepsort" / "config.yaml"
        raw = yaml.safe_load(yaml_path.read_text())

        if "config" not in raw:
            raise ValueError(f"Expected top-level 'config' in {yaml_path}")

        cfg_block = raw["config"]
        inf = cfg_block.get("inference_cfg", {})

        embedder = inf.get("embedder", "osnet_x0_25")

        # -----------------------------------------------------------
        # 1. Compute checkpoint directory relative to project root
        # -----------------------------------------------------------
        PROJECT_ROOT = config_root.parent          # e.g. /.../FinalICS
        CHECKPOINT_DIR = PROJECT_ROOT / "src" / "checkpoint" / "tracker"

        # -----------------------------------------------------------
        # 2. Resolve embedder_wts automatically for CLIP embedders
        # -----------------------------------------------------------
        embedder_wts: Optional[str] = None

        if embedder.startswith("clip_"):
            # Convert e.g. "clip_ViT-B/16" -> "clip_ViT-B-16.pt"
            ckpt_name = embedder.replace("/", "-") + ".pt"
            ckpt_path = CHECKPOINT_DIR / ckpt_name

            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"CLIP embedder '{embedder}' requires checkpoint:\n"
                    f"  {ckpt_path}\n"
                    f"But file was not found.\n"
                    f"Expected checkpoints inside:\n"
                    f"  {CHECKPOINT_DIR}\n"
                )
            else:
                print("Checkpoint found:", str(ckpt_path))

            embedder_wts = str(ckpt_path)
        # else: non-CLIP embedders (mobilenet, torchreid, osnet) – let DeepSort handle defaults

        # -----------------------------------------------------------
        # Build final config dataclass
        # -----------------------------------------------------------
        return cls(
            max_age=inf.get("max_age"),
            n_init=inf.get("n_init"),
            max_iou_distance=inf.get("max_iou_distance"),
            max_cosine_distance=inf.get("max_cosine_distance"),
            nms_max_overlap=inf.get("nms_max_overlap"),
            embedder=embedder,
            embedder_wts=embedder_wts,
            embedder_gpu=inf.get("embedder_gpu"),
            half=inf.get("half"),
            bgr=inf.get("bgr"),
            max_age_id=inf.get("max_age_id"),
            polygon=inf.get("polygon"),
        )