"""Tracker-specific configuration classes."""

import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

from src.tracker_config.base_tracker_config import BaseTrackerConfig, TrackerConfigFactory


# =========================
# ByteTrack
# =========================
@TrackerConfigFactory.register("ultralytics_bytetrack")
@dataclass
class UltralyticsBytetrackConfig(BaseTrackerConfig):
    """
    Configuration for ByteTrack tracker (Ultralytics).

    Fields here are aligned with ultralytics.trackers.byte_tracker.BYTETracker
    and the keys used in your YAML under `inference_cfg`.
    """
    track_buffer: int = 30
    track_thresh: float = 0.5
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    match_thresh: float = 0.8
    fuse_score: bool = True
    mot20: bool = False

    # extra meta
    frame_rate: int = 30
    proximity_thresh: Optional[float] = None
    appearance_thresh: Optional[float] = None

    def get_init_params(self) -> Dict[str, Any]:
        """Return parameters for ByteTrack initialization (become `args` for BYTETracker)."""
        params = {
            "track_buffer": self.track_buffer,
            "track_thresh": self.track_thresh,
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "match_thresh": self.match_thresh,
            "fuse_score": self.fuse_score,
            "mot20": self.mot20,
            "frame_rate": self.frame_rate,
        }
        # optional extras if you ever want them
        if self.proximity_thresh is not None:
            params["proximity_thresh"] = self.proximity_thresh
        if self.appearance_thresh is not None:
            params["appearance_thresh"] = self.appearance_thresh
        return params

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UltralyticsBytetrackConfig":
        return cls(**config_dict)

    @classmethod
    def from_model_name(cls, model_name: str, config_root: Path) -> "UltralyticsBytetrackConfig":
        """
        For model_name like 'bytetracker', always load:

            <config_root>/ultralytics/tracker/bytetrack/config.yaml
        """
        yaml_path = config_root / "ultralytics" / "tracker" / "bytetrack" / "config.yaml"
        raw = yaml.safe_load(yaml_path.read_text())

        if "config" not in raw:
            raise ValueError(f"Expected top-level 'config' key in {yaml_path}")

        cfg_block = raw["config"]
        inf = cfg_block.get("inference_cfg", {})

        return cls(
            track_buffer=inf.get("track_buffer", 30),
            track_thresh=inf.get("track_thresh", 0.5),
            track_high_thresh=inf.get("track_high_thresh", 0.6),
            track_low_thresh=inf.get("track_low_thresh", 0.1),
            new_track_thresh=inf.get("new_track_thresh", 0.7),
            match_thresh=inf.get("match_thresh", 0.8),
            fuse_score=inf.get("fuse_score", True),
            mot20=inf.get("mot20", False),
            frame_rate=30,  # or derive from somewhere else if needed
        )


# =========================
# BoTSORT
# =========================
@TrackerConfigFactory.register("ultralytics_botsort")
@dataclass
class UltralyticsBotsortConfig(BaseTrackerConfig):
    """
    Configuration for BoTSORT tracker (Ultralytics).

    Again, aligned with the args used in ultralytics.trackers.bot_sort.BOTSORT.
    """
    track_buffer: int = 30
    track_thresh: float = 0.5
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    match_thresh: float = 0.8
    fuse_score: bool = True
    mot20: bool = False

    cmc_method: str = "file"
    frame_rate: int = 30

    def get_init_params(self) -> Dict[str, Any]:
        """Return parameters for BoTSORT initialization."""
        return {
            "track_buffer": self.track_buffer,
            "track_thresh": self.track_thresh,
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "match_thresh": self.match_thresh,
            "fuse_score": self.fuse_score,
            "mot20": self.mot20,
            "cmc_method": self.cmc_method,
            "frame_rate": self.frame_rate,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UltralyticsBotsortConfig":
        return cls(**config_dict)

    @classmethod
    def from_model_name(cls, model_name: str, config_root: Path) -> "UltralyticsBotsortConfig":
        """
        For model_name like 'botsort', always load:

            <config_root>/ultralytics/tracker/botsort/config.yaml
        """
        yaml_path = config_root / "ultralytics" / "tracker" / "botsort" / "config.yaml"
        raw = yaml.safe_load(yaml_path.read_text())

        if "config" not in raw:
            raise ValueError(f"Expected top-level 'config' key in {yaml_path}")

        cfg_block = raw["config"]
        inf = cfg_block.get("inference_cfg", {})

        return cls(
            track_buffer=inf.get("track_buffer", 30),
            track_thresh=inf.get("track_thresh", 0.5),
            track_high_thresh=inf.get("track_high_thresh", 0.6),
            track_low_thresh=inf.get("track_low_thresh", 0.1),
            new_track_thresh=inf.get("new_track_thresh", 0.7),
            match_thresh=inf.get("match_thresh", 0.8),
            fuse_score=inf.get("fuse_score", True),
            mot20=inf.get("mot20", False),
            cmc_method="file",
            frame_rate=30,
        )