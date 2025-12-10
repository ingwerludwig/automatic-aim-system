"""Ultralytics-specific tracker model loader implementation."""
from __future__ import annotations

from typing import Any, Optional
from types import SimpleNamespace

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT

from src.tracker_config.ultralytics.config import (
    UltralyticsBytetrackConfig,
    UltralyticsBotsortConfig,
)
from src.tracker_model_loader.tracker_model_loader_registry import (
    tracker_model_loader_registry,
)
from src.tracker_model_loader.base_tracker_model_loader import BaseTrackerModelLoader


@tracker_model_loader_registry.register(
    "ultralytics",
    description="Tracker model loader for Ultralytics (ByteTrack / BoTSORT)",
)
class UltralyticsTrackerModelLoader(BaseTrackerModelLoader):
    """
    Ultralytics implementation of tracker model loader.

    Given a BaseTrackerConfig (ByteTrack or BoTSORT), this loader
    instantiates the appropriate Ultralytics tracker object and exposes
    a unified `update(...)` API.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._tracker: Optional[Any] = None  # BYTETracker or BOTSORT

    def load_model(self) -> Any:
        """
        Instantiate and return an Ultralytics tracker instance.

        Returns:
            An instance of BYTETracker or BOTSORT depending on config type.
        """
        cfg = self.config

        # ByteTrack
        if isinstance(cfg, UltralyticsBytetrackConfig):
            params = cfg.get_init_params()
            # Separate frame_rate from the rest
            frame_rate = params.pop("frame_rate", 30)
            args = SimpleNamespace(**params)

            self._tracker = BYTETracker(args=args, frame_rate=frame_rate)
            return self._tracker

        # BoTSORT
        if isinstance(cfg, UltralyticsBotsortConfig):
            params = cfg.get_init_params()
            frame_rate = params.pop("frame_rate", 30)
            args = SimpleNamespace(**params)

            self._tracker = BOTSORT(args=args, frame_rate=frame_rate)
            return self._tracker

        raise TypeError(
            f"UltralyticsTrackerModelLoader received unsupported config type: {type(cfg)}"
        )

    def update(self, results: Any, img: Any | None = None) -> Any:
        """
        Update the underlying Ultralytics tracker with new detections.

        This is a thin wrapper around `BYTETracker.update(...)` /
        `BOTSORT.update(...)` from Ultralytics.

        Args:
            results:
                - Typically the YOLO output for a single frame, e.g.:
                  * numpy array of detections [N, 6] (x1, y1, x2, y2, score, cls), or
                  * an Ultralytics `Results`-like structure, depending on how
                    you prepare detections for the tracker.
            img:
                - Original frame (H x W x 3 ndarray). Required for BoTSORT
                  when using ReID / GMC; optional for plain BYTETracker.

        Returns:
            Whatever the underlying tracker `.update(...)` returns
            (usually a list of active tracks / STrack objects).
        """
        # Lazy construction if `load_model()` was not explicitly called
        if self._tracker is None:
            self.load_model()

        # Both Ultralytics BYTETracker and BOTSORT have:
        #   update(self, results, img=None)
        # Convert tensors to numpy if needed (may already be numpy from det_to_tracker_ultralytics)
        if hasattr(results.xyxy, 'cpu'):
            results.xyxy = results.xyxy.cpu().numpy()
        if hasattr(results.xywh, 'cpu'):
            results.xywh = results.xywh.cpu().numpy()
        if hasattr(results.conf, 'cpu'):
            results.conf = results.conf.cpu().numpy()
        if hasattr(results.cls, 'cpu'):
            results.cls = results.cls.cpu().numpy()
        return self._tracker.update(results, img)

    @classmethod
    def get_name(cls) -> str:
        """Return the name of this loader."""
        return "ultralytics"
