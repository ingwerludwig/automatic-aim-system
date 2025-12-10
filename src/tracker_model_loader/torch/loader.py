"""DeepSORT-specific tracker model loader implementation."""

from __future__ import annotations

from typing import Any, Optional, List, Tuple

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.tracker_config.torch.config import TorchDeepsortConfig
from src.tracker_model_loader.tracker_model_loader_registry import (
    tracker_model_loader_registry,
)
from src.tracker_model_loader.base_tracker_model_loader import (
    BaseTrackerModelLoader,
)


@tracker_model_loader_registry.register(
    "torch",
    description="Tracker model loader for DeepSORT (deep-sort-realtime)",
)
class TorchDeepsortTrackerModelLoader(BaseTrackerModelLoader):
    """
    Model loader for DeepSORT (using deep-sort-realtime library).

    Given a TorchDeepsortConfig, this loader instantiates a DeepSort
    object and exposes a unified `update(...)` API compatible with
    tracking_core.process_frame_with_tracking.
    """

    def __init__(self, config: TorchDeepsortConfig) -> None:
        super().__init__(config)
        self._tracker: Optional[DeepSort] = None

    def load_model(self) -> DeepSort:
        """
        Instantiate and return a DeepSort tracker instance.
        """
        if not isinstance(self.config, TorchDeepsortConfig):
            raise TypeError(
                f"DeepSortTrackerModelLoader expected TorchDeepsortConfig, "
                f"got {type(self.config)}"
            )

        params = self.config.get_init_params()
        # If you ever need to adapt params, do it here
        self._tracker = DeepSort(**params)
        return self._tracker

    def _results_to_detections(
        self, results: Any
    ) -> List[Tuple[List[float], float, int]]:
        """
        Convert our detection wrapper (`det_to_tracker_*` output) into
        DeepSORT's expected detection format:

            ([x, y, w, h], confidence, class_id)

        Supports:
          - objects with .xyxy, .conf, .cls (numpy or torch-like)
          - numpy arrays shaped (N, 6) -> [x1, y1, x2, y2, score, cls]
          - list/tuple of the above
        """
        detections: List[Tuple[List[float], float, int]] = []

        # Case 1: object with xyxy/ conf / cls (like our ByteTrack wrapper)
        if (
            hasattr(results, "xyxy")
            and hasattr(results, "conf")
            and hasattr(results, "cls")
        ):
            xyxy = results.xyxy
            confs = results.conf
            clss = results.cls

            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
            if hasattr(confs, "cpu"):
                confs = confs.cpu().numpy()
            if hasattr(clss, "cpu"):
                clss = clss.cpu().numpy()

            xyxy = np.asarray(xyxy)
            confs = np.asarray(confs)
            clss = np.asarray(clss)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                w = x2 - x1
                h = y2 - y1
                detections.append(
                    ([float(x1), float(y1), float(w), float(h)],
                     float(confs[i]),
                     int(clss[i]))
                )
            return detections

        # Case 2: numpy array (N, 6) or list/tuple of such rows
        arr = None
        if isinstance(results, np.ndarray):
            arr = results
        elif isinstance(results, (list, tuple)) and len(results) > 0:
            # Possibly list of [x1, y1, x2, y2, score, cls]
            arr = np.asarray(results)

        if arr is not None:
            # Expect shape (N, >=6)
            for row in arr:
                if len(row) < 6:
                    continue
                x1, y1, x2, y2, score, cls = row[:6]
                w = x2 - x1
                h = y2 - y1
                detections.append(
                    ([float(x1), float(y1), float(w), float(h)],
                     float(score),
                     int(cls))
                )

            return detections

        raise TypeError(
            f"DeepSortTrackerModelLoader.update received unsupported "
            f"detections format: {type(results)}"
        )

    def update(self, results: Any, img: Any | None = None) -> Any:
        """
        Update the underlying DeepSORT tracker with new detections.

        Args:
            results:
                Output from det_to_tracker_ultralytics / det_to_tracker_torch
                or similar (must be convertible by _results_to_detections).
            img:
                Original frame (H x W x 3 ndarray, BGR).
                Required by DeepSort for embedding extraction.

        Returns:
            List of Track objects from DeepSort.update_tracks(...)
        """
        if self._tracker is None:
            self.load_model()

        detections = self._results_to_detections(results)
        tracks = self._tracker.update_tracks(detections, frame=img)
        return tracks

    @classmethod
    def get_name(cls) -> str:
        """Return the name of this loader."""
        return "deepsort"