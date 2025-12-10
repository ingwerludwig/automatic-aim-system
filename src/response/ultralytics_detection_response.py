# src/response/ultralytics_detection_response.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class UltralyticsDetectionsResponse:
    xyxy: np.ndarray      # [N, 4]
    xywh: np.ndarray      # [N, 4]
    conf: np.ndarray      # [N]
    cls: np.ndarray       # [N]

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx: Any) -> "UltralyticsDetectionsResponse":
        """
        Support indexing with:
          - boolean mask
          - slice
          - integer / list of integers
        """
        return UltralyticsDetectionsResponse(
            xyxy=self.xyxy[idx],
            xywh=self.xywh[idx],
            conf=self.conf[idx],
            cls=self.cls[idx],
        )


def det_to_tracker_ultralytics(results, min_conf: float = 0.3) -> UltralyticsDetectionsResponse:
    """
    Convert Ultralytics Results or Boxes to an adapter object
    that BYTETracker.update() can consume.
    """

    # Accept either Results or Boxes
    if hasattr(results, "boxes"):
        boxes = results.boxes
    else:
        boxes = results  # assume already Boxes-like

    # Move tensors to CPU and convert to numpy
    xyxy = boxes.xyxy.cpu().numpy()
    xywh = boxes.xywh.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls  = boxes.cls.cpu().numpy()

    # Filter by confidence
    keep = conf >= min_conf
    xyxy = xyxy[keep]
    xywh = xywh[keep]
    conf = conf[keep]
    cls  = cls[keep]

    return UltralyticsDetectionsResponse(
        xyxy=xyxy,
        xywh=xywh,
        conf=conf,
        cls=cls,
    )