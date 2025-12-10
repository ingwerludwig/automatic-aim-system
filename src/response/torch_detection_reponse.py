# src/response/torch_detection_response.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


@dataclass
class TorchDetectionsResponse:
    """
    Detection response container for torchvision-based detectors.

    Fields are intentionally shaped like the Ultralytics variant:
      - xyxy: [N, 4]  (x1, y1, x2, y2)
      - xywh: [N, 4]  (cx, cy, w, h)
      - conf: [N]
      - cls:  [N]
    """
    xyxy: np.ndarray      # [N, 4]
    xywh: np.ndarray      # [N, 4]
    conf: np.ndarray      # [N]
    cls: np.ndarray       # [N]

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx: Any) -> "TorchDetectionsResponse":
        """
        Support indexing with:
          - boolean mask
          - slice
          - integer / list of integers
        """
        return TorchDetectionsResponse(
            xyxy=self.xyxy[idx],
            xywh=self.xywh[idx],
            conf=self.conf[idx],
            cls=self.cls[idx],
        )


def det_to_tracker_torch(
    outputs: Dict[str, torch.Tensor],
    score_threshold: float = 0.5,
    person_only: bool = True,
) -> TorchDetectionsResponse:
    """
    Convert torchvision detection output dict into a TorchDetectionsResponse.

    Args:
        outputs: dict with keys 'boxes', 'scores', 'labels' (torch.Tensors)
        score_threshold: minimum confidence to keep a detection
        person_only: if True, keep only COCO 'person' class (label == 1)

    Returns:
        TorchDetectionsResponse with fields:
            xyxy: [N, 4]
            xywh: [N, 4]  (cx, cy, w, h)
            conf: [N]
            cls:  [N]
        If nothing remains after filtering, all arrays will have length 0.
    """
    boxes = outputs.get("boxes", None)
    scores = outputs.get("scores", None)
    labels = outputs.get("labels", None)

    # If outputs are missing anything, just return an empty response
    if boxes is None or scores is None or labels is None:
        empty_xy = np.zeros((0, 4), dtype=np.float32)
        empty_conf = np.zeros((0,), dtype=np.float32)
        empty_cls = np.zeros((0,), dtype=np.float32)
        return TorchDetectionsResponse(
            xyxy=empty_xy,
            xywh=empty_xy.copy(),
            conf=empty_conf,
            cls=empty_cls,
        )

    # Move to CPU & numpy
    boxes_np = boxes.detach().cpu().numpy()   # [N, 4] x1,y1,x2,y2
    scores_np = scores.detach().cpu().numpy() # [N]
    labels_np = labels.detach().cpu().numpy() # [N]

    # Confidence filter
    mask = scores_np >= score_threshold

    # Person-only filter (COCO: person = class id 1)
    if person_only:
        mask = mask & (labels_np == 1)

    if mask.sum() == 0:
        empty_xy = np.zeros((0, 4), dtype=np.float32)
        empty_conf = np.zeros((0,), dtype=np.float32)
        empty_cls = np.zeros((0,), dtype=np.float32)
        return TorchDetectionsResponse(
            xyxy=empty_xy,
            xywh=empty_xy.copy(),
            conf=empty_conf,
            cls=empty_cls,
        )

    xyxy = boxes_np[mask].astype(np.float32)       # [M, 4]
    conf = scores_np[mask].astype(np.float32)      # [M]
    cls  = labels_np[mask].astype(np.float32)      # [M]

    # Build xywh as (cx, cy, w, h)
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    return TorchDetectionsResponse(
        xyxy=xyxy,
        xywh=xywh,
        conf=conf,
        cls=cls,
    )