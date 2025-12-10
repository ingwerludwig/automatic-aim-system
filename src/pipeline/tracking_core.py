import cv2
import pickle
import zmq
from datetime import datetime

import torch
import time

from src.config_paths.model_paths import get_inference_config_path
from src.response.torch_detection_reponse import det_to_tracker_torch
from src.response.ultralytics_detection_response import det_to_tracker_ultralytics
from src.detector_config.ultralytics.config import UltralyticsDetectorConfig
from src.detector_config.torch.config import TorchDetectorConfig
from src.pipeline.builder.detector_builder import (
    load_global_inference_config as load_cfg_det,
    build_detector,
)
from src.pipeline.builder.tracker_builder import build_tracker


def create_zmq_publisher(port: int = 5002):
    """
    Create a ZMQ publisher socket for broadcasting tracking data.

    Args:
        port: Port to bind the publisher (default: 5002)

    Returns:
        zmq.Socket: Publisher socket
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    print(f"[TrackingPublisher] Broadcasting on port {port}")
    return socket


def build_detector_and_tracker(inference_cfg_name: str):
    """
    Build detector and tracker once from a named inference config.

    Returns:
        detector, det_cfg, tracker, trk_cfg
    """
    inference_cfg_path = get_inference_config_path(inference_cfg_name)
    global_cfg = load_cfg_det(inference_cfg_path)

    # NOTE:
    # Previously this forced "person only" for Ultralytics via classes=[0].
    # We keep it for backward-compatibility. Torch-based detectors will ignore
    # this and use their own `person_only` flag in TorchDetectorConfig.
    global_cfg.setdefault("detector", {})
    global_cfg["detector"].setdefault("classes", [0])

    detector, det_cfg = build_detector(global_cfg)
    tracker, trk_cfg = build_tracker(global_cfg)

    return detector, det_cfg, tracker, trk_cfg


def _run_ultralytics_detection(frame, detector, det_cfg: UltralyticsDetectorConfig):
    """
    Run Ultralytics (YOLO / RT-DETR) detection and convert to
    a UltralyticsDetectionsResponse for downstream trackers.
    """
    results_list = detector(
        frame,
        imgsz=getattr(det_cfg, "default_imgsz", 640),
        conf=getattr(det_cfg, "default_conf", 0.25),
        iou=getattr(det_cfg, "default_iou", 0.45),
        classes=getattr(det_cfg, "classes", None),
    )
    results = results_list[0]

    det_tracker_ultralytics = det_to_tracker_ultralytics(
        results,
        min_conf=getattr(det_cfg, "default_conf", 0.25),
    )
    return det_tracker_ultralytics


def _run_torch_detection(frame, detector, det_cfg: TorchDetectorConfig):
    """
    Run torchvision detector (e.g. Faster R-CNN ResNet50 FPN) and convert
    to a TorchDetectionsResponse for downstream trackers.
    """
    # frame: BGR uint8 HxWx3 (OpenCV)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To tensor [C,H,W], float32, 0-1
    img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

    device_str = det_cfg.device or "cuda"
    device = torch.device(device_str)
    img_tensor = img_tensor.to(device)

    detector.eval()
    with torch.no_grad():
        outputs = detector([img_tensor])[0]  # dict: boxes, labels, scores

    dets = det_to_tracker_torch(
        outputs,
        score_threshold=det_cfg.score_threshold,
        person_only=det_cfg.person_only,
    )
    return dets


def process_frame_with_tracking(
    frame,
    detector,
    det_cfg,
    tracker,
    tracked_objects=None,
    zmq_publisher=None,
    active_track_state=None,
):
    """
    Run detection (Ultralytics or Torch) + tracking (ByteTrack / BoTSORT / DeepSORT)
    on a single frame, draw the tracks on the frame (in-place), and return it.

    Args:
        frame: Input frame
        detector: Detector model
                 - Ultralytics YOLO / RT-DETR
                 - torchvision Faster R-CNN, etc.
        det_cfg: Detector configuration
                 - UltralyticsDetectorConfig or TorchDetectorConfig
        tracker: Tracker model loader (Ultralytics, DeepSORT, etc.)
                 - must expose .update(detections, img=frame)
        tracked_objects: Dictionary to maintain tracking history {track_id: metadata}
        zmq_publisher: ZMQ socket for publishing tracking data (optional)
        active_track_state: Dictionary to manage which track is currently being published
                            schema: {'current_active_id': int | None, 'all_track_ids': list[int]}
    """
    if tracked_objects is None:
        tracked_objects = {}
    if active_track_state is None:
        active_track_state = {"current_active_id": None, "all_track_ids": []}

    # 1) Run detection according to framework / config type
    if isinstance(det_cfg, UltralyticsDetectorConfig):
        det_tracker = _run_ultralytics_detection(frame, detector, det_cfg)
    elif isinstance(det_cfg, TorchDetectorConfig):
        det_tracker = _run_torch_detection(frame, detector, det_cfg)
    else:
        raise TypeError(
            f"Unsupported detector config type in tracking_core: {type(det_cfg)}"
        )

    # 2) Update tracker (ByteTrack / BoTSORT / DeepSORT) — works with both detection response types
    tracks = tracker.update(det_tracker, img=frame)

    # Collect current frame track data
    current_frame_tracks = {}
    current_track_ids = set()

    # 3) Process all tracks in current frame
    for t in tracks:
        # --- Support DeepSORT Track objects (deep-sort-realtime) ---
        # They usually have `.to_ltrb()` and `.track_id`
        if hasattr(t, "to_ltrb") and hasattr(t, "track_id"):
            # Optionally skip unconfirmed tracks, similar to ByteTrack
            if hasattr(t, "is_confirmed") and not t.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, t.to_ltrb())
            track_id = int(t.track_id)
        else:
            # --- Original ByteTrack / BoTSORT path (STrack or ndarray/list) ---
            x1, y1, x2, y2, track_id = t[:5]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            track_id = int(track_id)

        current_track_ids.add(track_id)

        # Calculate center point of bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Update tracking history
        if track_id not in tracked_objects:
            tracked_objects[track_id] = {
                "first_seen": datetime.now(),
                "count": 0,
                "last_position": (x1, y1, x2, y2),
                "last_center": (center_x, center_y),
                "center_history": [],
            }

        tracked_objects[track_id]["last_seen"] = datetime.now()
        tracked_objects[track_id]["count"] += 1
        tracked_objects[track_id]["last_position"] = (x1, y1, x2, y2)
        tracked_objects[track_id]["last_center"] = (center_x, center_y)
        tracked_objects[track_id]["center_history"].append((center_x, center_y))

        # Store current frame track data
        current_frame_tracks[track_id] = {
            "track_id": track_id,
            "center": (center_x, center_y),
            "bbox": (x1, y1, x2, y2),
            "timestamp": datetime.now().timestamp(),
        }

        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # 4) Publish tracking data in queue-based format
    if zmq_publisher is not None:
        # Add new track IDs to the queue
        for tid in current_track_ids:
            if tid not in active_track_state["all_track_ids"]:
                active_track_state["all_track_ids"].append(tid)

        # If no active track ID is set, use the first in queue
        if not active_track_state["all_track_ids"]:
            # No tracks at all - send empty array
            print("No tracked objects - sending empty array\n")
            zmq_publisher.send_multipart(
                [
                    b"bounding_box",
                    pickle.dumps([]),
                ]
            )
        else:
            # Get the first track ID in queue (index 0)
            first_track_id = active_track_state["all_track_ids"][0]

            # Check if first track is still present in current frame
            if first_track_id in current_frame_tracks:
                # Track is present - send its data
                tracking_data = [current_frame_tracks[first_track_id]]
                print(
                    f"Sending track ID {first_track_id} (queue position 0):\n"
                    f"{tracking_data}\n"
                )
                zmq_publisher.send_multipart(
                    [
                        b"bounding_box",
                        pickle.dumps(tracking_data),
                    ]
                )
            else:
                # Track disappeared - send empty array and remove from queue
                print(
                    f"Track ID {first_track_id} disappeared - "
                    f"sending empty array and removing from queue\n"
                )
                zmq_publisher.send_multipart(
                    [
                        b"bounding_box",
                        pickle.dumps([]),
                    ]
                )
                # Pop the first element (remove disappeared track)
                active_track_state["all_track_ids"].pop(0)
                print(f"Queue updated: {active_track_state['all_track_ids']}\n")
            time.sleep(0.02)

    return frame