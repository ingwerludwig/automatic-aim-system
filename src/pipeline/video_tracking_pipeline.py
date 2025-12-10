import cv2

from src.pipeline.tracking_core import (
    build_detector_and_tracker,
    create_zmq_publisher,
    process_frame_with_tracking,
)


def run_video_tracking(
    video_path: str,
    inference_cfg_name: str,
    publish_port: int = 5002,
):
    """
    Pipeline: read frames from a local video file, run YOLO + ByteTrack,
    visualize results, and publish selected tracking data via ZeroMQ.

    Args:
        video_path: Path to video file
        inference_cfg_name: Name of inference config
        publish_port: ZMQ port for publishing tracking data (default: 5002)
    """
    # Build detector & tracker once
    detector, det_cfg, tracker, trk_cfg = build_detector_and_tracker(
        inference_cfg_name
    )

    # Initialize tracking state
    tracked_objects = {}
    active_track_state = {"current_active_id": None, "all_track_ids": []}

    # Initialize ZMQ publisher for tracking data
    zmq_publisher = create_zmq_publisher(publish_port)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_path}")

    cv2.namedWindow("Tracking (Video)", cv2.WINDOW_NORMAL)

    # Set the window to full screen
    cv2.setWindowProperty("Tracking (Video)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame_with_tracking(
                frame,
                detector,
                det_cfg,
                tracker,
                tracked_objects=tracked_objects,
                zmq_publisher=zmq_publisher,
                active_track_state=active_track_state,
            )

            cv2.imshow("Tracking (Video)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        zmq_publisher.close()

    # Print tracking summary
    print("\n=== Tracking Summary ===")
    print(f"Total unique objects tracked: {len(tracked_objects)}")
    for track_id, info in tracked_objects.items():
        duration = (info["last_seen"] - info["first_seen"]).total_seconds()
        print(f"ID {track_id}: {info['count']} frames, {duration:.2f}s duration")