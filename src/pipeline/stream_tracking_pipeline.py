import cv2

from src.hardware.zmq_receiver import StreamReceiver
from src.pipeline.tracking_core import (
    build_detector_and_tracker,
    create_zmq_publisher,
    process_frame_with_tracking,
)


def run_stream_tracking(
    source_ip: str,
    source_port: int,
    inference_cfg_name: str,
    publish_port: int = 5002,
):
    """
    Pipeline: receive frames from ZeroMQ stream (e.g. Raspberry Pi),
    run YOLO + ByteTrack per frame, visualize, and publish tracking data.

    Args:
        source_ip: IP address of the video stream source
        source_port: Port of the video stream source
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

    # Connect to stream
    receiver = StreamReceiver(source_ip, source_port)
    if not receiver.connect():
        raise RuntimeError("Failed to connect to ZeroMQ stream")

    def frame_callback(frame, metadata):
        """
        Called for each incoming frame from the stream.
        We run detection + tracking and then display the result.
        """
        processed = process_frame_with_tracking(
            frame,
            detector,
            det_cfg,
            tracker,
            tracked_objects=tracked_objects,
            zmq_publisher=zmq_publisher,
            active_track_state=active_track_state,
        )

        # Optional overlays from metadata
        frame_id = metadata.get("frame_id", 0)
        fps = metadata.get("fps", 0.0)

        cv2.putText(
            processed,
            f"Frame: {frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            processed,
            f"Stream FPS: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            processed,
            f"Tracked IDs: {len(tracked_objects)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Tracking (Stream)", processed)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            receiver.stop_receiving()

    try:
        # Blocking receive loop
        receiver.start_receiving(frame_callback=frame_callback)
    finally:
        zmq_publisher.close()

    # Print tracking summary after stream ends
    print("\n=== Tracking Summary ===")
    print(f"Total unique objects tracked: {len(tracked_objects)}")
    for track_id, info in tracked_objects.items():
        duration = (info["last_seen"] - info["first_seen"]).total_seconds()
        print(f"ID {track_id}: {info['count']} frames, {duration:.2f}s duration")