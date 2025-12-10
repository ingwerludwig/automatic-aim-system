import base64
import pickle
import threading
import time
from typing import Optional, Callable

import cv2
import numpy as np
import zmq


class StreamReceiver:
    """
    ZeroMQ video stream receiver.

    - Connects to a PUB socket (Raspberry Pi or other sender)
    - Receives frames (JPEG+base64 or pickled numpy array)
    - Optionally calls a user-defined callback for each frame:
        callback(frame: np.ndarray, metadata: dict)
    - If callback is None, it will just display the raw frame with overlay.
    """

    def __init__(self, source_ip: str, source_port: int):
        self.source_ip = source_ip
        self.source_port = source_port
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None

        self.receiving: bool = False
        self.frame_callback: Optional[Callable] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count: int = 0

        self._receiving_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """Connect to the ZeroMQ publisher."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)

            connect_address = f"tcp://{self.source_ip}:{self.source_port}"
            self.socket.connect(connect_address)

            # Subscribe to topic "video_frame"
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "video_frame")

            # Set receive timeout (ms)
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)

            print(f"[StreamReceiver] Connected to {connect_address}")
            return True

        except Exception as e:
            print(f"[StreamReceiver] Failed to connect: {e}")
            return False

    def _decode_frame(self, message: dict) -> Optional[np.ndarray]:
        """Decode frame from message dict."""
        try:
            encoding = message.get("encoding", "base64_jpeg")
            data = message["data"]

            if encoding == "base64_jpeg":
                jpg_data = base64.b64decode(data)
                frame = cv2.imdecode(
                    np.frombuffer(jpg_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            elif encoding == "pickle":
                frame = pickle.loads(data)
            else:
                raise ValueError(f"Unknown encoding: {encoding}")

            return frame

        except Exception as e:
            print(f"[StreamReceiver] Error decoding frame: {e}")
            return None

    def start_receiving(self, frame_callback: Optional[Callable] = None):
        """
        Blocking loop that receives frames.

        If frame_callback is provided, it will be called for each frame:
            frame_callback(frame: np.ndarray, metadata: dict)

        If not provided, a default display window will be used.
        """
        if self.socket is None:
            raise RuntimeError("Call connect() before start_receiving()")

        self.frame_callback = frame_callback
        self.receiving = True
        self.frame_count = 0

        print("[StreamReceiver] Start receiving frames. Press 'q' in window to stop.")

        try:
            while self.receiving:
                try:
                    # Expect 2-part message: topic, payload
                    topic, message_data = self.socket.recv_multipart()
                    message = pickle.loads(message_data)

                    frame = self._decode_frame(message)
                    if frame is None:
                        continue

                    self.current_frame = frame
                    self.frame_count += 1

                    if self.frame_callback is not None:
                        # Your pipeline handles drawing & imshow
                        self.frame_callback(frame, message)
                    else:
                        # Default simple viewer
                        self._display_frame(frame, message)

                except zmq.Again:
                    # Timeout -> no frame in this period
                    # Just continue looping
                    continue
                except Exception as e:
                    print(f"[StreamReceiver] Error receiving frame: {e}")
                    continue

        except KeyboardInterrupt:
            print("\n[StreamReceiver] Interrupted by user.")
        finally:
            self.stop_receiving()

    def _display_frame(self, frame: np.ndarray, message: dict):
        """Default viewer if no callback is provided."""
        display = frame.copy()

        frame_id = message.get("frame_id", 0)
        fps = message.get("fps", 0.0)
        ts = message.get("timestamp", time.time())
        latency_ms = (time.time() - ts) * 1000.0

        cv2.putText(display, f"Frame: {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Latency: {latency_ms:.1f} ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("ZeroMQ Stream Receiver", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            self.stop_receiving()

    def start_receiving_threaded(self, frame_callback: Optional[Callable] = None) -> threading.Thread:
        """Start receiving in a separate thread."""
        self._receiving_thread = threading.Thread(
            target=self.start_receiving,
            args=(frame_callback,),
            daemon=True,
        )
        self._receiving_thread.start()
        return self._receiving_thread

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame (may be None)."""
        return self.current_frame

    def stop_receiving(self):
        """Stop receiving and clean up resources."""
        if not self.receiving:
            return

        print("[StreamReceiver] Stopping receiver...")
        self.receiving = False

        # Do not join if we're in the same thread
        if self._receiving_thread is not None and threading.current_thread() != self._receiving_thread:
            self._receiving_thread.join(timeout=2.0)

        if self.socket is not None:
            self.socket.close()
            self.socket = None

        if self.context is not None:
            self.context.term()
            self.context = None

        cv2.destroyAllWindows()
        print("[StreamReceiver] Stopped.")