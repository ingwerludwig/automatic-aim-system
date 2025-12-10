import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pipeline.video_tracking_pipeline import run_video_tracking
from src.pipeline.stream_tracking_pipeline import run_stream_tracking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO + ByteTrack pipeline")

    parser.add_argument(
        "--inference_cfg_name",
        required=True,
        help="Name of inference config (used by get_inference_config_path)",
    )

    # Option A: local video file
    parser.add_argument(
        "--video_path",
        help="Path to local video file (if using file-based pipeline)",
    )

    # Option B: ZeroMQ stream (Raspberry Pi, etc.)
    parser.add_argument(
        "--stream_ip",
        help="Source IP address for ZeroMQ video stream",
    )
    parser.add_argument(
        "--stream_port",
        type=int,
        help="Source port for ZeroMQ video stream",
    )

    # ZMQ publisher for tracking data
    parser.add_argument(
        "--publish_port",
        type=int,
        default=5002,
        help="Port for publishing tracking data via ZMQ (default: 5002)",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Prefer stream if both are given
    if args.stream_ip and args.stream_port:
        print(
            f"[Main] Running stream tracking from {args.stream_ip}:{args.stream_port} "
            f"with inference_cfg_name='{args.inference_cfg_name}'"
        )
        print(f"[Main] Publishing tracking data on port {args.publish_port}")
        run_stream_tracking(
            source_ip=args.stream_ip,
            source_port=args.stream_port,
            inference_cfg_name=args.inference_cfg_name,
            publish_port=args.publish_port,
        )
    elif args.video_path:
        print(
            f"[Main] Running video tracking from '{args.video_path}' "
            f"with inference_cfg_name='{args.inference_cfg_name}'"
        )
        print(f"[Main] Publishing tracking data on port {args.publish_port}")
        run_video_tracking(
            video_path=args.video_path,
            inference_cfg_name=args.inference_cfg_name,
            publish_port=args.publish_port,
        )
    else:
        raise SystemExit(
            "You must provide either:\n"
            "  --video_path /path/to/file.mp4\n"
            "  OR\n"
            "  --stream_ip 1.2.3.4 --stream_port 5001"
        )


if __name__ == "__main__":
    main()