# Video Object Detection and Tracking Pipeline

**Charlie Team** - Intelligent Control System Course Project

## Authors
- Ida Bagus Kade Rainata Putra - M11312806
- Ingwer Ludwig Nommensen - M11302839
- Tzu-Chien Joey Tseng - M11412009

## Description

A modular pipeline for real-time object detection and tracking in video streams. The system supports multiple detection backends (Ultralytics and Torchvision) and tracking algorithms (ByteTrack and DeepSORT), with flexible configuration and ZeroMQ-based stream processing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Extending the Pipeline](#extending-the-pipeline)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a configurable video processing pipeline that combines object detection and tracking capabilities. The architecture uses a factory pattern with dynamic registration, enabling easy integration of new detectors and trackers without modifying core pipeline logic.

## Features

### Supported Detectors
- Ultralytics models: YOLO (v8, v11, v12)
- Torchvision models: Faster R-CNN

### Supported Trackers
- ByteTrack: Fast, simple online multi-object tracking
- DeepSORT: Deep learning-based multi-object tracking with re-identification

### Input Sources
- Local video files (MP4, AVI, etc.)
- Live ZeroMQ video streams

### Output
- Real-time tracking data published via ZeroMQ
- Configurable inference parameters per detector/tracker combination

## Prerequisites

- Python 3.8 or higher
- Conda or Miniconda (recommended)
- CUDA-capable GPU (optional, recommended for real-time performance)
- CUDA Toolkit 11.x or 12.x (if using GPU acceleration)

## Installation

### 1. Create and Activate Environment

```bash
conda create -n tracking python=3.8
conda activate tracking
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- ultralytics: YOLO models and built-in trackers
- torch, torchvision: PyTorch framework and Torchvision models
- deep-sort-realtime: DeepSORT implementation
- opencv-python: Video I/O and processing
- numpy: Numerical operations
- zmq: ZeroMQ messaging
- tensorboard: Training and inference logging
- torchreid, openai-clip: Re-identification feature extractors
- gdown: Pretrained model download utility

### 3. Model Weights

Ultralytics models are downloaded automatically on first use. For Torchvision models or custom DeepSORT embedders, place checkpoint files in:
```
src/checkpoint/detector/  # Torchvision model weights
src/checkpoint/tracker/   # DeepSORT embedder weights
```

## Usage

The pipeline is invoked through `invoke_pipeline.py` with two operational modes:

### Mode 1: Video File Processing

Process a local video file:

```bash
python invoke_pipeline.py \
    --inference_cfg_name yolov12n_bytetrack \
    --video_path data/sample_video.mp4 \
    --publish_port 5002
```

### Mode 2: Live Stream Processing

Process a ZeroMQ video stream:

```bash
python invoke_pipeline.py \
    --inference_cfg_name fasterrcnn_deepsort \
    --stream_ip 192.168.1.100 \
    --stream_port 5001 \
    --publish_port 5002
```

### Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--inference_cfg_name` | Yes | Configuration file name from `config/inference/` (without .yaml extension) |
| `--video_path` | Conditional | Path to video file (required if not using stream mode) |
| `--stream_ip` | Conditional | Source IP for ZeroMQ stream (required for stream mode) |
| `--stream_port` | Conditional | Source port for ZeroMQ stream (required for stream mode) |
| `--publish_port` | No | Port for publishing tracking results (default: 5002) |

### Available Configurations

Pre-configured inference setups in `config/inference/`:
- `yolov12n_bytetrack.yaml`: YOLOv12 nano with ByteTrack
- `yolov12n_deepsort.yaml`: YOLOv12 nano with DeepSORT
- `fasterrcnn_bytetrack.yaml`: Faster R-CNN with ByteTrack
- `fasterrcnn_deepsort.yaml`: Faster R-CNN with DeepSORT

## Configuration

The configuration system uses a three-tier hierarchy:

```
config/
├── inference/           # Top-level configs (detector and tracker pairs)
├── torch/               # Torchvision-specific configs
│   ├── detector/        # Faster R-CNN, RetinaNet, FCOS configs
│   └── tracker/         # DeepSORT configs
└── ultralytics/         # Ultralytics-specific configs
    ├── detector/        # YOLO\
    └── tracker/         # ByteTrack, BoTSORT configs
```

### Configuration Hierarchy

1. Inference configs reference specific detector and tracker configs
2. Detector configs specify model architecture, weights, and inference parameters
3. Tracker configs specify tracking algorithm and hyperparameters

### Example Inference Config

```yaml
detector_cfg_name: yolov12n
tracker_cfg_name: bytetrack
```

### Creating Custom Configurations

1. Create detector config in `config/torch/detector/` or `config/ultralytics/detector/`
2. Create tracker config in `config/torch/tracker/` or `config/ultralytics/tracker/`
3. Create inference config in `config/inference/` that references both
4. Run pipeline with `--inference_cfg_name your_config_name`

## Project Structure

```
src/
├── config_paths/            # Configuration file path utilities
├── detector_config/         # Detector configuration abstractions
├── detector_model_loader/   # Detector instantiation logic
├── tracker_config/          # Tracker configuration abstractions
├── tracker_model_loader/    # Tracker instantiation logic
├── pipeline/                # Core pipeline implementations
│   ├── builder/             # Detector and tracker factory builders
│   ├── video_tracking_pipeline.py
│   ├── stream_tracking_pipeline.py
│   └── tracking_core.py
├── bbox_converter/          # Bounding box format conversions
├── response/                # Detection output standardization
├── registry/                # Dynamic component registration system
├── checkpoint/              # Model weight storage
└── hardware/                # Hardware-specific utilities
```

### Architecture Overview

The pipeline implements a layered architecture with clear separation of concerns:

1. Configuration Layer: YAML-based configs parsed into dataclass objects
2. Registry Layer: Automatic discovery and registration of detector/tracker implementations
3. Builder Layer: Factory pattern for instantiating detectors and trackers from configs
4. Pipeline Layer: Orchestrates video I/O, detection, tracking, and result publishing
5. Response Layer: Standardizes detection outputs across different frameworks

## Extending the Pipeline

### Adding New Detectors or Trackers

Each component (detector/tracker) requires implementation of:
- Configuration class (in `src/detector_config/` or `src/tracker_config/`)
- Model loader class (in `src/detector_model_loader/` or `src/tracker_model_loader/`)

Detailed extension guides:
- Detector configurations: [src/detector_config/README.md](src/detector_config/README.md)
- Detector loaders: [src/detector_model_loader/README.md](src/detector_model_loader/README.md)
- Tracker configurations: [src/tracker_config/README.md](src/tracker_config/README.md)
- Tracker loaders: [src/tracker_model_loader/README.md](src/tracker_model_loader/README.md)

## Troubleshooting

### CUDA Out of Memory Errors

Reduce memory usage by:
- Using smaller model variants (e.g., yolov12n instead of yolov12x)
- Lowering input resolution in detector config
- Switching to CPU inference by setting `device: cpu` in config files
- Enabling FP16 inference where supported

### Module Import Errors

The project root is automatically added to Python path in `invoke_pipeline.py`. If running individual modules, manually add:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/FinalICS"
```

### Configuration Not Found

Ensure config file names match exactly (case-sensitive):
- Config files are in `config/inference/`
- Use filename without `.yaml` extension for `--inference_cfg_name`
- Example: `yolov12n_bytetrack` not `yolov12n_bytetrack.yaml`

### Model Download Failures

Ultralytics models download automatically. For manual downloads:
- Torchvision weights: Place in `src/checkpoint/detector/`
- DeepSORT embedder weights: Place in `src/checkpoint/tracker/`
- Use `gdown` utility for Google Drive hosted models

### ZeroMQ Connection Issues

Check:
- Firewall settings allow connections on specified ports
- IP addresses and ports match between publisher and subscriber
- ZeroMQ is properly installed (`pip install zmq`)

## License

This project uses multiple open-source libraries. Refer to individual licenses:
- Ultralytics: AGPL-3.0
- PyTorch: BSD-style license
- deep-sort-realtime: MIT License
- OpenCV: Apache 2.0 License
