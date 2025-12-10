# Tracker Model Loader Module

This module provides a flexible factory pattern for loading tracker models from different frameworks and algorithms. It works in tandem with the tracker_config module to instantiate trackers based on configuration objects.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Available Loaders](#available-loaders)
- [Usage Examples](#usage-examples)
- [Adding New Loaders](#adding-new-loaders)

## Overview

The tracker model loader module:
- Loads tracking models/algorithms based on configuration objects
- Provides a unified interface across different frameworks and algorithms
- Uses registry pattern for dynamic loader registration
- Handles tracker initialization and parameter setup
- Exposes a common `update()` API for tracking operations

## Architecture

### Core Components

```
tracker_model_loader/
├── base_tracker_model_loader.py      # Abstract base class
├── tracker_model_loader_registry.py  # Registry implementation
├── __init__.py                        # Factory functions & exports
├── torch/                             # Torch-based trackers
│   └── loader.py                      # TorchDeepsortTrackerModelLoader
└── ultralytics/                       # Ultralytics trackers
    └── loader.py                      # UltralyticsTrackerModelLoader
```

### Key Classes

1. **BaseTrackerModelLoader**: Abstract base class defining the interface all loaders must implement
2. **Registry (tracker_model_loader_registry)**: Registry for registering and retrieving loaders
3. **Factory Functions**: `create_tracker_model_loader()` and `get_available_tracker_model_loaders()`
4. **Framework-specific loaders**: Concrete implementations (TorchDeepsortTrackerModelLoader, UltralyticsTrackerModelLoader)

## How It Works

### 1. Loader Registration

Model loaders register themselves using the `@tracker_model_loader_registry.register()` decorator:

```python
@tracker_model_loader_registry.register(
    "torch",
    description="Tracker model loader for DeepSORT (deep-sort-realtime)"
)
class TorchDeepsortTrackerModelLoader(BaseTrackerModelLoader):
    # Implementation...
    pass
```

This automatically adds the loader to the registry, making it available through the factory function.

### 2. Tracker Loading Flow

```
1. Configuration object (from tracker_config) → Factory function
2. create_tracker_model_loader(loader_name, config)
3. Registry retrieves appropriate loader class
4. Loader instantiated with config
5. loader.load_model() returns initialized tracker
6. loader.update(detections, img) performs tracking
```

### 3. Integration with Config Module

The model loader expects a configuration object from the tracker_config module:

```python
# 1. Load configuration
config = TrackerConfigLoader.from_yaml("config/ultralytics/tracker/bytetrack/config.yaml")

# 2. Create tracker loader with config
loader = create_tracker_model_loader(loader_name="ultralytics", config=config)

# 3. Load tracker
tracker = loader.load_model()

# 4. Use tracker
tracks = loader.update(detections, frame)
```

### 4. Unified Update API

All loaders expose a common `update()` method for tracking:

```python
def update(self, results: Any, img: Any | None = None) -> Any:
    """
    Update tracker with new detections.

    Args:
        results: Detection results (format depends on framework)
        img: Original frame (required for some trackers)

    Returns:
        List of active tracks
    """
    pass
```

## Available Loaders

### 1. Ultralytics Tracker Loader

**Loader Name**: `ultralytics`

**Supported Trackers**:
- ByteTrack (BYTETracker)
- BoTSORT (BOTSORT)

**Configuration Types**:
- `UltralyticsBytetrackConfig`
- `UltralyticsBotsortConfig`

**What It Does**:
1. Determines tracker type from config
2. Creates SimpleNamespace args object from config parameters
3. Instantiates BYTETracker or BOTSORT with args and frame_rate
4. Provides update() wrapper that handles detection format conversion

**Key Code** (`ultralytics/loader.py`):
```python
def load_model(self) -> Any:
    cfg = self.config

    # ByteTrack
    if isinstance(cfg, UltralyticsBytetrackConfig):
        params = cfg.get_init_params()
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

def update(self, results: Any, img: Any | None = None) -> Any:
    if self._tracker is None:
        self.load_model()

    # Convert tensors to numpy if needed
    if hasattr(results.xyxy, 'cpu'):
        results.xyxy = results.xyxy.cpu().numpy()
    # ... more conversions

    return self._tracker.update(results, img)
```

**Input Format**:
- `results`: Object with `.xyxy`, `.xywh`, `.conf`, `.cls` attributes
- `img`: Optional frame for BoTSORT (H x W x 3 numpy array)

**Output Format**:
- List of `STrack` objects with tracking information

### 2. Torch/DeepSORT Tracker Loader

**Loader Name**: `torch`

**Supported Trackers**:
- DeepSORT (deep-sort-realtime)

**Configuration Type**: `TorchDeepsortConfig`

**What It Does**:
1. Extracts parameters from TorchDeepsortConfig
2. Instantiates DeepSort tracker with parameters
3. Provides update() wrapper that converts detection formats
4. Handles multiple detection input formats

**Key Code** (`torch/loader.py`):
```python
def load_model(self) -> DeepSort:
    if not isinstance(self.config, TorchDeepsortConfig):
        raise TypeError(f"Expected TorchDeepsortConfig, got {type(self.config)}")

    params = self.config.get_init_params()
    self._tracker = DeepSort(**params)
    return self._tracker

def _results_to_detections(self, results: Any) -> List[Tuple[List[float], float, int]]:
    """
    Convert detection results to DeepSORT format: ([x, y, w, h], confidence, class_id)

    Supports:
    - Objects with .xyxy, .conf, .cls attributes
    - Numpy arrays (N, 6) -> [x1, y1, x2, y2, score, cls]
    """
    # ... conversion logic

def update(self, results: Any, img: Any | None = None) -> Any:
    if self._tracker is None:
        self.load_model()

    detections = self._results_to_detections(results)
    tracks = self._tracker.update_tracks(detections, frame=img)
    return tracks
```

**Input Format**:
- `results`:
  - Object with `.xyxy`, `.conf`, `.cls` attributes, OR
  - Numpy array (N, 6) with format [x1, y1, x2, y2, score, cls]
- `img`: Required frame (H x W x 3 numpy array, BGR format)

**Output Format**:
- List of `Track` objects from deep-sort-realtime

## Usage Examples

### Basic Usage

```python
from src.tracker_config import TrackerConfigLoader
from src.tracker_model_loader import create_tracker_model_loader

# Load configuration
config = TrackerConfigLoader.from_yaml(
    "config/ultralytics/tracker/bytetrack/config.yaml"
)

# Create and use loader
loader = create_tracker_model_loader(loader_name="ultralytics", config=config)
tracker = loader.load_model()

# Tracker is now ready for use
print(f"Tracker loaded: {type(tracker)}")
print(f"Loader name: {loader.get_name()}")
```

### Full Tracking Pipeline

```python
from src.detector_config import DetectorConfigLoader
from src.detector_model_loader import create_detector_model_loader
from src.tracker_config import TrackerConfigLoader
from src.tracker_model_loader import create_tracker_model_loader
import cv2

# 1. Setup detector
det_config = DetectorConfigLoader.from_yaml("config/ultralytics/detector/yolov8n/config.yaml")
det_loader = create_detector_model_loader("ultralytics", det_config)
detector = det_loader.load_model()

# 2. Setup tracker
track_config = TrackerConfigLoader.from_yaml("config/ultralytics/tracker/bytetrack/config.yaml")
track_loader = create_tracker_model_loader("ultralytics", track_config)
tracker = track_loader.load_model()

# 3. Process video
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    detections = detector(frame)

    # Track
    tracks = track_loader.update(detections, frame)

    # Process tracks
    for track in tracks:
        if hasattr(track, 'is_confirmed') and track.is_confirmed():
            x1, y1, x2, y2 = track.tlbr
            track_id = track.track_id
            print(f"Track {track_id}: [{x1}, {y1}, {x2}, {y2}]")

cap.release()
```

### Loading Different Trackers

```python
# ByteTrack
bytetrack_config = TrackerConfigLoader.from_yaml(
    "config/ultralytics/tracker/bytetrack/config.yaml"
)
bytetrack_loader = create_tracker_model_loader("ultralytics", bytetrack_config)
bytetrack = bytetrack_loader.load_model()

# BoTSORT
botsort_config = TrackerConfigLoader.from_yaml(
    "config/ultralytics/tracker/botsort/config.yaml"
)
botsort_loader = create_tracker_model_loader("ultralytics", botsort_config)
botsort = botsort_loader.load_model()

# DeepSORT
deepsort_config = TrackerConfigLoader.from_yaml(
    "config/torch/tracker/deepsort/config.yaml"
)
deepsort_loader = create_tracker_model_loader("torch", deepsort_config)
deepsort = deepsort_loader.load_model()
```

### Checking Available Loaders

```python
from src.tracker_model_loader import get_available_tracker_model_loaders

available = get_available_tracker_model_loaders()
print(f"Available loaders: {available}")
# Output: ['ultralytics', 'torch']
```

### Using Loader Class Directly

```python
from src.tracker_config.ultralytics.config import UltralyticsBytetrackConfig
from src.tracker_model_loader.ultralytics.loader import UltralyticsTrackerModelLoader

# Create config
config = UltralyticsBytetrackConfig(
    track_buffer=50,
    track_thresh=0.6,
    match_thresh=0.9
)

# Use loader directly
loader = UltralyticsTrackerModelLoader(config)
tracker = loader.load_model()
```

## Adding New Loaders

Follow these steps to add a loader for a new tracking algorithm:

### Step 1: Create Loader Class

Create `src/tracker_model_loader/<framework>/loader.py`:

```python
from typing import Any, Optional

from src.tracker_config.<framework>.config import MyTrackerConfig
from src.tracker_model_loader.base_tracker_model_loader import BaseTrackerModelLoader
from src.tracker_model_loader.tracker_model_loader_registry import tracker_model_loader_registry


@tracker_model_loader_registry.register(
    "myframework",
    description="Tracker model loader for My Tracker"
)
class MyFrameworkTrackerModelLoader(BaseTrackerModelLoader):
    """
    My Framework implementation of tracker model loader.

    Given a MyTrackerConfig, this loader instantiates
    the appropriate tracker.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._tracker: Optional[Any] = None

    def load_model(self) -> Any:
        """
        Instantiate and return a My Framework tracker instance.

        Returns:
            An instance of My Framework tracker.
        """
        cfg = self.config

        # Type check
        if not isinstance(cfg, MyTrackerConfig):
            raise TypeError(
                f"MyFrameworkTrackerModelLoader received unsupported config type: {type(cfg)}"
            )

        # Get parameters from config
        params = cfg.get_init_params()

        # Instantiate tracker
        self._tracker = MyTracker(
            max_age=params["max_age"],
            match_threshold=params["match_threshold"],
            # ... other parameters
        )

        return self._tracker

    def update(self, results: Any, img: Any | None = None) -> Any:
        """
        Update the tracker with new detections.

        Args:
            results: Detection results
            img: Optional frame image

        Returns:
            List of active tracks
        """
        # Lazy loading
        if self._tracker is None:
            self.load_model()

        # Convert detection format if needed
        detections = self._convert_detections(results)

        # Update tracker
        tracks = self._tracker.update(detections, img)

        return tracks

    def _convert_detections(self, results: Any) -> Any:
        """Convert detection results to tracker's expected format."""
        # Implement conversion logic
        pass

    @classmethod
    def get_name(cls) -> str:
        """Return the name of this loader."""
        return "myframework"
```

### Step 2: Create __init__.py

Create `src/tracker_model_loader/<framework>/__init__.py`:

```python
from src.tracker_model_loader.myframework.loader import MyFrameworkTrackerModelLoader

__all__ = ["MyFrameworkTrackerModelLoader"]
```

### Step 3: Register in Main __init__.py

Import your loader in `src/tracker_model_loader/__init__.py`:

```python
# Import concrete implementations so they register themselves
from src.tracker_model_loader.ultralytics.loader import UltralyticsTrackerModelLoader  # noqa: F401
from src.tracker_model_loader.torch.loader import TorchDeepsortTrackerModelLoader  # noqa: F401
from src.tracker_model_loader.myframework.loader import MyFrameworkTrackerModelLoader  # noqa: F401
```

### Step 4: Implement Required Methods

Your loader class must implement:

1. **`__init__(self, config: BaseTrackerConfig)`** - Initialize with config
2. **`load_model(self) -> Any`** - Load and return tracker instance
3. **`update(self, results: Any, img: Any | None = None) -> Any`** - Update tracker with detections
4. **`get_name(cls) -> str`** - Class method returning loader name

### Step 5: Handle Detection Format Conversion

Implement logic to convert detection formats:

```python
def _convert_detections(self, results: Any) -> List[Detection]:
    """
    Convert various detection formats to tracker's expected format.

    Supports:
    - Ultralytics Results objects
    - Torch detection outputs
    - Numpy arrays
    """
    detections = []

    # Case 1: Object with .xyxy, .conf, .cls
    if hasattr(results, "xyxy") and hasattr(results, "conf"):
        xyxy = results.xyxy
        confs = results.conf
        clss = results.cls

        # Convert to numpy if needed
        if hasattr(xyxy, 'cpu'):
            xyxy = xyxy.cpu().numpy()
        # ... more conversions

        for i in range(len(xyxy)):
            det = Detection(
                bbox=xyxy[i],
                confidence=confs[i],
                class_id=int(clss[i])
            )
            detections.append(det)

    # Case 2: Numpy array
    elif isinstance(results, np.ndarray):
        # ... handle numpy format

    return detections
```

### Step 6: Test Your Implementation

```python
from src.tracker_config import TrackerConfigLoader
from src.tracker_model_loader import (
    create_tracker_model_loader,
    get_available_tracker_model_loaders
)

# Test registration
available = get_available_tracker_model_loaders()
assert "myframework" in available

# Test loading
config = TrackerConfigLoader.from_yaml(
    "config/myframework/tracker/mytracker/config.yaml"
)
loader = create_tracker_model_loader("myframework", config)
tracker = loader.load_model()

# Test tracking
detections = # ... get detections
frame = # ... get frame
tracks = loader.update(detections, frame)

print(f"Active tracks: {len(tracks)}")
```

## Best Practices

### 1. Type Checking

Always validate the config type:
```python
if not isinstance(cfg, MyTrackerConfig):
    raise TypeError(f"Expected MyTrackerConfig, got {type(cfg)}")
```

### 2. Lazy Loading

Support lazy initialization:
```python
def update(self, results, img=None):
    if self._tracker is None:
        self.load_model()
    # ... proceed with update
```

### 3. Format Conversion

Handle multiple detection formats gracefully:
```python
def _convert_detections(self, results):
    # Try multiple formats
    if hasattr(results, "xyxy"):
        return self._from_object(results)
    elif isinstance(results, np.ndarray):
        return self._from_numpy(results)
    else:
        raise TypeError(f"Unsupported detection format: {type(results)}")
```

### 4. Error Handling

Provide informative error messages:
```python
if len(detections) == 0:
    # Return empty tracks instead of erroring
    return []

if img is None and self.requires_image:
    raise ValueError(
        f"{self.get_name()} tracker requires frame image for appearance features"
    )
```

### 5. Tensor to Numpy Conversion

Always convert GPU tensors to numpy:
```python
if hasattr(results.xyxy, 'cpu'):
    results.xyxy = results.xyxy.cpu().numpy()
if hasattr(results.conf, 'cpu'):
    results.conf = results.conf.cpu().numpy()
```

### 6. Track Output Format

Document what your `update()` method returns:
```python
def update(self, results, img=None) -> List[Track]:
    """
    Update tracker with detections.

    Returns:
        List of Track objects, each with:
        - track_id: Unique track identifier
        - tlbr: Bounding box [top, left, bottom, right]
        - is_confirmed(): Whether track is confirmed
        - ... other attributes
    """
    pass
```

### 7. Multiple Tracker Support

Support multiple algorithms in one loader:
```python
def load_model(self):
    if isinstance(self.config, ByteTrackConfig):
        return self._load_bytetrack()
    elif isinstance(self.config, BoTSORTConfig):
        return self._load_botsort()
    else:
        raise TypeError(f"Unsupported config: {type(self.config)}")
```

## Common Patterns

### Pattern 1: Bbox Format Conversion

```python
def _xyxy_to_xywh(self, xyxy):
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]

def _xywh_to_xyxy(self, xywh):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = xywh
    return [x, y, x + w, y + h]
```

### Pattern 2: Detection Filtering

```python
def _filter_detections(self, detections, min_conf=0.3):
    """Filter detections by confidence."""
    return [det for det in detections if det.confidence >= min_conf]
```

### Pattern 3: Track Post-Processing

```python
def update(self, results, img=None):
    raw_tracks = self._tracker.update(detections, img)

    # Filter confirmed tracks only
    confirmed_tracks = [t for t in raw_tracks if t.is_confirmed()]

    # Sort by track ID
    confirmed_tracks.sort(key=lambda t: t.track_id)

    return confirmed_tracks
```

## Troubleshooting

### Loader Not Found
```
ValueError: Unknown tracker model loader 'myloader'
```
**Solution**: Ensure you imported your loader in `__init__.py` and used the `@tracker_model_loader_registry.register()` decorator.

### Wrong Config Type
```
TypeError: MyLoader received unsupported config type: ByteTrackConfig
```
**Solution**: Ensure you're passing the correct config type to the loader.

### Detection Format Error
```
TypeError: Unsupported detection format
```
**Solution**: Implement detection format conversion in your loader, or ensure detections are in the expected format.

### Missing Frame
```
ValueError: DeepSORT requires frame image for appearance features
```
**Solution**: Pass the frame image to `update()`: `loader.update(detections, img=frame)`

### GPU/CPU Mismatch
```
RuntimeError: Tensor on cuda:0 but expected on cpu
```
**Solution**: Convert tensors to numpy or ensure consistent device usage.

## Integration Example

Here's a complete example showing integration with the config module and a detection pipeline:

```python
from pathlib import Path
import cv2
from src.detector_config import DetectorConfigLoader
from src.detector_model_loader import create_detector_model_loader
from src.tracker_config import TrackerConfigLoader
from src.tracker_model_loader import create_tracker_model_loader

# Setup
det_config = DetectorConfigLoader.from_yaml("config/ultralytics/detector/yolov8n/config.yaml")
det_loader = create_detector_model_loader("ultralytics", det_config)
detector = det_loader.load_model()

track_config = TrackerConfigLoader.from_yaml("config/ultralytics/tracker/bytetrack/config.yaml")
track_loader = create_tracker_model_loader("ultralytics", track_config)

# Process video
cap = cv2.VideoCapture("video.mp4")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detect objects
    detections = detector(frame)

    # Track objects
    tracks = track_loader.update(detections, frame)

    # Draw tracks
    for track in tracks:
        if hasattr(track, 'tlbr'):
            x1, y1, x2, y2 = map(int, track.tlbr)
            track_id = track.track_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Related Documentation

- [Tracker Config README](../tracker_config/README.md) - How to create and configure tracker configs
- [Detector Model Loader README](../detector_model_loader/README.md) - Similar pattern for detector loaders
- [Main README](../../README.md) - Overall project documentation
