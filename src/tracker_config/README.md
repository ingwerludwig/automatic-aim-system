# Tracker Configuration Module

This module provides a flexible factory pattern for managing tracker configurations across different frameworks and algorithms (ByteTrack, BoTSORT, DeepSORT, etc.). It enables easy addition of new tracker types through a registry system.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Available Trackers](#available-trackers)
- [Usage Examples](#usage-examples)
- [Adding New Trackers](#adding-new-trackers)

## Overview

The tracker configuration module serves as a configuration layer that:
- Defines tracker-specific parameters (matching thresholds, buffer sizes, etc.)
- Provides a unified interface for different tracking algorithms
- Uses factory pattern for dynamic tracker instantiation
- Loads configurations from YAML files
- Automatically registers new tracker types via decorators

## Architecture

### Core Components

```
tracker_config/
├── base_tracker_config.py        # Abstract base class & factory
├── __init__.py                    # Config loader & exports
├── torch/                         # Torch-based trackers
│   └── config.py                  # TorchDeepsortConfig
└── ultralytics/                   # Ultralytics trackers
    └── config.py                  # ByteTrack & BoTSORT configs
```

### Key Classes

1. **BaseTrackerConfig**: Abstract base class defining the interface all tracker configs must implement
2. **TrackerConfigFactory**: Factory class with registry for creating tracker configs
3. **TrackerConfigLoader**: Utility class for loading configs from YAML files
4. **Algorithm-specific configs**: Concrete implementations (TorchDeepsortConfig, UltralyticsBytetrackConfig, UltralyticsBotsortConfig)

## How It Works

### 1. Configuration Registration

Tracker configurations register themselves using the `@TrackerConfigFactory.register()` decorator:

```python
@TrackerConfigFactory.register("ultralytics_bytetrack")
@dataclass
class UltralyticsBytetrackConfig(BaseTrackerConfig):
    # Configuration fields...
    pass
```

This automatically adds the config to the factory's registry, making it available for instantiation.

### 2. Loading from YAML

Configurations are stored in YAML files under `config/`:

```
config/
├── torch/
│   └── tracker/
│       └── deepsort/
│           └── config.yaml
└── ultralytics/
    └── tracker/
        ├── bytetrack/
        │   └── config.yaml
        └── botsort/
            └── config.yaml
```

YAML structure:
```yaml
config:
  model_init_name: bytetracker
  inference_cfg:
    track_buffer: 30
    track_thresh: 0.5
    # ... other parameters
```

### 3. Configuration Flow

```
1. YAML file → TrackerConfigLoader.from_yaml()
2. Extract tracker_type from YAML
3. TrackerConfigFactory.create(tracker_type, config_dict)
4. Factory returns appropriate config instance
5. Config instance provides get_init_params() for model loader
```

### 4. Three Ways to Create Configs

#### Method 1: From YAML file (Recommended)
```python
from src.tracker_config import TrackerConfigLoader

config = TrackerConfigLoader.from_yaml("config/ultralytics/tracker/bytetrack/config.yaml")
```

#### Method 2: From model name (Uses convention)
```python
from src.tracker_config.ultralytics.config import UltralyticsBytetrackConfig
from pathlib import Path

config = UltralyticsBytetrackConfig.from_model_name(
    model_name="bytetrack",
    config_root=Path("config")
)
# Automatically looks for: config/ultralytics/tracker/bytetrack/config.yaml
```

#### Method 3: From dictionary
```python
from src.tracker_config import TrackerConfigFactory

config_dict = {
    "track_buffer": 30,
    "track_thresh": 0.5,
    "match_thresh": 0.8,
}

config = TrackerConfigFactory.create("ultralytics_bytetrack", config_dict)
```

## Available Trackers

### 1. ByteTrack (Ultralytics)

**Type**: `ultralytics_bytetrack`

**Description**: Simple, fast, and strong multi-object tracking algorithm. Uses only detection scores and IoU distance.

**Key Parameters**:
```python
track_buffer: int              # Number of frames to keep lost tracks (default: 30)
track_thresh: float            # Threshold for first association (default: 0.5)
track_high_thresh: float       # High detection threshold (default: 0.6)
track_low_thresh: float        # Low detection threshold (default: 0.1)
new_track_thresh: float        # Threshold for creating new tracks (default: 0.7)
match_thresh: float            # IoU matching threshold (default: 0.8)
fuse_score: bool               # Fuse detection and tracking scores (default: True)
mot20: bool                    # Use MOT20 settings (default: False)
frame_rate: int                # Video frame rate (default: 30)
```

**Example Config** (`config/ultralytics/tracker/bytetrack/config.yaml`):
```yaml
config:
  model_init_name: "bytetracker"
  inference_cfg:
    track_buffer: 30
    track_thresh: 0.5
    track_high_thresh: 0.6
    track_low_thresh: 0.1
    new_track_thresh: 0.7
    match_thresh: 0.8
    fuse_score: true
    mot20: false
```

**When to Use**:
- Fast inference speed required
- Simple appearance-free tracking
- Good detection quality available
- Real-time applications

### 2. BoTSORT (Ultralytics)

**Type**: `ultralytics_botsort`

**Description**: Enhanced ByteTrack with camera motion compensation (CMC) and better handling of occlusions.

**Key Parameters**:
```python
track_buffer: int              # Number of frames to keep lost tracks (default: 30)
track_thresh: float            # Threshold for first association (default: 0.5)
track_high_thresh: float       # High detection threshold (default: 0.6)
track_low_thresh: float        # Low detection threshold (default: 0.1)
new_track_thresh: float        # Threshold for creating new tracks (default: 0.7)
match_thresh: float            # IoU matching threshold (default: 0.8)
fuse_score: bool               # Fuse detection and tracking scores (default: True)
mot20: bool                    # Use MOT20 settings (default: False)
cmc_method: str                # Camera motion compensation method (default: "file")
frame_rate: int                # Video frame rate (default: 30)
```

**Example Config** (`config/ultralytics/tracker/botsort/config.yaml`):
```yaml
config:
  model_init_name: "botsort"
  inference_cfg:
    track_buffer: 30
    track_thresh: 0.5
    track_high_thresh: 0.6
    track_low_thresh: 0.1
    new_track_thresh: 0.7
    match_thresh: 0.8
    fuse_score: true
    mot20: false
    cmc_method: "file"
```

**When to Use**:
- Moving camera scenarios
- Need better occlusion handling
- Can afford slightly higher computation
- Crowded scenes

### 3. DeepSORT (Torch)

**Type**: `torch_deepsort`

**Description**: Deep learning-based appearance descriptor combined with Kalman filtering and Hungarian algorithm. Excellent for handling occlusions and ID switches.

**Key Parameters**:
```python
# Core DeepSORT parameters
max_age: int                    # Max frames to keep track alive without match (default: 30)
n_init: int                     # Frames needed to confirm a track (default: 3)
max_iou_distance: float         # Max IoU distance for matching (default: 0.7)
max_cosine_distance: float      # Max cosine distance for appearance (default: 0.5)
nms_max_overlap: float          # NMS overlap threshold (default: 1.0)

# Embedder configuration
embedder: str                   # Embedder type (default: "osnet_x0_25")
                               # Options: 'mobilenet', 'torchreid', 'osnet_x0_25',
                               #          'clip_RN50', 'clip_RN101', 'clip_RN50x4',
                               #          'clip_RN50x16', 'clip_ViT-B/32', 'clip_ViT-B/16'
embedder_gpu: bool              # Use GPU for embedder (default: True)
half: bool                      # Use FP16 for embedder (default: True)
bgr: bool                       # Input is BGR format (default: True)

# Optional parameters
max_age_id: Optional[int]       # Max age before ID is dropped (default: None)
polygon: bool                   # Use polygon detection (default: False)
embedder_wts: Optional[str]     # Path to embedder weights (auto-resolved for CLIP)
```

**Example Config** (`config/torch/tracker/deepsort/config.yaml`):
```yaml
config:
  model_init_name: "deepsort"
  inference_cfg:
    max_age: 30
    n_init: 3
    max_iou_distance: 0.7
    max_cosine_distance: 0.5
    nms_max_overlap: 1.0

    embedder: "clip_RN50x4"
    embedder_gpu: true
    half: true
    bgr: true

    max_age_id: null
    polygon: false
```

**When to Use**:
- Need appearance-based matching
- Heavy occlusions expected
- ID preservation is critical
- Can afford higher computation cost

**Embedder Options**:
- `mobilenet`: Fast, lightweight (no GPU needed)
- `torchreid`: Medium quality, moderate speed
- `osnet_x0_25`: Good balance of speed and quality
- `clip_RN50`: CLIP ResNet-50 (requires checkpoint)
- `clip_RN101`: CLIP ResNet-101 (requires checkpoint)
- `clip_RN50x4`: CLIP ResNet-50x4 (requires checkpoint)
- `clip_RN50x16`: CLIP ResNet-50x16 (requires checkpoint)
- `clip_ViT-B/32`: CLIP ViT-B/32 (requires checkpoint)
- `clip_ViT-B/16`: CLIP ViT-B/16 (requires checkpoint)

**CLIP Checkpoint Setup**:
When using CLIP embedders, place checkpoint files in:
```
src/checkpoint/tracker/
├── clip_RN50.pt
├── clip_RN101.pt
├── clip_RN50x4.pt
├── clip_RN50x16.pt
├── clip_ViT-B-32.pt
└── clip_ViT-B-16.pt
```

The config class automatically resolves checkpoint paths.

## Usage Examples

### Loading a Configuration

```python
from src.tracker_config import TrackerConfigLoader

# Load ByteTrack config
bytetrack_config = TrackerConfigLoader.from_yaml(
    "config/ultralytics/tracker/bytetrack/config.yaml"
)

# Load DeepSORT config
deepsort_config = TrackerConfigLoader.from_yaml(
    "config/torch/tracker/deepsort/config.yaml"
)

# Load BoTSORT config
botsort_config = TrackerConfigLoader.from_yaml(
    "config/ultralytics/tracker/botsort/config.yaml"
)
```

### Using Configuration with Model Loader

```python
from src.tracker_config import TrackerConfigLoader
from src.tracker_model_loader import create_tracker_model_loader

# 1. Load config
config = TrackerConfigLoader.from_yaml(
    "config/ultralytics/tracker/bytetrack/config.yaml"
)

# 2. Create model loader with config
loader = create_tracker_model_loader(loader_name="ultralytics", config=config)

# 3. Load tracker
tracker = loader.load_model()
```

### Checking Available Trackers

```python
from src.tracker_config import TrackerConfigLoader

available = TrackerConfigLoader.get_available_trackers()
print(f"Available tracker types: {available}")
# Output: ['ultralytics_bytetrack', 'ultralytics_botsort', 'torch_deepsort']
```

### Creating Config from Dictionary

```python
from src.tracker_config import TrackerConfigFactory

# ByteTrack config
config = TrackerConfigFactory.create("ultralytics_bytetrack", {
    "track_buffer": 50,
    "track_thresh": 0.6,
    "match_thresh": 0.9,
})

params = config.get_init_params()
print(params)
```

## Adding New Trackers

Follow these steps to add a new tracking algorithm:

### Step 1: Create Config Class

Create a new file in `src/tracker_config/<framework>/config.py` or add to existing:

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

from src.tracker_config.base_tracker_config import (
    BaseTrackerConfig,
    TrackerConfigFactory
)

@TrackerConfigFactory.register("myframework_mytracker")
@dataclass
class MyTrackerConfig(BaseTrackerConfig):
    """Configuration for My Tracker algorithm."""

    # Define your config fields
    max_age: int = 30
    match_threshold: float = 0.7
    feature_dim: int = 128
    # ... more fields

    def get_init_params(self) -> Dict[str, Any]:
        """Return parameters for tracker initialization."""
        return {
            "max_age": self.max_age,
            "match_threshold": self.match_threshold,
            "feature_dim": self.feature_dim,
            # ... more parameters
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MyTrackerConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_model_name(cls, model_name: str, config_root: Path) -> "MyTrackerConfig":
        """Load config from standardized path."""
        yaml_path = (
            config_root
            / "myframework"
            / "tracker"
            / model_name
            / "config.yaml"
        )

        raw = yaml.safe_load(yaml_path.read_text())

        if "config" not in raw:
            raise ValueError(f"Expected 'config' key in {yaml_path}")

        cfg_block = raw["config"]
        inf = cfg_block.get("inference_cfg", {})

        return cls(
            max_age=inf.get("max_age", 30),
            match_threshold=inf.get("match_threshold", 0.7),
            feature_dim=inf.get("feature_dim", 128),
            # ... map other parameters
        )
```

### Step 2: Register Multiple Trackers (Optional)

You can register multiple tracker configs in the same file:

```python
@TrackerConfigFactory.register("myframework_tracker1")
@dataclass
class MyFrameworkTracker1Config(BaseTrackerConfig):
    # ... implementation

@TrackerConfigFactory.register("myframework_tracker2")
@dataclass
class MyFrameworkTracker2Config(BaseTrackerConfig):
    # ... implementation
```

### Step 3: Create __init__.py

Create or update `src/tracker_config/<framework>/__init__.py`:

```python
from src.tracker_config.myframework.config import (
    MyTrackerConfig,
    # ... other configs
)

__all__ = ["MyTrackerConfig"]
```

### Step 4: Register in Main __init__.py

Import your config in `src/tracker_config/__init__.py`:

```python
# Import concrete implementations to trigger registration
from src.tracker_config.torch.config import TorchDeepsortConfig  # noqa: F401
from src.tracker_config.ultralytics.config import (  # noqa: F401
    UltralyticsBytetrackConfig,
    UltralyticsBotsortConfig,
)
from src.tracker_config.myframework.config import MyTrackerConfig  # noqa: F401
```

### Step 5: Create Config YAML Template

Create directory structure and config file:
```
config/myframework/tracker/mytracker/config.yaml
```

```yaml
config:
  model_init_name: "mytracker"

  inference_cfg:
    max_age: 30
    match_threshold: 0.7
    feature_dim: 128
    # ... other parameters
```

### Step 6: Test Your Implementation

```python
from src.tracker_config import TrackerConfigLoader

# Test loading
config = TrackerConfigLoader.from_yaml(
    "config/myframework/tracker/mytracker/config.yaml"
)

# Test factory
available = TrackerConfigLoader.get_available_trackers()
assert "myframework_mytracker" in available

# Test parameters
params = config.get_init_params()
print(params)
```

## Best Practices

1. **Use Dataclasses**: Leverage Python dataclasses for clean configuration definitions

2. **Provide Defaults**: Always provide sensible default values based on paper recommendations

3. **Document Parameters**: Add docstrings explaining what each parameter does and typical ranges

4. **Validate Inputs**: Check for invalid configurations (e.g., negative thresholds, invalid ranges)

5. **Follow Naming Conventions**:
   - Config class: `<Framework><Algorithm>Config`
   - Registry name: `<framework>_<algorithm>` (lowercase)
   - YAML path: `config/<framework>/tracker/<algorithm>/config.yaml`

6. **Keep Framework-Specific Logic Isolated**: Don't mix tracking algorithm logic into config classes

7. **Support Multiple Loading Methods**: Implement all three classmethods

8. **Handle Optional Features**: Use `Optional[T]` for parameters that may not always be present

## Common Patterns

### Pattern 1: Optional Parameters
```python
max_age_id: Optional[int] = None

def get_init_params(self) -> Dict[str, Any]:
    params = {"max_age": self.max_age}
    if self.max_age_id is not None:
        params["max_age_id"] = self.max_age_id
    return params
```

### Pattern 2: Path Resolution
```python
@classmethod
def from_model_name(cls, model_name: str, config_root: Path) -> "MyConfig":
    PROJECT_ROOT = config_root.parent
    CHECKPOINT_DIR = PROJECT_ROOT / "src" / "checkpoint" / "tracker"

    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return cls(checkpoint_path=str(checkpoint_path))
```

### Pattern 3: Embedder Selection
```python
embedder: str = "osnet_x0_25"

# Validate embedder choice
VALID_EMBEDDERS = ["mobilenet", "torchreid", "osnet_x0_25", "clip_RN50"]

def __post_init__(self):
    if self.embedder not in VALID_EMBEDDERS:
        raise ValueError(f"Invalid embedder: {self.embedder}")
```

## Troubleshooting

### Configuration Not Found
```
ValueError: Unknown tracker type: mytracker
```
**Solution**: Ensure you imported your config class in `__init__.py` and used the `@TrackerConfigFactory.register()` decorator.

### YAML Loading Error
```
ValueError: Expected top-level 'config' key in config.yaml
```
**Solution**: Ensure your YAML file has the correct structure with `config:` as the top-level key.

### Missing Checkpoint
```
FileNotFoundError: CLIP embedder requires checkpoint: /path/to/clip_RN50x4.pt
```
**Solution**: Download and place the required checkpoint file in `src/checkpoint/tracker/`.

### Parameter Mismatch
```
TypeError: __init__() got an unexpected keyword argument 'unknown_param'
```
**Solution**: Check that all parameters in your YAML file match the dataclass fields.

## Tuning Guidelines

### ByteTrack Tuning

- **Increase `track_buffer`**: For videos with more occlusions
- **Increase `match_thresh`**: For tighter matching (fewer ID switches, more track fragmentation)
- **Decrease `track_low_thresh`**: To recover more low-confidence detections
- **Adjust `new_track_thresh`**: Higher = more conservative track creation

### BoTSORT Tuning

- Same as ByteTrack, plus:
- **`cmc_method`**: "file" for camera motion compensation, "none" to disable

### DeepSORT Tuning

- **Increase `max_age`**: For longer occlusions
- **Increase `n_init`**: To be more conservative (fewer false positives)
- **Decrease `max_cosine_distance`**: For stricter appearance matching
- **Choose embedder**: Faster (mobilenet) vs. better quality (CLIP models)

## Related Documentation

- [Tracker Model Loader README](../tracker_model_loader/README.md) - How configs are used to load trackers
- [Detector Config README](../detector_config/README.md) - Similar pattern for detector configs
- [Main README](../../README.md) - Overall project documentation
