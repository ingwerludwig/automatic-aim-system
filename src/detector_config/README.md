# Detector Configuration Module

This module provides a flexible factory pattern for managing detector configurations across different frameworks (Ultralytics, Torch/Torchvision, etc.). It enables easy addition of new detector types and configurations through a registry system.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Available Detectors](#available-detectors)
- [Usage Examples](#usage-examples)
- [Adding New Detectors](#adding-new-detectors)

## Overview

The detector configuration module serves as a configuration layer that:
- Defines detector-specific parameters (model paths, hyperparameters, etc.)
- Provides a unified interface for different detector frameworks
- Uses factory pattern for dynamic detector instantiation
- Loads configurations from YAML files
- Automatically registers new detector types via decorators

## Architecture

### Core Components

```
detector_config/
├── base_detector_config.py       # Abstract base class & factory
├── __init__.py                    # Config loader & exports
├── torch/                         # Torch-specific configs
│   └── config.py                  # TorchDetectorConfig
└── ultralytics/                   # Ultralytics-specific configs
    └── config.py                  # UltralyticsDetectorConfig
```

### Key Classes

1. **BaseDetectorConfig**: Abstract base class defining the interface all detector configs must implement
2. **DetectorConfigFactory**: Factory class with registry for creating detector configs
3. **DetectorConfigLoader**: Utility class for loading configs from YAML files
4. **Framework-specific configs**: Concrete implementations (TorchDetectorConfig, UltralyticsDetectorConfig)

## How It Works

### 1. Configuration Registration

Detector configurations register themselves using the `@DetectorConfigFactory.register()` decorator:

```python
@DetectorConfigFactory.register("torch")
@dataclass
class TorchDetectorConfig(BaseDetectorConfig):
    # Configuration fields...
    pass
```

This automatically adds the config to the factory's registry, making it available for instantiation.

### 2. Loading from YAML

Configurations are stored in YAML files under `config/`:

```
config/
├── torch/
│   └── detector/
│       └── fasterrcnn/
│           └── config.yaml
└── ultralytics/
    └── detector/
        └── yolov8n/
            └── config.yaml
```

YAML structure:
```yaml
config:
  model_init_name: fasterrcnn_resnet50_fpn
  weights: null
  weights_path: /path/to/weights.pth

  inference_cfg:
    device: cuda
    score_threshold: 0.7
    person_only: true
    # ... other parameters
```

### 3. Configuration Flow

```
1. YAML file → DetectorConfigLoader.from_yaml()
2. Extract detector_type from YAML
3. DetectorConfigFactory.create(detector_type, config_dict)
4. Factory returns appropriate config instance
5. Config instance provides get_init_params() for model loader
```

### 4. Three Ways to Create Configs

#### Method 1: From YAML file (Recommended)
```python
from src.detector_config import DetectorConfigLoader

config = DetectorConfigLoader.from_yaml("config/torch/detector/fasterrcnn/config.yaml")
```

#### Method 2: From model name (Uses convention)
```python
from src.detector_config.torch.config import TorchDetectorConfig
from pathlib import Path

config = TorchDetectorConfig.from_model_name(
    model_name="fasterrcnn",
    config_root=Path("config")
)
# Automatically looks for: config/torch/detector/fasterrcnn/config.yaml
```

#### Method 3: From dictionary
```python
from src.detector_config import DetectorConfigFactory

config_dict = {
    "model_fn": "fasterrcnn_resnet50_fpn",
    "device": "cuda",
    "score_threshold": 0.7,
}

config = DetectorConfigFactory.create("torch", config_dict)
```

## Available Detectors

### 1. Torch/Torchvision Detectors

**Type**: `torch`

**Supported Models**:
- Faster R-CNN (ResNet50 FPN, ResNet50 FPN V2, MobileNetV3)
- RetinaNet
- FCOS
- SSD
- And other torchvision.models.detection models

**Key Parameters**:
```python
model_fn: str              # torchvision model function name
weights: Optional[str]     # "DEFAULT" or weights enum
weights_path: Optional[str]  # Path to custom .pth file
device: str                # "cuda", "cpu", etc.
score_threshold: float     # Confidence threshold
person_only: bool          # Filter for COCO person class only

# Faster R-CNN specific:
box_score_thresh: float
box_nms_thresh: float
box_detections_per_img: int
rpn_nms_thresh: float
# ... more RPN parameters
```

**Example Config** (`config/torch/detector/fasterrcnn/config.yaml`):
```yaml
config:
  model_init_name: fasterrcnn_resnet50_fpn
  weights: null
  weights_path: /path/to/fasterrcnn_resnet50_fpn_coco.pth

  inference_cfg:
    device: cuda
    score_threshold: 0.7
    person_only: true
    box_detections_per_img: 5
```

### 2. Ultralytics Detectors

**Type**: `ultralytics`

**Supported Models**:
- YOLOv8 (n, s, m, l, x variants)
- YOLOv9, YOLOv10, YOLOv11
- RT-DETR
- Custom trained models

**Key Parameters**:
```python
model: str                  # Model path or name (e.g., "yolov8n.pt")
device: Optional[str]       # "cuda:0", "cpu", "mps"
half: bool                  # Enable FP16 inference
fuse: bool                  # Fuse Conv+BN for speed
default_imgsz: int          # Input image size
default_conf: float         # Confidence threshold
default_iou: float          # IoU threshold for NMS
classes: Optional[List[int]]  # Filter specific classes
task: Optional[str]         # "detect", "segment", "pose", etc.
```

**Example Config** (`config/ultralytics/detector/yolov8n/config.yaml`):
```yaml
config:
  model_init_name: yolov8n.pt

  inference_cfg:
    device: cuda
    fp16: true
    fuse: true
    imgsz: 640
    conf: 0.25
    iou: 0.45
    classes: [0]  # Person only
    task: detect
```

## Usage Examples

### Loading a Configuration

```python
from src.detector_config import DetectorConfigLoader

# Load Torch detector config
torch_config = DetectorConfigLoader.from_yaml(
    "config/torch/detector/fasterrcnn/config.yaml"
)

# Load Ultralytics detector config
yolo_config = DetectorConfigLoader.from_yaml(
    "config/ultralytics/detector/yolov8n/config.yaml"
)
```

### Using Configuration with Model Loader

```python
from src.detector_config import DetectorConfigLoader
from src.detector_model_loader import create_detector_model_loader

# 1. Load config
config = DetectorConfigLoader.from_yaml("config/torch/detector/fasterrcnn/config.yaml")

# 2. Create model loader with config
loader = create_detector_model_loader(loader_name="torch", config=config)

# 3. Load model
model = loader.load_model()
```

### Checking Available Detectors

```python
from src.detector_config import DetectorConfigLoader

available = DetectorConfigLoader.get_available_detectors()
print(f"Available detector types: {available}")
# Output: ['torch', 'ultralytics']
```

## Adding New Detectors

Follow these steps to add a new detector framework:

### Step 1: Create Config Class

Create a new file in `src/detector_config/<framework>/config.py`:

```python
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path

from src.detector_config.base_detector_config import (
    BaseDetectorConfig,
    DetectorConfigFactory
)

@DetectorConfigFactory.register("myframework")
@dataclass
class MyFrameworkDetectorConfig(BaseDetectorConfig):
    """Configuration for My Framework detector."""

    # Define your config fields
    model_name: str = "default_model"
    device: str = "cuda"
    threshold: float = 0.5
    # ... more fields

    def get_init_params(self) -> Dict[str, Any]:
        """Return parameters for detector initialization."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "threshold": self.threshold,
            # ... more parameters
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MyFrameworkDetectorConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_model_name(cls, model_name: str, config_root: Path) -> "MyFrameworkDetectorConfig":
        """Load config from standardized path."""
        import yaml

        yaml_path = (
            config_root
            / "myframework"
            / "detector"
            / model_name
            / "config.yaml"
        )

        with yaml_path.open("r") as f:
            raw = yaml.safe_load(f)

        if "config" not in raw:
            raise ValueError(f"Expected 'config' key in {yaml_path}")

        cfg_block = raw["config"]
        inf = cfg_block.get("inference_cfg", {})

        return cls(
            model_name=cfg_block.get("model_init_name", "default_model"),
            device=inf.get("device", "cuda"),
            threshold=inf.get("threshold", 0.5),
            # ... map other parameters
        )
```

### Step 2: Create __init__.py

Create `src/detector_config/<framework>/__init__.py`:

```python
from src.detector_config.myframework.config import MyFrameworkDetectorConfig

__all__ = ["MyFrameworkDetectorConfig"]
```

### Step 3: Register in Main __init__.py

Import your config in `src/detector_config/__init__.py`:

```python
# Import concrete implementations to trigger registration
from src.detector_config.torch.config import TorchDetectorConfig  # noqa: F401
from src.detector_config.ultralytics.config import UltralyticsDetectorConfig  # noqa: F401
from src.detector_config.myframework.config import MyFrameworkDetectorConfig  # noqa: F401
```

### Step 4: Create Config YAML Template

Create directory structure and config file:
```
config/myframework/detector/mymodel/config.yaml
```

```yaml
config:
  model_init_name: mymodel_v1

  inference_cfg:
    device: cuda
    threshold: 0.5
    # ... other parameters
```

### Step 5: Test Your Implementation

```python
from src.detector_config import DetectorConfigLoader

# Test loading
config = DetectorConfigLoader.from_yaml(
    "config/myframework/detector/mymodel/config.yaml"
)

# Test factory
available = DetectorConfigLoader.get_available_detectors()
assert "myframework" in available

# Test parameters
params = config.get_init_params()
print(params)
```

## Best Practices

1. **Use Dataclasses**: Leverage Python dataclasses for clean configuration definitions

2. **Provide Defaults**: Always provide sensible default values for optional parameters

3. **Document Parameters**: Add docstrings explaining what each parameter does

4. **Validate Inputs**: Check for invalid configurations in `from_dict()` or `get_init_params()`

5. **Follow Naming Conventions**:
   - Config class: `<Framework>DetectorConfig`
   - Registry name: `<framework>` (lowercase)
   - YAML path: `config/<framework>/detector/<model>/config.yaml`

6. **Keep Framework-Specific Logic Isolated**: Put framework-specific code in the respective config class, not in base classes

7. **Support Multiple Loading Methods**: Implement all three classmethods (`from_dict`, `from_model_name`, and custom YAML loader if needed)

## Common Patterns

### Optional Parameters
```python
box_score_thresh: Optional[float] = None

def get_init_params(self) -> Dict[str, Any]:
    params = {}
    if self.box_score_thresh is not None:
        params["box_score_thresh"] = self.box_score_thresh
    return params
```

### Nested Configuration
```yaml
config:
  model_init_name: mymodel

  inference_cfg:
    device: cuda

    advanced:
      param1: value1
      param2: value2
```

### Path Resolution
```python
from pathlib import Path

weights_path = cfg_block.get("weights_path")
if weights_path:
    weights_path = Path(weights_path).expanduser().resolve()
```

## Troubleshooting

### Configuration Not Found
```
ValueError: Unknown detector type: mydetector
```
**Solution**: Make sure you imported your config class in `__init__.py` and used the `@DetectorConfigFactory.register()` decorator.

### YAML Loading Error
```
ValueError: Expected top-level 'config' key in config.yaml
```
**Solution**: Ensure your YAML file has the correct structure with `config:` as the top-level key.

### Parameter Mismatch
```
TypeError: __init__() got an unexpected keyword argument 'unknown_param'
```
**Solution**: Check that all parameters in your YAML file match the dataclass fields, or handle them in `from_dict()`.

## Related Documentation

- [Detector Model Loader README](../detector_model_loader/README.md) - How configs are used to load models
- [Main README](../../README.md) - Overall project documentation
