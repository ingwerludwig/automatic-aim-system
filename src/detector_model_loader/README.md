# Detector Model Loader Module

This module provides a flexible factory pattern for loading detector models from different frameworks (Ultralytics, Torchvision, etc.). It works in tandem with the detector_config module to instantiate models based on configuration objects.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Available Loaders](#available-loaders)
- [Usage Examples](#usage-examples)
- [Adding New Loaders](#adding-new-loaders)

## Overview

The detector model loader module:
- Loads detection models based on configuration objects
- Provides a unified interface across different frameworks
- Uses registry pattern for dynamic loader registration
- Handles model initialization, weight loading, and device placement
- Attaches runtime metadata to models for inference

## Architecture

### Core Components

```
detector_model_loader/
├── base_detector_model_loader.py    # Abstract base class
├── detector_model_loader_registry.py # Registry implementation
├── __init__.py                       # Factory functions & exports
├── torch/                            # Torch-specific loader
│   └── loader.py                     # TorchDetectorModelLoader
└── ultralytics/                      # Ultralytics-specific loader
    └── loader.py                     # UltralyticsDetectorModelLoader
```

### Key Classes

1. **BaseDetectorModelLoader**: Abstract base class defining the interface all loaders must implement
2. **Registry (detector_model_loader_registry)**: Registry for registering and retrieving loaders
3. **Factory Functions**: `create_detector_model_loader()` and `get_available_detector_model_loaders()`
4. **Framework-specific loaders**: Concrete implementations (TorchDetectorModelLoader, UltralyticsDetectorModelLoader)

## How It Works

### 1. Loader Registration

Model loaders register themselves using the `@detector_model_loader_registry.register()` decorator:

```python
@detector_model_loader_registry.register(
    "torch",
    description="Detector model loader for torchvision-based models"
)
class TorchDetectorModelLoader(BaseDetectorModelLoader):
    # Implementation...
    pass
```

This automatically adds the loader to the registry, making it available through the factory function.

### 2. Model Loading Flow

```
1. Configuration object (from detector_config) → Factory function
2. create_detector_model_loader(loader_name, config)
3. Registry retrieves appropriate loader class
4. Loader instantiated with config
5. loader.load_model() returns initialized model
```

### 3. Integration with Config Module

The model loader expects a configuration object from the detector_config module:

```python
# 1. Load configuration
config = DetectorConfigLoader.from_yaml("config/torch/detector/fasterrcnn/config.yaml")

# 2. Create model loader with config
loader = create_detector_model_loader(loader_name="torch", config=config)

# 3. Load model
model = loader.load_model()
```

### 4. Model Metadata

Loaders attach useful metadata to the loaded model for downstream inference:

```python
# Torch loader attaches:
model._score_threshold = 0.7
model._person_only = True
model._device_str = "cuda"

# Ultralytics loader attaches:
model._default_imgsz = 640
model._default_conf = 0.25
model._default_iou = 0.45
model._default_classes = [0]
```

## Available Loaders

### 1. Torch/Torchvision Loader

**Loader Name**: `torch`

**Supported Models**: All torchvision.models.detection models
- Faster R-CNN variants
- RetinaNet
- FCOS
- SSD
- etc.

**Configuration Type**: `TorchDetectorConfig`

**What It Does**:
1. Resolves the torchvision model function (e.g., `fasterrcnn_resnet50_fpn`)
2. Loads weights from built-in enum or custom .pth file
3. Forwards Faster R-CNN hyperparameters (box_score_thresh, rpn_nms_thresh, etc.)
4. Moves model to specified device (CUDA/CPU)
5. Sets model to eval mode
6. Attaches inference metadata

**Key Code** (`torch/loader.py`):
```python
def load_model(self) -> Any:
    params = self.config.get_init_params()

    # Get torchvision model function
    model_fn_name = params["model_fn"]
    model_fn = getattr(torchvision.models.detection, model_fn_name)

    # Prepare kwargs for Faster R-CNN
    detector_kwargs = {}
    for key in ["box_score_thresh", "box_nms_thresh", ...]:
        if params.get(key) is not None:
            detector_kwargs[key] = params[key]

    # Load with custom or built-in weights
    if params["weights_path"]:
        model = model_fn(weights=None, **detector_kwargs)
        state_dict = torch.load(params["weights_path"], map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        model = model_fn(weights=params["weights"], **detector_kwargs)

    # Move to device and eval mode
    device = torch.device(params["device"])
    model.to(device)
    model.eval()

    # Attach metadata
    model._score_threshold = params.get("score_threshold", 0.3)
    model._person_only = params.get("person_only", True)

    return model
```

### 2. Ultralytics Loader

**Loader Name**: `ultralytics`

**Supported Models**:
- YOLO variants (v8, v9, v10, v11)
- RT-DETR
- Custom trained models

**Configuration Type**: `UltralyticsDetectorConfig`

**What It Does**:
1. Instantiates YOLO model with model path
2. Sets device via overrides
3. Optionally fuses Conv+BN layers for speed
4. Enables FP16 inference if requested
5. Attaches default inference parameters

**Key Code** (`ultralytics/loader.py`):
```python
def load_model(self) -> Any:
    params = self.config.get_init_params()

    # Create YOLO instance
    model_path = params["model"]
    model = YOLO(model_path)

    # Set device
    device = params.get("device")
    if device:
        model.overrides['device'] = device

    # Fuse layers for speed
    if params.get("fuse", False):
        model.fuse()

    # Enable FP16
    if params.get("half", False):
        model.overrides['half'] = True

    # Attach default inference parameters
    model._default_imgsz = params.get("default_imgsz", 640)
    model._default_conf = params.get("default_conf", 0.25)
    model._default_iou = params.get("default_iou", 0.45)
    model._default_classes = params.get("classes", None)

    return model
```

## Usage Examples

### Basic Usage

```python
from src.detector_config import DetectorConfigLoader
from src.detector_model_loader import create_detector_model_loader

# Load configuration
config = DetectorConfigLoader.from_yaml(
    "config/torch/detector/fasterrcnn/config.yaml"
)

# Create and use loader
loader = create_detector_model_loader(loader_name="torch", config=config)
model = loader.load_model()

# Model is now ready for inference
print(f"Model loaded: {type(model)}")
print(f"Device: {model._device_str}")
print(f"Score threshold: {model._score_threshold}")
```

### Loading Different Framework Models

```python
# Load Torch model
torch_config = DetectorConfigLoader.from_yaml(
    "config/torch/detector/fasterrcnn/config.yaml"
)
torch_loader = create_detector_model_loader("torch", torch_config)
torch_model = torch_loader.load_model()

# Load Ultralytics model
yolo_config = DetectorConfigLoader.from_yaml(
    "config/ultralytics/detector/yolov8n/config.yaml"
)
yolo_loader = create_detector_model_loader("ultralytics", yolo_config)
yolo_model = yolo_loader.load_model()
```

### Checking Available Loaders

```python
from src.detector_model_loader import get_available_detector_model_loaders

available = get_available_detector_model_loaders()
print(f"Available loaders: {available}")
# Output: ['torch', 'ultralytics']
```

### Using Loader Class Directly

```python
from src.detector_config.torch.config import TorchDetectorConfig
from src.detector_model_loader.torch.loader import TorchDetectorModelLoader

# Create config
config = TorchDetectorConfig(
    model_fn="fasterrcnn_resnet50_fpn",
    weights="DEFAULT",
    device="cuda",
    score_threshold=0.7
)

# Use loader directly
loader = TorchDetectorModelLoader(config)
model = loader.load_model()
```

## Adding New Loaders

Follow these steps to add a loader for a new framework:

### Step 1: Create Loader Class

Create `src/detector_model_loader/<framework>/loader.py`:

```python
from typing import Any

from src.detector_config.<framework>.config import MyFrameworkDetectorConfig
from src.detector_model_loader.base_detector_model_loader import BaseDetectorModelLoader
from src.detector_model_loader.detector_model_loader_registry import detector_model_loader_registry


@detector_model_loader_registry.register(
    "myframework",
    description="Detector model loader for My Framework"
)
class MyFrameworkDetectorModelLoader(BaseDetectorModelLoader):
    """
    My Framework implementation of detector model loader.

    Given a MyFrameworkDetectorConfig, this loader instantiates
    the appropriate model.
    """

    def load_model(self) -> Any:
        """
        Instantiate and return a My Framework detector model.

        Returns:
            An instance of My Framework model.
        """
        cfg = self.config

        # Type check
        if not isinstance(cfg, MyFrameworkDetectorConfig):
            raise TypeError(
                f"MyFrameworkDetectorModelLoader received unsupported config type: {type(cfg)}"
            )

        # Get parameters from config
        params = cfg.get_init_params()

        # Load model using your framework's API
        model = my_framework.load_model(
            model_name=params["model_name"],
            device=params["device"],
            # ... other parameters
        )

        # Attach metadata for inference
        model._threshold = params.get("threshold", 0.5)
        model._device = params["device"]

        return model

    @classmethod
    def get_name(cls) -> str:
        """Return the name of this loader."""
        return "myframework"
```

### Step 2: Create __init__.py

Create `src/detector_model_loader/<framework>/__init__.py`:

```python
from src.detector_model_loader.myframework.loader import MyFrameworkDetectorModelLoader

__all__ = ["MyFrameworkDetectorModelLoader"]
```

### Step 3: Register in Main __init__.py

Import your loader in `src/detector_model_loader/__init__.py`:

```python
# Import concrete implementations so they register themselves
from src.detector_model_loader.ultralytics.loader import UltralyticsDetectorModelLoader  # noqa: F401
from src.detector_model_loader.torch.loader import TorchDetectorModelLoader  # noqa: F401
from src.detector_model_loader.myframework.loader import MyFrameworkDetectorModelLoader  # noqa: F401
```

### Step 4: Implement Required Methods

Your loader class must implement:

1. **`__init__(self, config: BaseDetectorConfig)`** - Usually inherited from base class
2. **`load_model(self) -> Any`** - Main method to load and return model
3. **`get_name(cls) -> str`** - Class method returning loader name

### Step 5: Test Your Implementation

```python
from src.detector_config import DetectorConfigLoader
from src.detector_model_loader import (
    create_detector_model_loader,
    get_available_detector_model_loaders
)

# Test registration
available = get_available_detector_model_loaders()
assert "myframework" in available

# Test loading
config = DetectorConfigLoader.from_yaml(
    "config/myframework/detector/mymodel/config.yaml"
)
loader = create_detector_model_loader("myframework", config)
model = loader.load_model()

# Test model
print(f"Model type: {type(model)}")
print(f"Model device: {model._device}")
```

## Best Practices

### 1. Type Checking

Always validate the config type:
```python
if not isinstance(cfg, MyFrameworkDetectorConfig):
    raise TypeError(f"Expected MyFrameworkDetectorConfig, got {type(cfg)}")
```

### 2. Error Handling

Provide informative error messages:
```python
if not hasattr(my_framework, model_fn_name):
    raise ValueError(
        f"My Framework has no model '{model_fn_name}'. "
        f"Check your 'model_init_name' in config.yaml."
    )
```

### 3. Device Handling

Support multiple device types:
```python
device_str = params.get("device", "cpu")
device = torch.device(device_str)  # For PyTorch
# or
device = params.get("device", "cpu")  # For frameworks with string devices
```

### 4. Weight Loading

Support both built-in and custom weights:
```python
if params.get("weights_path"):
    # Custom weights
    model = framework.load_model(model_name, weights=None)
    model.load_weights(params["weights_path"])
else:
    # Built-in weights
    model = framework.load_model(model_name, weights=params["weights"])
```

### 5. Model Preparation

Put model in appropriate mode:
```python
model.eval()  # For inference
# or
model.train()  # If you support training mode
```

### 6. Metadata Attachment

Attach useful runtime information:
```python
model._score_threshold = params.get("score_threshold", 0.5)
model._device = device_str
model._model_name = params["model_name"]
```

### 7. Lazy Loading

Consider lazy loading for efficiency:
```python
def __init__(self, config):
    super().__init__(config)
    self._model = None

def load_model(self):
    if self._model is None:
        self._model = self._load_model_internal()
    return self._model
```

## Common Patterns

### Pattern 1: Conditional Initialization

```python
# Only pass non-None parameters
kwargs = {}
for key in ["param1", "param2", "param3"]:
    value = params.get(key)
    if value is not None:
        kwargs[key] = value

model = framework.Model(**kwargs)
```

### Pattern 2: Device Migration

```python
# PyTorch style
device = torch.device(params["device"])
model.to(device)

# or framework-specific style
model.to_device(params["device"])
```

### Pattern 3: Post-Processing Setup

```python
# Attach post-processing parameters
model._score_threshold = params.get("score_threshold", 0.5)
model._nms_threshold = params.get("nms_threshold", 0.45)
model._max_detections = params.get("max_detections", 100)
```

## Troubleshooting

### Loader Not Found
```
ValueError: Unknown detector model loader 'myloader'
```
**Solution**: Ensure you imported your loader in `__init__.py` and used the `@detector_model_loader_registry.register()` decorator.

### Wrong Config Type
```
TypeError: MyLoader received unsupported config type: TorchDetectorConfig
```
**Solution**: Ensure you're passing the correct config type to the loader. Each loader expects a specific config type.

### Model Not Found
```
ValueError: torchvision.models.detection has no attribute 'invalid_model'
```
**Solution**: Check the `model_init_name` in your config file matches available models in the framework.

### Weight Loading Error
```
RuntimeError: Error loading state_dict
```
**Solution**: Ensure weight file exists, is compatible with model architecture, and device has enough memory.

### Device Error
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Use CPU: `device: cpu`
- Enable FP16: `half: true`
- Use smaller model variant

## Integration Example

Here's a complete example showing integration with the config module:

```python
from pathlib import Path
from src.detector_config import DetectorConfigLoader
from src.detector_model_loader import create_detector_model_loader

# Define config path
config_path = "config/torch/detector/fasterrcnn/config.yaml"

# Load configuration
config = DetectorConfigLoader.from_yaml(config_path)
print(f"Loaded config for: {config.model_fn}")

# Create model loader
loader = create_detector_model_loader(
    loader_name="torch",
    config=config
)
print(f"Created loader: {loader.get_name()}")

# Load model
model = loader.load_model()
print(f"Model loaded on device: {model._device_str}")

# Model is ready for inference
# Use with your detection pipeline
```

## Related Documentation

- [Detector Config README](../detector_config/README.md) - How to create and configure detector configs
- [Tracker Model Loader README](../tracker_model_loader/README.md) - Similar pattern for tracker loaders
- [Main README](../../README.md) - Overall project documentation
