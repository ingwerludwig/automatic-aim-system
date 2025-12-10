"""Main configuration loader for detectors."""

from typing import Dict, Any, List
import yaml

from src.detector_config.base_detector_config import BaseDetectorConfig, DetectorConfigFactory

# Re-export for convenience
__all__ = [
    "BaseDetectorConfig",
    "DetectorConfigFactory",
    "DetectorConfigLoader",
]


class DetectorConfigLoader:
    """Main class to load detector configurations from YAML."""

    @staticmethod
    def from_yaml(config_path: str) -> BaseDetectorConfig:
        """Load detector configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            raise ValueError("Empty configuration file")

        # Extract detector type
        detector_type = config_data.get('detector_type')
        if not detector_type:
            raise ValueError("Missing 'detector_type' in configuration")

        # Remove detector_type from config dict as it's not a init parameter
        config_dict = {k: v for k, v in config_data.items() if k != 'detector_type'}

        # Use factory to create appropriate config
        return DetectorConfigFactory.create(detector_type, config_dict)

    @staticmethod
    def get_available_detectors() -> List[str]:
        """Get list of available detector types."""
        return DetectorConfigFactory.get_available_detectors()