"""Main configuration loader for trackers."""

from typing import Dict, Any, List
import yaml

from src.tracker_config.base_tracker_config import BaseTrackerConfig, TrackerConfigFactory

# Re-export for convenience
__all__ = [
    "BaseTrackerConfig",
    "TrackerConfigFactory",
    "TrackerConfigLoader",
]


class TrackerConfigLoader:
    """Main class to load tracker configurations from YAML."""

    @staticmethod
    def from_yaml(config_path: str) -> BaseTrackerConfig:
        """Load tracker configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            raise ValueError("Empty configuration file")

        # Extract tracker type
        tracker_type = config_data.get('tracker_type')
        if not tracker_type:
            raise ValueError("Missing 'tracker_type' in configuration")

        # Remove tracker_type from config dict as it's not a init parameter
        config_dict = {k: v for k, v in config_data.items() if k != 'tracker_type'}

        # Use factory to create appropriate config
        return TrackerConfigFactory.create(tracker_type, config_dict)

    @staticmethod
    def get_available_trackers() -> List[str]:
        """Get list of available tracker types."""
        return TrackerConfigFactory.get_available_trackers()