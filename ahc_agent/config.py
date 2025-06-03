import os
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """
    Configuration manager for AHCAgent.
    """

    def __init__(self, config_path_or_dict: Optional[Union[str, Dict[str, Any]]] = None):
        """
        Initialize configuration.

        Args:
            config_path_or_dict: Path to configuration file or configuration dictionary
        """
        self.config_file_path: Optional[str] = None
        self.config: Dict[str, Any] = {}

        # Determine the path to the default configuration file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(base_dir, "default_config.yaml")

        # Load default configuration
        if os.path.exists(default_config_path):
            with open(default_config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # This case should ideally not happen if default_config.yaml is part of the package
            print(f"Warning: Default config file {default_config_path} not found. Initializing with empty config.")
            self.config = {}

        # Load user-provided configuration from file or dictionary
        if isinstance(config_path_or_dict, str):
            self.config_file_path = config_path_or_dict
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path) as f:
                    user_config = yaml.safe_load(f)
                self._merge_configs(self.config, user_config)
            else:
                print(f"Warning: Config file {self.config_file_path} not found. Using default settings.")
        elif isinstance(config_path_or_dict, dict):
            self._merge_configs(self.config, config_path_or_dict)
        elif config_path_or_dict is None:
            # If no config is provided, defaults from default_config.yaml are already loaded.
            pass

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key (dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Split key by dots
        keys = key.split(".")

        # Navigate through config
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key (dot notation for nested keys)
            value: Configuration value
        """
        # Split key by dots
        keys = key.split(".")

        # Navigate through config
        config = self.config
        for _, k in enumerate(keys[:-1]):
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]

        # Set value
        config[keys[-1]] = value

    def save(self, path: str) -> None:
        """
        Save configuration to a file.

        Args:
            path: Path to save configuration
        """
        try:
            with open(path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except (OSError, yaml.YAMLError) as e:
            print(f"Error saving configuration to {path}: {e!s}")

    def export(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def import_config(self, config: Dict[str, Any]) -> None:
        """
        Import configuration from dictionary.

        Args:
            config: Configuration dictionary
        """
        self._merge_configs(self.config, config)

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Merge override configuration into base configuration.

        Args:
            base: Base configuration
            override: Override configuration
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
