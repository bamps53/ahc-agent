"""
Default configuration for AHCAgent CLI.
"""

import os
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """
    Configuration manager for AHCAgent CLI.
    """

    def __init__(self, config_path_or_dict: Optional[Union[str, Dict[str, Any]]] = None):
        """
        Initialize configuration.

        Args:
            config_path_or_dict: Path to configuration file or configuration dictionary
        """
        # Default configuration
        self.config = {
            "llm": {"provider": "litellm", "model": "gpt-4", "temperature": 0.7, "max_tokens": 4000, "timeout": 60},
            "docker": {
                "enabled": True,
                "image": "mcr.microsoft.com/devcontainers/rust:1-1-bullseye",
                "cpp_compiler": "g++",
                "cpp_flags": "-std=c++17 -O2 -Wall",
            },
            "workspace": {"base_dir": "~/ahc_workspace"},
            "evolution": {
                "max_generations": 30,
                "population_size": 10,
                "time_limit_seconds": 1800,
                "score_plateau_generations": 5,
            },
            "analyzer": {"detailed_analysis": True},
            "strategist": {"detailed_strategy": True},
            "debugger": {"execution_timeout": 10},
            "problem_logic": {"test_cases_count": 3},
            "batch": {"parallel": 1, "output_dir": "~/ahc_batch"},
        }

        # Load configuration from file or dictionary
        if config_path_or_dict:
            if isinstance(config_path_or_dict, str):
                # Load from file
                try:
                    with open(config_path_or_dict) as f:
                        loaded_config = yaml.safe_load(f)
                        if loaded_config:
                            self._merge_configs(self.config, loaded_config)
                except (OSError, yaml.YAMLError) as e:
                    print(f"Error loading configuration from {config_path_or_dict}: {e!s}")
            elif isinstance(config_path_or_dict, dict):
                # Load from dictionary
                self._merge_configs(self.config, config_path_or_dict)

        # Override with environment variables
        self._load_from_env()

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

    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.

        Environment variables should be in the format AHC_SECTION_KEY.
        For example, AHC_LLM_MODEL will override llm.model.
        """
        for env_key, env_value in os.environ.items():
            if env_key.startswith("AHC_"):
                # Convert environment variable name to config key
                # AHC_LLM_MODEL -> llm.model
                config_key = env_key[4:].lower().replace("_", ".")

                # Convert value to appropriate type
                if env_value.lower() == "true":
                    value = True
                elif env_value.lower() == "false":
                    value = False
                elif env_value.isdigit():
                    value = int(env_value)
                elif env_value.replace(".", "", 1).isdigit() and env_value.count(".") == 1:
                    value = float(env_value)
                else:
                    value = env_value

                # Set config value
                self.set(config_key, value)
