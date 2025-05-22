"""
Configuration management for AHCAgent CLI.

This module provides utilities for managing configuration.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

from .utils.file_io import read_yaml, write_yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # LLM settings
    "llm": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.2,
        "max_tokens": 4096,
        "timeout": 60
    },
    
    # Docker settings
    "docker": {
        "image": "mcr.microsoft.com/devcontainers/rust:1-1-bullseye",
        "mount_path": "/workspace",
        "timeout": 300
    },
    
    # Evolution settings
    "evolution": {
        "max_generations": 30,
        "population_size": 10,
        "time_limit_seconds": 1800,
        "score_plateau_generations": 5
    },
    
    # C++ compilation settings
    "cpp": {
        "compiler": "g++",
        "flags": "-std=c++17 -O2 -Wall",
        "execution_timeout": 10
    },
    
    # Workspace settings
    "workspace": {
        "base_dir": "~/ahc_workspace",
        "keep_history": True,
        "max_sessions": 10
    }
}

class Config:
    """
    Configuration manager for AHCAgent CLI.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file (default: None, use environment variable or default)
        """
        self.config_file = config_file or os.environ.get("AHCAGENT_CONFIG")
        self.config = DEFAULT_CONFIG.copy()
        
        # Load configuration from file if specified
        if self.config_file and os.path.exists(self.config_file):
            self._load_config()
        
        # Override with environment variables
        self._override_from_env()
        
        logger.debug(f"Initialized configuration: {self.config}")
    
    def _load_config(self) -> None:
        """
        Load configuration from file.
        """
        try:
            logger.info(f"Loading configuration from {self.config_file}")
            file_config = read_yaml(self.config_file)
            
            # Update config with file values
            self._update_config(self.config, file_config)
            
            logger.debug(f"Loaded configuration: {self.config}")
        
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_file}: {str(e)}")
            logger.warning("Using default configuration")
    
    def _update_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Update configuration recursively.
        
        Args:
            target: Target configuration
            source: Source configuration
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_config(target[key], value)
            else:
                # Update value
                target[key] = value
    
    def _override_from_env(self) -> None:
        """
        Override configuration with environment variables.
        """
        # LLM settings
        if os.environ.get("AHCAGENT_LLM_PROVIDER"):
            self.config["llm"]["provider"] = os.environ.get("AHCAGENT_LLM_PROVIDER")
        
        if os.environ.get("AHCAGENT_LLM_MODEL"):
            self.config["llm"]["model"] = os.environ.get("AHCAGENT_LLM_MODEL")
        
        # Docker settings
        if os.environ.get("AHCAGENT_DOCKER_IMAGE"):
            self.config["docker"]["image"] = os.environ.get("AHCAGENT_DOCKER_IMAGE")
        
        if os.environ.get("AHCAGENT_NO_DOCKER") == "1":
            self.config["docker"]["enabled"] = False
        else:
            self.config["docker"]["enabled"] = True
        
        # Workspace settings
        if os.environ.get("AHCAGENT_WORKSPACE"):
            self.config["workspace"]["base_dir"] = os.environ.get("AHCAGENT_WORKSPACE")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            
            return value
        
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (dot notation for nested keys)
            value: Configuration value
        """
        keys = key.split(".")
        target = self.config
        
        # Navigate to the target dictionary
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            elif not isinstance(target[k], dict):
                target[k] = {}
            
            target = target[k]
        
        # Set the value
        target[keys[-1]] = value
    
    def save(self, config_file: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_file: Path to configuration file (default: None, use self.config_file)
        """
        file_path = config_file or self.config_file
        
        if not file_path:
            raise ValueError("No configuration file specified")
        
        try:
            logger.info(f"Saving configuration to {file_path}")
            write_yaml(file_path, self.config)
            
            # Update config_file if a new file was specified
            if config_file:
                self.config_file = config_file
            
            logger.debug(f"Saved configuration: {self.config}")
        
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
            raise
    
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
        self.config = DEFAULT_CONFIG.copy()
        self._update_config(self.config, config)
        
        logger.debug(f"Imported configuration: {self.config}")
