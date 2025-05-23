"""
Unit tests for Config class.
"""

import os
import yaml
import pytest
import tempfile
from unittest.mock import patch

from ahc_agent_cli.config import Config

class TestConfig:
    """
    Tests for Config class.
    """
    
    @pytest.fixture
    def sample_config(self):
        """
        Create a sample configuration dictionary.
        """
        return {
            "llm": {
                "provider": "litellm",
                "model": "gpt-4",
                "temperature": 0.7
            },
            "docker": {
                "enabled": True,
                "image": "test-image"
            },
            "workspace": {
                "base_dir": "/tmp/workspace"
            }
        }
    
    def test_init_empty(self):
        """
        Test initialization with empty config.
        """
        config = Config()
        
        # Check default values
        assert config.get("llm.provider") == "litellm"
        assert config.get("llm.model") == "gpt-4"
        assert config.get("docker.enabled") is True
        assert config.get("workspace.base_dir") == "~/ahc_workspace"
    
    def test_init_with_file(self, sample_config):
        """
        Test initialization with config file.
        """
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
            yaml.dump(sample_config, temp)
            temp_path = temp.name
        
        try:
            # Initialize config with file
            config = Config(temp_path)
            
            # Check values
            assert config.get("llm.provider") == "litellm"
            assert config.get("llm.model") == "gpt-4"
            assert config.get("llm.temperature") == 0.7
            assert config.get("docker.enabled") is True
            assert config.get("docker.image") == "test-image"
            assert config.get("workspace.base_dir") == "/tmp/workspace"
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_init_with_dict(self, sample_config):
        """
        Test initialization with config dictionary.
        """
        # Initialize config with dictionary
        config = Config(sample_config)
        
        # Check values
        assert config.get("llm.provider") == "litellm"
        assert config.get("llm.model") == "gpt-4"
        assert config.get("llm.temperature") == 0.7
        assert config.get("docker.enabled") is True
        assert config.get("docker.image") == "test-image"
        assert config.get("workspace.base_dir") == "/tmp/workspace"
    
    def test_get(self, sample_config):
        """
        Test get method.
        """
        # Initialize config
        config = Config(sample_config)
        
        # Test get with dot notation
        assert config.get("llm.provider") == "litellm"
        assert config.get("llm.model") == "gpt-4"
        assert config.get("docker.enabled") is True
        
        # Test get with dictionary
        assert config.get("llm") == {
            "provider": "litellm",
            "model": "gpt-4",
            "temperature": 0.7
        }
        
        # Test get with default value
        assert config.get("non_existent", "default") == "default"
        
        # Test get with non-existent key
        assert config.get("non_existent") is None
    
    def test_set(self, sample_config):
        """
        Test set method.
        """
        # Initialize config
        config = Config(sample_config)
        
        # Test set with dot notation
        config.set("llm.model", "gpt-3.5-turbo")
        assert config.get("llm.model") == "gpt-3.5-turbo"
        
        # Test set with new key
        config.set("new_key", "new_value")
        assert config.get("new_key") == "new_value"
        
        # Test set with nested new key
        config.set("new_section.new_key", "new_value")
        assert config.get("new_section.new_key") == "new_value"
        assert config.get("new_section") == {"new_key": "new_value"}
    
    def test_save_and_load(self, sample_config):
        """
        Test save and load methods.
        """
        # Initialize config
        config = Config(sample_config)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
        
        try:
            # Save config
            config.save(temp_path)
            
            # Load config
            new_config = Config(temp_path)
            
            # Check values
            assert new_config.get("llm.provider") == "litellm"
            assert new_config.get("llm.model") == "gpt-4"
            assert new_config.get("llm.temperature") == 0.7
            assert new_config.get("docker.enabled") is True
            assert new_config.get("docker.image") == "test-image"
            assert new_config.get("workspace.base_dir") == "/tmp/workspace"
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_export(self, sample_config):
        """
        Test export method.
        """
        # Initialize config
        config = Config(sample_config)
        
        # Export config
        exported = config.export()
        
        # Check exported config
        assert exported == sample_config
    
    def test_import_config(self, sample_config):
        """
        Test import_config method.
        """
        # Initialize empty config
        config = Config()
        
        # Import config
        config.import_config(sample_config)
        
        # Check values
        assert config.get("llm.provider") == "litellm"
        assert config.get("llm.model") == "gpt-4"
        assert config.get("llm.temperature") == 0.7
        assert config.get("docker.enabled") is True
        assert config.get("docker.image") == "test-image"
        assert config.get("workspace.base_dir") == "/tmp/workspace"
    
    @patch.dict(os.environ, {"AHC_LLM_MODEL": "gpt-3.5-turbo", "AHC_DOCKER_ENABLED": "false"})
    def test_env_vars(self):
        """
        Test environment variable overrides.
        """
        # Initialize config
        config = Config()
        
        # Check values from environment variables
        assert config.get("llm.model") == "gpt-3.5-turbo"
        assert config.get("docker.enabled") is False
    
    def test_merge_configs(self):
        """
        Test merging of configs.
        """
        # Create base config
        base_config = {
            "llm": {
                "provider": "litellm",
                "model": "gpt-4",
                "temperature": 0.7
            },
            "docker": {
                "enabled": True
            }
        }
        
        # Create override config
        override_config = {
            "llm": {
                "model": "gpt-3.5-turbo",
                "max_tokens": 1000
            },
            "new_section": {
                "new_key": "new_value"
            }
        }
        
        # Initialize config with base
        config = Config(base_config)
        
        # Import override
        config.import_config(override_config)
        
        # Check merged values
        assert config.get("llm.provider") == "litellm"  # Unchanged
        assert config.get("llm.model") == "gpt-3.5-turbo"  # Overridden
        assert config.get("llm.temperature") == 0.7  # Unchanged
        assert config.get("llm.max_tokens") == 1000  # Added
        assert config.get("docker.enabled") is True  # Unchanged
        assert config.get("new_section.new_key") == "new_value"  # Added
