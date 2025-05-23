"""
Unit tests for LLM client.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from ahc_agent_cli.utils.llm import LLMClient

class TestLLMClient:
    """
    Tests for LLMClient.
    """
    
    @pytest.fixture
    def llm_client(self):
        """
        Create a LLMClient instance for testing.
        """
        config = {
            "provider": "litellm",
            "model": "gpt-4",
            "api_key": "test_key",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        return LLMClient(config)
    
    @patch("litellm.completion")
    async def test_generate(self, mock_completion, llm_client):
        """
        Test generate method.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response
        
        # Call generate
        response = await llm_client.generate("Test prompt")
        
        # Check result
        assert response == "Test response"
        
        # Check call arguments
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert call_args["messages"][0]["content"] == "Test prompt"
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 1000
    
    @patch("litellm.completion")
    async def test_generate_json(self, mock_completion, llm_client):
        """
        Test generate_json method.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value"}'
        mock_completion.return_value = mock_response
        
        # Call generate_json
        response = await llm_client.generate_json("Test prompt")
        
        # Check result
        assert response == {"key": "value"}
        
        # Check call arguments
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert "Test prompt" in call_args["messages"][0]["content"]
        assert "JSON" in call_args["messages"][0]["content"]
    
    @patch("litellm.completion")
    async def test_generate_json_invalid(self, mock_completion, llm_client):
        """
        Test generate_json method with invalid JSON response.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Invalid JSON'
        mock_completion.return_value = mock_response
        
        # Call generate_json
        with pytest.raises(json.JSONDecodeError):
            await llm_client.generate_json("Test prompt")
    
    @patch("litellm.completion")
    async def test_generate_with_error(self, mock_completion, llm_client):
        """
        Test generate method with error.
        """
        # Mock error
        mock_completion.side_effect = Exception("Test error")
        
        # Call generate
        with pytest.raises(Exception) as excinfo:
            await llm_client.generate("Test prompt")
        
        # Check error
        assert "Test error" in str(excinfo.value)
    
    @patch("litellm.completion")
    async def test_generate_with_retry(self, mock_completion, llm_client):
        """
        Test generate method with retry.
        """
        # Mock responses
        mock_completion.side_effect = [
            Exception("Rate limit exceeded"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Test response"))])
        ]
        
        # Call generate
        response = await llm_client.generate("Test prompt")
        
        # Check result
        assert response == "Test response"
        
        # Check call count
        assert mock_completion.call_count == 2
    
    @patch("litellm.completion")
    async def test_generate_with_max_retries(self, mock_completion, llm_client):
        """
        Test generate method with max retries exceeded.
        """
        # Mock error
        mock_completion.side_effect = Exception("Rate limit exceeded")
        
        # Call generate
        with pytest.raises(Exception) as excinfo:
            await llm_client.generate("Test prompt")
        
        # Check error
        assert "Rate limit exceeded" in str(excinfo.value)
        
        # Check call count
        assert mock_completion.call_count == llm_client.max_retries + 1
    
    def test_init_with_env_vars(self):
        """
        Test initialization with environment variables.
        """
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = "env_test_key"
        
        # Create client with minimal config
        config = {
            "provider": "litellm",
            "model": "gpt-4"
        }
        client = LLMClient(config)
        
        # Check that API key was loaded from environment
        assert client.api_key == "env_test_key"
        
        # Clean up
        del os.environ["OPENAI_API_KEY"]
    
    def test_init_with_invalid_config(self):
        """
        Test initialization with invalid config.
        """
        # Create client with invalid config
        config = {
            "provider": "invalid_provider",
            "model": "gpt-4"
        }
        
        # Check that initialization raises error
        with pytest.raises(ValueError) as excinfo:
            LLMClient(config)
        
        # Check error message
        assert "provider" in str(excinfo.value).lower()
