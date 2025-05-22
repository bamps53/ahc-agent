"""
LLM communication utilities for AHCAgent CLI.

This module provides utilities for communicating with LLM APIs using LiteLLM.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union

import litellm

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for communicating with LLM APIs using LiteLLM.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Configuration dictionary
                - provider: LLM provider (default: from env var AHCAGENT_LLM_PROVIDER or "openai")
                - model: LLM model (default: from env var AHCAGENT_LLM_MODEL or "gpt-4")
                - api_key: API key (default: from env var based on provider)
                - temperature: Temperature (default: 0.2)
                - max_tokens: Maximum tokens (default: 4096)
                - timeout: Timeout in seconds (default: 60)
        """
        self.config = config or {}
        
        # Get provider from config or env var or default
        self.provider = self.config.get("provider") or os.environ.get("AHCAGENT_LLM_PROVIDER") or "openai"
        
        # Get model from config or env var or default
        self.model = self.config.get("model") or os.environ.get("AHCAGENT_LLM_MODEL") or "gpt-4"
        
        # Get API key from config or env var based on provider
        api_key_env_var = f"{self.provider.upper()}_API_KEY"
        self.api_key = self.config.get("api_key") or os.environ.get(api_key_env_var)
        
        # Set API key in environment if not already set
        if self.api_key and not os.environ.get(api_key_env_var):
            os.environ[api_key_env_var] = self.api_key
        
        # Get other parameters
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.timeout = self.config.get("timeout", 60)
        
        # Configure litellm
        litellm.set_verbose = self.config.get("verbose", False)
        
        logger.info(f"Initialized LLM client with provider: {self.provider}, model: {self.model}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Prompt text
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Generated text
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "timeout": kwargs.get("timeout", self.timeout)
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value
            
            # Log request
            logger.debug(f"Sending request to LLM: {self.model}")
            
            # Send request
            response = await litellm.acompletion(**params)
            
            # Extract and return text
            text = response.choices[0].message.content
            
            logger.debug(f"Received response from LLM: {len(text)} chars")
            
            return text
        
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate JSON using the LLM.
        
        Args:
            prompt: Prompt text
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Generated JSON as dictionary
        """
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no other text."
        
        # Generate text
        text = await self.generate(json_prompt, **kwargs)
        
        try:
            # Extract JSON from text
            # First, try to find JSON block
            import re
            json_match = re.search(r'```json(.*?)```', text, re.DOTALL)
            
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                # If no JSON block found, use the entire text
                json_text = text.strip()
            
            # Parse JSON
            result = json.loads(json_text)
            
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            logger.error(f"Raw text: {text}")
            
            # Try to fix common JSON errors
            try:
                # Replace single quotes with double quotes
                fixed_text = text.replace("'", '"')
                
                # Add missing quotes around keys
                fixed_text = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', fixed_text)
                
                # Parse fixed JSON
                result = json.loads(fixed_text)
                
                logger.info("Successfully fixed and parsed JSON")
                
                return result
            
            except Exception:
                logger.error("Failed to fix JSON")
                raise ValueError(f"LLM did not return valid JSON: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error processing JSON response: {str(e)}")
            raise
    
    async def generate_with_retries(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        """
        Generate text with retries.
        
        Args:
            prompt: Prompt text
            max_retries: Maximum number of retries
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Generated text
        """
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                return await self.generate(prompt, **kwargs)
            
            except Exception as e:
                retries += 1
                last_error = e
                
                logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** retries)
        
        logger.error(f"Failed after {max_retries} retries: {str(last_error)}")
        raise last_error
