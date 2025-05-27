"""
LLM communication utilities for AHCAgent CLI.

This module provides utilities for communicating with LLM APIs using LiteLLM.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import litellm

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1


class LLMClient:
    """
    Client for communicating with LLM APIs using LiteLLM.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            config: Configuration dictionary
                - provider: LLM provider (default: from env var AHCAGENT_LLM_PROVIDER or "openai")
                - model: LLM model (default: from env var AHCAGENT_LLM_MODEL or "o4-mini")
                - api_key: API key (default: from env var based on provider)
                - temperature: Temperature (default: 0.2)
                - max_tokens: Maximum tokens (default: 4096)
                - timeout: Timeout in seconds (default: 60)
                - max_retries: Maximum number of retries (default: 2)
                - retry_delay_seconds: Delay between retries in seconds (default: 1)
        """
        if config is None:
            config = {}

        self.config = config

        # Get provider from config or env var or default
        self.provider = self.config.get("provider") or os.environ.get("AHCAGENT_LLM_PROVIDER") or "openai"

        # Get model from config or env var or default
        self.model = self.config.get("model") or os.environ.get("AHCAGENT_LLM_MODEL") or "o4-mini"

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

        # Get retry parameters
        self.max_retries = self.config.get("max_retries", DEFAULT_MAX_RETRIES)
        self.retry_delay_seconds = self.config.get("retry_delay_seconds", DEFAULT_RETRY_DELAY)

        # Configure litellm
        litellm.set_verbose = self.config.get("verbose", False)

        # Set API key and base URL if provided
        if self.api_key:
            os.environ[f"{self.model.upper()}_API_KEY"] = self.api_key

        # Validate model and provider
        try:
            # litellm.validate_params(model=self.model) # これはパラメータの型チェックが主
            # モデルが実際に利用可能かを確認するより直接的な方法として model_cost を使う
            # これにより、モデルが存在しない場合にエラーが発生することを期待
            if self.provider == "litellm":  # litellm provider の場合はモデル名だけで良い
                litellm.model_cost[self.model]
            else:  # 特定のプロバイダーの場合は model を 'provider/model' の形にするか、litellm が解釈できるようにする
                # ここでは単純化のため、モデル名がプロバイダープレフィックスを持つか、
                # litellm が直接解決できることを期待する。
                # より厳密には litellm.get_model_info(self.model) なども使えるが、
                # model_cost が存在チェックとして機能する。
                qualified_model_name = f"{self.provider}/{self.model}" if not self.model.startswith(self.provider + "/") else self.model
                if qualified_model_name in litellm.model_cost or self.model in litellm.model_cost:
                    pass  # OK
                else:
                    # もし provider が OpenAI 互換エンドポイントを指す場合、モデル検証は難しい
                    # ここでは litellm が知っているモデルかどうかを基本とする
                    raise KeyError(f"Model '{self.model}' (or '{qualified_model_name}') not found in litellm.model_cost.")

        except KeyError as e:
            logger.warning(f"Model validation failed for {self.model} (KeyError): {e!s}. Proceeding with caution.")
        except RuntimeError as e:
            logger.warning(f"Model validation failed for {self.model} (Unexpected Error): {e!s}. Proceeding with caution.")

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
                "timeout": kwargs.get("timeout", self.timeout),
            }

            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value

            # Log request
            logger.debug(f"Sending request to LLM: {self.model}")

            # Send request with retry logic
            current_retry = 0
            while True:
                try:
                    response = await litellm.acompletion(**params)
                    text = response.choices[0].message.content
                    logger.debug(f"Received response from LLM: {len(text)} chars")
                    return text
                except Exception as e:
                    logger.error(f"Error generating text (attempt {current_retry + 1}/{self.max_retries + 1}): {e!s}")
                    if current_retry < self.max_retries:
                        current_retry += 1
                        logger.info(f"Retrying in {self.retry_delay_seconds} seconds...")
                        await asyncio.sleep(self.retry_delay_seconds)
                    else:
                        logger.error("Max retries reached. Raising exception.")
                        raise

        except Exception as e:
            logger.error(f"Error generating text: {e!s}")
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
            json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
            json_text_content = json_match.group(1).strip() if json_match else text.strip()
            return json.loads(json_text_content)

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e!s}")
            logger.error(f"Raw text: {text}")

            # Try to fix common JSON errors
            try:
                # Replace single quotes with double quotes
                fixed_text = text.replace("'", '"')

                # Add missing quotes around keys
                fixed_text = re.sub(r"(\s*)(\w+)(\s*):(\s*)", r'\1"\2"\3:\4', fixed_text)

                # Parse fixed JSON
                result = json.loads(fixed_text)

                logger.info("Successfully fixed and parsed JSON")

                return result

            except json.JSONDecodeError as fix_err:
                logger.error(f"Failed to fix JSON after attempting common corrections: {fix_err!s}")
                # Raise the original error 'e' to provide context of the first failure
                raise ValueError(f"LLM did not return valid JSON: {e!s}") from e

        except (TypeError, AttributeError, IndexError, ValueError) as e:
            logger.error(f"Error processing JSON response: {e!s}")
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

                logger.warning(f"Retry {retries}/{max_retries} after error: {e!s}")

                # Exponential backoff
                await asyncio.sleep(2**retries)

        logger.error(f"Failed after {max_retries} retries: {last_error!s}")
        raise last_error
