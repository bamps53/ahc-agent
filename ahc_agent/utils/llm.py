"""
LLM communication utilities for AHCAgent.

This module provides utilities for communicating with LLM APIs using LiteLLM.
"""

import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
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
                    logger.warning(
                        f"Model '{self.model}' with provider '{self.provider}' not found in litellm.model_cost. "
                        f"This may cause issues if the model is not supported by litellm."
                    )
        except Exception as e:
            logger.warning(f"Error validating model '{self.model}' with provider '{self.provider}': {e!s}")
            logger.warning("Continuing anyway, but this may cause issues if the model is not supported by litellm.")

        # ワークスペースディレクトリ(外部から設定される)
        self._workspace_dir = None

        logger.info(f"Initialized LLM client with provider: {self.provider}, model: {self.model}")

    def set_workspace_dir(self, workspace_dir):
        """
        ワークスペースディレクトリを設定する

        Args:
            workspace_dir: ワークスペースディレクトリのパス
        """
        if workspace_dir:
            self._workspace_dir = workspace_dir

    def _ensure_log_directory(self, resolved_workspace_dir_or_none=None):
        """
        Ensures the log directory exists based on a precedence of configurations.

        Args:
            resolved_workspace_dir_or_none: The workspace directory explicitly passed to
                                            generate/generate_json or set via set_workspace_dir().
                                            Can be None.

        Returns:
            Path to the log directory, or None if logging is disabled or directory creation fails.
        """
        if os.getenv("AHCAGENT_LLM_LOGGING_DISABLED", "false").lower() == "true":
            logger.debug("LLMClient: Logging is disabled by AHCAGENT_LLM_LOGGING_DISABLED.")
            return None

        base_dir_for_logs = None

        # Priority 1: Explicit workspace_dir from method arguments or client instance
        if resolved_workspace_dir_or_none is not None:
            base_dir_for_logs = Path(resolved_workspace_dir_or_none)
            logger.debug(f"LLMClient: Using explicit workspace directory for logs: {base_dir_for_logs}")
            print(f"[_ensure_log_directory] Using explicit workspace_dir: {resolved_workspace_dir_or_none}")
        else:
            # Priority 2: Test mode temp workspace (if no explicit workspace was resolved)
            test_mode_temp_workspace = os.getenv("AHCAGENT_TEST_MODE_TEMP_WORKSPACE")
            if test_mode_temp_workspace:
                base_dir_for_logs = Path(test_mode_temp_workspace)
                print(f"[_ensure_log_directory] Using AHCAGENT_TEST_MODE_TEMP_WORKSPACE: {test_mode_temp_workspace}")
                logger.debug(f"LLMClient: Using test mode temp workspace (AHCAGENT_TEST_MODE_TEMP_WORKSPACE) for logs: {base_dir_for_logs}")
            else:
                # Priority 3: Fallback to CWD (should be rare if app configures SolveService properly or _workspace_dir is set)
                logger.warning("LLMClient: Workspace directory not resolved. Falling back to CWD for logs.")
                base_dir_for_logs = Path.cwd()
                print(f"[_ensure_log_directory] Falling back to CWD: {base_dir_for_logs}")

        log_dir = base_dir_for_logs / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"LLMClient: Ensured log directory exists at {log_dir}")
            print(f"[_ensure_log_directory] Final log_dir: {log_dir}")
            return log_dir
        except OSError as e:
            logger.error(f"LLMClient: Error creating log directory {log_dir}: {e!s}")
            return None

    def _save_llm_log(self, log_dir, prompt, response, params=None, error=None):
        """
        Saves the LLM call details to a log file.
        """
        if os.getenv("AHCAGENT_LLM_LOGGING_DISABLED", "false").lower() == "true":
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = log_dir / f"llm_call_{timestamp}.json"

        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": self.model,
            "provider": self.provider,
            "prompt": prompt,
            "response": response,
            "params": params,
            "error": error,
        }

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"LLM call log saved to {log_file}")
            logger.info(f"LLM logs will be saved to: {log_dir}")
        except Exception as e:
            logger.error(f"Error saving LLM call log: {e!s}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: Prompt text
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            Generated text
        """
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

        # Resolve workspace directory for logging
        # Priority 1: workspace_dir from kwargs
        # Priority 2: self._workspace_dir (set via set_workspace_dir or constructor)
        resolved_workspace_dir = kwargs.get("workspace_dir", self._workspace_dir)
        log_dir = self._ensure_log_directory(resolved_workspace_dir)

        response_text = None
        error_msg = None

        try:
            # Send request with retry logic
            current_retry = 0
            while True:
                try:
                    response = await litellm.acompletion(**params)
                    response_text = response.choices[0].message.content
                    logger.debug(f"Received response from LLM: {len(response_text)} chars")
                    break
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating text (attempt {current_retry + 1}/{self.max_retries + 1}): {e!s}")
                    if current_retry < self.max_retries:
                        current_retry += 1
                        logger.info(f"Retrying in {self.retry_delay_seconds} seconds...")
                        await asyncio.sleep(self.retry_delay_seconds)
                    else:
                        logger.error("Max retries reached. Raising exception.")
                        raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating text: {e!s}")
            raise
        finally:
            # Save log regardless of success or failure
            if log_dir:  # Check if log_dir was successfully created
                self._save_llm_log(
                    log_dir=log_dir,
                    prompt=prompt,
                    response=response_text,
                    params={k: v for k, v in params.items() if k != "messages"},
                    error=error_msg,
                )

        return response_text

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

        # Get workspace directory from kwargs
        # workspace_dir = kwargs.get("workspace_dir") # log_dir is not used in this method
        # log_dir = self._ensure_log_directory(workspace_dir) # log_dir is not used in this method

        result = None
        # error_msg = None # error_msg is not used in this method

        try:
            # Extract JSON from text
            # First, try to find JSON block
            json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
            json_text_content = json_match.group(1).strip() if json_match else text.strip()
            return json.loads(json_text_content)

        except json.JSONDecodeError as e:
            # error_msg = str(e) # error_msg is not used
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
                # error_msg = f"Failed to fix JSON: {fix_err}" # error_msg is not used
                logger.error(f"Failed to fix JSON after attempting common corrections: {fix_err!s}")
                # Raise the original error 'e' to provide context of the first failure
                raise ValueError(f"LLM did not return valid JSON: {e!s}") from e

        except (TypeError, AttributeError, IndexError, ValueError) as e:
            # error_msg = str(e) # error_msg is not used in this method
            logger.error(f"Error processing JSON response: {e!s}")
            raise
        finally:
            # The LLM call is logged by self.generate().
            # Additional logging for JSON parsing status, if needed, should be handled differently.
            pass
