"""
LLM communication utilities for AHCAgent.

This module provides utilities for communicating with LLM APIs using LiteLLM.
"""

import asyncio
import datetime
import json  # Keep json for _save_llm_log and for dumping schema in prompt
import logging
import os
from pathlib import Path

# import re # re is confirmed to be no longer needed
from typing import Any, Dict, Optional, Type  # Added Type

import litellm
from pydantic import BaseModel, ValidationError  # Added BaseModel and ValidationError

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

    async def generate_json(self, prompt: str, pydantic_model: Type[BaseModel], **kwargs) -> BaseModel:
        """
        Generate JSON using the LLM and parse it into a Pydantic model.

        Args:
            prompt: Prompt text.
            pydantic_model: The Pydantic model to validate and parse the response into.
            **kwargs: Additional parameters to pass to the LLM.
                      Includes 'workspace_dir' for logging location.

        Returns:
            An instance of the provided Pydantic model.

        Raises:
            ValueError: If the LLM response is not valid JSON or does not match the Pydantic model schema.
        """
        # Get the JSON schema from the Pydantic model
        json_schema = pydantic_model.model_json_schema()

        # Add JSON instruction to prompt, including the schema
        # Using json.dumps to ensure the schema is correctly formatted as a string within the prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON that conforms to the following schema:\n{json.dumps(json_schema)}"

        # Prepare parameters for litellm.acompletion, including response_format
        current_params = kwargs.copy()  # Avoid modifying the original kwargs
        current_params["response_format"] = {"type": "json_object"}

        # Resolve workspace directory for logging
        resolved_workspace_dir = current_params.get("workspace_dir", self._workspace_dir)
        log_dir = self._ensure_log_directory(resolved_workspace_dir)

        raw_llm_response_content = None
        error_for_log = None

        try:
            # Call self.generate, passing the modified prompt and response_format.
            # self.generate will handle the actual call to litellm.acompletion,
            # retries, and basic logging of the LLM interaction.
            raw_llm_response_content = await self.generate(json_prompt, **current_params)

            # Validate and parse the JSON response using the Pydantic model
            response_model_instance = pydantic_model.model_validate_json(raw_llm_response_content)
            logger.debug(f"Successfully validated and parsed JSON response into {pydantic_model.__name__}")
            return response_model_instance

        except ValidationError as e:
            error_for_log = f"Pydantic validation error: {e}"
            logger.error(error_for_log)
            logger.error(f"Raw LLM response: {raw_llm_response_content}")
            # Re-raise as ValueError for consistent error handling by the caller
            raise ValueError(
                f"LLM response failed Pydantic validation for {pydantic_model.__name__}: {e!s}. Raw response: {raw_llm_response_content}"
            ) from e
        except Exception as e:  # Catch other potential errors (e.g., from self.generate)
            error_for_log = str(e)
            logger.error(f"Error in generate_json for {pydantic_model.__name__}: {e!s}")
            if raw_llm_response_content:  # Log raw response if available
                logger.error(f"Raw LLM response (if available): {raw_llm_response_content}")
            # Re-raise the exception. If it's a ValueError from self.generate(), it's already informative.
            # Otherwise, wrap it to indicate failure in this specific method.
            if not isinstance(e, ValueError):
                raise ValueError(f"Failed to generate or parse JSON for {pydantic_model.__name__}: {e!s}") from e
            # Is already a ValueError, likely from self.generate() or the ValidationError above
            raise
        finally:
            # Log the attempt if an error occurred and we have a log_dir.
            # self.generate already logs its own execution. This log is specifically for generate_json's context,
            # especially if validation fails after a successful generation.
            if error_for_log and log_dir:
                # Filter params for logging to avoid duplication or overly large objects
                params_for_log = {
                    k: v
                    for k, v in current_params.items()
                    if k not in ["messages", "api_key", "schema"]  # schema can be large
                }
                # Ensure response_format is logged as a string if it's an object
                if "response_format" in params_for_log and isinstance(params_for_log["response_format"], dict):
                    params_for_log["response_format"] = json.dumps(params_for_log["response_format"])

                self._save_llm_log(
                    log_dir=log_dir,
                    prompt=json_prompt,
                    response=raw_llm_response_content,  # Log the raw response from LLM
                    params=params_for_log,
                    error=error_for_log,
                )
