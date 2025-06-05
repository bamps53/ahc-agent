"""
Unit tests for LLM client.
"""

import json
import os
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel
import pytest

from ahc_agent.utils.llm import LLMClient


class TestLLMClient:
    """
    Tests for LLMClient.
    """

    @pytest.fixture()
    def llm_client(self):
        """
        Create a LLMClient instance for testing.
        """
        config = {"provider": "litellm", "model": "o4-mini", "api_key": "test_key", "temperature": 0.7, "max_tokens": 1000}
        return LLMClient(config)

    @patch("ahc_agent.utils.llm.litellm.acompletion")
    @pytest.mark.asyncio()
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
        assert call_args["model"] == "o4-mini"
        assert call_args["messages"][0]["content"] == "Test prompt"
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 1000

    @patch("ahc_agent.utils.llm.LLMClient.generate")  # Mocking the internal self.generate call
    @pytest.mark.asyncio()
    async def test_generate_json_structured(self, mock_llm_generate, llm_client):  # Renamed mock_generate to mock_llm_generate for clarity
        """
        Test generate_json method with Pydantic model.
        """
        # Expected data and Pydantic model instance
        expected_data = TestModel(name="Test Name", value=123)
        # Mock the response from self.generate to return a JSON string
        mock_llm_generate.return_value = expected_data.model_dump_json()

        # Call generate_json with the Pydantic model
        response_model_instance = await llm_client.generate_json(  # llm_client is now the fixture
            "Test prompt for structured data", pydantic_model=TestModel
        )

        # Check result: type and content
        assert isinstance(response_model_instance, TestModel)
        assert response_model_instance.name == expected_data.name
        assert response_model_instance.value == expected_data.value

        # Check call arguments to self.generate
        mock_llm_generate.assert_called_once()
        call_args = mock_llm_generate.call_args[0]  # Positional arguments
        call_kwargs = mock_llm_generate.call_args[1]  # Keyword arguments

        assert "Test prompt for structured data" in call_args[0]  # The prompt
        expected_schema = TestModel.model_json_schema()
        assert json.dumps(expected_schema) in call_args[0]  # The schema string in the prompt

        # Verify that response_format with schema was passed to self.generate
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_object"
        assert call_kwargs["response_format"]["schema"] == expected_schema

    @patch("ahc_agent.utils.llm.LLMClient.generate")  # Mocking the internal self.generate call
    @pytest.mark.asyncio()
    async def test_generate_json_invalid_pydantic(self, mock_llm_generate, llm_client):  # llm_client is fixture, mock_llm_generate
        """
        Test generate_json method with a response that's valid JSON but doesn't match the Pydantic model.
        """
        # Mock response: valid JSON, but incorrect structure for TestModel (e.g., missing 'value')
        mock_llm_generate.return_value = '{"name": "Test Name Only"}'

        with pytest.raises(ValueError) as excinfo:
            await llm_client.generate_json("Test prompt for invalid structure", pydantic_model=TestModel)  # llm_client is fixture

        assert "LLM response failed Pydantic validation for TestModel" in str(excinfo.value)
        assert "Field required" in str(excinfo.value)  # Pydantic's error for missing field

    @patch("ahc_agent.utils.llm.LLMClient.generate")  # Mocking the internal self.generate call
    @pytest.mark.asyncio()
    async def test_generate_json_not_json_string(self, mock_llm_generate, llm_client):  # llm_client is fixture, mock_llm_generate
        """
        Test generate_json method with a response that is not a valid JSON string at all.
        """
        # Mock response: not a JSON string
        mock_llm_generate.return_value = "This is not JSON."

        with pytest.raises(ValueError) as excinfo:
            await llm_client.generate_json("Test prompt for non-JSON response", pydantic_model=TestModel)  # llm_client is fixture

        # Depending on how model_validate_json handles fundamentally malformed strings,
        # the error might be directly from Pydantic or a wrapped one.
        # Pydantic v2's model_validate_json is robust and will raise a ValidationError.
        assert "LLM response failed Pydantic validation for TestModel" in str(excinfo.value)
        # The underlying error from Pydantic will likely mention JSON decoding issues.
        # Check for Pydantic's specific json_invalid type or "Expecting value" for broader compatibility.
        error_string = str(excinfo.value).lower()
        assert "json_invalid" in error_string or "expecting value" in error_string

    @patch("ahc_agent.utils.llm.litellm.acompletion")  # This test is for the raw generate, so litellm.acompletion is correct mock
    @pytest.mark.asyncio()
    async def test_generate_with_error(self, mock_litellm_acompletion, llm_client):  # Renamed for clarity
        """
        Test generate method with error.
        """
        # Mock error
        mock_litellm_acompletion.side_effect = Exception("Test error")

        # Call generate
        with pytest.raises(Exception) as excinfo:
            await llm_client.generate("Test prompt")

        # Check error
        assert "Test error" in str(excinfo.value)

    @patch("ahc_agent.utils.llm.litellm.acompletion")  # Correct mock for raw generate
    @pytest.mark.asyncio()
    async def test_generate_with_retry(self, mock_litellm_acompletion, llm_client):  # Renamed for clarity
        """
        Test generate method with retry.
        """
        # Mock responses
        mock_litellm_acompletion.side_effect = [
            Exception("Rate limit exceeded"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Test response"))]),
        ]

        # Call generate
        response = await llm_client.generate("Test prompt")

        # Check result
        assert response == "Test response"

        # Check call count
        assert mock_litellm_acompletion.call_count == 2

    @patch("ahc_agent.utils.llm.litellm.acompletion")  # Correct mock for raw generate
    @pytest.mark.asyncio()
    async def test_generate_with_max_retries(self, mock_litellm_acompletion, llm_client):  # Renamed for clarity
        """
        Test generate method with max retries exceeded.
        """
        # Mock error
        mock_litellm_acompletion.side_effect = Exception("Rate limit exceeded")

        # Call generate
        with pytest.raises(Exception) as excinfo:
            await llm_client.generate("Test prompt")

        # Check error
        assert "Rate limit exceeded" in str(excinfo.value)

        # Check call count
        assert mock_litellm_acompletion.call_count == llm_client.max_retries + 1

    def test_init_with_env_vars(self):
        """
        Test initialization with environment variables.
        """
        # Set environment variables
        os.environ["LITELLM_API_KEY"] = "env_test_key"

        # Create client with minimal config
        config = {"provider": "litellm", "model": "o4-mini"}
        client = LLMClient(config)

        # Check that API key was loaded from environment
        assert client.api_key == "env_test_key"

        # Clean up
        del os.environ["LITELLM_API_KEY"]

    def test_init_with_invalid_config(self, caplog):
        """
        Test LLMClient initialization with invalid config (invalid model).
        """
        invalid_config = {"provider": "openai", "model": "this-model-does-not-exist-for-sure", "api_key": "sk-dummykey"}
        client = LLMClient(config=invalid_config)
        assert client is not None  # クライアントが初期化されることを確認

        # ログメッセージの確認
        captured_logs = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert any("not found in litellm.model_cost" in log for log in captured_logs)
        assert any("this-model-does-not-exist-for-sure" in log for log in captured_logs)

    def test_init_with_custom_validator(self):
        pass  # メソッド本体が欠落していたため、一時的に pass を追加

    @pytest.mark.asyncio
    async def test_ensure_log_directory(self):
        client = LLMClient()

        # 一時ディレクトリを使用してテスト
        with tempfile.TemporaryDirectory() as temp_dir:
            # ワークスペースディレクトリを設定
            log_dir = client._ensure_log_directory(temp_dir)

            # ログディレクトリが作成されたことを確認
            assert log_dir.exists()
            assert log_dir.is_dir()
            assert log_dir.name == "logs"
            assert log_dir.parent == Path(temp_dir)

    @pytest.mark.asyncio
    async def test_save_llm_log(self):
        client = LLMClient()

        # 一時ディレクトリを使用してテスト
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            log_dir.mkdir(exist_ok=True)

            # テスト用のデータ
            prompt = "テスト用プロンプト"
            response = "テスト用レスポンス"
            params = {"temperature": 0.2}

            # ログを保存
            client._save_llm_log(log_dir, prompt, response, params)

            # ログファイルが作成されたことを確認
            log_files = list(log_dir.glob("llm_call_*.json"))
            assert len(log_files) == 1

            # ログファイルの内容を確認
            with open(log_files[0], encoding="utf-8") as f:
                log_data = json.load(f)

            assert log_data["prompt"] == prompt
            assert log_data["response"] == response
            assert log_data["params"] == params
            assert "timestamp" in log_data
            assert log_data["model"] == client.model
            assert log_data["provider"] == client.provider

    @pytest.mark.asyncio
    async def test_set_workspace_dir(self):
        client = LLMClient()

        # ワークスペースディレクトリを設定
        test_dir = "/test/workspace"
        client.set_workspace_dir(test_dir)

        # 設定されたことを確認
        assert client._workspace_dir == test_dir

    @pytest.mark.asyncio
    async def test_generate_with_workspace_dir(self):
        client = LLMClient()

        # モックレスポンスを設定
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "テストレスポンス"

        # 一時ディレクトリを使用してテスト
        with tempfile.TemporaryDirectory() as temp_dir, patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion, patch.object(
            client, "_save_llm_log"
        ) as mock_save_log:
            mock_acompletion.return_value = mock_response

            # ワークスペースディレクトリを設定してテキスト生成
            response = await client.generate("テストプロンプト", workspace_dir=temp_dir)

            # 結果を確認
            assert response == "テストレスポンス"

            # ログが保存されたことを確認
            mock_save_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_instance_workspace_dir(self):
        client = LLMClient()

        # モックレスポンスを設定
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "テストレスポンス"

        # 一時ディレクトリを使用してテスト
        with tempfile.TemporaryDirectory() as temp_dir:
            # インスタンス変数にワークスペースディレクトリを設定
            client.set_workspace_dir(temp_dir)

            # litellm.acompletionをモック化
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion, patch.object(client, "_save_llm_log") as mock_save_log:
                mock_acompletion.return_value = mock_response

                response = await client.generate("テストプロンプト")

                # 結果を確認
                assert response == "テストレスポンス"

                # ログが保存されたことを確認
                mock_save_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_json_with_workspace_dir(self, llm_client):  # Use fixture
        # Expected data and Pydantic model instance
        expected_data = TestModel(name="WS Test", value=456)

        # Mock the internal 'generate' call within 'generate_json'
        with patch.object(llm_client, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = expected_data.model_dump_json()

            with tempfile.TemporaryDirectory() as temp_dir:
                # Call generate_json with the Pydantic model and workspace_dir
                response_model_instance = await llm_client.generate_json(
                    "Test prompt for workspace_dir", pydantic_model=TestModel, workspace_dir=temp_dir
                )

                # Check result
                assert isinstance(response_model_instance, TestModel)
                assert response_model_instance.name == expected_data.name
                assert response_model_instance.value == expected_data.value

                # Check that 'generate' was called and 'workspace_dir' was in its kwargs
                mock_generate.assert_called_once()
                call_kwargs = mock_generate.call_args[1]
                assert call_kwargs["workspace_dir"] == temp_dir
                # Also check that response_format was passed correctly
                assert "response_format" in call_kwargs
                assert call_kwargs["response_format"]["type"] == "json_object"
                assert call_kwargs["response_format"]["schema"] == TestModel.model_json_schema()

    @pytest.mark.asyncio
    async def test_generate_with_logging_disabled(self):
        client = LLMClient()

        # モックレスポンスを設定
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "テストレスポンス"

        # 一時ディレクトリを使用してテスト
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {"AHCAGENT_LLM_LOGGING_DISABLED": "true"}), patch(
            "litellm.acompletion", new_callable=AsyncMock
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            # ワークスペースディレクトリを設定してテキスト生成
            response = await client.generate("テストプロンプト", workspace_dir=temp_dir)

            # 結果を確認
            assert response == "テストレスポンス"

            # ログディレクトリが空であることを確認
            log_dir = Path(temp_dir) / "logs"
            assert not log_dir.exists() or not any(log_dir.iterdir())


class TestModel(BaseModel):
    name: str
    value: int
