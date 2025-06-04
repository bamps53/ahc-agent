"""
Unit tests for LLM client.
"""

import json
import os
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

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

    @patch("ahc_agent.utils.llm.litellm.acompletion")
    @pytest.mark.asyncio()
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
        assert call_args["model"] == "o4-mini"
        assert "Test prompt" in call_args["messages"][0]["content"]
        assert "JSON" in call_args["messages"][0]["content"]

    @patch("ahc_agent.utils.llm.litellm.acompletion")
    @pytest.mark.asyncio()
    async def test_generate_json_invalid(self, mock_completion, llm_client):
        """
        Test generate_json method with invalid JSON response.
        """
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid JSON"
        mock_completion.return_value = mock_response

        # Call generate_json
        with pytest.raises(ValueError) as excinfo:
            await llm_client.generate_json("Test prompt")
        assert "LLM did not return valid JSON" in str(excinfo.value)
        assert "Expecting value: line 1 column 1 (char 0)" in str(excinfo.value)  # Underlying json.JSONDecodeError message

    @patch("ahc_agent.utils.llm.litellm.acompletion")
    @pytest.mark.asyncio()
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

    @patch("ahc_agent.utils.llm.litellm.acompletion")
    @pytest.mark.asyncio()
    async def test_generate_with_retry(self, mock_completion, llm_client):
        """
        Test generate method with retry.
        """
        # Mock responses
        mock_completion.side_effect = [
            Exception("Rate limit exceeded"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Test response"))]),
        ]

        # Call generate
        response = await llm_client.generate("Test prompt")

        # Check result
        assert response == "Test response"

        # Check call count
        assert mock_completion.call_count == 2

    @patch("ahc_agent.utils.llm.litellm.acompletion")
    @pytest.mark.asyncio()
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
    async def test_generate_json_with_workspace_dir(self):
        client = LLMClient()

        # モックレスポンスを設定
        json_content = '{"key": "value"}'

        # 一時ディレクトリを使用してテスト
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(client, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = json_content

            # ワークスペースディレクトリを設定してJSON生成
            response = await client.generate_json("テストプロンプト", workspace_dir=temp_dir)

            # 結果を確認
            assert response == {"key": "value"}

            # generateが呼ばれたことを確認
            mock_generate.assert_called_once()
            # workspace_dirが渡されたことを確認
            assert mock_generate.call_args[1]["workspace_dir"] == temp_dir

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
