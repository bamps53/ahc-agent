"""
CLIコマンドにおけるワークスペースコンテキストの扱いに関するテスト。
"""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
import pytest
import yaml

from ahc_agent.cli import cli


class TestCLIWorkspaceContext:
    """
    CLIコマンドにおけるワークスペースコンテキストの扱いに関するテスト。
    """

    @pytest.fixture()
    def runner(self):
        """
        テスト用のCLIランナーを作成する。
        """
        return CliRunner()

    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.KnowledgeBase")
    @patch("ahc_agent.cli.StatusService")
    def test_status_command_with_workspace_arg(self, MockStatusService, MockKnowledgeBase, MockConfig, runner, tmp_path):
        """
        status コマンドがワークスペース引数を受け取り、そのパスからコンテキストを決定できることをテスト。
        """
        # モックの設定
        mock_global_config_instance = MockConfig.return_value
        mock_global_config_instance.get.return_value = {}

        mock_workspace_dir = tmp_path / "mock_status_ws"
        mock_workspace_dir.mkdir(exist_ok=True)

        # ahc_config.yamlファイルを作成
        dummy_ws_cfg_path = mock_workspace_dir / "ahc_config.yaml"
        with open(dummy_ws_cfg_path, "w") as f:
            yaml.dump({"contest_id": "status_contest"}, f)

        # ワークスペース固有の設定をモック
        mock_ws_config = MockConfig.return_value
        mock_ws_config.get.side_effect = lambda key, default=None: "status_contest" if key == "contest_id" else default
        MockConfig.side_effect = [mock_global_config_instance, mock_ws_config]

        mock_kb_instance = MockKnowledgeBase.return_value
        mock_status_service_instance = MockStatusService.return_value
        mock_status_service_instance.get_status.return_value = [
            "=== Session Status ===",
            "Session ID: test-session",
            "Status: Complete",
        ]

        # ワークスペース引数を指定してstatusコマンドを実行
        result = runner.invoke(cli, ["status", "--workspace", str(mock_workspace_dir), "test-session"])

        assert result.exit_code == 0
        # KnowledgeBaseが正しいワークスペースパスとproblem_idで初期化されたことを確認
        MockKnowledgeBase.assert_called_once_with(str(mock_workspace_dir), problem_id="status_contest")
        # StatusServiceが正しく初期化されたことを確認
        MockStatusService.assert_called_once_with(mock_global_config_instance, mock_kb_instance)
        # get_statusが正しく呼び出されたことを確認
        mock_status_service_instance.get_status.assert_called_once_with(session_id="test-session", watch=False)
        # 出力を確認
        assert "Session ID: test-session" in result.output
        assert "Status: Complete" in result.output

    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.KnowledgeBase")
    @patch("ahc_agent.cli.SubmitService")
    def test_submit_command_with_workspace_arg(self, MockSubmitService, MockKnowledgeBase, MockConfig, runner, tmp_path):
        """
        submit コマンドがワークスペース引数を受け取り、そのパスからコンテキストを決定できることをテスト。
        """
        # モックの設定
        mock_global_config_instance = MockConfig.return_value
        mock_global_config_instance.get.return_value = {}

        mock_workspace_dir = tmp_path / "mock_submit_ws"
        mock_workspace_dir.mkdir(exist_ok=True)

        # ahc_config.yamlファイルを作成
        dummy_ws_cfg_path = mock_workspace_dir / "ahc_config.yaml"
        with open(dummy_ws_cfg_path, "w") as f:
            yaml.dump({"contest_id": "submit_contest"}, f)

        # ワークスペース固有の設定をモック
        mock_ws_config = MockConfig.return_value
        mock_ws_config.get.side_effect = lambda key, default=None: "submit_contest" if key == "contest_id" else default
        MockConfig.side_effect = [mock_global_config_instance, mock_ws_config]

        mock_kb_instance = MockKnowledgeBase.return_value
        mock_submit_service_instance = MockSubmitService.return_value
        output_file_name = "solution.cpp"
        expected_output_path_str = str(Path(tmp_path) / output_file_name)

        mock_submit_service_instance.submit_solution.return_value = {
            "session_id": "test-session",
            "output_path": expected_output_path_str,
            "solution_code": "// Test code",
            "score": 100,
        }

        # ワークスペース引数を指定してsubmitコマンドを実行
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["submit", "--workspace", str(mock_workspace_dir), "test-session", "--output", output_file_name])

            assert result.exit_code == 0
            # KnowledgeBaseが正しいワークスペースパスとproblem_idで初期化されたことを確認
            MockKnowledgeBase.assert_called_once_with(str(mock_workspace_dir), problem_id="submit_contest")
            # SubmitServiceが正しく初期化されたことを確認
            MockSubmitService.assert_called_once_with(mock_global_config_instance, mock_kb_instance)
            # submit_solutionが正しく呼び出されたことを確認
            mock_submit_service_instance.submit_solution.assert_called_once_with(session_id="test-session", output_path=output_file_name)
            # 出力を確認
            assert f"Best solution for session test-session (Score: 100) written to {expected_output_path_str}" in result.output

    @patch("ahc_agent.cli.os")
    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.KnowledgeBase")
    @patch("ahc_agent.cli.StatusService")
    def test_status_command_with_current_dir_as_workspace(self, MockStatusService, MockKnowledgeBase, MockConfig, MockOS, runner, tmp_path):
        """
        status コマンドがワークスペース引数を受け取らない場合、カレントディレクトリをワークスペースとして使用することをテスト。
        """
        # モックの設定
        mock_global_config_instance = MockConfig.return_value
        mock_global_config_instance.get.return_value = {}

        mock_workspace_dir = tmp_path / "mock_current_dir_ws"
        mock_workspace_dir.mkdir(exist_ok=True)

        # カレントディレクトリのモック
        MockOS.getcwd.return_value = str(mock_workspace_dir)

        # ahc_config.yamlファイルを作成
        dummy_ws_cfg_path = mock_workspace_dir / "ahc_config.yaml"
        with open(dummy_ws_cfg_path, "w") as f:
            yaml.dump({"contest_id": "current_dir_contest"}, f)

        # ワークスペース固有の設定をモック
        mock_ws_config_instance = MockConfig.return_value
        mock_ws_config_instance.get.side_effect = lambda key, default=None: "current_dir_contest" if key == "contest_id" else default
        MockConfig.side_effect = [mock_global_config_instance, mock_ws_config_instance]

        mock_kb_instance = MockKnowledgeBase.return_value
        mock_status_service_instance = MockStatusService.return_value
        mock_status_service_instance.get_status.return_value = [
            "=== Session Status ===",
            "Session ID: test-session",
            "Status: Complete",
        ]

        # ワークスペース引数を指定せずにstatusコマンドを実行
        result = runner.invoke(cli, ["status", "test-session"])

        assert result.exit_code == 0
        # カレントディレクトリが取得されたことを確認
        MockOS.getcwd.assert_called_once()
        # KnowledgeBaseが正しいワークスペースパスとproblem_idで初期化されたことを確認
        MockKnowledgeBase.assert_called_once_with(str(mock_workspace_dir), problem_id="current_dir_contest")
        # StatusServiceが正しく初期化されたことを確認
        MockStatusService.assert_called_once_with(mock_global_config_instance, mock_kb_instance)
        # get_statusが正しく呼び出されたことを確認
        mock_status_service_instance.get_status.assert_called_once_with(session_id="test-session", watch=False)
        # 出力を確認
        assert "Session ID: test-session" in result.output
        assert "Status: Complete" in result.output

    @patch("ahc_agent.cli.os")
    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.KnowledgeBase")
    @patch("ahc_agent.cli.SubmitService")
    def test_submit_command_with_current_dir_as_workspace(self, MockSubmitService, MockKnowledgeBase, MockConfig, MockOS, runner, tmp_path):
        """
        submit コマンドがワークスペース引数を受け取らない場合、カレントディレクトリをワークスペースとして使用することをテスト。
        """
        # モックの設定
        mock_global_config_instance = MockConfig.return_value
        mock_global_config_instance.get.return_value = {}

        mock_workspace_dir = tmp_path / "mock_current_dir_ws"
        mock_workspace_dir.mkdir(exist_ok=True)

        # カレントディレクトリのモック
        MockOS.getcwd.return_value = str(mock_workspace_dir)

        # ahc_config.yamlファイルを作成
        dummy_ws_cfg_path = mock_workspace_dir / "ahc_config.yaml"
        with open(dummy_ws_cfg_path, "w") as f:
            yaml.dump({"contest_id": "current_dir_contest"}, f)

        # ワークスペース固有の設定をモック
        mock_ws_config_instance = MockConfig.return_value
        mock_ws_config_instance.get.side_effect = lambda key, default=None: "current_dir_contest" if key == "contest_id" else default
        MockConfig.side_effect = [mock_global_config_instance, mock_ws_config_instance]

        mock_kb_instance = MockKnowledgeBase.return_value
        mock_submit_service_instance = MockSubmitService.return_value
        output_file_name = "solution.cpp"
        expected_output_path_str = str(Path(tmp_path) / output_file_name)

        mock_submit_service_instance.submit_solution.return_value = {
            "session_id": "test-session",
            "output_path": expected_output_path_str,
            "solution_code": "// Test code",
            "score": 100,
        }

        # ワークスペース引数を指定せずにsubmitコマンドを実行
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["submit", "test-session", "--output", output_file_name])

            assert result.exit_code == 0
            # カレントディレクトリが取得されたことを確認
            MockOS.getcwd.assert_called_once()
            # KnowledgeBaseが正しいワークスペースパスとproblem_idで初期化されたことを確認
            MockKnowledgeBase.assert_called_once_with(str(mock_workspace_dir), problem_id="current_dir_contest")
            # SubmitServiceが正しく初期化されたことを確認
            MockSubmitService.assert_called_once_with(mock_global_config_instance, mock_kb_instance)
            # submit_solutionが正しく呼び出されたことを確認
            mock_submit_service_instance.submit_solution.assert_called_once_with(session_id="test-session", output_path=output_file_name)
            # 出力を確認
            assert f"Best solution for session test-session (Score: 100) written to {expected_output_path_str}" in result.output
