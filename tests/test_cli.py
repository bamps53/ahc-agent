"""
Unit tests for CLI module.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner
import pytest
import yaml

# Main CLI entrypoint
from ahc_agent.cli import cli
from ahc_agent.config import Config


class TestCLI:
    """
    Tests for CLI module.
    """

    @pytest.fixture()
    def runner(self):
        """
        Create a CLI runner for testing.
        """
        return CliRunner()

    def test_cli_help(self, runner):
        """
        Test CLI help command.
        """
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AHCAgent" in result.output
        assert "init" in result.output
        assert "solve" in result.output

    @patch("ahc_agent.cli.InitService")
    def test_init_command(self, MockInitService, runner):
        """Test basic init command flow."""
        mock_init_service_instance = MockInitService.return_value
        mock_init_service_instance.initialize_project = MagicMock(
            return_value={
                "project_dir": "/mocked/workspace/ahc001",
                "config_file_path": "/mocked/workspace/ahc001/config.yaml",
                "contest_id": "ahc001",
            }
        )

        result = runner.invoke(cli, ["init", "ahc001", "--workspace", "./workspace"])

        assert result.exit_code == 0
        MockInitService.assert_called_once_with()

        mock_init_service_instance.initialize_project.assert_called_once_with(
            contest_id="ahc001", workspace="./workspace", html_file=None, url="https://atcoder.jp/contests/ahc001/tasks/ahc001_a"
        )
        assert "Project for contest 'ahc001' initialized successfully" in result.output
        assert "/mocked/workspace/ahc001" in result.output

    @patch("ahc_agent.cli.InitService")
    def test_init_default_workspace(self, MockInitService, runner: CliRunner, tmp_path: Path):
        """Test init command with default workspace (uses contest_id as dir name)."""
        contest_id = "ahc999"

        mock_init_service_instance = MockInitService.return_value
        # Simulate the service creating paths relative to the execution directory (tmp_path here)
        # When runner.isolated_filesystem is used, CWD becomes that isolated dir.
        # InitService's default workspace logic (os.getcwd() / contest_id) will use this.
        # So, expected_project_dir should be relative to the isolated CWD.
        # For simplicity, we'll assume the service returns an absolute-like path for the mock.
        expected_project_dir = Path(tmp_path) / contest_id
        expected_config_path = expected_project_dir / "config.yaml"

        mock_init_service_instance.initialize_project = MagicMock(
            return_value={
                "project_dir": str(expected_project_dir),
                "config_file_path": str(expected_config_path),
                "contest_id": contest_id,
            }
        )

        with runner.isolated_filesystem(temp_dir=tmp_path) as _:
            result = runner.invoke(cli, ["init", contest_id])

            assert result.exit_code == 0
            mock_init_service_instance.initialize_project.assert_called_once_with(
                contest_id=contest_id, workspace=None, html_file=None, url=f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
            )
            assert f"Project for contest '{contest_id}' initialized successfully" in result.output
            # The output path in the message comes from the mocked return value
            assert str(expected_project_dir) in result.output

    @patch("ahc_agent.cli.InitService")
    def test_init_with_workspace(self, MockInitService, runner: CliRunner, tmp_path: Path):
        """Test init command with a specified workspace."""
        contest_id = "ahc998"
        workspace_name = "my_custom_workspace"

        mock_init_service_instance = MockInitService.return_value
        expected_project_dir = tmp_path / workspace_name  # Service returns absolute-like path
        expected_config_path = expected_project_dir / "config.yaml"

        mock_init_service_instance.initialize_project = MagicMock(
            return_value={
                "project_dir": str(expected_project_dir),
                "config_file_path": str(expected_config_path),
                "contest_id": contest_id,
            }
        )

        with runner.isolated_filesystem(temp_dir=tmp_path) as _:
            result = runner.invoke(cli, ["init", contest_id, "--workspace", workspace_name])

            assert result.exit_code == 0
            mock_init_service_instance.initialize_project.assert_called_once_with(
                contest_id=contest_id, workspace=workspace_name, html_file=None, url=f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
            )
            assert f"Project for contest '{contest_id}' initialized successfully" in result.output
            assert str(expected_project_dir) in result.output

    # Docker関連の機能を削除したため、このテストは不要になりました

    @patch("ahc_agent.cli.InitService")
    def test_init_command_with_existing_target_dir_as_file(self, MockInitService, runner, tmp_path):
        contest_id = "ahc888"

        # This test assumes that the CLI's InitService will handle the file existence check.
        # The CLI command catches RuntimeError from the service.
        mock_init_service_instance = MockInitService.return_value
        # Simulate the service raising an error because the target path is a file
        target_path_as_file_simulated_by_service = tmp_path / contest_id  # Path service would try to create
        mock_init_service_instance.initialize_project.side_effect = RuntimeError(
            f"Error creating project directory: '{target_path_as_file_simulated_by_service}' already exists and is a file or non-empty directory."
        )

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            # Create the conflicting file within the isolated directory
            actual_conflicting_file = Path(td) / contest_id
            actual_conflicting_file.write_text("This is a file, not a directory.")

            result = runner.invoke(cli, ["init", contest_id])  # Workspace is implicitly contest_id in td

            assert result.exit_code == 1
            assert "Error during project initialization" in result.output
            # The error message from the service (containing the path) should be in the output
            assert str(target_path_as_file_simulated_by_service) in result.output

    @patch("ahc_agent.cli.DockerManager")
    @patch("ahc_agent.cli.LLMClient")
    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.KnowledgeBase")
    @patch("ahc_agent.cli.SolveService")
    def test_solve_command(self, MockSolveService, MockKnowledgeBase, MockCliConfig, MockLLMClient, MockDockerManager, runner):
        mock_global_config_instance = MagicMock(spec=Config)
        mock_global_config_instance.get.return_value = {}

        mock_workspace_config_instance = MagicMock(spec=Config)
        mock_workspace_config_instance.get.side_effect = lambda key, default=None: {
            "contest_id": "test_contest",
            "workspace.base_dir": "/mocked_workspace_path",
            "llm": {},
            "docker": {},
        }.get(key, default)
        mock_workspace_config_instance.config_file_path = "/mocked_workspace_path/config.yaml"

        # solve コマンド内で Config がインスタンス化される際のモックを設定
        MockCliConfig.return_value = mock_workspace_config_instance

        mock_llm_instance = MockLLMClient.return_value
        mock_docker_instance = MockDockerManager.return_value
        mock_kb_instance = MockKnowledgeBase.return_value

        mock_solve_service_instance = MockSolveService.return_value
        mock_solve_service_instance.run_solve_session = AsyncMock(return_value=None)

        with runner.isolated_filesystem() as temp_dir:
            workspace_path = Path(temp_dir)
            problem_file = workspace_path / "problem.md"
            problem_file.write_text("# Test Problem")

            config_file = workspace_path / "config.yaml"
            config_file.write_text(yaml.dump({"contest_id": "test_contest"}))

            result = runner.invoke(cli, ["solve", str(workspace_path)])

            assert result.exit_code == 0
            # solve コマンド内で Config が config_file_path を引数に1回だけ呼び出されることを確認
            MockCliConfig.assert_called_once_with(str(config_file))
            MockLLMClient.assert_called_once_with({})
            MockDockerManager.assert_called_once_with({})
            MockKnowledgeBase.assert_called_once_with(str(workspace_path), problem_id="test_contest")
            MockSolveService.assert_called_once_with(mock_llm_instance, mock_docker_instance, mock_workspace_config_instance, mock_kb_instance)
            mock_solve_service_instance.run_solve_session.assert_called_once()
            call_args = mock_solve_service_instance.run_solve_session.call_args
            assert call_args[1]["problem_text"] == "# Test Problem"
            assert call_args[1]["session_id"] is None
            assert call_args[1]["interactive"] is False
            assert f"Solving problem in workspace: {workspace_path}" in result.output

    @patch("ahc_agent.cli.DockerManager")
    @patch("ahc_agent.cli.LLMClient")
    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.KnowledgeBase")
    @patch("ahc_agent.cli.SolveService")
    def test_solve_command_with_workspace(
        self, MockSolveService, MockKnowledgeBase, MockCliConfig, MockLLMClient, MockDockerManager, runner, tmp_path
    ):
        contest_id = "ahc999"
        workspace_dir = tmp_path / contest_id
        workspace_dir.mkdir()

        problem_text_content = "# AHC999 Problem"
        problem_file = workspace_dir / "problem.md"
        problem_file.write_text(problem_text_content)

        config_file_content = {"contest_id": contest_id}  # Simplified
        config_file = workspace_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_file_content, f)

        mock_global_config_instance = MagicMock(spec=Config)
        mock_global_config_instance.get.return_value = {}
        mock_workspace_config_instance = MagicMock(spec=Config)
        mock_workspace_config_instance.get.side_effect = lambda key, default=None: {
            "contest_id": contest_id,
            "workspace.base_dir": str(workspace_dir),
            "llm": {},
            "docker": {},
        }.get(key, default)
        mock_workspace_config_instance.config_file_path = str(config_file)
        # In solve command tests, Config is instantiated once inside solve()
        MockCliConfig.return_value = mock_workspace_config_instance

        mock_llm_instance = MockLLMClient.return_value
        mock_docker_instance = MockDockerManager.return_value
        mock_kb_instance = MockKnowledgeBase.return_value

        mock_solve_service_instance = MockSolveService.return_value
        mock_solve_service_instance.run_solve_session = AsyncMock()

        result = runner.invoke(cli, ["solve", str(workspace_dir)])

        assert result.exit_code == 0
        mock_workspace_config_instance.set.assert_called_with("workspace.base_dir", str(workspace_dir))
        MockKnowledgeBase.assert_called_once_with(str(workspace_dir), problem_id=contest_id)
        MockSolveService.assert_called_once_with(mock_llm_instance, mock_docker_instance, mock_workspace_config_instance, mock_kb_instance)
        mock_solve_service_instance.run_solve_session.assert_called_once_with(problem_text=problem_text_content, session_id=None, interactive=False)
        assert f"Solving problem in workspace: {workspace_dir}" in result.output

    @patch("ahc_agent.cli.DockerManager")
    @patch("ahc_agent.cli.LLMClient")
    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.KnowledgeBase")
    @patch("ahc_agent.cli.SolveService")
    def test_solve_command_uses_tools_in_files_simplified(
        self, MockSolveService, MockKnowledgeBase, MockCliConfig, MockLLMClient, MockDockerManager, runner, tmp_path
    ):
        workspace_dir = tmp_path / "ahc_test_workspace_tools"
        workspace_dir.mkdir()
        tools_dir = workspace_dir / "tools"
        tools_dir.mkdir()
        tools_in_dir = tools_dir / "in"
        tools_in_dir.mkdir()
        (tools_in_dir / "test01.txt").write_text("input data for test01")

        problem_file = workspace_dir / "problem.md"
        problem_file.write_text("# Test Problem with Tools")
        config_file = workspace_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"contest_id": "tools_test_contest"}, f)

        mock_global_config_instance = MagicMock(spec=Config)
        mock_global_config_instance.get.return_value = {}
        mock_workspace_config_instance = MagicMock(spec=Config)
        mock_workspace_config_instance.get.side_effect = lambda key, default=None: {
            "contest_id": "tools_test_contest",
            "workspace.base_dir": str(workspace_dir),
            "llm": {},
            "docker": {},
        }.get(key, default)
        mock_workspace_config_instance.config_file_path = str(config_file)
        # In solve command tests, Config is instantiated once inside solve()
        MockCliConfig.return_value = mock_workspace_config_instance

        mock_solve_service_instance = MockSolveService.return_value
        mock_solve_service_instance.run_solve_session = AsyncMock()

        result = runner.invoke(cli, ["solve", str(workspace_dir)])

        assert result.exit_code == 0
        mock_solve_service_instance.run_solve_session.assert_called_once()
