"""
Unit tests for CLI module.
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call 

from click.testing import CliRunner
import pytest
import yaml

# Main CLI entrypoint
from ahc_agent.cli import cli 
from ahc_agent.config import Config 
from ahc_agent.core.knowledge import KnowledgeBase


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
        assert "AHCAgent CLI" in result.output
        assert "init" in result.output
        assert "solve" in result.output
        assert "status" in result.output
        assert "submit" in result.output
        assert "batch" in result.output
        assert "docker" in result.output
        assert "config" in result.output


    @patch("ahc_agent.cli.InitService")
    def test_init_command(self, MockInitService, runner):
        """Test basic init command flow."""
        mock_init_service_instance = MockInitService.return_value
        mock_init_service_instance.initialize_project = MagicMock(
            return_value={
                "project_dir": "/mocked/workspace/ahc001",
                "config_file_path": "/mocked/workspace/ahc001/ahc_config.yaml",
                "contest_id": "ahc001",
            }
        )

        result = runner.invoke(cli, ["init", "ahc001", "--workspace", "./workspace"])
        
        assert result.exit_code == 0
        MockInitService.assert_called_once() 
        assert isinstance(MockInitService.call_args[0][0], Config)

        mock_init_service_instance.initialize_project.assert_called_once_with(
            contest_id="ahc001", template=None, docker_image=None, workspace="./workspace"
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
        expected_config_path = expected_project_dir / "ahc_config.yaml"

        mock_init_service_instance.initialize_project = MagicMock(
            return_value={
                "project_dir": str(expected_project_dir),
                "config_file_path": str(expected_config_path),
                "contest_id": contest_id,
            }
        )
        
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            result = runner.invoke(cli, ["init", contest_id]) 

            assert result.exit_code == 0
            mock_init_service_instance.initialize_project.assert_called_once_with(
                contest_id=contest_id, template=None, docker_image=None, workspace=None 
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
        expected_project_dir = tmp_path / workspace_name # Service returns absolute-like path
        expected_config_path = expected_project_dir / "ahc_config.yaml"

        mock_init_service_instance.initialize_project = MagicMock(
            return_value={
                "project_dir": str(expected_project_dir),
                "config_file_path": str(expected_config_path),
                "contest_id": contest_id,
            }
        )

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            result = runner.invoke(cli, ["init", contest_id, "--workspace", workspace_name])

            assert result.exit_code == 0
            mock_init_service_instance.initialize_project.assert_called_once_with(
                contest_id=contest_id, template=None, docker_image=None, workspace=workspace_name
            )
            assert f"Project for contest '{contest_id}' initialized successfully" in result.output
            assert str(expected_project_dir) in result.output


    @patch("ahc_agent.cli.InitService")
    def test_init_with_custom_template_and_image(self, MockInitService, runner: CliRunner, tmp_path: Path):
        contest_id = "ahc997"
        custom_template = "cpp_pro"
        custom_image = "my_cpp_env:1.0"

        mock_init_service_instance = MockInitService.return_value
        # Assuming default workspace naming if not provided
        expected_project_dir = Path(tmp_path) / contest_id 
        expected_config_path = expected_project_dir / "ahc_config.yaml"
        
        mock_init_service_instance.initialize_project = MagicMock(
            return_value={
                "project_dir": str(expected_project_dir),
                "config_file_path": str(expected_config_path),
                "contest_id": contest_id,
                "template": custom_template, 
                "docker_image": custom_image
            }
        )
        
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            result = runner.invoke(cli, ["init", contest_id, "--template", custom_template, "--docker-image", custom_image])

            assert result.exit_code == 0
            mock_init_service_instance.initialize_project.assert_called_once_with(
                contest_id=contest_id, template=custom_template, docker_image=custom_image, workspace=None
            )
            assert f"Project for contest '{contest_id}' initialized successfully" in result.output


    @patch("ahc_agent.cli.InitService")
    def test_init_command_with_existing_target_dir_as_file(self, MockInitService, runner, tmp_path):
        contest_id = "ahc888"
        
        # This test assumes that the CLI's InitService will handle the file existence check.
        # The CLI command catches RuntimeError from the service.
        mock_init_service_instance = MockInitService.return_value
        # Simulate the service raising an error because the target path is a file
        target_path_as_file_simulated_by_service = tmp_path / contest_id # Path service would try to create
        mock_init_service_instance.initialize_project.side_effect = RuntimeError(
            f"Error creating project directory: '{target_path_as_file_simulated_by_service}' already exists and is a file or non-empty directory."
        )

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            # Create the conflicting file within the isolated directory
            actual_conflicting_file = Path(td) / contest_id
            actual_conflicting_file.write_text("This is a file, not a directory.")

            result = runner.invoke(cli, ["init", contest_id]) # Workspace is implicitly contest_id in td

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
            "workspace.base_dir": "/mocked_workspace_path" 
        }.get(key, default)
        mock_workspace_config_instance.config_file_path = "/mocked_workspace_path/ahc_config.yaml"

        MockCliConfig.side_effect = [mock_global_config_instance, mock_workspace_config_instance]

        mock_llm_instance = MockLLMClient.return_value
        mock_docker_instance = MockDockerManager.return_value
        mock_kb_instance = MockKnowledgeBase.return_value
        
        mock_solve_service_instance = MockSolveService.return_value
        mock_solve_service_instance.run_solve_session = AsyncMock(return_value=None)

        with runner.isolated_filesystem() as temp_dir:
            workspace_path = Path(temp_dir)
            problem_file = workspace_path / "problem.md"
            problem_file.write_text("# Test Problem")
            
            config_file = workspace_path / "ahc_config.yaml"
            config_file.write_text(yaml.dump({"contest_id": "test_contest"}))

            result = runner.invoke(cli, ["solve", str(workspace_path)])

            assert result.exit_code == 0
            MockCliConfig.assert_any_call(None) 
            MockCliConfig.assert_any_call(str(config_file)) 
            MockLLMClient.assert_called_once_with({}) 
            MockDockerManager.assert_called_once_with({})
            MockKnowledgeBase.assert_called_once_with(str(workspace_path), problem_id="test_contest")
            MockSolveService.assert_called_once_with(
                mock_llm_instance, mock_docker_instance, mock_workspace_config_instance, mock_kb_instance
            )
            mock_solve_service_instance.run_solve_session.assert_called_once()
            call_args = mock_solve_service_instance.run_solve_session.call_args
            assert call_args[1]['problem_text'] == "# Test Problem"
            assert call_args[1]['session_id'] is None
            assert call_args[1]['interactive'] is False
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

        config_file_content = { "contest_id": contest_id } # Simplified
        config_file = workspace_dir / "ahc_config.yaml"
        with open(config_file, "w") as f: yaml.dump(config_file_content, f)

        mock_global_config_instance = MagicMock(spec=Config); mock_global_config_instance.get.return_value = {} 
        mock_workspace_config_instance = MagicMock(spec=Config)
        mock_workspace_config_instance.get.side_effect = lambda key, default=None: {
            "contest_id": contest_id, "workspace.base_dir": str(workspace_dir)
        }.get(key, default)
        mock_workspace_config_instance.config_file_path = str(config_file)
        MockCliConfig.side_effect = [mock_global_config_instance, mock_workspace_config_instance]
        
        mock_llm_instance = MockLLMClient.return_value
        mock_docker_instance = MockDockerManager.return_value
        mock_kb_instance = MockKnowledgeBase.return_value
        
        mock_solve_service_instance = MockSolveService.return_value
        mock_solve_service_instance.run_solve_session = AsyncMock()

        result = runner.invoke(cli, ["solve", str(workspace_dir)])

        assert result.exit_code == 0
        mock_workspace_config_instance.set.assert_called_with("workspace.base_dir", str(workspace_dir))
        MockKnowledgeBase.assert_called_once_with(str(workspace_dir), problem_id=contest_id)
        MockSolveService.assert_called_once_with(
            mock_llm_instance, mock_docker_instance, mock_workspace_config_instance, mock_kb_instance
        )
        mock_solve_service_instance.run_solve_session.assert_called_once_with(
            problem_text=problem_text_content, session_id=None, interactive=False
        )
        assert f"Solving problem in workspace: {workspace_dir}" in result.output

    @patch("ahc_agent.cli.DockerManager")
    @patch("ahc_agent.cli.LLMClient")
    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.KnowledgeBase")
    @patch("ahc_agent.cli.SolveService")
    def test_solve_command_uses_tools_in_files_simplified(
        self, MockSolveService, MockKnowledgeBase, MockCliConfig, MockLLMClient, MockDockerManager, runner, tmp_path
    ):
        workspace_dir = tmp_path / "ahc_test_workspace_tools"; workspace_dir.mkdir()
        tools_dir = workspace_dir / "tools"; tools_dir.mkdir()
        tools_in_dir = tools_dir / "in"; tools_in_dir.mkdir()
        (tools_in_dir / "test01.txt").write_text("input data for test01")

        problem_file = workspace_dir / "problem.md"; problem_file.write_text("# Test Problem with Tools")
        config_file = workspace_dir / "ahc_config.yaml"
        with open(config_file, "w") as f: yaml.dump({"contest_id": "tools_test_contest"}, f)

        mock_global_config_instance = MagicMock(spec=Config); mock_global_config_instance.get.return_value = {}
        mock_workspace_config_instance = MagicMock(spec=Config)
        mock_workspace_config_instance.get.side_effect = lambda key, default=None: {
            "contest_id": "tools_test_contest", "workspace.base_dir": str(workspace_dir)
        }.get(key, default)
        mock_workspace_config_instance.config_file_path = str(config_file)
        MockCliConfig.side_effect = [mock_global_config_instance, mock_workspace_config_instance]

        mock_solve_service_instance = MockSolveService.return_value
        mock_solve_service_instance.run_solve_session = AsyncMock()

        result = runner.invoke(cli, ["solve", str(workspace_dir)])

        assert result.exit_code == 0
        mock_solve_service_instance.run_solve_session.assert_called_once()


    @patch("ahc_agent.cli.Config") 
    @patch("ahc_agent.cli.KnowledgeBase") 
    @patch("ahc_agent.cli.StatusService")
    def test_status_command(self, MockStatusService, MockKnowledgeBase, MockConfig, runner, tmp_path):
        mock_global_config_instance = MockConfig.return_value
        # Simulate workspace.base_dir being set in global config
        mock_workspace_dir = tmp_path / "mock_status_ws"
        mock_global_config_instance.get.side_effect = lambda key, default=None: \
            mock_workspace_dir if key == "workspace.base_dir" else default
        
        mock_workspace_dir.mkdir(exist_ok=True) # Ensure dir exists for KB

        mock_kb_instance = MockKnowledgeBase.return_value
        mock_status_service_instance = MockStatusService.return_value
        mock_status_service_instance.get_status.return_value = [
            "=== Session Status ===", "Session ID: test-session", "Status: Complete"
        ]
        
        result = runner.invoke(cli, ["status", "test-session"])

        assert result.exit_code == 0
        # Check Config (global) was used for workspace_base_dir
        mock_global_config_instance.get.assert_any_call("workspace.base_dir")
        # Check KB instantiation
        MockKnowledgeBase.assert_called_once_with(str(mock_workspace_dir), problem_id=mock_workspace_dir.name)
        # Check StatusService instantiation
        MockStatusService.assert_called_once_with(mock_global_config_instance, mock_kb_instance)
        mock_status_service_instance.get_status.assert_called_once_with(session_id="test-session", watch=False) 
        assert "Session ID: test-session" in result.output
        assert "Status: Complete" in result.output

    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.KnowledgeBase")
    @patch("ahc_agent.cli.SubmitService")
    def test_submit_command(self, MockSubmitService, MockKnowledgeBase, MockConfig, runner, tmp_path):
        mock_global_config_instance = MockConfig.return_value
        mock_workspace_dir = tmp_path / "mock_submit_ws"
        mock_global_config_instance.get.side_effect = lambda key, default=None: \
            mock_workspace_dir if key == "workspace.base_dir" else default
        
        mock_workspace_dir.mkdir(exist_ok=True)
        # Create a dummy ahc_config.yaml in the workspace for problem_id determination
        dummy_ws_cfg_path = mock_workspace_dir / "ahc_config.yaml"
        with open(dummy_ws_cfg_path, "w") as f: yaml.dump({"contest_id": "submit_contest"}, f)


        mock_kb_instance = MockKnowledgeBase.return_value
        mock_submit_service_instance = MockSubmitService.return_value
        # Path for output file, relative to isolated fs temp dir 'td'
        output_file_name = "solution.cpp" 
        expected_output_path_str = str(Path(tmp_path) / output_file_name) # Construct absolute-like path for assertion

        mock_submit_service_instance.submit_solution.return_value = {
            "session_id": "test-session", "output_path": expected_output_path_str,
            "solution_code": "// Test code", "score": 100,
        }
        
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            # Invoke with output path relative to 'td'
            result = runner.invoke(cli, ["submit", "test-session", "--output", output_file_name])

            assert result.exit_code == 0
            # Check KB instantiation. Problem_id comes from dummy_ws_cfg_path.
            MockKnowledgeBase.assert_called_once_with(str(mock_workspace_dir), problem_id="submit_contest")
            MockSubmitService.assert_called_once_with(mock_global_config_instance, mock_kb_instance)
            # Service receives the path as given to CLI, it might resolve it internally.
            # For the test, we check it was called with the path we provided.
            mock_submit_service_instance.submit_solution.assert_called_once_with(
                session_id="test-session", output_path=output_file_name 
            )
            assert f"Best solution for session test-session (Score: 100) written to {expected_output_path_str}" in result.output


    @patch("ahc_agent.cli.Config") # For main cli context
    @patch("ahc_agent.cli.DockerManager") # For main cli context
    @patch("ahc_agent.cli.DockerService")
    def test_docker_status_command(self, MockDockerService, MockMainDockerManager, MockMainConfig, runner):
        mock_config_instance = MockMainConfig.return_value 
        mock_config_instance.get.return_value = {} # For config.get('docker')

        mock_docker_manager_instance = MockMainDockerManager.return_value
        
        mock_docker_service_instance = MockDockerService.return_value
        mock_docker_service_instance.get_status.return_value = {
            "docker_available": True, "test_successful": True, 
            "message": "Docker is available and test successful."
        }

        result = runner.invoke(cli, ["docker", "status"])

        assert result.exit_code == 0
        MockMainDockerManager.assert_called_once_with({}) # Check instantiation in cli main group
        MockDockerService.assert_called_once_with(mock_docker_manager_instance) # Service gets manager instance
        mock_docker_service_instance.get_status.assert_called_once_with()
        assert "Docker is available and test successful." in result.output

    @patch("ahc_agent.cli.Config") 
    @patch("ahc_agent.cli.DockerManager") 
    @patch("ahc_agent.cli.DockerService")
    def test_docker_setup_command(self, MockDockerService, MockMainDockerManager, MockMainConfig, runner):
        mock_config_instance = MockMainConfig.return_value
        mock_config_instance.get.return_value = {}

        mock_docker_manager_instance = MockMainDockerManager.return_value
        # DockerManager.image_name used in cli.py's docker_setup, ensure mock has it or it's part of config
        mock_docker_manager_instance.image_name = "test_image:latest" 
        
        mock_docker_service_instance = MockDockerService.return_value
        mock_docker_service_instance.setup_environment.return_value = True # Simulate success

        result = runner.invoke(cli, ["docker", "setup"])

        assert result.exit_code == 0
        MockDockerService.assert_called_once_with(mock_docker_manager_instance)
        mock_docker_service_instance.setup_environment.assert_called_once_with()
        assert "Docker image pulled successfully" in result.output


    @patch("ahc_agent.cli.Config") # For main cli context
    @patch("ahc_agent.cli.DockerManager") # For main cli context
    @patch("ahc_agent.cli.LLMClient") # For main cli context
    @patch("ahc_agent.cli.BatchService")
    def test_batch_command(self, MockBatchService, MockLLMClient, MockMainDockerManager, MockMainConfig, runner, tmp_path):
        # Mock instances created in main cli group
        mock_config_instance = MockMainConfig.return_value
        mock_config_instance.get.return_value = {} # For llm, docker, batch default configs
        
        mock_llm_client_instance = MockLLMClient.return_value
        mock_docker_manager_instance = MockMainDockerManager.return_value

        # BatchService mock
        mock_batch_service_instance = MockBatchService.return_value
        # run_batch_experiments_service is async
        mock_batch_service_instance.run_batch_experiments_service = AsyncMock(return_value=[
            {"experiment_id": "exp1", "best_score": 100, "error": None},
            {"experiment_id": "exp2", "best_score": 0, "error": "Something went wrong"}
        ])

        # Create a dummy batch config file
        batch_config_file = tmp_path / "batch_config.yaml"
        with open(batch_config_file, "w") as f:
            yaml.dump({"common": {}, "problems": [], "parameter_sets": [], "experiments": []}, f)

        result = runner.invoke(cli, ["batch", str(batch_config_file)])

        assert result.exit_code == 0
        MockBatchService.assert_called_once_with(
            mock_llm_client_instance, mock_docker_manager_instance, mock_config_instance
        )
        mock_batch_service_instance.run_batch_experiments_service.assert_called_once_with(
            batch_config_path=str(batch_config_file),
            output_dir_override=None,
            parallel_override=None
        )
        assert "Batch processing completed." in result.output
        assert "Total experiments processed: 2" in result.output
        assert "Successful: 1" in result.output
        assert "Failed: 1" in result.output

    @patch("ahc_agent.cli.Config")
    def test_config_get_command(self, MockMainCliConfig, runner):
        mock_config_instance = MockMainCliConfig.return_value
        mock_config_instance.get.return_value = "o4-mini"
        result = runner.invoke(cli, ["config", "get", "llm.model"])
        assert result.exit_code == 0
        mock_config_instance.get.assert_called_with("llm.model")
        assert "llm.model = o4-mini" in result.output

    @patch("ahc_agent.cli.Config")
    def test_config_set_command(self, MockMainCliConfig, runner):
        mock_config_instance = MockMainCliConfig.return_value
        result = runner.invoke(cli, ["config", "set", "llm.model", "gpt-3.5-turbo"])
        assert result.exit_code == 0
        mock_config_instance.set.assert_called_with("llm.model", "gpt-3.5-turbo")
        assert "Set llm.model = gpt-3.5-turbo" in result.output
