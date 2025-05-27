"""
Unit tests for CLI module.
"""

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner
import pytest
import yaml

# Assuming these are the correct import paths after refactoring
from ahc_agent.cli import _solve_problem, cli 
from ahc_agent.config import Config
from ahc_agent.core.session_store import SessionStore # For type hinting and spec
from ahc_agent.core.heuristic_knowledge_base import HeuristicKnowledgeBase # For type hinting and spec


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

    @patch("ahc_agent.cli.Config")
    def test_init_command(self, mock_config, runner):
        """
        Test init command.
        """
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        with runner.isolated_filesystem() as temp_dir:
            def mock_get_side_effect(key, default=None):
                if key == "workspace.base_dir":
                    return os.path.join(temp_dir, "workspace")
                if key == "template":
                    return "default"
                if key == "docker.image":
                    return "ubuntu:latest"
                return MagicMock()
            mock_config_instance.get.side_effect = mock_get_side_effect
            runner.invoke(cli, ["init", "ahc001", "--workspace", "./workspace"])
            # Add more specific assertions if needed, e.g., file creation

    @patch("ahc_agent.cli.ProblemLogic") # Mock the temp ProblemLogic for parsing
    @patch("ahc_agent.cli._solve_problem") # Mock the main _solve_problem async function
    @patch("ahc_agent.cli.asyncio.run")   # Mock asyncio.run
    def test_solve_command(self, mock_asyncio_run, mock_solve_problem_coroutine, mock_problem_logic_temp, runner):
        """
        Test solve command basic flow.
        """
        mock_pl_temp_instance = mock_problem_logic_temp.return_value
        mock_pl_temp_instance.parse_problem_statement = AsyncMock(return_value={"title": "parsed_title_for_test"})
        
        mock_solve_problem_coroutine.return_value = AsyncMock() # _solve_problem itself returns an awaitable
    
        def run_coro_side_effect(coro, *_args, **_kwargs):
            # Simplified: just return if coro is an AsyncMock, or run if it's a real coroutine (not expected here)
            if isinstance(coro, AsyncMock):
                return MagicMock() # Or some other suitable mock for the result of asyncio.run
            return asyncio.get_event_loop().run_until_complete(coro)

        mock_asyncio_run.side_effect = run_coro_side_effect
    
        with runner.isolated_filesystem() as temp_dir:
            problem_file_path = Path(temp_dir) / "problem.md"
            problem_file_path.write_text("# Test Problem\n\nThis is a test problem.")
            
            config_file_path = Path(temp_dir) / "ahc_config.yaml"
            # IMPORTANT: Ensure workspace.base_dir is correctly set for SessionManager
            # The 'solve' command expects 'temp_dir' to be the problem directory.
            # SessionManager(sm_workspace_dir, sm_problem_id)
            # sm_workspace_dir = problem_dir_path.parent -> temp_dir.parent
            # sm_problem_id = problem_dir_path.name -> temp_dir.name
            # So, config's workspace.base_dir should be temp_dir for consistency.
            with open(config_file_path, "w") as f:
                yaml.dump({
                    "contest_id": Path(temp_dir).name, # Use the temp_dir name as contest_id
                    "workspace": {"base_dir": str(temp_dir)}, # Critical for SessionManager context
                    "llm": {"model": "o4-mini", "api_key": "mock_key_prevent_error"}, # Prevent LLM errors
                    "heuristic_knowledge_base": {"global_path": str(Path(temp_dir) / "global_hkb_mock")}
                }, f)
            (Path(temp_dir) / "global_hkb_mock").mkdir(exist_ok=True)


            result = runner.invoke(cli, ["solve", temp_dir]) # temp_dir is the workspace path for solve

            assert result.exit_code == 0, f"CLI command failed: {result.output}"
            # Allow for multiple asyncio.run calls: one for parsing, one for _solve_problem
            assert mock_asyncio_run.call_count > 0 
            mock_solve_problem_coroutine.assert_called_once()
            mock_problem_logic_temp.assert_called_once() 
            mock_pl_temp_instance.parse_problem_statement.assert_called_once()

    @patch("ahc_agent.cli.SessionManager")
    def test_status_command(self, MockSessionManager, runner):
        mock_sm_instance = MockSessionManager.return_value
        mock_ss_instance = MagicMock(spec=SessionStore)
        mock_sm_instance.get_session_store.return_value = mock_ss_instance
        
        mock_ss_instance.session_id = "test-session"
        mock_ss_instance.problem_id = "Test Problem"
        mock_ss_instance.get_session_metadata.return_value = {
            "session_id": "test-session", "problem_id": "Test Problem",
            "created_at": time.time(), "updated_at": time.time(),
            "status": "completed", "best_score": 100
        }
        mock_ss_instance.get_problem_analysis.return_value = {"title": "Test Problem"}
        mock_ss_instance.get_solution_strategy.return_value = {"approach": "Test approach"}
        mock_ss_instance.get_evolution_log.return_value = {"generations_completed": 10, "best_score_in_log": 100, "duration_seconds": 60}
        mock_ss_instance.get_best_solution.return_value = {"code": "// Test code", "score": 100, "solution_id": "s1"}

        with runner.isolated_filesystem() as temp_dir_status:
            dummy_config_path = Path(temp_dir_status) / "ahc_config.yaml"
            # The status command will try to load this config to find 'workspace.base_dir'
            # This base_dir is then used to instantiate SessionManager
            with open(dummy_config_path, "w") as f:
                 yaml.dump({"workspace": {"base_dir": str(temp_dir_status)}}, f)
            
            # The SessionManager is initialized with (Path(temp_dir_status).parent, Path(temp_dir_status).name)
            # So, the problem_id for SessionManager will be temp_dir_status.name.
            # Ensure the mocked problem_id matches or handle this in SessionManager mock.
            mock_sm_instance.problem_id = Path(temp_dir_status).name # Align problem_id

            result = runner.invoke(cli, ["--config", str(dummy_config_path), "status", "test-session"])
        
        assert result.exit_code == 0, f"CLI error: {result.output}"
        assert "Session ID: test-session" in result.output
        MockSessionManager.assert_called_once_with(str(Path(temp_dir_status).parent), Path(temp_dir_status).name)
        mock_sm_instance.get_session_store.assert_called_once_with("test-session")

    @patch("ahc_agent.cli.SessionManager")
    def test_submit_command(self, MockSessionManager, runner):
        mock_sm_instance = MockSessionManager.return_value
        mock_ss_instance = MagicMock(spec=SessionStore)
        mock_sm_instance.get_session_store.return_value = mock_ss_instance
        mock_ss_instance.get_best_solution.return_value = {"code": "// Test code", "score": 100, "solution_id": "best_sol_id"}

        with runner.isolated_filesystem() as temp_dir_submit:
            dummy_config_path = Path(temp_dir_submit) / "ahc_config.yaml"
            with open(dummy_config_path, "w") as f:
                 yaml.dump({"workspace": {"base_dir": str(temp_dir_submit)}}, f)
            
            result = runner.invoke(cli, ["--config", str(dummy_config_path), "submit", "test-session", "--output", "solution.cpp"])

            assert result.exit_code == 0, f"CLI error: {result.output}"
            # Check output file *inside* the isolated_filesystem context
            solution_file_path = Path(temp_dir_submit) / "solution.cpp"
            assert solution_file_path.is_file()
            with open(solution_file_path) as f:
                content = f.read()
                assert content == "// Test code"
        
        assert "Score: 100" in result.output # This can remain outside
        MockSessionManager.assert_called_once_with(str(Path(temp_dir_submit).parent), Path(temp_dir_submit).name)
        mock_sm_instance.get_session_store.assert_called_once_with("test-session")

    @patch("ahc_agent.cli.SessionManager")
    def test_stop_command(self, MockSessionManager, runner):
        mock_sm_instance = MockSessionManager.return_value
        mock_ss_instance = MagicMock(spec=SessionStore)
        mock_sm_instance.get_session_store.return_value = mock_ss_instance
        mock_ss_instance.update_session_metadata.return_value = True

        with runner.isolated_filesystem() as temp_dir_stop:
            dummy_config_path = Path(temp_dir_stop) / "ahc_config.yaml"
            with open(dummy_config_path, "w") as f:
                 yaml.dump({"workspace": {"base_dir": str(temp_dir_stop)}}, f)

            result = runner.invoke(cli, ["--config", str(dummy_config_path), "stop", "test-session"])

        assert result.exit_code == 0, f"CLI error: {result.output}"
        assert "Session test-session marked as stopped." in result.output
        MockSessionManager.assert_called_once_with(str(Path(temp_dir_stop).parent), Path(temp_dir_stop).name)
        mock_sm_instance.get_session_store.assert_called_once_with("test-session")
        mock_ss_instance.update_session_metadata.assert_called_once_with({"status": "stopped"})


    @patch("ahc_agent.cli.Config")
    def test_config_get_command(self, mock_config, runner):
        mock_config_instance = MagicMock()
        mock_config_instance.get.return_value = "o4-mini"
        mock_config.return_value = mock_config_instance
        runner.invoke(cli, ["config", "get", "llm.model"])
        mock_config_instance.get.assert_called_with("llm.model")

    @patch("ahc_agent.cli.Config")
    def test_config_set_command(self, mock_config, runner):
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        runner.invoke(cli, ["config", "set", "llm.model", "gpt-3.5-turbo"])
        mock_config_instance.set.assert_called_with("llm.model", "gpt-3.5-turbo")

    @patch("ahc_agent.cli.DockerManager")
    def test_docker_status_command(self, mock_docker_manager, runner):
        mock_docker_manager_instance = MagicMock()
        mock_docker_manager.return_value = mock_docker_manager_instance
        # Changed to check_docker_availability to match current cli.py
        mock_docker_manager_instance.check_docker_availability.return_value = True 
        mock_docker_manager_instance.run_command.return_value = {"success": True, "stdout": "Test success", "stderr": ""}


        result = runner.invoke(cli, ["docker", "status"])
        assert "Docker is available" in result.output
        assert "Docker test successful" in result.output # This check depends on run_command mock

    @patch("ahc_agent.cli.scrape_and_setup_problem")
    @patch("ahc_agent.cli.Config")
    def test_init_default_workspace(self, mock_config, mock_scraper, runner: CliRunner, tmp_path: Path):
        contest_id = "ahc999"
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        def mock_get_side_effect(key, default_val=None):
            if key == "template": return "default"
            if key == "docker.image": return "ubuntu:latest"
            return default_val
        mock_config_instance.get.side_effect = mock_get_side_effect
        def scraper_side_effect(url, base_output_dir):
            project_path = Path(base_output_dir)
            (project_path / "tools" / "in").mkdir(parents=True, exist_ok=True)
            (project_path / "tools" / "in" / "0000.txt").touch()
            return True
        mock_scraper.side_effect = scraper_side_effect
        os.chdir(tmp_path)
        result = runner.invoke(cli, ["init", contest_id])
        project_dir = tmp_path / contest_id
        assert project_dir.is_dir()
        config_file = project_dir / "ahc_config.yaml"
        assert config_file.is_file()
        with open(config_file) as f: project_config = yaml.safe_load(f)
        assert project_config["contest_id"] == contest_id
        expected_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
        mock_scraper.assert_called_once_with(expected_url, str(project_dir))
        assert (project_dir / "tools" / "in" / "0000.txt").is_file()

    @patch("ahc_agent.cli.scrape_and_setup_problem")
    @patch("ahc_agent.cli.Config")
    def test_init_with_workspace(self, mock_config, mock_scraper, runner: CliRunner, tmp_path: Path):
        contest_id = "ahc998"; workspace_name = "my_custom_workspace"
        mock_config_instance = MagicMock(); mock_config.return_value = mock_config_instance
        mock_config_instance.get.side_effect = lambda k, v=None: "default" if k=="template" else ("ubuntu:latest" if k=="docker.image" else v)
        mock_scraper.side_effect = lambda u,b: (Path(b)/"tools"/"in").mkdir(parents=True,exist_ok=True) or (Path(b)/"tools"/"in"/"0000.txt").touch() or True
        workspace_path_absolute = tmp_path / workspace_name
        os.chdir(tmp_path)
        result = runner.invoke(cli, ["init", contest_id, "--workspace", workspace_name])
        assert workspace_path_absolute.is_dir()
        config_file = workspace_path_absolute / "ahc_config.yaml"; assert config_file.is_file()
        with open(config_file) as f: project_config = yaml.safe_load(f)
        assert project_config["contest_id"] == contest_id
        mock_scraper.assert_called_once_with(f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a", str(workspace_path_absolute))

    @patch("ahc_agent.cli.scrape_and_setup_problem")
    def test_init_with_custom_template_and_image(self, mock_scraper, runner: CliRunner, tmp_path: Path):
        contest_id = "ahc997"; custom_template = "cpp_pro"; custom_image = "my_cpp_env:1.0"
        mock_scraper.side_effect = lambda u,b: (Path(b)/"tools"/"in").mkdir(parents=True,exist_ok=True) or (Path(b)/"tools"/"in"/"0000.txt").touch() or True
        os.chdir(tmp_path)
        result = runner.invoke(cli, ["init", contest_id, "--template", custom_template, "--docker-image", custom_image])
        project_dir = tmp_path / contest_id; assert project_dir.is_dir()
        config_file = project_dir / "ahc_config.yaml"; assert config_file.is_file()
        with open(config_file) as f: project_config = yaml.safe_load(f)
        assert project_config["template"] == custom_template and project_config["docker_image"] == custom_image

    @patch("ahc_agent.cli.ProblemLogic") # For the temporary parsing instance
    @patch("ahc_agent.cli._solve_problem") # Main async solve function
    @patch("ahc_agent.cli.asyncio.run")
    @patch("ahc_agent.cli.Config") # For the main Config loading in CLI
    def test_solve_command_with_workspace(
        self, mock_Config_class_arg, mock_asyncio_run_arg, mock_solve_problem_arg, mock_problem_logic_temp_arg, runner, tmp_path
    ):
        mock_pl_temp_instance = mock_problem_logic_temp_arg.return_value
        mock_pl_temp_instance.parse_problem_statement = AsyncMock(return_value={"title": "parsed_title_for_test_workspace"})

        contest_id = "ahc999"
        workspace_dir = tmp_path / contest_id 
        workspace_dir.mkdir()
        problem_text_content_from_file = (Path(__file__).parent / "data" / "ahc001" / "problem.md").read_text(encoding="utf-8")
        (workspace_dir / "problem.md").write_text(problem_text_content_from_file, encoding="utf-8")

        config_file_content = {
            "contest_id": contest_id, "template": "test_template",
            "docker": {"image": "test_image:latest"},
            "evolution": {"time_limit_seconds": 10},
            "workspace": {"base_dir": str(workspace_dir)}, 
            "llm": {"model": "o4-mini", "api_key": "mock_key"}, 
            "heuristic_knowledge_base": {"global_path": str(tmp_path / "global_hkb")}
        }
        config_file = workspace_dir / "ahc_config.yaml"
        config_file.write_text(yaml.dump(config_file_content))
        (tmp_path / "global_hkb").mkdir(exist_ok=True) # Ensure global HKB dir exists

        mock_config_instance_loaded_from_ahc_config = MagicMock(spec=Config)
        mock_config_instance_loaded_from_ahc_config.config_file_path = str(config_file)
        
        def loaded_config_get_side_effect(key, default=None):
            # Simplified: directly access parts of config_file_content
            parts = key.split('.')
            val = config_file_content
            try:
                for part in parts: val = val[part]
                return val
            except (KeyError, TypeError):
                # Fallback for structured keys like 'llm' if only 'llm.model' is in dict
                if key == "llm": return config_file_content.get("llm", default if default is not None else {})
                if key == "docker": return config_file_content.get("docker", default if default is not None else {})
                # ... add other top-level dicts if needed ...
                return default
        mock_config_instance_loaded_from_ahc_config.get.side_effect = loaded_config_get_side_effect
        mock_config_instance_loaded_from_ahc_config.set = MagicMock()
    
        mock_Config_class_arg.side_effect = [MagicMock(spec=Config), mock_config_instance_loaded_from_ahc_config]
        mock_solve_problem_arg.return_value = AsyncMock()
        mock_asyncio_run_arg.side_effect = lambda coro, *_a, **_kw: asyncio.get_event_loop().run_until_complete(coro) if not isinstance(coro, AsyncMock) else MagicMock()
            
        result = runner.invoke(cli, ["solve", str(workspace_dir)])

        assert result.exit_code == 0, f"CLI error: {result.output}"
        mock_solve_problem_arg.assert_called_once()
        
        called_args, _ = mock_solve_problem_arg.call_args
        assert called_args[0] is mock_config_instance_loaded_from_ahc_config
        assert called_args[1] == problem_text_content_from_file
        assert isinstance(called_args[2], SessionStore)
        assert isinstance(called_args[3], HeuristicKnowledgeBase)
        assert isinstance(called_args[4], HeuristicKnowledgeBase)
        assert called_args[5] is False

    @patch("ahc_agent.cli.scrape_and_setup_problem")
    @patch("ahc_agent.cli.Config")
    def test_init_command_with_existing_target_dir_as_file(self, MockConfig, mock_scrape_and_setup_problem, runner, tmp_path):
        mock_config_instance = MagicMock(); MockConfig.return_value = mock_config_instance
        mock_config_instance.get.side_effect = lambda key, default=None: default
        mock_scrape_and_setup_problem.return_value = None
        contest_id = "ahc888"
        target_path_as_file = tmp_path / contest_id
        target_path_as_file.write_text("This is a file, not a directory.")
        result = runner.invoke(cli, ["init", contest_id, "--workspace", str(target_path_as_file)])
        assert result.exit_code != 0
        assert "Error creating project directory" in result.output

    @patch("ahc_agent.cli.Config") # For the main config in _solve_problem
    @patch("ahc_agent.cli.EvolutionaryEngine")
    @patch("ahc_agent.cli.ImplementationDebugger")
    @patch("ahc_agent.cli.ProblemAnalyzer")   # Main ProblemAnalyzer used in _solve_problem
    @patch("ahc_agent.cli.SolutionStrategist")# Main SolutionStrategist used in _solve_problem
    @patch("ahc_agent.cli.ProblemLogic")      # Main ProblemLogic used in _solve_problem
    def test_solve_command_uses_tools_in_files(
        self, MockProblemLogic, MockSolutionStrategist, MockProblemAnalyzer,
        MockImplementationDebugger, MockEvolutionaryEngine, MockConfig, 
        runner, tmp_path,
    ):
        workspace_dir = tmp_path / "ahc_test_workspace_tools"
        tools_in_dir = workspace_dir / "tools" / "in"; tools_in_dir.mkdir(parents=True)
        test_input_content1 = "input data for test01"; (tools_in_dir / "test01.txt").write_text(test_input_content1)
        test_input_content2 = "input data for test02"; (tools_in_dir / "test02.txt").write_text(test_input_content2)
        problem_text_content = (Path(__file__).parent / "data" / "ahc001" / "problem.md").read_text(encoding="utf-8")
        (workspace_dir / "problem.md").write_text(problem_text_content)

        mock_config_for_solve = MagicMock(spec=Config)
        mock_config_for_solve.get.side_effect = lambda k, v=None: \
            str(workspace_dir) if k == "workspace.base_dir" else \
            ({"model": "test_m", "api_key": "test_k"} if k == "llm" else \
            ({"image": "test_i"} if k == "docker" else \
            ({} if k in ["evolution", "analyzer", "strategist", "debugger", "problem_logic"] else v)))

        mock_session_store = MagicMock(spec=SessionStore)
        mock_session_store.get_problem_analysis.return_value = None
        mock_session_store.get_solution_strategy.return_value = None
        mock_session_store.list_solutions.return_value = []
        mock_session_store.session_dir = str(workspace_dir / "knowledge" / "sessions" / "test_tools_session")
        mock_session_store.session_id = "test_session_tools_in_files" # Define the session_id attribute

        mock_problem_hkb = MagicMock(spec=HeuristicKnowledgeBase)
        mock_global_hkb = MagicMock(spec=HeuristicKnowledgeBase)

        MockProblemAnalyzer.return_value.analyze = AsyncMock(return_value={"title": "analyzed"})
        MockSolutionStrategist.return_value.develop_strategy = AsyncMock(return_value={"approach": "strat"})
        
        pl_instance_mock = MockProblemLogic.return_value
        pl_instance_mock.generate_initial_solution = AsyncMock(return_value="initial_code")
        pl_instance_mock.generate_test_cases = AsyncMock(return_value=[]) # Should NOT be called
        pl_instance_mock.create_score_calculator = AsyncMock(return_value=MagicMock(return_value=100.0))
        
        mock_debugger_instance = MockImplementationDebugger.return_value
        mock_debugger_instance.compile_and_test = AsyncMock(side_effect=[
            {"success": True, "execution_output": "out1", "execution_time": 0.1},
            {"success": True, "execution_output": "out2", "execution_time": 0.2},
        ])

        async def mock_evolve(pa, ss, ic, eval_func, ad):
            await eval_func(ic) # This will trigger test case loading logic
            return {"best_solution": "final", "best_score": 1.0, "evolution_log": {}, "generations_completed": 1}
        MockEvolutionaryEngine.return_value.evolve.side_effect = mock_evolve
        
        asyncio.run(
            _solve_problem(
                config=mock_config_for_solve, problem_text=problem_text_content,
                session_store=mock_session_store, problem_heuristic_kb=mock_problem_hkb,
                global_heuristic_kb=mock_global_hkb, interactive=False
            )
        )
        
        pl_instance_mock.generate_test_cases.assert_not_called()
        compile_calls = mock_debugger_instance.compile_and_test.call_args_list
        assert len(compile_calls) == 2
        received_inputs = {call[0][1] for call in compile_calls}
        assert received_inputs == {test_input_content1, test_input_content2}
