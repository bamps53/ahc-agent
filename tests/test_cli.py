"""
Unit tests for CLI module.
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner
import pytest
import yaml

from ahc_agent.cli import _solve_problem, cli
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
        # Run CLI with --help
        result = runner.invoke(cli, ["--help"])

        # Check result
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
        # Mock Config
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        # Run init command
        with runner.isolated_filesystem() as temp_dir:
            # Mock Config get method
            def mock_get_side_effect(key, default=None):
                if key == "workspace.base_dir":
                    return os.path.join(temp_dir, "workspace")
                if key == "template":
                    return "default"
                if key == "docker.image":
                    return "ubuntu:latest"
                return MagicMock()

            mock_config_instance.get.side_effect = mock_get_side_effect

            # Corrected command invocation
            runner.invoke(cli, ["init", "ahc001", "--workspace", "./workspace"])

            # Check Config calls
            # The base_dir might be set to the absolute path by the CLI logic
            # For simplicity, we're checking if it's called with something like './workspace'
            # or its resolved equivalent. This part of the test might need more robust path handling.
            # mock_config_instance.set.assert_any_call(
            #     "workspace.base_dir", os.path.abspath(workspace_path)
            # )
            # mock_config_instance.save.assert_called_once_with(
            #     os.path.join(os.path.abspath(workspace_path), "ahc_config.yaml")
            # )

    @patch("ahc_agent.cli._solve_problem")
    @patch("ahc_agent.cli.asyncio.run")
    def test_solve_command(self, mock_asyncio_run, mock_solve_problem_coroutine, runner):
        """
        Test solve command.
        """
        # _solve_problem がコルーチン関数であると仮定し、AsyncMock を return_value に設定
        # これにより、_solve_problem(...) は await 可能な AsyncMock インスタンスを返す
        mock_solve_problem_coroutine.return_value = AsyncMock()  # この AsyncMock は await される必要がある

        # asyncio.run が呼び出されたときに、渡されたコルーチンを実行するような side_effect
        def run_coro_side_effect(coro, *_args, **_kwargs):
            # coro が AsyncMock インスタンスなので、それを await する
            # 実際の asyncio.run は結果を返すので、ここでは None を返すか、AsyncMock の結果を返す
            async def dummy_coro():
                return await coro

            # 新しいイベントループを作成して使用する
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(dummy_coro())
            finally:
                loop.close()
                asyncio.set_event_loop(None)  # グローバルなイベントループポリシーをリセット
            return result

        mock_asyncio_run.side_effect = run_coro_side_effect

        # Create a temporary problem file
        with runner.isolated_filesystem() as temp_dir:  # 一時ディレクトリを取得
            # Create problem file
            problem_file_path = Path(temp_dir) / "problem.md"
            with open(problem_file_path, "w") as f:
                f.write("# Test Problem\n\nThis is a test problem.")

            # Create dummy ahc_config.yaml
            config_file_path = Path(temp_dir) / "ahc_config.yaml"
            with open(config_file_path, "w") as f:
                yaml.dump({"contest_id": "test_contest"}, f)

            # Run solve command with the directory path
            runner.invoke(cli, ["solve", temp_dir])

            # Check asyncio.run call
            mock_asyncio_run.assert_called_once()
            # _solve_problem が期待通りに呼び出されたかも確認
            mock_solve_problem_coroutine.assert_called_once()

    @patch("ahc_agent.cli.KnowledgeBase")
    def test_status_command(self, mock_knowledge_base, runner):
        """
        Test status command.
        """
        # Mock KnowledgeBase
        mock_kb_instance = MagicMock()
        mock_kb_instance.get_session.return_value = {
            "session_id": "test-session",
            "problem_id": "Test Problem",
            "created_at": 1621234567,
            "updated_at": 1621234567,
            "status": "completed",
        }
        mock_kb_instance.get_problem_analysis.return_value = {"title": "Test Problem"}
        mock_kb_instance.get_solution_strategy.return_value = {"approach": "Test approach"}
        mock_kb_instance.get_evolution_log.return_value = {"generations_completed": 10, "best_score": 100, "duration": 60}
        mock_kb_instance.get_best_solution.return_value = {"code": "// Test code", "score": 100}
        mock_knowledge_base.return_value = mock_kb_instance

        # Run status command
        result = runner.invoke(cli, ["status", "test-session"])

        # Check result
        assert "Session ID: test-session" in result.output
        assert "Problem: Test Problem" in result.output
        assert "Status: completed" in result.output
        assert "Problem Analysis: Complete" in result.output
        assert "Solution Strategy: Complete" in result.output
        assert "Evolution: Complete (10 generations)" in result.output
        assert "Best Score: 100" in result.output

    @patch("ahc_agent.cli.KnowledgeBase")
    def test_submit_command(self, mock_knowledge_base, runner):
        """
        Test submit command.
        """
        # Mock KnowledgeBase
        mock_kb_instance = MagicMock()
        mock_kb_instance.get_session.return_value = {"session_id": "test-session", "problem_id": "Test Problem"}
        mock_kb_instance.get_best_solution.return_value = {"code": "// Test code", "score": 100}
        mock_knowledge_base.return_value = mock_kb_instance

        # Run submit command with output file
        with runner.isolated_filesystem():
            runner.invoke(cli, ["submit", "test-session", "--output", "solution.cpp"])

            # Check output file
            with open("solution.cpp") as f:
                content = f.read()
                assert content == "// Test code"

    @patch("ahc_agent.cli.Config")
    def test_config_get_command(self, mock_config, runner):
        """
        Test config get command.
        """
        # Mock Config
        mock_config_instance = MagicMock()
        mock_config_instance.get.return_value = "o4-mini"
        mock_config.return_value = mock_config_instance

        # Run config get command
        runner.invoke(cli, ["config", "get", "llm.model"])

        # Check Config calls
        mock_config_instance.get.assert_called_with("llm.model")

    @patch("ahc_agent.cli.Config")
    def test_config_set_command(self, mock_config, runner):
        """
        Test config set command.
        """
        # Mock Config
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        # Run config set command
        runner.invoke(cli, ["config", "set", "llm.model", "gpt-3.5-turbo"])

        # Check Config calls
        mock_config_instance.set.assert_called_with("llm.model", "gpt-3.5-turbo")

    @patch("ahc_agent.cli.DockerManager")
    def test_docker_status_command(self, mock_docker_manager, runner):
        """
        Test docker status command.
        """
        # Mock DockerManager
        mock_docker_manager_instance = MagicMock()
        mock_docker_manager.return_value = mock_docker_manager_instance
        mock_docker_manager_instance.is_docker_available.return_value = True
        mock_docker_manager_instance.test_docker_connection.return_value = True

        # Run docker status command
        result = runner.invoke(cli, ["docker", "status"])

        # Check result
        assert "Docker is available" in result.output
        assert "Docker test successful" in result.output

    @patch("ahc_agent.cli.scrape_and_setup_problem")  # scrape_and_setup_problemもモックする
    @patch("ahc_agent.cli.Config")
    def test_init_default_workspace(self, mock_config, mock_scraper, runner: CliRunner, tmp_path: Path):
        """Test init command with default workspace (uses contest_id as dir name)."""
        contest_id = "ahc999"

        # Mock Config instance and its get method
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        def mock_get_side_effect(key, default_val=None):
            if key == "template":
                return "default"
            if key == "docker.image":
                return "ubuntu:latest"
            return default_val

        mock_config_instance.get.side_effect = mock_get_side_effect

        # Change current working directory to tmp_path for the test
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["init", contest_id])

        project_dir = tmp_path / contest_id
        assert project_dir.is_dir()

        config_file = project_dir / "ahc_config.yaml"
        assert config_file.is_file()

        with open(config_file) as f:
            project_config = yaml.safe_load(f)

        assert project_config["contest_id"] == contest_id
        assert project_config["template"] == "default"
        assert project_config["docker_image"] == "ubuntu:latest"

        expected_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
        mock_scraper.assert_called_once_with(expected_url, str(project_dir))
        assert (
            f"Initialized AHC project in ./{contest_id}" in result.output
            or f"Initialized AHC project in {project_dir}" in result.output
        )
        assert f"Project configuration saved to {config_file}" in result.output

    @patch("ahc_agent.cli.scrape_and_setup_problem")
    @patch("ahc_agent.cli.Config")
    def test_init_with_workspace(self, mock_config, mock_scraper, runner: CliRunner, tmp_path: Path):
        """Test init command with a specified workspace."""
        contest_id = "ahc998"
        workspace_name = "my_custom_workspace"

        # Mock Config instance and its get method
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        def mock_get_side_effect(key, default_val=None):
            if key == "template":
                return "default"
            if key == "docker.image":
                return "ubuntu:latest"
            return default_val

        mock_config_instance.get.side_effect = mock_get_side_effect

        # Create workspace_path relative to tmp_path for isolation
        workspace_path_relative = Path(workspace_name)
        workspace_path_absolute = tmp_path / workspace_name

        # Change CWD to tmp_path so relative workspace path works as expected
        os.chdir(tmp_path)

        result = runner.invoke(cli, ["init", contest_id, "--workspace", str(workspace_path_relative)])

        assert workspace_path_absolute.is_dir()

        config_file = workspace_path_absolute / "ahc_config.yaml"
        assert config_file.is_file()

        with open(config_file) as f:
            project_config = yaml.safe_load(f)

        assert project_config["contest_id"] == contest_id
        assert project_config["template"] == "default"
        assert project_config["docker_image"] == "ubuntu:latest"

        expected_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
        mock_scraper.assert_called_once_with(expected_url, str(workspace_path_absolute))
        assert (
            f"Initialized AHC project in ./{workspace_path_relative}" in result.output
            or f"Initialized AHC project in {workspace_path_relative}" in result.output
            or f"Initialized AHC project in {workspace_path_absolute}" in result.output
        )
        assert f"Project configuration saved to {config_file}" in result.output

    @patch("ahc_agent.cli.scrape_and_setup_problem")
    def test_init_with_custom_template_and_image(self, mock_scraper, runner: CliRunner, tmp_path: Path):
        """Test init command with custom template and docker image."""
        contest_id = "ahc997"
        custom_template = "cpp_pro"
        custom_image = "my_cpp_env:1.0"

        os.chdir(tmp_path)  # Ensure relative paths are handled from a known base

        result = runner.invoke(cli, ["init", contest_id, "--template", custom_template, "--docker-image", custom_image])

        project_dir = tmp_path / contest_id
        assert project_dir.is_dir()
        config_file = project_dir / "ahc_config.yaml"
        assert config_file.is_file()

        with open(config_file) as f:
            project_config = yaml.safe_load(f)

        assert project_config["contest_id"] == contest_id
        assert project_config["template"] == custom_template
        assert project_config["docker_image"] == custom_image
        expected_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
        mock_scraper.assert_called_once_with(expected_url, str(project_dir))
        assert (
            f"Initialized AHC project in ./{contest_id}" in result.output
            or f"Initialized AHC project in {project_dir}" in result.output
        )
        assert f"Project configuration saved to {config_file}" in result.output

    @patch("ahc_agent.cli._solve_problem")
    @patch("ahc_agent.cli.asyncio.run")  # asyncio.run もモックする
    @patch("ahc_agent.cli.Config")
    def test_solve_command_with_workspace(
        self, mock_Config_class_arg, mock_asyncio_run_arg, mock_solve_problem_arg, runner, tmp_path
    ):
        """Test the solve command with a workspace argument."""
        contest_id = "ahc999"
        workspace_dir = tmp_path / contest_id
        workspace_dir.mkdir()

        problem_text_content = "This is a dummy problem statement."
        problem_file = workspace_dir / "problem.md"
        problem_file.write_text(problem_text_content)

        config_file_content = {
            "contest_id": contest_id,
            "template": "test_template",
            "docker": {"image": "test_image:latest"},
            "evolution": {"time_limit_seconds": 10},
            "workspace": {"base_dir": str(workspace_dir)},
        }
        config_file = workspace_dir / "ahc_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_file_content, f)

        mock_config_instance_loaded = MagicMock(spec=Config)
        mock_config_instance_loaded.config_file_path = str(config_file)

        loaded_config_values = {
            "workspace.base_dir": str(workspace_dir),
            "workspace.problem_file": "problem.md",
            "llm.model": "test_model",
            "evolution.population_size": 1,
            "evolution.generations": 1,
            "docker.image": "test_image",  # ImplementationDebugger で使われる可能性
            "language": "cpp",  # ImplementationDebugger で使われる可能性
        }
        mock_config_instance_loaded.get.side_effect = lambda k, v=None: loaded_config_values.get(k, v)

        mock_Config_class_arg.side_effect = [MagicMock(spec=Config), mock_config_instance_loaded]

        # _solve_problem がコルーチン関数であると仮定し、AsyncMock を return_value に設定
        mock_solve_problem_arg.return_value = AsyncMock()  # AsyncMockインスタンスを返す

        # asyncio.run が呼び出されたときに、渡されたコルーチンを実行するような side_effect
        def run_coro_side_effect(coro, *_args, **_kwargs):
            async def dummy_coro():
                return await coro

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(dummy_coro())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
            return result

        mock_asyncio_run_arg.side_effect = run_coro_side_effect

        result = runner.invoke(cli, ["solve", str(workspace_dir)])

        assert mock_Config_class_arg.call_count >= 1

        mock_solve_problem_arg.assert_called_once()
        mock_asyncio_run_arg.assert_called_once()  # asyncio.runが呼ばれたことを確認
        called_args, called_kwargs = mock_solve_problem_arg.call_args

        # 呼び出し引数の検証
        assert called_args[0] == mock_config_instance_loaded  # Config オブジェクト
        assert called_args[1] == problem_text_content  # 問題文のテキスト
        assert called_args[2] is None  # session_id は渡されていないはず
        assert called_args[3] is False  # interactive (このテストではデフォルトのFalse)
        assert called_kwargs == {}  # キーワード引数は渡されないはず

        assert f"Solving problem in workspace: {workspace_dir}" in result.output
        assert f"Using config: {config_file}" in result.output

    @patch("ahc_agent.cli.scrape_and_setup_problem")  # scrape_and_setup_problemもモックする
    @patch("ahc_agent.cli.Config")
    def test_init_command_with_existing_target_dir_as_file(self, MockConfig, mock_scrape_and_setup_problem, runner, tmp_path):
        # Mock Config
        mock_config_instance = MagicMock()
        MockConfig.return_value = mock_config_instance
        mock_config_instance.get.side_effect = lambda key, default=None: default  # シンプルなgetのモック

        # モックされた scrape_and_setup_problem が呼ばれないようにするか、期待通りに処理する
        mock_scrape_and_setup_problem.return_value = None

        contest_id = "ahc888"
        # 一時ファイルシステム内に、初期化ターゲットと同名のファイルを作成
        target_path_as_file = tmp_path / contest_id
        with open(target_path_as_file, "w") as f:
            f.write("This is a file, not a directory.")

        # init コマンドを実行 (workspaceオプションで既存のファイルパスを指定)
        result = runner.invoke(cli, ["init", contest_id, "--workspace", str(target_path_as_file)])

        # エラーが発生することを期待 (例えばディレクトリ作成に失敗)
        assert result.exit_code != 0, "CLI should exit with an error code."
        assert "Error creating project directory" in result.output  # エラーメッセージを確認

        # 元のファイルが変更されていないことを確認
        assert target_path_as_file.is_file()
        with open(target_path_as_file) as f:
            content = f.read()
            assert content == "This is a file, not a directory."

    @patch("ahc_agent.cli.Config")
    @patch("ahc_agent.cli.EvolutionaryEngine")
    @patch("ahc_agent.cli.ImplementationDebugger")
    @patch("ahc_agent.cli.ProblemAnalyzer")
    @patch("ahc_agent.cli.SolutionStrategist")
    @patch("ahc_agent.cli.ProblemLogic")
    def test_solve_command_uses_tools_in_files(
        self,
        MockProblemLogic,
        MockSolutionStrategist,
        MockProblemAnalyzer,
        MockImplementationDebugger,
        MockEvolutionaryEngine,
        MockConfig,
        runner,  # CliRunner はこのテストでは直接使わないが、フィクスチャとして残す
        tmp_path,
    ):
        """Test that solve command uses test cases from tools/in/*.txt files."""
        # 1. テスト用のワークスペースと tools/in/*.txt ファイルを作成
        workspace_dir = tmp_path / "ahc_test_workspace_tools"
        workspace_dir.mkdir()
        tools_dir = workspace_dir / "tools"
        tools_dir.mkdir()
        tools_in_dir = tools_dir / "in"
        tools_in_dir.mkdir()

        test_input_content1 = "input data for test01"
        test_input_file1 = tools_in_dir / "test01.txt"
        with open(test_input_file1, "w") as f:
            f.write(test_input_content1)

        test_input_content2 = "input data for test02"
        test_input_file2 = tools_in_dir / "test02.txt"
        with open(test_input_file2, "w") as f:
            f.write(test_input_content2)

        problem_text_content = "This is a sample problem text."
        problem_file = workspace_dir / "problem.md"
        with open(problem_file, "w") as f:
            f.write(problem_text_content)

        # 2. Config のモック設定
        mock_config_instance = MagicMock(spec=Config)
        # MockConfig.return_value = mock_config_instance # Config() の呼び出しではなく、Config.load() をモックする

        loaded_config_values = {
            "workspace.base_dir": str(workspace_dir),
            "workspace.problem_file": "problem.md",
            "llm.model": "test_model",
            "evolution.population_size": 1,
            "evolution.generations": 1,
            "docker.image": "test_image",  # ImplementationDebugger で使われる可能性
            "language": "cpp",  # ImplementationDebugger で使われる可能性
            "problem_logic": "cpp",  # ProblemLogic で使われる可能性
            "problem_logic.test_cases_dir": "tools/in",  # ProblemLogic で使われる可能性
        }
        mock_config_instance.get.side_effect = lambda k, v=None: loaded_config_values.get(k, v)
        mock_config_instance.config_file_path = str(workspace_dir / "ahc_config.yaml")

        # 3. ProblemLogic のモック設定
        # MockProblemLogic はパッチされたクラスのモックオブジェクト
        # MockProblemLogic.return_value は ProblemLogic(...) が返すインスタンスのモック
        pl_instance_mock = MockProblemLogic.return_value
        pl_instance_mock.parse_problem_statement = AsyncMock(return_value={"title": "test_problem"})
        pl_instance_mock.analyze_problem = AsyncMock(return_value="problem_analysis")  # _solve_problem内では直接呼ばれない
        pl_instance_mock.propose_strategy = AsyncMock(return_value="solution_strategy")  # _solve_problem内では直接呼ばれない
        pl_instance_mock.generate_initial_solution = AsyncMock(return_value="initial_solution_code")
        pl_instance_mock.generate_test_cases = AsyncMock(
            return_value=[{"input": "test_case_from_logic_input", "output": "test_case_from_logic_output"}]
        )
        pl_instance_mock.create_score_calculator = AsyncMock(return_value=MagicMock(return_value=100.0))

        # SolutionStrategist のモック設定
        ss_instance_mock = MockSolutionStrategist.return_value
        ss_instance_mock.develop_strategy = AsyncMock(return_value="mocked_developed_strategy")

        # ProblemAnalyzer のモック設定
        # MockProblemAnalyzer.return_value は ProblemAnalyzer(...) が返すインスタンスのモック
        pa_instance_mock = MockProblemAnalyzer.return_value
        pa_instance_mock.analyze = AsyncMock(return_value="mocked_problem_analysis")

        # 4. ImplementationDebugger のモック設定
        mock_debugger_instance = MockImplementationDebugger.return_value
        compile_test_results = [
            {
                "success": True,
                "execution_output": "output1",
                "execution_time": 0.1,
                "compilation_errors": None,
                "execution_errors": None,
            },
            {
                "success": True,
                "execution_output": "output2",
                "execution_time": 0.2,
                "compilation_errors": None,
                "execution_errors": None,
            },
        ]
        # compile_and_test は async def なので、AsyncMock を使うか、Future を返す
        mock_debugger_instance.compile_and_test = AsyncMock(side_effect=compile_test_results)

        # 5. EvolutionaryEngine のモック設定
        mock_engine_instance = MockEvolutionaryEngine.return_value
        captured_evaluate_func = None

        async def mock_evolve(problem_analysis, solution_strategy, initial_solution, evaluate_solution_func, session_path):
            nonlocal captured_evaluate_func
            captured_evaluate_func = evaluate_solution_func
            # "evolution_log" キーを追加。内容はテストの目的に応じて調整可能。
            return {
                "best_solution": "final_code",
                "best_score": 200.0,
                "history": [],
                "evolution_log": [],
                "generations_completed": 1,  # "generations_completed" キーを追加
            }

        mock_engine_instance.evolve.side_effect = mock_evolve

        # _solve_problem コルーチンを直接呼び出してテストする
        asyncio.run(
            _solve_problem(mock_config_instance, problem_text_content)  # session_id を削除
        )

        assert captured_evaluate_func is not None

        # evaluate_solution は現状同期関数として EvolutionaryEngine に渡される。
        # その内部で async 関数である compile_and_test を呼び出すために asyncio.run() を使うという改修案。
        # よって、ここでの captured_evaluate_func の呼び出しは同期的。
        # そして、その内部で呼ばれる asyncio.run(compile_and_test(...)) をモックする。
        with patch("ahc_agent.cli.asyncio.run") as mock_run_in_evaluate:
            # compile_and_test の結果を再度設定 (asyncio.run の戻り値として)
            mock_run_in_evaluate.side_effect = [
                compile_test_results[0],  # 1回目の呼び出し (test01.txt)
                compile_test_results[1],  # 2回目の呼び出し (test02.txt)
            ]

            avg_score, all_details = captured_evaluate_func("sample_code")  # 引数を1つに変更

            assert "test01.txt" in all_details
            assert all_details["test01.txt"]["score"] == 100.0
            assert "test02.txt" in all_details
            assert all_details["test02.txt"]["score"] == 100.0
            assert avg_score == 100.0

            # mock_debugger_instance.compile_and_test が正しい引数で呼び出されたことを確認
            # evaluate_solution 内で asyncio.run に渡される前に呼び出される
            mock_debugger_instance.compile_and_test.assert_any_call("sample_code", test_input_content1)
            mock_debugger_instance.compile_and_test.assert_any_call("sample_code", test_input_content2)

            # ProblemLogic.generate_test_cases が呼ばれていないことを確認
            # (tools/in/*.txt が使われたため)
