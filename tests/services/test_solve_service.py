from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ahc_agent.config import Config
from ahc_agent.core.debugger import ImplementationDebugger
from ahc_agent.core.workspace_store import WorkspaceStore
from ahc_agent.services.solve_service import SolveService
from ahc_agent.utils.docker_manager import DockerManager
from ahc_agent.utils.llm import LLMClient


@pytest.fixture
def mock_llm_client(mocker):  # Use mocker for pytest-mock features
    return mocker.AsyncMock(spec=LLMClient)


@pytest.fixture
def mock_docker_manager(mocker):
    return mocker.MagicMock(spec=DockerManager)


@pytest.fixture
def mock_config(mocker, tmp_path):
    config = mocker.MagicMock(spec=Config)
    config.get.side_effect = lambda key, default=None: {
        "llm": {},
        "docker": {},
        "analyzer": {},
        "strategist": {},
        "evolution": {"time_limit_seconds": 10, "max_generations": 1, "population_size": 2, "score_plateau_generations": 1},
        "debugger": {},
        "problem_logic": {"test_cases_count": 1},  # Default number of test cases if not from tools/in
        "workspace.base_dir": str(tmp_path / "test_ws"),  # 作業ディレクトリとして使用される
        "contest_id": "test_contest",  # Default contest_id from config
    }.get(key, default)  # Return the default if key is not in the dict
    return config


@pytest.fixture
def mock_workspace_store(mocker):
    ws = mocker.MagicMock(spec=WorkspaceStore)
    # Default behaviors for a fresh solve
    ws.load_problem_text.return_value = "problem text from workspace"
    ws.problem_id = "test_problem_ws"
    ws.load_problem_analysis.return_value = None
    ws.load_solution_strategy.return_value = None
    ws.get_best_solution_code_and_meta.return_value = None  # For get_best_solution()
    ws.load_solution_code.return_value = None  # For loading specific solutions like 'initial'
    ws.load_solution_metadata.return_value = None  # For loading specific solutions like 'initial'

    # Mock for solutions_dir attribute
    # Used as: self.workspace_store.solutions_dir / "filename.cpp"
    # Then str(result_of_truediv)
    mock_final_solution_path = mocker.MagicMock(spec=Path)
    mock_final_solution_path.__str__.return_value = "mocked/solutions/best.cpp"
    ws.solutions_dir = mocker.MagicMock(spec=Path)
    ws.solutions_dir.__truediv__.return_value = mock_final_solution_path

    # Mock for get_workspace_dir() method, used for generations_dir
    # Used as: workspace_dir / "generations", then str(result)
    mock_generations_subdir_path = mocker.MagicMock(spec=Path)
    mock_generations_subdir_path.__str__.return_value = "mocked_workspace/generations"

    mock_workspace_root_path = mocker.MagicMock(spec=Path)
    # This mock_workspace_root_path is what get_workspace_dir() returns.
    # When (mock_workspace_root_path / "subdir") is called, it should return another path mock.
    # For example, if it's used for generations_dir = workspace_dir / "generations"
    mock_workspace_root_path.__truediv__.return_value = mock_generations_subdir_path
    ws.get_workspace_dir.return_value = mock_workspace_root_path

    return ws


# Patch all core modules used by SolveService
# These are patched at the location where they are imported by solve_service.py
@patch("ahc_agent.services.solve_service.ProblemLogic")
@patch("ahc_agent.services.solve_service.ImplementationDebugger")
@patch("ahc_agent.services.solve_service.EvolutionaryEngine")
@patch("ahc_agent.services.solve_service.SolutionStrategist")
@patch("ahc_agent.services.solve_service.ProblemAnalyzer")
@pytest.mark.asyncio
async def test_run_solve_fresh_solve(
    MockProblemAnalyzer,
    MockSolutionStrategist,
    MockEvolutionaryEngine,
    MockImplementationDebugger,
    MockProblemLogic,
    mock_llm_client,
    mock_docker_manager,
    mock_config,
    mock_workspace_store,
):
    # Arrange: Setup mock instances for core modules
    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Test Problem Analysis"})

    # Important: Make sure the service.problem_analyzer is the same as mock_pa_instance
    MockProblemAnalyzer.reset_mock()
    MockProblemAnalyzer.return_value = mock_pa_instance

    mock_ss_instance = MockSolutionStrategist.return_value
    mock_ss_instance.develop_strategy = AsyncMock(return_value={"approach": "Test Strategy"})

    # Important: Make sure the service.solution_strategist is the same as mock_ss_instance
    MockSolutionStrategist.reset_mock()
    MockSolutionStrategist.return_value = mock_ss_instance

    mock_pl_instance = MockProblemLogic.return_value
    mock_pl_instance.generate_initial_solution = AsyncMock(return_value="initial_code")
    # Simulate no tools/in files, so generate_test_cases is called
    mock_pl_instance.generate_test_cases = AsyncMock(return_value=[{"name": "t1", "input": "in1"}])
    mock_pl_instance.create_score_calculator = AsyncMock(return_value=MagicMock(return_value=100.0))

    # Important: Make sure the service.problem_logic is the same as mock_pl_instance
    MockProblemLogic.reset_mock()
    MockProblemLogic.return_value = mock_pl_instance

    # Mock ImplementationDebugger used by _evaluate_solution_wrapper
    mock_id_instance = MockImplementationDebugger.return_value
    mock_id_instance.compile_and_test = AsyncMock(return_value={"success": True, "execution_output": "out1", "execution_time": 0.1})

    # Mock EvolutionaryEngine
    mock_ee_instance = MockEvolutionaryEngine.return_value
    mock_ee_instance.evolve = AsyncMock(
        return_value={
            "best_solution": "best_code_evolved",
            "best_score": 200.0,
            "generations_completed": 1,
            "evolution_log": [{"gen": 1, "score": 200}],
        }
    )

    # SolveService instance
    service = SolveService(
        llm_client=mock_llm_client,
        docker_manager=mock_docker_manager,
        config=mock_config,
        workspace_store=mock_workspace_store,
    )

    # get_best_solutionがNoneを返すように設定し、generate_initial_solutionが呼ばれるようにする
    mock_workspace_store.get_best_solution.return_value = None

    problem_text_arg = "This is the problem statement from argument."

    # problem_textは引数から取得される

    # Act
    await service.run_solve(problem_text_arg, interactive=False)

    # Assert
    mock_pa_instance.analyze.assert_called_once_with(problem_text_arg)
    mock_workspace_store.save_problem_analysis.assert_called_once_with({"title": "Test Problem Analysis"})
    mock_workspace_store.save_solution_strategy.assert_called_once_with({"approach": "Test Strategy"})

    # Strategy, Solution generation
    mock_ss_instance.develop_strategy.assert_called_once_with({"title": "Test Problem Analysis"})

    mock_pl_instance.generate_initial_solution.assert_called_once_with({"title": "Test Problem Analysis"})
    # Note: save_solution for initial solution is not explicitly in SolveService, only for best.

    # Test case generation (assuming tools/in is empty or not found by default)
    # Patch Path.exists for tools/in to return False to force generation
    with patch("pathlib.Path.exists", return_value=False):
        # Re-run or check if already called. Since it's complex to re-run part, ensure mocks are set before main Act.
        # For this test, we'll assume the default path for Path.exists (if called) is False.
        # Or, more robustly, ensure the call happens within a specific patch context if needed.
        # The service's ProblemLogic instance is `self.problem_logic`
        # It's already mocked as mock_pl_instance.
        pass  # generate_test_cases should have been called if tools/in is effectively empty

    mock_pl_instance.generate_test_cases.assert_called_once_with({"title": "Test Problem Analysis"}, 3)  # 3が実際に使用されているテストケース数
    mock_pl_instance.create_score_calculator.assert_called_once_with({"title": "Test Problem Analysis"})

    # Evolution
    mock_ee_instance.evolve.assert_called_once()
    # Check some args of evolve, e.g., initial_solution_code
    assert mock_ee_instance.evolve.call_args[0][2] == "initial_code"  # initial_solution_code

    mock_workspace_store.save_evolution_log.assert_called_once_with([{"gen": 1, "score": 200}])
    mock_workspace_store.save_solution.assert_called_once_with("best", {"code": "best_code_evolved", "score": 200.0, "generation": 1})


@patch("ahc_agent.services.solve_service.ProblemLogic")
@patch("ahc_agent.services.solve_service.ImplementationDebugger")
@patch("ahc_agent.services.solve_service.EvolutionaryEngine")
@patch("ahc_agent.services.solve_service.SolutionStrategist")
@patch("ahc_agent.services.solve_service.ProblemAnalyzer")
@pytest.mark.asyncio
async def test_run_solve_resuming_solve(
    MockProblemAnalyzer,
    MockSolutionStrategist,
    MockEvolutionaryEngine,
    MockImplementationDebugger,
    MockProblemLogic,
    mock_llm_client,
    mock_docker_manager,
    mock_config,
    mock_workspace_store,
):
    # Arrange: KB returns existing data
    existing_analysis = {"title": "Existing Analysis"}
    existing_strategy = {"approach": "Existing Strategy"}
    existing_best_solution = {"code": "existing_best_code", "score": 150.0}

    mock_workspace_store.get_problem_analysis.return_value = existing_analysis
    mock_workspace_store.get_solution_strategy.return_value = existing_strategy
    mock_workspace_store.get_best_solution.return_value = existing_best_solution  # To be used as initial_solution_code
    mock_workspace_store.load_problem_analysis.return_value = existing_analysis
    mock_workspace_store.load_solution_strategy.return_value = existing_strategy

    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Test Problem Analysis"})

    # Important: Make sure the service.problem_analyzer is the same as mock_pa_instance
    MockProblemAnalyzer.reset_mock()
    MockProblemAnalyzer.return_value = mock_pa_instance

    mock_ss_instance = MockSolutionStrategist.return_value
    mock_ss_instance.develop_strategy = AsyncMock(return_value={"approach": "Test Strategy"})

    # Important: Make sure the service.solution_strategist is the same as mock_ss_instance
    MockSolutionStrategist.reset_mock()
    MockSolutionStrategist.return_value = mock_ss_instance

    mock_pl_instance = MockProblemLogic.return_value
    mock_pl_instance.generate_test_cases = AsyncMock(return_value=[{"name": "t1", "input": "in1"}])
    mock_pl_instance.create_score_calculator = AsyncMock(return_value=MagicMock(return_value=100.0))

    mock_id_instance = MockImplementationDebugger.return_value
    mock_id_instance.compile_and_test = AsyncMock(return_value={"success": True, "execution_output": "out1", "execution_time": 0.1})

    mock_ee_instance = MockEvolutionaryEngine.return_value
    mock_ee_instance.evolve = AsyncMock(
        return_value={
            "best_solution": "evolved_from_existing",
            "best_score": 250.0,
            "generations_completed": 1,
            "evolution_log": [],
        }
    )

    service = SolveService(
        llm_client=mock_llm_client,
        docker_manager=mock_docker_manager,
        config=mock_config,
        workspace_store=mock_workspace_store,
    )

    problem_text_arg = "problem text for resumed solve"  # 既存の解析と戦略がある場合のテスト用

    # Act
    await service.run_solve(problem_text_arg, interactive=False)

    # Assert
    # 既存の解析と戦略が読み込まれる
    mock_workspace_store.load_problem_analysis.assert_called_once()
    mock_workspace_store.load_solution_strategy.assert_called_once()

    # Analysis and Strategy should NOT be called as data exists
    mock_pa_instance.analyze.assert_not_called()
    mock_ss_instance.develop_strategy.assert_not_called()
    mock_pl_instance.generate_initial_solution.assert_not_called()  # Should use existing_best_solution.code

    # Test case and score calculator generation still happens
    # (Assuming tools/in is empty for this test for simplicity)
    mock_pl_instance.generate_test_cases.assert_called_once()
    mock_pl_instance.create_score_calculator.assert_called_once()

    # Evolution should still run, using existing_best_solution.code as initial
    mock_ee_instance.evolve.assert_called_once()
    assert mock_ee_instance.evolve.call_args[0][0] == existing_analysis  # problem_analysis
    assert mock_ee_instance.evolve.call_args[0][1] == existing_strategy  # solution_strategy
    assert mock_ee_instance.evolve.call_args[0][2] == "existing_best_code"  # initial_solution_code

    mock_workspace_store.save_solution.assert_called_once()  # For best solution from evolve


@patch("ahc_agent.services.solve_service.ProblemLogic")
@patch("ahc_agent.services.solve_service.ImplementationDebugger")
@patch("ahc_agent.services.solve_service.EvolutionaryEngine")
@patch("ahc_agent.services.solve_service.SolutionStrategist")
@patch("ahc_agent.services.solve_service.ProblemAnalyzer")
@patch("builtins.open", new_callable=MagicMock)  # Using MagicMock for open, not mock_open
@patch("pathlib.Path.glob")
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.exists", return_value=True)
@pytest.mark.asyncio
async def test_run_solve_tools_in_handling(
    mock_path_exists,
    mock_path_is_dir,
    mock_path_glob,
    mock_builtin_open,
    MockProblemAnalyzer,
    MockSolutionStrategist,
    MockEvolutionaryEngine,
    MockImplementationDebugger,
    MockProblemLogic,
    mock_llm_client,
    mock_docker_manager,
    mock_config,
    mock_workspace_store,
):
    # Arrange
    # Mocks for core modules (similar to fresh_solve, but focus on test case generation)
    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Analysis"})
    mock_ss_instance = MockSolutionStrategist.return_value
    mock_ss_instance.develop_strategy = AsyncMock(return_value={"strat": "Strategy"})
    mock_pl_instance = MockProblemLogic.return_value
    mock_pl_instance.generate_initial_solution = AsyncMock(return_value="initial_code_tools_in")
    mock_pl_instance.create_score_calculator = AsyncMock(return_value=MagicMock())
    mock_ee_instance = MockEvolutionaryEngine.return_value
    mock_ee_instance.evolve = AsyncMock(return_value={"best_solution": "code", "best_score": 1, "generations_completed": 1, "evolution_log": []})

    # Simulate tools/in files
    mock_file_content = "test_case_content_from_file"
    # Configure mock_open to simulate reading file content
    mock_builtin_open.return_value.__enter__.return_value.read.return_value = mock_file_content

    # Configure glob to return a list of mock Path objects
    mock_test_file_path = MagicMock(spec=Path)
    mock_test_file_path.name = "0000.txt"
    mock_path_glob.return_value = [mock_test_file_path]

    service = SolveService(
        llm_client=mock_llm_client,
        docker_manager=mock_docker_manager,
        config=mock_config,
        workspace_store=mock_workspace_store,
    )

    # Act
    await service.run_solve("problem_text_tools", interactive=False)

    # Assert
    # Key assertion: ProblemLogic.generate_test_cases should NOT be called
    mock_pl_instance.generate_test_cases.assert_not_called()

    # Check that Path.glob was called on the correct directory
    # The path is constructed as Path(self.config.get("workspace.base_dir")) / "tools" / "in"
    Path(mock_config.get("workspace.base_dir")) / "tools" / "in"
    # This is a bit tricky as Path() is called multiple times.
    # We can check that *a* Path instance had glob called, or be more specific if Path itself is patched.
    # For now, assume mock_path_glob is the glob method of the specific Path instance.
    # This requires patching Path instances' methods, not the class method globally, or complex side effects.
    # A simpler check might be on mock_path_glob if it's a direct patch to pathlib.Path.glob.
    # The current patching is `pathlib.Path.glob` which is a class method effectively.
    # We need to check if it was called on an instance that matches `expected_tools_in_path`.
    # This is hard with current patching. A more direct approach is to check if open was called.
    mock_builtin_open.assert_called_once_with(mock_test_file_path)  # Check if file was opened

    # Check that evolve was called (implies test cases were loaded and eval func was created)
    mock_ee_instance.evolve.assert_called_once()
    # To verify the loaded test cases were used, one would need to inspect the
    # lambda function passed to evolve, which is complex.
    # A simpler check is that generate_test_cases was NOT called.


@pytest.mark.asyncio
async def test_evaluate_solution_wrapper(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store):  # Need other mocks for ID
    # Arrange
    mock_id_instance = AsyncMock(spec=ImplementationDebugger)  # _evaluate_solution_wrapper uses an ID instance

    service = SolveService(llm_client=mock_llm_client, docker_manager=mock_docker_manager, config=mock_config, workspace_store=mock_workspace_store)

    # Override the service's debugger instance with our specific mock for this test
    # This is needed because _evaluate_solution_wrapper is called internally by the evolve callback,
    # but here we test it directly, so we need to provide the debugger it would expect.
    # The service initializes self.implementation_debugger in its __init__.
    # We can pass our mock_id_instance when creating SolveService, or patch it.
    # For direct test of wrapper, let's pass it if service allowed, or use the one from service.
    # The service instantiates its own ID. So, we need to patch the service's ID.
    service.implementation_debugger = mock_id_instance  # Replace the service's debugger

    test_code = "some_code"
    test_cases = [
        {"name": "tc1", "input": "in1"},
        {"name": "tc2", "input": "in2"},
        {"name": "tc3", "no_input_field": True},  # inputフィールドがない無効なテストケース
    ]
    mock_score_calculator = MagicMock()

    # Mock results from implementation_debugger.compile_and_test
    # 3つのテストケース全てに対応する値を設定
    mock_id_instance.compile_and_test.side_effect = [
        {"success": True, "execution_output": "out1", "execution_time": 0.1},  # tc1
        {"success": False, "compilation_errors": "compile_err", "execution_errors": None, "execution_time": 0},  # tc2
        # tc3は'input'フィールドがないため、実際には呼び出されないはずですが、
        # コードが変更された場合に備えて値を設定しておきます
        {"success": False, "compilation_errors": "tc3_error", "execution_errors": None, "execution_time": 0},  # tc3
    ]

    # Mock score_calculator behavior
    # 十分な数の戻り値を用意する
    mock_score_calculator.return_value = 100.0  # side_effectの代わりにreturn_valueを使用

    # Act
    avg_score, details = await service._evaluate_solution_wrapper(
        test_code,
        test_cases,
        mock_score_calculator,
        mock_id_instance,  # Pass the mock ID
    )

    # Assert
    assert mock_id_instance.compile_and_test.call_count == 2  # Called for tc1, tc2. Not for tc3.
    mock_id_instance.compile_and_test.assert_any_call(test_code, "in1")
    mock_id_instance.compile_and_test.assert_any_call(test_code, "in2")

    mock_score_calculator.assert_called_once_with("in1", "out1")  # Only for successful tc1

    assert avg_score == 100.0 / 3  # Total score / num_test_cases (including invalid one)
    assert details["tc1"]["score"] == 100.0
    assert details["tc2"]["score"] == 0
    assert "compile_err" in details["tc2"]["error"]
    assert details["tc3"]["score"] == 0  # Due to missing 'input'
    assert "Missing 'input' field" in details["tc3"]["error"]


@patch("ahc_agent.services.solve_service.ProblemLogic")
@patch("ahc_agent.services.solve_service.ImplementationDebugger")  # Not strictly needed if not testing evolve part
@patch("ahc_agent.services.solve_service.EvolutionaryEngine")  # Not strictly needed
@patch("ahc_agent.services.solve_service.SolutionStrategist")  # Not strictly needed
@patch("ahc_agent.services.solve_service.ProblemAnalyzer")
@patch("ahc_agent.services.solve_service.questionary.select")
@pytest.mark.asyncio
async def test_run_interactive_solve_basic_analyze_and_exit(
    mock_questionary_select,
    MockProblemAnalyzer,
    MockSolutionStrategist,
    MockEvolutionaryEngine,
    MockImplementationDebugger,
    MockProblemLogic,  # These are from service init, not directly used in this simple test
    mock_llm_client,
    mock_docker_manager,
    mock_config,
    mock_workspace_store,
):
    # Arrange
    # Service will use its own instances of core modules based on its __init__
    # We need to ensure these instances (or their mocks) behave as expected.
    # MockProblemAnalyzer is the class, MockProblemAnalyzer.return_value is the instance.
    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Interactive Analysis"})

    # Important: Make sure the service.problem_analyzer is the same as mock_pa_instance
    MockProblemAnalyzer.reset_mock()
    MockProblemAnalyzer.return_value = mock_pa_instance

    # Configure workspace for interactive solve
    # problem_text is fetched from workspace in interactive mode
    mock_workspace_store.load_problem_text.return_value = "interactive problem content"
    mock_workspace_store.problem_id = "interactive_problem"
    # Initial load_problem_analysis must return a value for interactive solve to start
    mock_workspace_store.load_problem_analysis.return_value = {"title": "Existing Analysis"}
    # Initial load_solution_strategy must return a value for interactive solve to start
    mock_workspace_store.load_solution_strategy.return_value = {"approach": "Existing Strategy"}

    service = SolveService(llm_client=mock_llm_client, docker_manager=mock_docker_manager, config=mock_config, workspace_store=mock_workspace_store)

    # Simulate user selecting "analyze" then "exit"
    mock_select = AsyncMock()
    mock_select.ask_async.side_effect = ["analyze", "exit"]
    mock_questionary_select.return_value = mock_select

    # Act
    # run_interactive_solve is called by run_solve if interactive=True
    # Or, we can test it directly if it's public (it is in the service)
    await service.run_interactive_solve()

    # Assert
    # Check that ProblemAnalyzer.analyze was called because user typed "analyze"
    # The instance used is service.problem_analyzer
    # This was set up by @patch('ahc_agent.services.solve_service.ProblemAnalyzer')
    # and its return_value (the instance mock) is mock_pa_instance.
    # モックがテスト中にリセットされているため、直接サービスのproblem_analyzerを確認する
    service.problem_analyzer.analyze.assert_called_once_with("interactive problem content")

    # Check that the analysis was saved
    mock_workspace_store.save_problem_analysis.assert_called_once_with({"title": "Interactive Analysis"})

    # Check that questionary.select was called twice
    assert mock_questionary_select.call_count == 2
    # First call for main menu, second call would be after selecting "analyze"
    mock_questionary_select.assert_called()


@pytest.mark.asyncio
async def test_run_analyze_step_success(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    # We need to ensure that the ProblemAnalyzer used by the service is the one we mock.
    # The SolveService __init__ creates its own ProblemAnalyzer.
    # So, we either patch ProblemAnalyzer globally (like other tests) or mock the instance post-init.

    # Patching globally for consistency with other tests in this file:
    with patch("ahc_agent.services.solve_service.ProblemAnalyzer") as MockProblemAnalyzerGlobal:
        mock_pa_instance_global = MockProblemAnalyzerGlobal.return_value
        mock_pa_instance_global.analyze = AsyncMock(return_value={"title": "Analyzed Problem"})

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)
        # The service's problem_analyzer will be an instance of the patched MockProblemAnalyzerGlobal

        problem_text_arg = "Test problem statement"

        # Act
        result = await solve_service.run_analyze_step(problem_text=problem_text_arg)

        # Assert
        mock_pa_instance_global.analyze.assert_called_once_with(problem_text_arg)
        mock_workspace_store.save_problem_analysis.assert_called_once_with({"title": "Analyzed Problem"})
        assert result == {"title": "Analyzed Problem"}


@pytest.mark.asyncio
async def test_run_analyze_step_no_problem_text_loads_from_store(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemAnalyzer") as MockProblemAnalyzerGlobal:
        mock_pa_instance_global = MockProblemAnalyzerGlobal.return_value
        mock_pa_instance_global.analyze = AsyncMock(return_value={"title": "Analyzed Stored Problem"})

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        stored_problem_text = "Problem text from store"
        # Configure the mock_workspace_store passed into SolveService
        mock_workspace_store.load_problem_text.return_value = stored_problem_text

        # Act
        result = await solve_service.run_analyze_step(problem_text=None)  # Pass None or empty string

        # Assert
        mock_workspace_store.load_problem_text.assert_called_once()
        mock_pa_instance_global.analyze.assert_called_once_with(stored_problem_text)
        mock_workspace_store.save_problem_analysis.assert_called_once_with({"title": "Analyzed Stored Problem"})
        assert result == {"title": "Analyzed Stored Problem"}


@pytest.mark.asyncio
async def test_run_analyze_step_no_problem_text_and_not_in_store(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemAnalyzer") as MockProblemAnalyzerGlobal:
        mock_pa_instance_global = MockProblemAnalyzerGlobal.return_value
        mock_pa_instance_global.analyze = AsyncMock()  # Should not be called

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        # Configure mock_workspace_store
        mock_workspace_store.load_problem_text.return_value = None  # Store also has no problem text

        # Act
        result = await solve_service.run_analyze_step(problem_text=None)

        # Assert
        mock_workspace_store.load_problem_text.assert_called_once()
        mock_pa_instance_global.analyze.assert_not_called()
        mock_workspace_store.save_problem_analysis.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_strategy_step_success(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.SolutionStrategist") as MockSolutionStrategistGlobal:
        mock_ss_instance_global = MockSolutionStrategistGlobal.return_value
        mock_ss_instance_global.develop_strategy = AsyncMock(return_value={"approach": "Test Strategy"})

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store.load_problem_analysis.return_value = sample_analysis_data

        # Act
        result = await solve_service.run_strategy_step()

        # Assert
        mock_workspace_store.load_problem_analysis.assert_called_once()
        mock_ss_instance_global.develop_strategy.assert_called_once_with(sample_analysis_data)
        mock_workspace_store.save_solution_strategy.assert_called_once_with({"approach": "Test Strategy"})
        assert result == {"approach": "Test Strategy"}


@pytest.mark.asyncio
async def test_run_strategy_step_no_analysis(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.SolutionStrategist") as MockSolutionStrategistGlobal:
        mock_ss_instance_global = MockSolutionStrategistGlobal.return_value
        mock_ss_instance_global.develop_strategy = AsyncMock()  # Should not be called

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        mock_workspace_store.load_problem_analysis.return_value = None  # No analysis data

        # Act
        result = await solve_service.run_strategy_step()

        # Assert
        mock_workspace_store.load_problem_analysis.assert_called_once()
        mock_ss_instance_global.develop_strategy.assert_not_called()
        mock_workspace_store.save_solution_strategy.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_testcases_step_load_from_tools_success(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal, patch(
        "ahc_agent.services.solve_service.Path"  # Patch where Path is looked up
    ) as MockPathClass:
        # Configure Path mock
        mock_path_base = MockPathClass.return_value
        mock_path_tools = MagicMock(spec=Path)
        mock_path_tools_in = MagicMock(spec=Path)

        mock_path_base.__truediv__.side_effect = lambda segment: mock_path_tools if segment == "tools" else MagicMock(spec=Path)
        mock_path_tools.__truediv__.side_effect = lambda segment: mock_path_tools_in if segment == "in" else MagicMock(spec=Path)

        mock_path_tools_in.exists.return_value = True
        mock_path_tools_in.is_dir.return_value = True
        # mock_tools_in_path_instance is now mock_path_tools_in for assertion purposes
        mock_tools_in_path_instance = mock_path_tools_in

        mock_file_path = MagicMock(spec=Path)
        mock_file_path.name = "0000.txt"
        mock_tools_in_path_instance.glob.return_value = [mock_file_path]

        mock_open_fn = mocker.patch("builtins.open", mocker.mock_open(read_data="test_input_data"))

        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.create_score_calculator = AsyncMock(return_value=MagicMock())  # Returns a mock calculator
        mock_pl_instance_global.generate_test_cases = AsyncMock(return_value=[])  # Ensure it's an AsyncMock even if not expected to be called

        # Ensure config mock returns a specific value for 'workspace.base_dir'
        def mock_config_get_side_effect(key, default=None):
            print(f"DEBUG: mock_config.get called with key='{key}', default='{default}'")
            if key == "workspace.base_dir":
                value = "/mocked/workspace/base"
                print(f"DEBUG: mock_config.get returning '{value}' for key '{key}'")
                return value
            print(f"DEBUG: mock_config.get returning default='{default}' for key '{key}'")
            return default  # or raise an error if unexpected keys are called

        mock_config.get.side_effect = mock_config_get_side_effect

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store.load_problem_analysis.return_value = sample_analysis_data

        # Act
        result = await solve_service.run_testcases_step(load_from_tools=True, num_to_generate=3)

        # Assert
        mock_workspace_store.load_problem_analysis.assert_called_once()
        # Path(...) was called, then exists(), is_dir(), glob()
        MockPathClass.assert_called()  # Check Path was instantiated
        mock_tools_in_path_instance.exists.assert_called_once()
        mock_tools_in_path_instance.is_dir.assert_called_once()
        mock_tools_in_path_instance.glob.assert_called_once_with("*.txt")

        # builtins.open was called
        mock_open_fn.assert_called_once_with(mock_file_path)

        mock_pl_instance_global.generate_test_cases.assert_not_called()
        mock_pl_instance_global.create_score_calculator.assert_called_once_with(sample_analysis_data)

        assert result is not None
        assert len(result["test_cases"]) == 1
        assert result["test_cases"][0]["input"] == "test_input_data"
        assert result["score_calculator"] is not None


@pytest.mark.asyncio
async def test_run_testcases_step_load_from_tools_empty_dir_generates(
    mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker
):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal, patch("pathlib.Path") as MockPathClass:
        mock_tools_in_path_instance = MockPathClass.return_value
        mock_tools_in_path_instance.exists.return_value = True
        mock_tools_in_path_instance.is_dir.return_value = True
        mock_tools_in_path_instance.glob.return_value = []  # Empty directory

        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_test_cases = AsyncMock(return_value=[{"name": "gen1", "input": "gen_in1"}])
        mock_pl_instance_global.create_score_calculator = AsyncMock(return_value=MagicMock())

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)
        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store.load_problem_analysis.return_value = sample_analysis_data

        # Act
        result = await solve_service.run_testcases_step(load_from_tools=True, num_to_generate=3)

        # Assert
        mock_pl_instance_global.generate_test_cases.assert_called_once_with(sample_analysis_data, 3)
        mock_pl_instance_global.create_score_calculator.assert_called_once_with(sample_analysis_data)
        assert result is not None
        assert len(result["test_cases"]) == 1
        assert result["test_cases"][0]["name"] == "gen1"


@pytest.mark.asyncio
async def test_run_testcases_step_generate_explicitly(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_test_cases = AsyncMock(return_value=[{"name": "gen1", "input": "gen_in1"}])
        mock_pl_instance_global.create_score_calculator = AsyncMock(return_value=MagicMock())

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)
        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store.load_problem_analysis.return_value = sample_analysis_data

        # Act
        result = await solve_service.run_testcases_step(load_from_tools=False, num_to_generate=5)

        # Assert
        mock_pl_instance_global.generate_test_cases.assert_called_once_with(sample_analysis_data, 5)
        mock_pl_instance_global.create_score_calculator.assert_called_once_with(sample_analysis_data)
        assert result is not None
        assert len(result["test_cases"]) == 1


@pytest.mark.asyncio
async def test_run_testcases_step_no_analysis(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)
        mock_workspace_store.load_problem_analysis.return_value = None  # No analysis

        # Act
        result = await solve_service.run_testcases_step(load_from_tools=False, num_to_generate=3)

        # Assert
        mock_pl_instance_global.generate_test_cases.assert_not_called()
        mock_pl_instance_global.create_score_calculator.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_testcases_step_generation_fails(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_test_cases = AsyncMock(return_value=[])  # Generation returns empty list
        # create_score_calculator might still be called if analysis exists, but the method should return None overall
        mock_pl_instance_global.create_score_calculator = AsyncMock(return_value=MagicMock())

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)
        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store.load_problem_analysis.return_value = sample_analysis_data

        # Act
        result = await solve_service.run_testcases_step(load_from_tools=False, num_to_generate=3)

        # Assert
        mock_pl_instance_global.generate_test_cases.assert_called_once_with(sample_analysis_data, 3)
        # If generate_test_cases returns [], the service method should return None before calling create_score_calculator.
        # Based on current service code: `if not test_cases: logger.error(...); return None`
        mock_pl_instance_global.create_score_calculator.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_initial_solution_step_success(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        expected_code = "def main():\n  print('hello')"
        mock_pl_instance_global.generate_initial_solution = AsyncMock(return_value=expected_code)

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        sample_analysis_data = {"title": "Analyzed Problem for Initial"}
        mock_workspace_store.load_problem_analysis.return_value = sample_analysis_data

        # Act
        result = await solve_service.run_initial_solution_step()

        # Assert
        mock_workspace_store.load_problem_analysis.assert_called_once()
        mock_pl_instance_global.generate_initial_solution.assert_called_once_with(sample_analysis_data)
        mock_workspace_store.save_solution.assert_called_once_with("initial", {"code": expected_code, "score": 0, "generation": 0})
        assert result == expected_code


@pytest.mark.asyncio
async def test_run_initial_solution_step_no_analysis(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_initial_solution = AsyncMock()

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)
        mock_workspace_store.load_problem_analysis.return_value = None  # No analysis

        # Act
        result = await solve_service.run_initial_solution_step()

        # Assert
        mock_pl_instance_global.generate_initial_solution.assert_not_called()
        mock_workspace_store.save_solution.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_evolve_step_success_with_override_code(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch(
        "ahc_agent.services.solve_service.ensure_directory"
    ) as mock_ensure_dir:  # Mock ensure_directory
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        expected_evolve_result = {
            "best_solution": "evolved_code",
            "best_score": 1000,
            "generations_completed": 10,
            "evolution_log": [{"gen": 10, "score": 1000}],
        }
        mock_ee_instance_global.evolve = AsyncMock(return_value=expected_evolve_result)

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        sample_analysis = {"title": "Evolve Analysis"}
        sample_strategy = {"approach": "Evolve Strategy"}
        mock_workspace_store.load_problem_analysis.return_value = sample_analysis
        mock_workspace_store.load_solution_strategy.return_value = sample_strategy

        mock_test_cases = [{"name": "tc1", "input": "in1"}]
        mock_score_calculator = MagicMock()
        initial_code_override_arg = "override_initial_code"

        # Act
        result = await solve_service.run_evolve_step(
            test_cases=mock_test_cases,
            score_calculator=mock_score_calculator,
            max_generations=10,
            population_size=20,
            time_limit_seconds=60,
            initial_code_override=initial_code_override_arg,
        )

        # Assert
        mock_workspace_store.load_problem_analysis.assert_called_once()
        mock_workspace_store.load_solution_strategy.assert_called_once()
        mock_ensure_dir.assert_called()  # Check generations directory creation helper

        mock_ee_instance_global.evolve.assert_called_once()
        call_args = mock_ee_instance_global.evolve.call_args[0]
        assert call_args[0] == sample_analysis
        assert call_args[1] == sample_strategy
        assert call_args[2] == initial_code_override_arg  # Initial code
        assert callable(call_args[3])  # eval_func
        # generations_dir path is based on workspace_store.get_workspace_dir()
        # We can check that it's a string path.
        assert isinstance(call_args[4], str)

        assert solve_service.evolutionary_engine.max_generations == 10
        assert solve_service.evolutionary_engine.population_size == 20
        assert solve_service.evolutionary_engine.time_limit_seconds == 60

        mock_workspace_store.save_evolution_log.assert_called_once_with(expected_evolve_result["evolution_log"])
        mock_workspace_store.save_solution.assert_called_once_with(
            "best",
            {
                "code": expected_evolve_result["best_solution"],
                "score": expected_evolve_result["best_score"],
                "generation": expected_evolve_result["generations_completed"],
            },
        )
        assert result == expected_evolve_result


@pytest.mark.asyncio
async def test_run_evolve_step_success_load_best_from_kb(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch(
        "ahc_agent.services.solve_service.ensure_directory"
    ):
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        mock_ee_instance_global.evolve = AsyncMock(
            return_value={"best_solution": "code", "best_score": 1, "generations_completed": 1, "evolution_log": []}
        )

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)
        mock_workspace_store.load_problem_analysis.return_value = {"title": "Analysis"}
        mock_workspace_store.load_solution_strategy.return_value = {"approach": "Strategy"}

        kb_best_code = {"code": "kb_best_code", "score": 100}
        mock_workspace_store.get_best_solution.return_value = kb_best_code
        # mock_workspace_store.get_solution was removed, load_solution_code/metadata will return None by default from fixture

        # Act
        await solve_service.run_evolve_step(
            test_cases=[{"name": "tc1"}],
            score_calculator=MagicMock(),
            max_generations=1,
            population_size=1,
            time_limit_seconds=1,
            initial_code_override=None,  # This is key for this test
        )

        # Assert
        mock_ee_instance_global.evolve.assert_called_once()
        assert mock_ee_instance_global.evolve.call_args[0][2] == kb_best_code["code"]


@pytest.mark.asyncio
async def test_run_evolve_step_success_load_initial_from_kb(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch(
        "ahc_agent.services.solve_service.ensure_directory"
    ):
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        mock_ee_instance_global.evolve = AsyncMock(
            return_value={"best_solution": "code", "best_score": 1, "generations_completed": 1, "evolution_log": []}
        )

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)
        mock_workspace_store.load_problem_analysis.return_value = {"title": "Analysis"}
        mock_workspace_store.load_solution_strategy.return_value = {"approach": "Strategy"}

        mock_workspace_store.get_best_solution.return_value = None  # No "best" from KB
        kb_initial_code_content = "initial_code_from_kb"
        kb_initial_meta = {"score": 99}
        mock_workspace_store.load_solution_code.return_value = kb_initial_code_content
        mock_workspace_store.load_solution_metadata.return_value = kb_initial_meta

        # Act
        await solve_service.run_evolve_step(
            test_cases=[{"name": "tc1"}],
            score_calculator=MagicMock(),
            max_generations=1,
            population_size=1,
            time_limit_seconds=1,
            initial_code_override=None,
        )

        # Assert
        mock_workspace_store.load_solution_code.assert_called_once_with("initial")
        mock_workspace_store.load_solution_metadata.assert_called_once_with("initial")
        mock_ee_instance_global.evolve.assert_called_once()
        assert mock_ee_instance_global.evolve.call_args[0][2] == kb_initial_code_content


@pytest.mark.asyncio
async def test_run_evolve_step_success_generate_new_initial(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch(
        "ahc_agent.services.solve_service.ProblemLogic"
    ) as MockProblemLogicGlobal, patch("ahc_agent.services.solve_service.ensure_directory"):
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        mock_ee_instance_global.evolve = AsyncMock(
            return_value={"best_solution": "code", "best_score": 1, "generations_completed": 1, "evolution_log": []}
        )

        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        generated_code = "generated_initial_code"
        mock_pl_instance_global.generate_initial_solution = AsyncMock(return_value=generated_code)

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        sample_analysis = {"title": "Analysis for Gen Initial"}
        mock_workspace_store.load_problem_analysis.return_value = sample_analysis
        mock_workspace_store.load_solution_strategy.return_value = {"approach": "Strategy"}
        mock_workspace_store.get_best_solution.return_value = None
        # mock_workspace_store.get_solution was removed, load_solution_code/metadata will return None by default from fixture

        # Act
        await solve_service.run_evolve_step(
            test_cases=[{"name": "tc1"}],
            score_calculator=MagicMock(),
            max_generations=1,
            population_size=1,
            time_limit_seconds=1,
            initial_code_override=None,
        )

        # Assert
        mock_pl_instance_global.generate_initial_solution.assert_called_once_with(sample_analysis)
        mock_workspace_store.save_solution.assert_any_call(  # Use any_call due to best solution also being saved
            "initial_for_evolve", {"code": generated_code, "score": 0, "generation": 0}
        )
        mock_ee_instance_global.evolve.assert_called_once()
        assert mock_ee_instance_global.evolve.call_args[0][2] == generated_code


@pytest.mark.parametrize("missing_item", ["analysis", "strategy", "test_cases", "score_calculator"])
@pytest.mark.asyncio
async def test_run_evolve_step_missing_prerequisites(missing_item, mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal:
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        # Setup defaults, then make one item missing
        mock_workspace_store.load_problem_analysis.return_value = {"title": "Analysis"}
        mock_workspace_store.load_solution_strategy.return_value = {"approach": "Strategy"}
        test_cases_arg = [{"name": "tc1"}]
        score_calculator_arg = MagicMock()

        if missing_item == "analysis":
            mock_workspace_store.load_problem_analysis.return_value = None
        elif missing_item == "strategy":
            mock_workspace_store.load_solution_strategy.return_value = None
        elif missing_item == "test_cases":
            test_cases_arg = None
        elif missing_item == "score_calculator":
            score_calculator_arg = None

        # Act
        result = await solve_service.run_evolve_step(
            test_cases=test_cases_arg, score_calculator=score_calculator_arg, max_generations=1, population_size=1, time_limit_seconds=1
        )

        # Assert
        mock_ee_instance_global.evolve.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_evolve_step_eval_func_integration(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch.object(
        SolveService, "_evaluate_solution_wrapper", new_callable=AsyncMock
    ) as mock_eval_wrapper, patch("ahc_agent.services.solve_service.ensure_directory"):
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        # Let evolve capture the passed eval_func
        captured_eval_func = None

        async def capture_func_then_return(*args, **kwargs):
            nonlocal captured_eval_func
            captured_eval_func = args[3]  # eval_func is the 4th positional arg
            return {"best_solution": "code", "best_score": 1, "generations_completed": 1, "evolution_log": []}

        mock_ee_instance_global.evolve = AsyncMock(side_effect=capture_func_then_return)

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        mock_workspace_store.load_problem_analysis.return_value = {"title": "Analysis"}
        mock_workspace_store.load_solution_strategy.return_value = {"approach": "Strategy"}
        mock_test_cases = [{"name": "tc1_eval_func", "input": "in_eval"}]
        mock_score_calculator = MagicMock()

        # Act
        await solve_service.run_evolve_step(
            test_cases=mock_test_cases, score_calculator=mock_score_calculator, max_generations=1, population_size=1, time_limit_seconds=1
        )

        # Assert that evolve was called (which it was, by side_effect)
        mock_ee_instance_global.evolve.assert_called_once()
        assert captured_eval_func is not None

        # Now, call the captured eval_func to see if it calls _evaluate_solution_wrapper
        test_code_for_eval = "test_code_for_eval_func"
        await captured_eval_func(test_code_for_eval)

        mock_eval_wrapper.assert_called_once_with(
            test_code_for_eval,
            mock_test_cases,
            mock_score_calculator,
            solve_service.implementation_debugger,  # The actual debugger instance from the service
        )


@pytest.mark.asyncio
async def test_run_initial_solution_step_generation_fails(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store, mocker):
    # Arrange
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_initial_solution = AsyncMock(return_value=None)  # Generation fails

        solve_service = SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store)

        sample_analysis_data = {"title": "Analyzed Problem for Initial Fail"}
        mock_workspace_store.load_problem_analysis.return_value = sample_analysis_data

        # Act
        result = await solve_service.run_initial_solution_step()

        # Assert
        mock_pl_instance_global.generate_initial_solution.assert_called_once_with(sample_analysis_data)
        mock_workspace_store.save_solution.assert_not_called()
        assert result is None
