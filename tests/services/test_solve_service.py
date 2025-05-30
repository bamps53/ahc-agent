from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from ahc_agent.config import Config
from ahc_agent.core.debugger import ImplementationDebugger
from ahc_agent.core.session_store import SessionStore
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
        "workspace.base_dir": str(tmp_path / "test_ws"),  # Used for session_dir in evolve and tools/in path
        "contest_id": "test_contest",  # Default contest_id from config
    }.get(key, default)  # Return the default if key is not in the dict
    return config


@pytest.fixture
def mock_session_store(mocker):
    kb = mocker.MagicMock(spec=SessionStore)
    # Default behaviors for a fresh solve
    kb.get_session.return_value = {
        "problem_text": "problem text from session",
        "problem_id": "test_problem_kb",
        "metadata": {"problem_text": "problem text from session metadata"},
    }
    kb.get_problem_analysis.return_value = None
    kb.get_solution_strategy.return_value = None
    kb.get_best_solution.return_value = None
    kb.create_session.return_value = "test_session_id_created"
    return kb


# Patch all core modules used by SolveService
# These are patched at the location where they are imported by solve_service.py
@patch("ahc_agent.services.solve_service.ProblemLogic")
@patch("ahc_agent.services.solve_service.ImplementationDebugger")
@patch("ahc_agent.services.solve_service.EvolutionaryEngine")
@patch("ahc_agent.services.solve_service.SolutionStrategist")
@patch("ahc_agent.services.solve_service.ProblemAnalyzer")
@pytest.mark.asyncio
async def test_run_solve_session_fresh_solve_no_session_id(
    MockProblemAnalyzer,
    MockSolutionStrategist,
    MockEvolutionaryEngine,
    MockImplementationDebugger,
    MockProblemLogic,
    mock_llm_client,
    mock_docker_manager,
    mock_config,
    mock_session_store,
):
    # Arrange: Setup mock instances for core modules
    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Test Problem Analysis"})

    mock_ss_instance = MockSolutionStrategist.return_value
    mock_ss_instance.develop_strategy = AsyncMock(return_value={"approach": "Test Strategy"})

    mock_pl_instance = MockProblemLogic.return_value
    # parse_problem_statement is called when session_id is None to create a new session
    mock_pl_instance.parse_problem_statement = AsyncMock(return_value={"title": "Parsed Problem Title", "problem_id": "parsed_problem_id"})
    mock_pl_instance.generate_initial_solution = AsyncMock(return_value="initial_code")
    # Simulate no tools/in files, so generate_test_cases is called
    mock_pl_instance.generate_test_cases = AsyncMock(return_value=[{"name": "t1", "input": "in1"}])
    mock_pl_instance.create_score_calculator = AsyncMock(return_value=MagicMock(return_value=100.0))

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
    service = SolveService(llm_client=mock_llm_client, docker_manager=mock_docker_manager, config=mock_config, session_store=mock_session_store)

    problem_text_arg = "This is the problem statement from argument."

    # Simulate KB behavior for creating a new session (session_id=None)
    # get_session will be called by session_store.create_session internally if it's part of its logic,
    # but SolveService calls create_session if session_id is None.
    # The important mock here is create_session.
    # The problem_text for analysis should come from the argument.

    # Act
    await service.run_solve_session(problem_text_arg, session_id=None, interactive=False)

    # Assert
    # Session creation
    mock_pl_instance.parse_problem_statement.assert_called_once_with(problem_text_arg)
    mock_session_store.create_session.assert_called_once_with(
        "Parsed Problem Title",  # from parse_problem_statement result
        {
            "problem_text": problem_text_arg,
            "parsed_info": {"title": "Parsed Problem Title", "problem_id": "parsed_problem_id"},
            "problem_id": "test_contest",  # from mock_config.get("contest_id")
        },
    )

    # Analysis, Strategy, Solution generation
    mock_pa_instance.analyze.assert_called_once_with(problem_text_arg)
    mock_session_store.save_problem_analysis.assert_called_once_with("test_session_id_created", {"title": "Test Problem Analysis"})

    mock_ss_instance.develop_strategy.assert_called_once_with({"title": "Test Problem Analysis"})
    mock_session_store.save_solution_strategy.assert_called_once_with("test_session_id_created", {"approach": "Test Strategy"})

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

    mock_session_store.save_evolution_log.assert_called_once_with("test_session_id_created", [{"gen": 1, "score": 200}])
    mock_session_store.save_solution.assert_called_once_with(
        "test_session_id_created", "best", {"code": "best_code_evolved", "score": 200.0, "generation": 1}
    )


@patch("ahc_agent.services.solve_service.ProblemLogic")
@patch("ahc_agent.services.solve_service.ImplementationDebugger")
@patch("ahc_agent.services.solve_service.EvolutionaryEngine")
@patch("ahc_agent.services.solve_service.SolutionStrategist")
@patch("ahc_agent.services.solve_service.ProblemAnalyzer")
@pytest.mark.asyncio
async def test_run_solve_session_resuming_solve(
    MockProblemAnalyzer,
    MockSolutionStrategist,
    MockEvolutionaryEngine,
    MockImplementationDebugger,
    MockProblemLogic,
    mock_llm_client,
    mock_docker_manager,
    mock_config,
    mock_session_store,
):
    # Arrange: KB returns existing data
    existing_analysis = {"title": "Existing Analysis"}
    existing_strategy = {"approach": "Existing Strategy"}
    existing_best_solution = {"code": "existing_best_code", "score": 150.0}

    mock_session_store.get_session.return_value = {
        "problem_text": "problem from session",
        "metadata": {"problem_text": "problem from session metadata"},
    }
    mock_session_store.get_problem_analysis.return_value = existing_analysis
    mock_session_store.get_solution_strategy.return_value = existing_strategy
    mock_session_store.get_best_solution.return_value = existing_best_solution  # To be used as initial_solution_code

    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_ss_instance = MockSolutionStrategist.return_value
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

    service = SolveService(llm_client=mock_llm_client, docker_manager=mock_docker_manager, config=mock_config, session_store=mock_session_store)

    problem_text_arg = "problem text for resumed session"  # This might come from KB session if session_id is provided
    session_id_arg = "existing_session_id"

    # Act
    await service.run_solve_session(problem_text_arg, session_id=session_id_arg, interactive=False)

    # Assert
    mock_session_store.get_session.assert_called_once_with(session_id_arg)
    mock_session_store.create_session.assert_not_called()  # Should not create new session

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

    mock_session_store.save_solution.assert_called_once()  # For best solution from evolve


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
async def test_run_solve_session_tools_in_handling(
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
    mock_session_store,
):
    # Arrange
    # Mocks for core modules (similar to fresh_solve, but focus on test case generation)
    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Analysis"})
    mock_ss_instance = MockSolutionStrategist.return_value
    mock_ss_instance.develop_strategy = AsyncMock(return_value={"strat": "Strategy"})
    mock_pl_instance = MockProblemLogic.return_value
    mock_pl_instance.parse_problem_statement = AsyncMock(return_value={"title": "Parsed"})
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

    service = SolveService(llm_client=mock_llm_client, docker_manager=mock_docker_manager, config=mock_config, session_store=mock_session_store)

    # Act
    await service.run_solve_session("problem_text_tools", session_id=None, interactive=False)

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
async def test_evaluate_solution_wrapper(mock_llm_client, mock_docker_manager, mock_config, mock_session_store):  # Need other mocks for ID
    # Arrange
    mock_id_instance = AsyncMock(spec=ImplementationDebugger)  # _evaluate_solution_wrapper uses an ID instance

    service = SolveService(llm_client=mock_llm_client, docker_manager=mock_docker_manager, config=mock_config, session_store=mock_session_store)

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
@patch("builtins.input")  # To mock user input
@pytest.mark.asyncio
async def test_run_interactive_session_basic_analyze_and_exit(
    mock_input,
    MockProblemAnalyzer,
    MockSolutionStrategist,
    MockEvolutionaryEngine,
    MockImplementationDebugger,
    MockProblemLogic,  # These are from service init, not directly used in this simple test
    mock_llm_client,
    mock_docker_manager,
    mock_config,
    mock_session_store,
):
    # Arrange
    # Service will use its own instances of core modules based on its __init__
    # We need to ensure these instances (or their mocks) behave as expected.
    # MockProblemAnalyzer is the class, MockProblemAnalyzer.return_value is the instance.
    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Interactive Analysis"})

    # Configure KB for interactive session
    session_id_interactive = "interactive_session_1"
    # problem_text is fetched from session metadata in interactive mode
    mock_session_store.get_session.return_value = {
        "session_id": session_id_interactive,
        "problem_id": "interactive_problem",
        "metadata": {"problem_text": "interactive problem content"},
    }
    # Initial get_problem_analysis returns None, so 'analyze' command will trigger analysis
    mock_session_store.get_problem_analysis.return_value = None

    service = SolveService(llm_client=mock_llm_client, docker_manager=mock_docker_manager, config=mock_config, session_store=mock_session_store)

    # Simulate user typing "analyze" then "exit"
    mock_input.side_effect = ["analyze", "exit"]

    # Act
    # run_interactive_session is called by run_solve_session if interactive=True
    # Or, we can test it directly if it's public (it is in the service)
    await service.run_interactive_session(session_id_interactive)

    # Assert
    mock_session_store.get_session.assert_called_with(session_id_interactive)

    # Check that ProblemAnalyzer.analyze was called because user typed "analyze"
    # The instance used is service.problem_analyzer
    # This was set up by @patch('ahc_agent.services.solve_service.ProblemAnalyzer')
    # and its return_value (the instance mock) is mock_pa_instance.
    mock_pa_instance.analyze.assert_called_once_with("interactive problem content")

    # Check that the analysis was saved
    mock_session_store.save_problem_analysis.assert_called_once_with(session_id_interactive, {"title": "Interactive Analysis"})

    # Check that input was called twice
    assert mock_input.call_count == 2
    mock_input.assert_has_calls([call("\nEnter command: "), call("\nEnter command: ")])
