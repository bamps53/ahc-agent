from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ahc_agent.config import Config
from ahc_agent.core.debugger import ImplementationDebugger
from ahc_agent.core.workspace_store import WorkspaceStore
from ahc_agent.services.solve_service import SolveService
from ahc_agent.utils.docker_manager import DockerManager
from ahc_agent.utils.llm import LLMClient
from typing import Dict, Any, List, Callable, Tuple, Optional # New tests require these


# --- Existing Fixtures (renamed to avoid conflict if any, though not strictly necessary after reset_all) ---
@pytest.fixture
def mock_llm_client_existing(mocker):
    return mocker.AsyncMock(spec=LLMClient)


@pytest.fixture
def mock_docker_manager_existing(mocker):
    return mocker.MagicMock(spec=DockerManager)


@pytest.fixture
def mock_config_existing(mocker, tmp_path):
    config = mocker.MagicMock(spec=Config)
    config.get.side_effect = lambda key, default=None: {
        "llm": {},
        "docker": {},
        "analyzer": {},
        "strategist": {},
        "evolution": {"time_limit_seconds": 10, "max_generations": 1, "population_size": 2, "score_plateau_generations": 1},
        "debugger": {},
        "problem_logic": {"test_cases_count": 1},
        "workspace.base_dir": str(tmp_path / "test_ws"),
        "contest_id": "test_contest",
    }.get(key, default)
    return config


@pytest.fixture
def mock_workspace_store_existing(mocker):
    ws = mocker.MagicMock(spec=WorkspaceStore)
    ws.load_problem_text.return_value = "problem text from workspace"
    ws.problem_id = "test_problem_ws"
    ws.load_problem_analysis.return_value = None
    ws.load_solution_strategy.return_value = None
    ws.get_best_solution_code_and_meta.return_value = None
    ws.get_best_solution.return_value = None # Added to reflect actual usage in original tests
    ws.load_solution_code.return_value = None
    ws.load_solution_metadata.return_value = None

    mock_final_solution_path = mocker.MagicMock(spec=Path)
    mock_final_solution_path.__str__.return_value = "mocked/solutions/best.cpp"
    ws.solutions_dir = mocker.MagicMock(spec=Path)
    ws.solutions_dir.__truediv__.return_value = mock_final_solution_path

    mock_generations_subdir_path = mocker.MagicMock(spec=Path)
    mock_generations_subdir_path.__str__.return_value = "mocked_workspace/generations"

    mock_workspace_root_path = mocker.MagicMock(spec=Path)
    mock_workspace_root_path.__truediv__.return_value = mock_generations_subdir_path
    ws.get_workspace_dir.return_value = mock_workspace_root_path

    return ws

# --- Existing Tests (adapted to use _existing fixtures for clarity) ---

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
    mock_llm_client_existing,
    mock_docker_manager_existing,
    mock_config_existing,
    mock_workspace_store_existing,
):
    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Test Problem Analysis"})
    MockProblemAnalyzer.reset_mock() # Ensure fresh mock for this test
    MockProblemAnalyzer.return_value = mock_pa_instance

    mock_ss_instance = MockSolutionStrategist.return_value
    mock_ss_instance.develop_strategy = AsyncMock(return_value={"approach": "Test Strategy"})
    MockSolutionStrategist.reset_mock()
    MockSolutionStrategist.return_value = mock_ss_instance

    mock_pl_instance = MockProblemLogic.return_value
    mock_pl_instance.generate_initial_solution = AsyncMock(return_value="initial_code")
    mock_pl_instance.generate_test_cases = AsyncMock(return_value=[{"name": "t1", "input": "in1"}])
    # Ensure score_calculator returns a tuple (score, error_message)
    mock_score_calculator_actual_func = MagicMock(return_value=(100.0, None))
    mock_pl_instance.create_score_calculator = AsyncMock(return_value=mock_score_calculator_actual_func)
    MockProblemLogic.reset_mock()
    MockProblemLogic.return_value = mock_pl_instance

    mock_id_instance = MockImplementationDebugger.return_value
    # Ensure compile_and_test returns all expected keys
    mock_id_instance.compile_and_test = AsyncMock(return_value={
        "success": True, "compilation_success": True, "compilation_errors": None,
        "execution_success": True, "execution_output": "out1", "execution_errors": None,
        "execution_time": 0.1, "fixed_code": "initial_code"
    })
    MockImplementationDebugger.reset_mock()
    MockImplementationDebugger.return_value = mock_id_instance

    mock_ee_instance = MockEvolutionaryEngine.return_value
    mock_ee_instance.evolve = AsyncMock(
        return_value={
            "best_solution_code": "best_code_evolved", # Key changed
            "best_score": 200.0,
            "best_solution_details": {}, # Added
            "generations_completed": 1,
            "evolution_log": [{"gen": 1, "score": 200}],
        }
    )
    MockEvolutionaryEngine.reset_mock()
    MockEvolutionaryEngine.return_value = mock_ee_instance

    service = SolveService(
        llm_client=mock_llm_client_existing,
        docker_manager=mock_docker_manager_existing,
        config=mock_config_existing,
        workspace_store=mock_workspace_store_existing,
    )
    mock_workspace_store_existing.get_best_solution.return_value = None
    problem_text_arg = "This is the problem statement from argument."

    await service.run_solve(problem_text_arg, interactive=False)

    mock_pa_instance.analyze.assert_called_once_with(problem_text_arg)
    mock_workspace_store_existing.save_problem_analysis.assert_called_once_with({"title": "Test Problem Analysis"})
    mock_workspace_store_existing.save_solution_strategy.assert_called_once_with({"approach": "Test Strategy"})
    mock_ss_instance.develop_strategy.assert_called_once_with({"title": "Test Problem Analysis"})
    mock_pl_instance.generate_initial_solution.assert_called_once_with({"title": "Test Problem Analysis"})

    with patch("pathlib.Path.exists", return_value=False): # Simulate tools/in not existing
        # This part is a bit tricky as run_solve calls test case generation internally.
        # We need to ensure the conditions are right for generate_test_cases to be called.
        # The call to run_solve will trigger this logic.
        pass

    # The number of generated test cases might depend on config, ensure it matches.
    # From mock_config_existing, "problem_logic": {"test_cases_count": 1}
    # However, the original test_run_solve_fresh_solve asserted 3. Let's keep 1 as per config.
    num_expected_test_cases = mock_config_existing.get("problem_logic.test_cases_count", 3) # Default to 3 if not in config for some reason
    mock_pl_instance.generate_test_cases.assert_called_once_with({"title": "Test Problem Analysis"}, num_expected_test_cases)
    mock_pl_instance.create_score_calculator.assert_called_once_with({"title": "Test Problem Analysis"})

    mock_ee_instance.evolve.assert_called_once()
    assert mock_ee_instance.evolve.call_args[0][2] == "initial_code"
    mock_workspace_store_existing.save_evolution_log.assert_called_once_with([{"gen": 1, "score": 200}])
    mock_workspace_store_existing.save_solution.assert_called_once_with(
        "best",
        {"code": "best_code_evolved", "score": 200.0, "generation": 1, "details": {}} # Added details
    )


@patch("ahc_agent.services.solve_service.ProblemLogic")
@patch("ahc_agent.services.solve_service.ImplementationDebugger")
@patch("ahc_agent.services.solve_service.EvolutionaryEngine")
@patch("ahc_agent.services.solve_service.SolutionStrategist")
@patch("ahc_agent.services.solve_service.ProblemAnalyzer")
@pytest.mark.asyncio
async def test_run_solve_resuming_solve(
    MockProblemAnalyzer, MockSolutionStrategist, MockEvolutionaryEngine,
    MockImplementationDebugger, MockProblemLogic,
    mock_llm_client_existing, mock_docker_manager_existing,
    mock_config_existing, mock_workspace_store_existing
):
    existing_analysis = {"title": "Existing Analysis"}
    existing_strategy = {"approach": "Existing Strategy"}
    existing_best_solution = {"code": "existing_best_code", "score": 150.0, "details": {}} # Added details

    mock_workspace_store_existing.load_problem_analysis.return_value = existing_analysis
    mock_workspace_store_existing.load_solution_strategy.return_value = existing_strategy
    mock_workspace_store_existing.get_best_solution.return_value = existing_best_solution

    mock_pa_instance = MockProblemAnalyzer.return_value # Not called
    mock_ss_instance = MockSolutionStrategist.return_value # Not called
    mock_pl_instance = MockProblemLogic.return_value
    mock_pl_instance.generate_test_cases = AsyncMock(return_value=[{"name": "t1", "input": "in1"}])
    mock_score_calculator_actual_func = MagicMock(return_value=(100.0, None))
    mock_pl_instance.create_score_calculator = AsyncMock(return_value=mock_score_calculator_actual_func)

    mock_id_instance = MockImplementationDebugger.return_value # Called by wrapper
    mock_id_instance.compile_and_test = AsyncMock(return_value={
        "success": True, "compilation_success": True, "execution_output": "out1",
        "execution_time": 0.1, "compilation_errors": None, "execution_errors": None,
        "execution_success": True, "fixed_code": "existing_best_code"
    })

    mock_ee_instance = MockEvolutionaryEngine.return_value
    mock_ee_instance.evolve = AsyncMock(return_value={
        "best_solution_code": "evolved_from_existing", "best_score": 250.0,
        "best_solution_details": {}, "generations_completed": 1, "evolution_log": []
    })

    service = SolveService(
        llm_client=mock_llm_client_existing, docker_manager=mock_docker_manager_existing,
        config=mock_config_existing, workspace_store=mock_workspace_store_existing
    )
    problem_text_arg = "problem text for resumed solve"

    await service.run_solve(problem_text_arg, interactive=False)

    mock_workspace_store_existing.load_problem_analysis.assert_called_once()
    mock_workspace_store_existing.load_solution_strategy.assert_called_once()
    mock_pa_instance.analyze.assert_not_called()
    mock_ss_instance.develop_strategy.assert_not_called()
    mock_pl_instance.generate_initial_solution.assert_not_called()
    mock_pl_instance.generate_test_cases.assert_called_once()
    mock_pl_instance.create_score_calculator.assert_called_once()
    mock_ee_instance.evolve.assert_called_once()
    assert mock_ee_instance.evolve.call_args[0][0] == existing_analysis
    assert mock_ee_instance.evolve.call_args[0][1] == existing_strategy
    assert mock_ee_instance.evolve.call_args[0][2] == "existing_best_code"
    mock_workspace_store_existing.save_solution.assert_called_once()


@patch("ahc_agent.services.solve_service.ProblemLogic")
@patch("ahc_agent.services.solve_service.ImplementationDebugger")
@patch("ahc_agent.services.solve_service.EvolutionaryEngine")
@patch("ahc_agent.services.solve_service.SolutionStrategist")
@patch("ahc_agent.services.solve_service.ProblemAnalyzer")
@patch("builtins.open", new_callable=MagicMock)
@patch("pathlib.Path.glob")
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.exists", return_value=True)
@pytest.mark.asyncio
async def test_run_solve_tools_in_handling(
    mock_path_exists, mock_path_is_dir, mock_path_glob, mock_builtin_open,
    MockProblemAnalyzer, MockSolutionStrategist, MockEvolutionaryEngine,
    MockImplementationDebugger, MockProblemLogic,
    mock_llm_client_existing, mock_docker_manager_existing,
    mock_config_existing, mock_workspace_store_existing
):
    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Analysis"})
    mock_ss_instance = MockSolutionStrategist.return_value
    mock_ss_instance.develop_strategy = AsyncMock(return_value={"strat": "Strategy"})
    mock_pl_instance = MockProblemLogic.return_value
    mock_pl_instance.generate_initial_solution = AsyncMock(return_value="initial_code_tools_in")
    mock_score_calculator_actual_func = MagicMock(return_value=(100.0, None))
    mock_pl_instance.create_score_calculator = AsyncMock(return_value=mock_score_calculator_actual_func)
    mock_ee_instance = MockEvolutionaryEngine.return_value
    mock_ee_instance.evolve = AsyncMock(return_value={
        "best_solution_code": "code", "best_score": 1,
        "best_solution_details": {}, "generations_completed": 1, "evolution_log": []
    })

    mock_file_content = "test_case_content_from_file"
    mock_builtin_open.return_value.__enter__.return_value.read.return_value = mock_file_content
    mock_test_file_path = MagicMock(spec=Path)
    mock_test_file_path.name = "0000.txt"
    mock_path_glob.return_value = [mock_test_file_path]

    service = SolveService(
        llm_client=mock_llm_client_existing, docker_manager=mock_docker_manager_existing,
        config=mock_config_existing, workspace_store=mock_workspace_store_existing
    )
    await service.run_solve("problem_text_tools", interactive=False)
    mock_pl_instance.generate_test_cases.assert_not_called()
    mock_builtin_open.assert_called_once_with(mock_test_file_path)
    mock_ee_instance.evolve.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_solution_wrapper_existing_modified( # Renamed from _existing
    mock_llm_client_existing, mock_docker_manager_existing,
    mock_config_existing, mock_workspace_store_existing
):
    mock_id_instance = AsyncMock(spec=ImplementationDebugger)
    service = SolveService(
        llm_client=mock_llm_client_existing, docker_manager=mock_docker_manager_existing,
        config=mock_config_existing, workspace_store=mock_workspace_store_existing
    )
    service.implementation_debugger = mock_id_instance # Override debugger
    test_code = "some_code"
    test_cases = [
        {"name": "tc1", "input": "in1"},
        {"name": "tc2", "input": "in2"},
        {"name": "tc3", "no_input_field": True},
    ]
    # score_calculator_func now returns a tuple (score, error_message)
    mock_score_calculator_func = AsyncMock(return_value=(100.0, None))

    mock_id_instance.compile_and_test.side_effect = [
        {"success": True, "compilation_success": True, "execution_output": "out1", "execution_time": 0.1, "execution_errors": None, "compilation_errors": None, "execution_success": True},
        {"success": False, "compilation_errors": "compile_err", "execution_errors": None, "execution_time": 0, "compilation_success": False, "execution_success": False, "execution_output": None},
    ]

    avg_score, details = await service._evaluate_solution_wrapper(
        test_code, test_cases, mock_score_calculator_func, mock_id_instance,
    )
    assert mock_id_instance.compile_and_test.call_count == 2
    mock_id_instance.compile_and_test.assert_any_call(test_code, "in1")
    mock_id_instance.compile_and_test.assert_any_call(test_code, "in2")
    mock_score_calculator_func.assert_called_once_with("in1", "out1")
    assert avg_score == pytest.approx(100.0 / 3)
    assert details["tc1"]["score"] == 100.0
    assert "score_calculation_error" not in details["tc1"]
    assert details["tc2"]["score"] == 0.0
    assert "compile_err" in details["tc2"]["error"]
    assert details["tc3"]["score"] == 0.0
    assert "Missing 'input' field" in details["tc3"]["error"]


@patch("ahc_agent.services.solve_service.ProblemLogic")
@patch("ahc_agent.services.solve_service.ImplementationDebugger")
@patch("ahc_agent.services.solve_service.EvolutionaryEngine")
@patch("ahc_agent.services.solve_service.SolutionStrategist")
@patch("ahc_agent.services.solve_service.ProblemAnalyzer")
@patch("ahc_agent.services.solve_service.questionary.select")
@pytest.mark.asyncio
async def test_run_interactive_solve_basic_analyze_and_exit(
    mock_questionary_select, MockProblemAnalyzer, MockSolutionStrategist,
    MockEvolutionaryEngine, MockImplementationDebugger, MockProblemLogic,
    mock_llm_client_existing, mock_docker_manager_existing,
    mock_config_existing, mock_workspace_store_existing
):
    mock_pa_instance = MockProblemAnalyzer.return_value
    mock_pa_instance.analyze = AsyncMock(return_value={"title": "Interactive Analysis"})
    MockProblemAnalyzer.reset_mock()
    MockProblemAnalyzer.return_value = mock_pa_instance
    mock_workspace_store_existing.load_problem_text.return_value = "interactive problem content"
    mock_workspace_store_existing.problem_id = "interactive_problem"
    mock_workspace_store_existing.load_problem_analysis.return_value = {"title": "Existing Analysis"}
    mock_workspace_store_existing.load_solution_strategy.return_value = {"approach": "Existing Strategy"}

    service = SolveService(
        llm_client=mock_llm_client_existing, docker_manager=mock_docker_manager_existing,
        config=mock_config_existing, workspace_store=mock_workspace_store_existing
    )
    mock_select = AsyncMock()
    mock_select.ask_async.side_effect = ["analyze", "exit"]
    mock_questionary_select.return_value = mock_select
    await service.run_interactive_solve(problem_text_initial=None)
    service.problem_analyzer.analyze.assert_called_once_with("interactive problem content")
    mock_workspace_store_existing.save_problem_analysis.assert_called_once_with({"title": "Interactive Analysis"})
    assert mock_questionary_select.call_count == 2


@pytest.mark.asyncio
async def test_run_analyze_step_success(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemAnalyzer") as MockProblemAnalyzerGlobal:
        mock_pa_instance_global = MockProblemAnalyzerGlobal.return_value
        mock_pa_instance_global.analyze = AsyncMock(return_value={"title": "Analyzed Problem"})
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        problem_text_arg = "Test problem statement"
        result = await solve_service.run_analyze_step(problem_text=problem_text_arg)
        mock_pa_instance_global.analyze.assert_called_once_with(problem_text_arg)
        mock_workspace_store_existing.save_problem_analysis.assert_called_once_with({"title": "Analyzed Problem"})
        assert result == {"title": "Analyzed Problem"}


@pytest.mark.asyncio
async def test_run_analyze_step_no_problem_text_loads_from_store(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemAnalyzer") as MockProblemAnalyzerGlobal:
        mock_pa_instance_global = MockProblemAnalyzerGlobal.return_value
        mock_pa_instance_global.analyze = AsyncMock(return_value={"title": "Analyzed Stored Problem"})
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        stored_problem_text = "Problem text from store"
        mock_workspace_store_existing.load_problem_text.return_value = stored_problem_text
        result = await solve_service.run_analyze_step(problem_text=None)
        mock_workspace_store_existing.load_problem_text.assert_called_once()
        mock_pa_instance_global.analyze.assert_called_once_with(stored_problem_text)
        mock_workspace_store_existing.save_problem_analysis.assert_called_once_with({"title": "Analyzed Stored Problem"})
        assert result == {"title": "Analyzed Stored Problem"}


@pytest.mark.asyncio
async def test_run_analyze_step_no_problem_text_and_not_in_store(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemAnalyzer") as MockProblemAnalyzerGlobal:
        mock_pa_instance_global = MockProblemAnalyzerGlobal.return_value
        mock_pa_instance_global.analyze = AsyncMock()
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        mock_workspace_store_existing.load_problem_text.return_value = None
        result = await solve_service.run_analyze_step(problem_text=None)
        mock_workspace_store_existing.load_problem_text.assert_called_once()
        mock_pa_instance_global.analyze.assert_not_called()
        mock_workspace_store_existing.save_problem_analysis.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_strategy_step_success(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.SolutionStrategist") as MockSolutionStrategistGlobal:
        mock_ss_instance_global = MockSolutionStrategistGlobal.return_value
        mock_ss_instance_global.develop_strategy = AsyncMock(return_value={"approach": "Test Strategy"})
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store_existing.load_problem_analysis.return_value = sample_analysis_data
        result = await solve_service.run_strategy_step()
        mock_workspace_store_existing.load_problem_analysis.assert_called_once()
        mock_ss_instance_global.develop_strategy.assert_called_once_with(sample_analysis_data)
        mock_workspace_store_existing.save_solution_strategy.assert_called_once_with({"approach": "Test Strategy"})
        assert result == {"approach": "Test Strategy"}


@pytest.mark.asyncio
async def test_run_strategy_step_no_analysis(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.SolutionStrategist") as MockSolutionStrategistGlobal:
        mock_ss_instance_global = MockSolutionStrategistGlobal.return_value
        mock_ss_instance_global.develop_strategy = AsyncMock()
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        mock_workspace_store_existing.load_problem_analysis.return_value = None
        result = await solve_service.run_strategy_step()
        mock_workspace_store_existing.load_problem_analysis.assert_called_once()
        mock_ss_instance_global.develop_strategy.assert_not_called()
        mock_workspace_store_existing.save_solution_strategy.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_testcases_step_load_from_tools_success(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal, patch(
        "ahc_agent.services.solve_service.Path"
    ) as MockPathClass:
        mock_path_base = MockPathClass.return_value
        mock_path_tools = MagicMock(spec=Path)
        mock_path_tools_in = MagicMock(spec=Path)
        mock_path_base.__truediv__.side_effect = lambda segment: mock_path_tools if segment == "tools" else MagicMock(spec=Path)
        mock_path_tools.__truediv__.side_effect = lambda segment: mock_path_tools_in if segment == "in" else MagicMock(spec=Path)
        mock_path_tools_in.exists.return_value = True
        mock_path_tools_in.is_dir.return_value = True
        mock_tools_in_path_instance = mock_path_tools_in
        mock_file_path = MagicMock(spec=Path)
        mock_file_path.name = "0000.txt"
        mock_tools_in_path_instance.glob.return_value = [mock_file_path]
        mock_open_fn = mocker.patch("builtins.open", mocker.mock_open(read_data="test_input_data"))
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.create_score_calculator = AsyncMock(return_value=MagicMock())
        mock_pl_instance_global.generate_test_cases = AsyncMock(return_value=[])
        def mock_config_get_side_effect(key, default=None):
            if key == "workspace.base_dir": return "/mocked/workspace/base"
            # Provide default for problem_logic.test_cases_count if not explicitly set for this test
            if key == "problem_logic.test_cases_count": return default if default is not None else 1
            return default
        mock_config_existing.get.side_effect = mock_config_get_side_effect
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store_existing.load_problem_analysis.return_value = sample_analysis_data
        result = await solve_service.run_testcases_step(load_from_tools=True, num_to_generate=3)
        mock_workspace_store_existing.load_problem_analysis.assert_called_once()
        MockPathClass.assert_called()
        mock_tools_in_path_instance.exists.assert_called_once()
        mock_tools_in_path_instance.is_dir.assert_called_once()
        mock_tools_in_path_instance.glob.assert_called_once_with("*.txt")
        mock_open_fn.assert_called_once_with(mock_file_path)
        mock_pl_instance_global.generate_test_cases.assert_not_called()
        mock_pl_instance_global.create_score_calculator.assert_called_once_with(sample_analysis_data)
        assert result is not None
        assert len(result["test_cases"]) == 1
        assert result["test_cases"][0]["input"] == "test_input_data"
        assert result["score_calculator"] is not None


@pytest.mark.asyncio
async def test_run_testcases_step_load_from_tools_empty_dir_generates(
    mock_llm_client_existing, mock_docker_manager_existing,
    mock_config_existing, # Keep this for other config values
    mock_workspace_store_existing,
    mocker, # For general mocking
    tmp_path # Pytest fixture for temporary directory
):
    # Configure mock_config_existing to use tmp_path for workspace.base_dir
    # This ensures Path.cwd() is not called if base_dir is always provided.
    test_workspace_path = tmp_path / "test_ws_empty_tools"
    test_workspace_path.mkdir()

    # Create an empty tools/in directory
    tools_in_dir = test_workspace_path / "tools" / "in"
    tools_in_dir.mkdir(parents=True, exist_ok=True)

    # Temporarily override the config to use this specific path for this test
    original_config_get_side_effect = mock_config_existing.get.side_effect
    def side_effect_func(key, default=None):
        if key == "workspace.base_dir":
            return str(test_workspace_path)
        # Call the original side_effect for other keys if it was set
        if callable(original_config_get_side_effect):
            return original_config_get_side_effect(key, default)
        # Fallback if original_config_get_side_effect was None or not callable
        # This might happen if the fixture was not using a side_effect initially
        # For this specific mock_config_existing, it was using a side_effect.
        # We need to replicate its original behavior for other keys.
        # The original side_effect was:
        # lambda k, d=None: { ... }.get(k, d)
        # We can't directly call that here without knowing the dict.
        # So, we assume that if it's not workspace.base_dir, the original complex side_effect
        # from mock_config_existing fixture should handle it.
        # This part is tricky. A better way is to have a more granular config mock or
        # ensure the original side_effect lambda is accessible and callable.
        # For now, we'll assume the original side_effect handles other keys correctly.
        # A simpler mock_config_existing would make this easier.
        # Let's assume the original mock_config_existing.get.side_effect can be called.
        return original_config_get_side_effect(key, default)


    mock_config_existing.get.side_effect = side_effect_func

    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_test_cases = AsyncMock(return_value=[{"name": "gen1", "input": "gen_in1"}])
        mock_pl_instance_global.create_score_calculator = AsyncMock(return_value=MagicMock())

        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store_existing.load_problem_analysis.return_value = sample_analysis_data

        result = await solve_service.run_testcases_step(load_from_tools=True, num_to_generate=3)

        mock_pl_instance_global.generate_test_cases.assert_called_once_with(sample_analysis_data, 3)
        mock_pl_instance_global.create_score_calculator.assert_called_once_with(sample_analysis_data)
        assert result is not None
        assert len(result["test_cases"]) == 1
        assert result["test_cases"][0]["name"] == "gen1"

    # Restore original side_effect
    if original_config_get_side_effect: # Check if it was actually set
        mock_config_existing.get.side_effect = original_config_get_side_effect


@pytest.mark.asyncio
async def test_run_testcases_step_generate_explicitly(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_test_cases = AsyncMock(return_value=[{"name": "gen1", "input": "gen_in1"}])
        mock_pl_instance_global.create_score_calculator = AsyncMock(return_value=MagicMock())
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store_existing.load_problem_analysis.return_value = sample_analysis_data
        result = await solve_service.run_testcases_step(load_from_tools=False, num_to_generate=5)
        mock_pl_instance_global.generate_test_cases.assert_called_once_with(sample_analysis_data, 5)
        mock_pl_instance_global.create_score_calculator.assert_called_once_with(sample_analysis_data)
        assert result is not None
        assert len(result["test_cases"]) == 1


@pytest.mark.asyncio
async def test_run_testcases_step_no_analysis(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        mock_workspace_store_existing.load_problem_analysis.return_value = None
        result = await solve_service.run_testcases_step(load_from_tools=False, num_to_generate=3)
        mock_pl_instance_global.generate_test_cases.assert_not_called()
        mock_pl_instance_global.create_score_calculator.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_testcases_step_generation_fails(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_test_cases = AsyncMock(return_value=[])
        mock_pl_instance_global.create_score_calculator = AsyncMock(return_value=MagicMock())
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        sample_analysis_data = {"title": "Analyzed Problem"}
        mock_workspace_store_existing.load_problem_analysis.return_value = sample_analysis_data
        result = await solve_service.run_testcases_step(load_from_tools=False, num_to_generate=3)
        mock_pl_instance_global.generate_test_cases.assert_called_once_with(sample_analysis_data, 3)
        mock_pl_instance_global.create_score_calculator.assert_not_called() # Because test_cases is empty
        assert result is None


@pytest.mark.asyncio
async def test_run_initial_solution_step_success(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        expected_code = "def main():\n  print('hello')"
        mock_pl_instance_global.generate_initial_solution = AsyncMock(return_value=expected_code)
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        sample_analysis_data = {"title": "Analyzed Problem for Initial"}
        mock_workspace_store_existing.load_problem_analysis.return_value = sample_analysis_data
        result = await solve_service.run_initial_solution_step()
        mock_workspace_store_existing.load_problem_analysis.assert_called_once()
        mock_pl_instance_global.generate_initial_solution.assert_called_once_with(sample_analysis_data)
        mock_workspace_store_existing.save_solution.assert_called_once_with("initial", {"code": expected_code, "score": 0, "generation": 0, "details": None}) # Added details
        assert result == expected_code


@pytest.mark.asyncio
async def test_run_initial_solution_step_no_analysis(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_initial_solution = AsyncMock()
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        mock_workspace_store_existing.load_problem_analysis.return_value = None
        result = await solve_service.run_initial_solution_step()
        mock_pl_instance_global.generate_initial_solution.assert_not_called()
        mock_workspace_store_existing.save_solution.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_evolve_step_success_with_override_code(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch(
        "ahc_agent.services.solve_service.ensure_directory"
    ) as mock_ensure_dir:
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        expected_evolve_result = {
            "best_solution_code": "evolved_code", "best_score": 1000,
            "best_solution_details": {}, "generations_completed": 10,
            "evolution_log": [{"gen": 10, "score": 1000}],
        }
        mock_ee_instance_global.evolve = AsyncMock(return_value=expected_evolve_result)
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        sample_analysis = {"title": "Evolve Analysis"}
        sample_strategy = {"approach": "Evolve Strategy"}
        mock_workspace_store_existing.load_problem_analysis.return_value = sample_analysis
        mock_workspace_store_existing.load_solution_strategy.return_value = sample_strategy
        mock_test_cases = [{"name": "tc1", "input": "in1"}]
        mock_score_calculator = MagicMock()
        initial_code_override_arg = "override_initial_code"
        result = await solve_service.run_evolve_step(
            test_cases=mock_test_cases, score_calculator=mock_score_calculator,
            max_generations=10, population_size=20, time_limit_seconds=60,
            initial_code_override=initial_code_override_arg,
        )
        mock_workspace_store_existing.load_problem_analysis.assert_called_once()
        mock_workspace_store_existing.load_solution_strategy.assert_called_once()
        mock_ensure_dir.assert_called()
        mock_ee_instance_global.evolve.assert_called_once()
        call_args = mock_ee_instance_global.evolve.call_args[0]
        assert call_args[0] == sample_analysis
        assert call_args[1] == sample_strategy
        assert call_args[2] == initial_code_override_arg
        assert callable(call_args[3])
        assert isinstance(call_args[4], str)
        assert solve_service.evolutionary_engine.max_generations == 10
        assert solve_service.evolutionary_engine.population_size == 20
        assert solve_service.evolutionary_engine.time_limit_seconds == 60
        mock_workspace_store_existing.save_evolution_log.assert_called_once_with(expected_evolve_result["evolution_log"])
        mock_workspace_store_existing.save_solution.assert_called_once_with(
            "best", {
                "code": expected_evolve_result["best_solution_code"],
                "score": expected_evolve_result["best_score"],
                "generation": expected_evolve_result["generations_completed"],
                "details": expected_evolve_result["best_solution_details"],
            },
        )
        assert result == expected_evolve_result


@pytest.mark.asyncio
async def test_run_evolve_step_success_load_best_from_kb(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch(
        "ahc_agent.services.solve_service.ensure_directory"
    ):
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        mock_ee_instance_global.evolve = AsyncMock(
            return_value={"best_solution_code": "code", "best_score": 1, "best_solution_details":{}, "generations_completed": 1, "evolution_log": []}
        )
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        mock_workspace_store_existing.load_problem_analysis.return_value = {"title": "Analysis"}
        mock_workspace_store_existing.load_solution_strategy.return_value = {"approach": "Strategy"}
        kb_best_code = {"code": "kb_best_code", "score": 100, "details": {}}
        mock_workspace_store_existing.get_best_solution.return_value = kb_best_code
        await solve_service.run_evolve_step(
            test_cases=[{"name": "tc1"}], score_calculator=MagicMock(),
            max_generations=1, population_size=1, time_limit_seconds=1,
            initial_code_override=None,
        )
        mock_ee_instance_global.evolve.assert_called_once()
        assert mock_ee_instance_global.evolve.call_args[0][2] == kb_best_code["code"]


@pytest.mark.asyncio
async def test_run_evolve_step_success_load_initial_from_kb(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch(
        "ahc_agent.services.solve_service.ensure_directory"
    ):
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        mock_ee_instance_global.evolve = AsyncMock(
            return_value={"best_solution_code": "code", "best_score": 1, "best_solution_details":{}, "generations_completed": 1, "evolution_log": []}
        )
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        mock_workspace_store_existing.load_problem_analysis.return_value = {"title": "Analysis"}
        mock_workspace_store_existing.load_solution_strategy.return_value = {"approach": "Strategy"}
        mock_workspace_store_existing.get_best_solution.return_value = None
        kb_initial_code_content = "initial_code_from_kb"
        kb_initial_meta = {"score": 0, "generation": 0, "details": None} # Ensure details is present
        mock_workspace_store_existing.load_solution_code.return_value = kb_initial_code_content
        mock_workspace_store_existing.load_solution_metadata.return_value = kb_initial_meta
        await solve_service.run_evolve_step(
            test_cases=[{"name": "tc1"}], score_calculator=MagicMock(),
            max_generations=1, population_size=1, time_limit_seconds=1,
            initial_code_override=None,
        )
        mock_workspace_store_existing.load_solution_code.assert_called_once_with("initial")
        mock_workspace_store_existing.load_solution_metadata.assert_called_once_with("initial")
        mock_ee_instance_global.evolve.assert_called_once()
        assert mock_ee_instance_global.evolve.call_args[0][2] == kb_initial_code_content


@pytest.mark.asyncio
async def test_run_evolve_step_success_generate_new_initial(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch(
        "ahc_agent.services.solve_service.ProblemLogic"
    ) as MockProblemLogicGlobal, patch("ahc_agent.services.solve_service.ensure_directory"):
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        mock_ee_instance_global.evolve = AsyncMock(
            return_value={"best_solution_code": "code", "best_score": 1, "best_solution_details":{}, "generations_completed": 1, "evolution_log": []}
        )
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        generated_code = "generated_initial_code"
        mock_pl_instance_global.generate_initial_solution = AsyncMock(return_value=generated_code)
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        sample_analysis = {"title": "Analysis for Gen Initial"}
        mock_workspace_store_existing.load_problem_analysis.return_value = sample_analysis
        mock_workspace_store_existing.load_solution_strategy.return_value = {"approach": "Strategy"}
        mock_workspace_store_existing.get_best_solution.return_value = None
        mock_workspace_store_existing.load_solution_code.return_value = None # Ensure no initial code from KB
        await solve_service.run_evolve_step(
            test_cases=[{"name": "tc1"}], score_calculator=MagicMock(),
            max_generations=1, population_size=1, time_limit_seconds=1,
            initial_code_override=None,
        )
        mock_pl_instance_global.generate_initial_solution.assert_called_once_with(sample_analysis)
        mock_workspace_store_existing.save_solution.assert_any_call(
            "initial_for_evolve", {"code": generated_code, "score": 0, "generation": 0, "details":None}
        )
        mock_ee_instance_global.evolve.assert_called_once()
        assert mock_ee_instance_global.evolve.call_args[0][2] == generated_code


@pytest.mark.parametrize("missing_item", ["analysis", "strategy", "test_cases", "score_calculator"])
@pytest.mark.asyncio
async def test_run_evolve_step_missing_prerequisites(missing_item, mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal:
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        mock_workspace_store_existing.load_problem_analysis.return_value = {"title": "Analysis"}
        mock_workspace_store_existing.load_solution_strategy.return_value = {"approach": "Strategy"}
        test_cases_arg = [{"name": "tc1"}]
        score_calculator_arg = MagicMock()
        if missing_item == "analysis": mock_workspace_store_existing.load_problem_analysis.return_value = None
        elif missing_item == "strategy": mock_workspace_store_existing.load_solution_strategy.return_value = None
        elif missing_item == "test_cases": test_cases_arg = [] # Empty list instead of None for this check
        elif missing_item == "score_calculator": score_calculator_arg = None
        result = await solve_service.run_evolve_step(
            test_cases=test_cases_arg, score_calculator=score_calculator_arg, max_generations=1, population_size=1, time_limit_seconds=1
        )
        mock_ee_instance_global.evolve.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_run_evolve_step_eval_func_integration(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.EvolutionaryEngine") as MockEvolutionaryEngineGlobal, patch.object(
        SolveService, "_evaluate_solution_wrapper", new_callable=AsyncMock
    ) as mock_eval_wrapper, patch("ahc_agent.services.solve_service.ensure_directory"):
        mock_ee_instance_global = MockEvolutionaryEngineGlobal.return_value
        captured_eval_func = None
        async def capture_func_then_return(*args, **kwargs):
            nonlocal captured_eval_func
            captured_eval_func = args[3]
            return {"best_solution_code": "code", "best_score": 1, "best_solution_details":{}, "generations_completed": 1, "evolution_log": []}
        mock_ee_instance_global.evolve = AsyncMock(side_effect=capture_func_then_return)
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        mock_workspace_store_existing.load_problem_analysis.return_value = {"title": "Analysis"}
        mock_workspace_store_existing.load_solution_strategy.return_value = {"approach": "Strategy"}
        mock_workspace_store_existing.get_best_solution.return_value = {"code":"initial", "score":0, "details":{}} # Provide initial code
        mock_test_cases = [{"name": "tc1_eval_func", "input": "in_eval"}]
        mock_score_calculator = MagicMock()
        await solve_service.run_evolve_step(
            test_cases=mock_test_cases, score_calculator=mock_score_calculator, max_generations=1, population_size=1, time_limit_seconds=1
        )
        mock_ee_instance_global.evolve.assert_called_once()
        assert captured_eval_func is not None
        test_code_for_eval = "test_code_for_eval_func"
        await captured_eval_func(test_code_for_eval)
        mock_eval_wrapper.assert_called_once_with(
            test_code_for_eval,
            mock_test_cases,
            mock_score_calculator,
            solve_service.implementation_debugger,
        )


@pytest.mark.asyncio
async def test_run_initial_solution_step_generation_fails(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing, mocker):
    with patch("ahc_agent.services.solve_service.ProblemLogic") as MockProblemLogicGlobal:
        mock_pl_instance_global = MockProblemLogicGlobal.return_value
        mock_pl_instance_global.generate_initial_solution = AsyncMock(return_value=None)
        solve_service = SolveService(mock_llm_client_existing, mock_docker_manager_existing, mock_config_existing, mock_workspace_store_existing)
        sample_analysis_data = {"title": "Analyzed Problem for Initial Fail"}
        mock_workspace_store_existing.load_problem_analysis.return_value = sample_analysis_data
        result = await solve_service.run_initial_solution_step()
        mock_pl_instance_global.generate_initial_solution.assert_called_once_with(sample_analysis_data)
        mock_workspace_store_existing.save_solution.assert_not_called()
        assert result is None


# --- New Fixtures and Tests for _evaluate_solution_wrapper (from previous request) ---

@pytest.fixture
def mock_llm_client() -> LLMClient:
    return MagicMock(spec=LLMClient)

@pytest.fixture
def mock_docker_manager() -> DockerManager:
    return MagicMock(spec=DockerManager)

@pytest.fixture
def mock_config() -> Config:
    cfg = MagicMock(spec=Config)
    cfg.get = MagicMock(side_effect=lambda key, default=None: {
        "workspace.base_dir": "/tmp/ahc_workspace_new", # Different from existing to avoid clash
        "analyzer": {}, "strategist": {}, "evolution": {},
        "debugger": {}, "problem_logic": {}
    }.get(key, default))
    return cfg

@pytest.fixture
def mock_workspace_store() -> WorkspaceStore:
    ws = MagicMock(spec=WorkspaceStore)
    ws.get_workspace_dir = MagicMock(return_value=MagicMock(spec=Path))
    return ws

@pytest.fixture
def mock_implementation_debugger() -> ImplementationDebugger: # Not directly used by solve_service fixture but good for consistency
    debugger = MagicMock(spec=ImplementationDebugger)
    debugger.compile_and_test = AsyncMock()
    return debugger

@pytest.fixture
def solve_service(
    mock_llm_client: LLMClient,
    mock_docker_manager: DockerManager,
    mock_config: Config,
    mock_workspace_store: WorkspaceStore,
) -> SolveService:
    # This fixture is specifically for the new _evaluate_solution_wrapper tests.
    # It patches dependencies more locally.

    # Mock instance for ProblemLogic that will be returned by the patched ProblemLogic class
    mock_problem_logic_instance = MagicMock(spec=SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store).problem_logic.__class__) # Get class from an instance
    mock_problem_logic_instance.create_score_calculator = AsyncMock() # Mock its methods as needed

    # Mock instance for ImplementationDebugger
    local_mock_implementation_debugger = MagicMock(spec=SolveService(mock_llm_client, mock_docker_manager, mock_config, mock_workspace_store).implementation_debugger.__class__)
    local_mock_implementation_debugger.compile_and_test = AsyncMock()

    with patch('ahc_agent.services.solve_service.ProblemAnalyzer', MagicMock()), \
         patch('ahc_agent.services.solve_service.SolutionStrategist', MagicMock()), \
         patch('ahc_agent.services.solve_service.EvolutionaryEngine', MagicMock()), \
         patch('ahc_agent.services.solve_service.ImplementationDebugger', return_value=local_mock_implementation_debugger), \
         patch('ahc_agent.services.solve_service.ProblemLogic', return_value=mock_problem_logic_instance):

        service = SolveService(
            llm_client=mock_llm_client,
            docker_manager=mock_docker_manager,
            config=mock_config,
            workspace_store=mock_workspace_store,
        )
        # Allow tests to access these specific mocks if needed
        service.problem_logic_mock = mock_problem_logic_instance
        service.implementation_debugger_mock = local_mock_implementation_debugger
        # The service will use the `local_mock_implementation_debugger` due to the patch.
        # So, `service.implementation_debugger` IS `local_mock_implementation_debugger`.
    return service


@pytest.mark.anyio
async def test_evaluate_wrapper_success(solve_service: SolveService):
    code = "valid_code"
    test_cases = [{"name": "test1", "input": "input1"}]

    solve_service.implementation_debugger.compile_and_test.return_value = {
        "success": True, "compilation_success": True, "compilation_errors": None,
        "execution_success": True, "execution_output": "output1", "execution_errors": None,
        "execution_time": 0.1, "fixed_code": code
    }
    mock_score_calculator_func = AsyncMock(return_value=(100.0, None))

    avg_score, details = await solve_service._evaluate_solution_wrapper(
        code, test_cases, mock_score_calculator_func, solve_service.implementation_debugger
    )

    assert avg_score == 100.0
    assert "test1" in details
    assert details["test1"]["score"] == 100.0
    assert details["test1"]["execution_output"] == "output1"
    assert "score_calculation_error" not in details["test1"]
    solve_service.implementation_debugger.compile_and_test.assert_called_once_with(code, "input1")
    mock_score_calculator_func.assert_called_once_with("input1", "output1")

@pytest.mark.anyio
async def test_evaluate_wrapper_compile_error(solve_service: SolveService):
    code = "compile_error_code"
    test_cases = [{"name": "test1", "input": "input1"}]
    compile_error_msg = "Syntax Error at line 1"

    solve_service.implementation_debugger.compile_and_test.return_value = {
        "success": False, "compilation_success": False, "compilation_errors": compile_error_msg,
        "execution_success": False, "execution_output": None, "execution_errors": None,
        "execution_time": None, "fixed_code": None
    }
    mock_score_calculator_func = AsyncMock()

    avg_score, details = await solve_service._evaluate_solution_wrapper(
        code, test_cases, mock_score_calculator_func, solve_service.implementation_debugger
    )

    assert avg_score == 0.0
    assert "test1" in details
    assert details["test1"]["score"] == 0.0
    assert details["test1"]["compilation_errors"] == compile_error_msg
    assert details["test1"]["error"] == compile_error_msg
    mock_score_calculator_func.assert_not_called()

@pytest.mark.anyio
async def test_evaluate_wrapper_runtime_error(solve_service: SolveService):
    code = "runtime_error_code"
    test_cases = [{"name": "test1", "input": "input1"}]
    runtime_error_msg = "Segmentation fault"

    solve_service.implementation_debugger.compile_and_test.return_value = {
        "success": False, "compilation_success": True, "compilation_errors": None,
        "execution_success": False, "execution_output": "", "execution_errors": runtime_error_msg,
        "execution_time": 0.05, "fixed_code": code
    }
    mock_score_calculator_func = AsyncMock()

    avg_score, details = await solve_service._evaluate_solution_wrapper(
        code, test_cases, mock_score_calculator_func, solve_service.implementation_debugger
    )

    assert avg_score == 0.0
    assert "test1" in details
    assert details["test1"]["score"] == 0.0
    assert details["test1"]["execution_errors"] == runtime_error_msg
    assert details["test1"]["error"] == runtime_error_msg
    mock_score_calculator_func.assert_not_called()

@pytest.mark.anyio
async def test_evaluate_wrapper_score_calculation_error(solve_service: SolveService):
    code = "valid_code_score_error"
    test_cases = [{"name": "test1", "input": "input1"}]
    score_calc_error_msg = "Invalid format in output for scoring"

    solve_service.implementation_debugger.compile_and_test.return_value = {
        "success": True, "compilation_success": True, "compilation_errors": None,
        "execution_success": True, "execution_output": "malformed_output", "execution_errors": None,
        "execution_time": 0.1, "fixed_code": code
    }
    mock_score_calculator_func = AsyncMock(return_value=(0.0, score_calc_error_msg))

    avg_score, details = await solve_service._evaluate_solution_wrapper(
        code, test_cases, mock_score_calculator_func, solve_service.implementation_debugger
    )

    assert avg_score == 0.0
    assert "test1" in details
    assert details["test1"]["score"] == 0.0
    assert details["test1"]["score_calculation_error"] == score_calc_error_msg
    mock_score_calculator_func.assert_called_once_with("input1", "malformed_output")

@pytest.mark.anyio
async def test_evaluate_wrapper_multiple_test_cases(solve_service: SolveService):
    code = "multi_test_code"
    test_cases = [
        {"name": "tc1_success", "input": "in1"},
        {"name": "tc2_runtime_error", "input": "in2"},
        {"name": "tc3_score_error", "input": "in3"},
    ]

    solve_service.implementation_debugger.compile_and_test.side_effect = [
        {"success": True, "compilation_success": True, "execution_output": "out1", "execution_time": 0.1, "fixed_code": code, "compilation_errors":None, "execution_errors":None, "execution_success":True},
        {"success": False, "compilation_success": True, "execution_errors": "Segfault", "fixed_code": code, "compilation_errors":None, "execution_output":None, "execution_time":None, "execution_success":False},
        {"success": True, "compilation_success": True, "execution_output": "out3_malformed", "execution_time": 0.2, "fixed_code": code, "compilation_errors":None, "execution_errors":None, "execution_success":True},
    ]

    score_calc_error_msg_tc3 = "Bad output for score"
    mock_score_calculator_func = AsyncMock(side_effect=[
        (150.0, None),
        (0.0, score_calc_error_msg_tc3) # This corresponds to the third compile_and_test call (tc3_score_error)
    ])

    avg_score, details = await solve_service._evaluate_solution_wrapper(
        code, test_cases, mock_score_calculator_func, solve_service.implementation_debugger
    )

    assert solve_service.implementation_debugger.compile_and_test.call_count == 3
    # mock_score_calculator_func is called for tc1 and tc3 (both have success:True in compile_and_test)
    assert mock_score_calculator_func.call_count == 2

    assert avg_score == pytest.approx(50.0) # (150.0 + 0.0 + 0.0) / 3

    assert "tc1_success" in details
    assert details["tc1_success"]["score"] == 150.0
    assert "score_calculation_error" not in details["tc1_success"]

    assert "tc2_runtime_error" in details
    assert details["tc2_runtime_error"]["score"] == 0.0
    assert details["tc2_runtime_error"]["execution_errors"] == "Segfault"

    assert "tc3_score_error" in details
    assert details["tc3_score_error"]["score"] == 0.0
    assert details["tc3_score_error"]["score_calculation_error"] == score_calc_error_msg_tc3

@pytest.mark.anyio
async def test_evaluate_wrapper_no_test_cases(solve_service: SolveService):
    avg_score, details = await solve_service._evaluate_solution_wrapper(
        "any_code", [], AsyncMock(), solve_service.implementation_debugger
    )
    assert avg_score == 0.0
    assert "warning" in details
    assert details["warning"] == "No test cases provided"

@pytest.mark.anyio
async def test_evaluate_wrapper_test_case_missing_input(solve_service: SolveService):
    test_cases = [{"name": "no_input_tc"}]
    avg_score, details = await solve_service._evaluate_solution_wrapper(
        "any_code", test_cases, AsyncMock(), solve_service.implementation_debugger
    )
    assert avg_score == 0.0
    assert "no_input_tc" in details
    assert details["no_input_tc"]["error"] == "Missing 'input' field"
    assert details["no_input_tc"]["score"] == 0.0

# End of file
