import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call, mock_open
from pathlib import Path
import asyncio
import json
import yaml
import os # For os.makedirs

from ahc_agent.services.batch_service import BatchService
from ahc_agent.config import Config
from ahc_agent.utils.llm import LLMClient
from ahc_agent.utils.docker_manager import DockerManager
from ahc_agent.core.knowledge import KnowledgeBase
from ahc_agent.core.analyzer import ProblemAnalyzer
from ahc_agent.core.strategist import SolutionStrategist
from ahc_agent.core.engine import EvolutionaryEngine
from ahc_agent.core.debugger import ImplementationDebugger
from ahc_agent.core.problem_logic import ProblemLogic

@pytest.fixture
def mock_llm_client_bs(mocker): # Suffix _bs for BatchService test
    return mocker.AsyncMock(spec=LLMClient)

@pytest.fixture
def mock_docker_manager_bs(mocker):
    return mocker.MagicMock(spec=DockerManager)

@pytest.fixture
def mock_config_bs(mocker):
    config = mocker.MagicMock(spec=Config)
    # Base config that export() will return
    base_exported_config = { 
        "llm": {"model": "default_llm"}, 
        "docker": {"image": "default_docker"}, 
        "analyzer": {"default_setting": "analyzer_val"}, 
        "strategist": {"default_setting": "strategist_val"},
        "evolution": {"max_generations": 10, "population_size": 20, "time_limit_seconds": 1800}, 
        "debugger": {"default_setting": "debugger_val"}, 
        "problem_logic": {"test_cases_count": 3, "language": "cpp"},
        "batch": {"output_dir": "~/ahc_batch_default", "parallel": 1} # Default batch settings
    }
    config.export.return_value = base_exported_config.copy() # Return a copy
    
    # Make .get behave as if the export was the actual config dict for global settings
    config.get.side_effect = lambda key, default=None: base_exported_config.get(key, default)
    return config

class TestBatchService:

    # Test Static Methods
    def test_format_duration(self):
        assert BatchService._format_duration(None) == "Unknown"
        assert BatchService._format_duration("invalid") == "Invalid duration"
        assert BatchService._format_duration(0) == "0s"
        assert BatchService._format_duration(59) == "59s"
        assert BatchService._format_duration(60) == "1m 0s"
        assert BatchService._format_duration(3661) == "1h 1m 1s"

    def test_set_nested_dict(self):
        d = {}
        BatchService._set_nested_dict(d, ["a", "b", "c"], 1)
        assert d == {"a": {"b": {"c": 1}}}
        BatchService._set_nested_dict(d, ["a", "x"], 2)
        assert d == {"a": {"b": {"c": 1}, "x": 2}}
        BatchService._set_nested_dict(d, ["y"], 3)
        assert d == {"a": {"b": {"c": 1}, "x": 2}, "y": 3}
        # Test overriding
        BatchService._set_nested_dict(d, ["a", "b", "c"], "override")
        assert d["a"]["b"]["c"] == "override"
        # Test setting on a non-dict (should replace)
        BatchService._set_nested_dict(d, ["a", "b"], "not_a_dict_anymore")
        assert d["a"]["b"] == "not_a_dict_anymore"
        # Test with empty keys list (should ideally not happen or raise error, current impl returns d)
        assert BatchService._set_nested_dict(d, [], "ignored") == d


    @pytest.mark.asyncio
    async def test_evaluate_solution_for_experiment(self, mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs):
        service = BatchService(mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs)
        mock_debugger_instance = AsyncMock(spec=ImplementationDebugger)
        
        test_code = "code_for_eval"
        test_cases = [{"name": "t1", "input": "in1"}, {"name": "t2", "input": "in2"}]
        mock_score_calc = MagicMock()

        mock_debugger_instance.compile_and_test.side_effect = [
            AsyncMock(return_value={"success": True, "execution_output": "out1", "execution_time": 0.1}),
            AsyncMock(return_value={"success": False, "compilation_errors": "err", "execution_errors": None})
        ]
        mock_score_calc.side_effect = [100.0] # Only for the successful one

        avg_score, details = await service._evaluate_solution_for_experiment(
            test_code, test_cases, mock_score_calc, mock_debugger_instance
        )

        assert mock_debugger_instance.compile_and_test.call_count == 2
        mock_score_calc.assert_called_once_with("in1", "out1")
        assert avg_score == 50.0 # (100.0 + 0) / 2
        assert details["t1"]["score"] == 100.0
        assert details["t2"]["score"] == 0
        assert "err" in details["t2"]["error"]

    @patch('ahc_agent.services.batch_service.ProblemLogic')
    @patch('ahc_agent.services.batch_service.ImplementationDebugger')
    @patch('ahc_agent.services.batch_service.EvolutionaryEngine')
    @patch('ahc_agent.services.batch_service.SolutionStrategist')
    @patch('ahc_agent.services.batch_service.ProblemAnalyzer')
    @patch('ahc_agent.services.batch_service.KnowledgeBase')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.is_file', return_value=True) # Assume problem file exists
    @patch('os.makedirs')
    @pytest.mark.asyncio
    async def test_run_single_experiment_service_success(
        self, mock_os_makedirs, mock_path_is_file, mock_file_open_bs, # Renamed mock_file_open
        MockKnowledgeBase, MockProblemAnalyzer, MockSolutionStrategist, MockEvolutionaryEngine,
        MockImplementationDebugger, MockProblemLogic,
        mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs, tmp_path
    ):
        service = BatchService(mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs)
        
        experiment_id = "exp_success"
        problem_file_path_str = str(tmp_path / "problems" / "P001.md")
        problem_config = {"name": "P001", "path": problem_file_path_str}
        # Parameter set overrides a value from the base config's 'evolution' section
        parameter_set_config = {"name": "ParamSetA", "evolution.max_generations": 50} 
        experiment_dir_path = tmp_path / experiment_id

        mock_file_open_bs.side_effect = [
            mock_open(read_data="Problem P001 Text").return_value, # For reading problem file
            mock_open().return_value  # For writing result.json
        ]

        mock_kb_instance = MockKnowledgeBase.return_value
        mock_kb_instance.create_session.return_value = "session_exp_success"

        mock_pa_instance = MockProblemAnalyzer.return_value; mock_pa_instance.analyze = AsyncMock(return_value={"analysis_key": "analysis_val"})
        mock_ss_instance = MockSolutionStrategist.return_value; mock_ss_instance.develop_strategy = AsyncMock(return_value={"strategy_key": "strategy_val"})
        mock_pl_instance = MockProblemLogic.return_value
        mock_pl_instance.generate_initial_solution = AsyncMock(return_value="initial_code_single_exp")
        mock_pl_instance.generate_test_cases = AsyncMock(return_value=[{"input": "tc_single_exp"}])
        mock_pl_instance.create_score_calculator = AsyncMock(return_value=MagicMock(return_value=100.0))
        
        mock_id_instance = MockImplementationDebugger.return_value
        mock_id_instance.compile_and_test = AsyncMock(return_value={"success": True, "execution_output": "out_single_exp", "execution_time": 0.1})

        mock_ee_instance = MockEvolutionaryEngine.return_value
        mock_ee_instance.evolve = AsyncMock(return_value={
            "best_solution": "best_code_single_exp", "best_score": 120.0,
            "generations_completed": 45, "evolution_log": {"log_data": []}
        })

        result = await service._run_single_experiment_service(experiment_id, problem_config, parameter_set_config, experiment_dir_path)

        # Assert config override was applied when instantiating EvolutionaryEngine
        evo_engine_constructor_args = MockEvolutionaryEngine.call_args[0]
        evo_engine_config_passed = evo_engine_constructor_args[1] # Config dict is the second arg
        assert evo_engine_config_passed['max_generations'] == 50 # Overridden value
        assert evo_engine_config_passed['population_size'] == 20 # From base config

        # Assert other module instantiations
        # Example: ProblemAnalyzer should receive the 'analyzer' part of the config
        pa_constructor_args = MockProblemAnalyzer.call_args[0]
        pa_config_passed = pa_constructor_args[1]
        assert pa_config_passed['default_setting'] == "analyzer_val"


        mock_kb_instance.create_session.assert_called_once()
        mock_pa_instance.analyze.assert_called_once_with("Problem P001 Text")
        mock_ee_instance.evolve.assert_called_once()
        
        # Assert result.json writing (check that open was called for it)
        # The second open call is for result.json
        result_json_path = experiment_dir_path / "result.json"
        # Check for the specific call to open for result.json
        # This is a bit complex with multiple open calls. A more robust way is to check json.dump call.
        with patch('json.dump') as mock_json_dump:
             # Re-run or just check if the call was made to json.dump
             # For this, we'd need to run again or ensure the test setup is such that we can inspect this.
             # Let's assume the open mock is sufficient for now.
             # Or, we can check the call list of mock_file_open_bs
             assert any(call(result_json_path, "w", encoding="utf-8") in mock_file_open_bs.call_args_list for call_obj in mock_file_open_bs.mock_calls if call_obj[0] == '')


        assert result["experiment_id"] == experiment_id
        assert result["best_score"] == 120.0
        assert result["generations"] == 45
        assert result["error"] is None

    @patch('pathlib.Path.is_file', return_value=False) # Problem file does not exist
    @pytest.mark.asyncio
    async def test_run_single_experiment_service_problem_file_not_found(
        self, mock_path_is_file, mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs, tmp_path
    ):
        service = BatchService(mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs)
        experiment_id = "exp_no_file"
        problem_config = {"name": "P_NoFile", "path": str(tmp_path / "non_existent_problem.md")}
        parameter_set_config = {"name": "ParamSetB"}
        experiment_dir_path = tmp_path / experiment_id

        result = await service._run_single_experiment_service(experiment_id, problem_config, parameter_set_config, experiment_dir_path)

        assert result["error"] is not None
        assert "Problem file not found" in result["error"]
        assert result["best_score"] == float('-inf')

    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open) # For batch config reading
    @patch('os.makedirs')
    @patch.object(BatchService, '_run_single_experiment_service', new_callable=AsyncMock) # Mock the helper
    @pytest.mark.asyncio
    async def test_run_batch_experiments_service_valid_config(
        self, mock_run_single_exp, mock_os_makedirs_batch, mock_file_open_batch, mock_yaml_safe_load,
        mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs, tmp_path
    ):
        service = BatchService(mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs)
        
        batch_config_path_str = str(tmp_path / "batch.yaml")
        mock_file_open_batch.return_value = mock_open(read_data="---\nversion: 1").return_value # For batch.yaml

        batch_config_data = {
            "common": {"output_dir": str(tmp_path / "batch_output"), "parallel": 2},
            "problems": [{"name": "P1", "path": str(tmp_path / "p1.md")}],
            "parameter_sets": [{"name": "PS1", "evolution.param": 10}],
            "experiments": [{"problem": "P1", "parameter_set": "PS1", "repeats": 2}]
        }
        mock_yaml_safe_load.return_value = batch_config_data
        
        # Mock return values for each call to _run_single_experiment_service
        mock_run_single_exp.side_effect = [
            AsyncMock(return_value={"experiment_id": "P1_PS1_repeat1", "best_score": 100, "error": None}),
            AsyncMock(return_value={"experiment_id": "P1_PS1_repeat2", "best_score": 110, "error": None}),
        ]

        # Patch json.dump for summary.json
        with patch('json.dump') as mock_json_dump_summary:
            results = await service.run_batch_experiments_service(batch_config_path_str)

            assert mock_run_single_exp.call_count == 2
            # Check calls to _run_single_experiment_service (example for first call)
            first_call_args = mock_run_single_exp.call_args_list[0][0]
            assert first_call_args[0] == "P1_PS1_repeat1" # experiment_id
            assert first_call_args[1]["name"] == "P1" # problem_config
            assert first_call_args[2]["name"] == "PS1" # parameter_set_config
            
            assert len(results) == 2
            assert results[0]["best_score"] == 100
            assert results[1]["best_score"] == 110

            # Check summary.json was written
            expected_summary_path = Path(batch_config_data["common"]["output_dir"]) / "summary.json"
            # mock_file_open_batch was used for batch.yaml, json.dump uses its own open
            # So, we check mock_json_dump_summary call
            assert mock_json_dump_summary.call_count == 1
            # First arg of first call to json.dump is the data, second is the file stream
            dumped_summary_data = mock_json_dump_summary.call_args[0][0]
            assert dumped_summary_data == results
            # The file stream argument is harder to check precisely without more complex mocking of open for json.dump

    @patch('builtins.open', side_effect=FileNotFoundError("Batch config not found!"))
    @pytest.mark.asyncio
    async def test_run_batch_experiments_service_config_not_found(
        self, mock_file_open_err, mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs, tmp_path
    ):
        service = BatchService(mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs)
        results = await service.run_batch_experiments_service(str(tmp_path / "non_existent_batch.yaml"))
        assert len(results) == 1
        assert "Failed to load batch configuration" in results[0]["error"]

    @patch('yaml.safe_load', return_value={"experiments": []}) # Empty experiments
    @patch('builtins.open', new_callable=mock_open)
    @pytest.mark.asyncio
    async def test_run_batch_experiments_service_empty_experiments(
        self, mock_file_open_empty, mock_yaml_empty, mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs, tmp_path
    ):
        service = BatchService(mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs)
        results = await service.run_batch_experiments_service(str(tmp_path / "empty_batch.yaml"))
        assert results == [] # Expect empty list back

    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(BatchService, '_run_single_experiment_service', new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_run_batch_experiments_service_problem_or_param_set_missing(
        self, mock_run_single_exp_miss, mock_file_open_miss, mock_yaml_miss,
        mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs, tmp_path
    ):
        service = BatchService(mock_llm_client_bs, mock_docker_manager_bs, mock_config_bs)
        batch_config_data = {
            "problems": [{"name": "P1", "path": "p1.md"}],
            "parameter_sets": [{"name": "PS1"}],
            "experiments": [
                {"problem": "P_NonExistent", "parameter_set": "PS1"}, # Problem not in "problems"
                {"problem": "P1", "parameter_set": "PS_NonExistent"}  # ParamSet not in "parameter_sets"
            ]
        }
        mock_yaml_miss.return_value = batch_config_data
        
        await service.run_batch_experiments_service(str(tmp_path / "missing_ref_batch.yaml"))
        
        # _run_single_experiment_service should not have been called as references are invalid
        mock_run_single_exp_miss.assert_not_called()
