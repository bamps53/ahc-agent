import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile # Required for execute_cpp_code's inner workings, though not directly in test logic
import os # Required for execute_cpp_code's inner workings

# Assuming ahc_agent modules are importable from the test environment
from ahc_agent.core.problem_logic import ProblemLogic
from ahc_agent.core.docker_manager import DockerManager # For type hinting if needed, and for mock target
from ahc_agent.utils.llm import LLMClient # For ProblemLogic constructor


@pytest.fixture
def mock_llm_client():
    return MagicMock(spec=LLMClient)

@pytest.fixture
def mock_docker_manager():
    manager = MagicMock(spec=DockerManager)
    manager.compile_cpp = AsyncMock()
    manager.run_cpp = AsyncMock()
    return manager

@pytest.fixture
def problem_logic(mock_llm_client, mock_docker_manager):
    # ProblemLogic might try to access config on LLMClient, so ensure it's there
    mock_llm_client.config = {}
    # Similarly, ProblemLogic itself might take a config
    return ProblemLogic(llm_client=mock_llm_client, config={})


class TestProblemLogicExecuteCppCode:
    @pytest.mark.asyncio
    async def test_execute_success(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        mock_docker_manager.compile_cpp.return_value = {
            "success": True,
            "stdout": "Compiled successfully",
            "stderr": "",
            # execute_cpp_code itself constructs the executable path within its temp_dir
        }
        mock_docker_manager.run_cpp.return_value = {
            "status": "success", # Changed from "success": True to match evaluate_solution_code's expectation
            "stdout": "Hello World",
            "stderr": "",
            "execution_time": 0.05,
        }

        cpp_code = "#include <iostream>\nint main() { std::cout << \"Hello World\" << std::endl; return 0; }"
        input_data = "test_input"
        timeout = 10

        result = await problem_logic.execute_cpp_code(cpp_code, input_data, mock_docker_manager, timeout)

        assert result["compilation_success"] is True
        assert result["compilation_stdout"] == "Compiled successfully"
        assert result["execution_success"] is True
        assert result["execution_stdout"] == "Hello World"
        assert result["execution_time"] == 0.05
        assert result["error_type"] is None
        mock_docker_manager.compile_cpp.assert_called_once()
        mock_docker_manager.run_cpp.assert_called_once()
        # Check that executable_path is a string (actual path depends on tempfile)
        assert isinstance(result["executable_path"], str)


    @pytest.mark.asyncio
    async def test_execute_compilation_error(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        mock_docker_manager.compile_cpp.return_value = {
            "success": False,
            "stdout": "Compilation failed",
            "stderr": "Syntax Error on line 1",
        }

        cpp_code = "invalid code"
        input_data = "test_input"
        timeout = 10

        result = await problem_logic.execute_cpp_code(cpp_code, input_data, mock_docker_manager, timeout)

        assert result["compilation_success"] is False
        assert result["compilation_stderr"] == "Syntax Error on line 1"
        assert result["execution_success"] is False # Should not attempt execution
        assert result["error_type"] == "compilation"
        assert result["error_message"] == "Syntax Error on line 1"
        mock_docker_manager.compile_cpp.assert_called_once()
        mock_docker_manager.run_cpp.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        mock_docker_manager.compile_cpp.return_value = {"success": True, "stdout": "", "stderr": ""}
        mock_docker_manager.run_cpp.return_value = {
            "status": "runtime_error", # Changed key
            "stdout": "",
            "stderr": "Segmentation fault",
            "execution_time": 0.02,
        }

        cpp_code = "valid code"
        input_data = "trigger_runtime_error"
        timeout = 10

        result = await problem_logic.execute_cpp_code(cpp_code, input_data, mock_docker_manager, timeout)

        assert result["compilation_success"] is True
        assert result["execution_success"] is False
        assert result["execution_stderr"] == "Segmentation fault"
        assert result["error_type"] == "runtime"
        assert result["error_message"] == "Segmentation fault"
        mock_docker_manager.run_cpp.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_timeout_error(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        mock_docker_manager.compile_cpp.return_value = {"success": True, "stdout": "", "stderr": ""}
        mock_docker_manager.run_cpp.return_value = {
            "status": "timeout", # Changed key
            "stdout": "Partial output",
            "stderr": "", # Timeout might not produce stderr
            "execution_time": 10.0, # Reflects timeout duration
        }
        cpp_code = "long running code"
        input_data = "test_input"
        timeout = 10 # This timeout is passed to run_cpp

        result = await problem_logic.execute_cpp_code(cpp_code, input_data, mock_docker_manager, timeout)

        assert result["compilation_success"] is True
        assert result["execution_success"] is False
        assert result["error_type"] == "timeout"
        assert result["error_message"] == "Execution timed out."
        assert result["execution_stdout"] == "Partial output"
        mock_docker_manager.run_cpp.assert_called_once()

class TestProblemLogicEvaluateSolutionCode:

    # --- Mock Scorer Functions ---
    def sync_scorer_success(self, input_str, output_str):
        return 100.0

    async def async_scorer_success(self, input_str, output_str):
        await asyncio.sleep(0) # Simulate async
        return 150.0

    def sync_scorer_exception(self, input_str, output_str):
        raise ValueError("Scorer failed")

    async def async_scorer_error_tuple(self, input_str, output_str):
        await asyncio.sleep(0)
        return 50.0, "Minor issue in scoring"

    def sync_scorer_error_tuple_fixed_score(self, input_str, output_str):
        return 75.0, "Error details for 75 score"


    @pytest.mark.asyncio
    async def test_evaluate_all_success_single_case_sync_scorer(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"name": "case1", "input": "input1"}]

        # Mocking execute_cpp_code directly for evaluate_solution_code tests
        # This means we are unit testing evaluate_solution_code in isolation from execute_cpp_code's full logic.
        # The alternative is to mock docker_manager.compile_cpp and docker_manager.run_cpp,
        # but that would make these tests more like integration tests for the two ProblemLogic methods.
        # The prompt asks for unit tests for *both*, implying some level of isolation.
        mock_execute_cpp_code = AsyncMock(return_value={
            "compilation_success": True, "execution_success": True,
            "execution_stdout": "output1", "execution_stderr": "", "execution_time": 0.1,
            "compilation_stdout": "", "compilation_stderr": "", "error_type": None, "error_message": ""
        })

        with patch.object(problem_logic, 'execute_cpp_code', mock_execute_cpp_code):
            result = await problem_logic.evaluate_solution_code(
                cpp_code="dummy_code",
                test_cases=test_cases,
                score_calculator_func=self.sync_scorer_success,
                docker_manager=mock_docker_manager, # Still needed by signature, though execute_cpp_code is mocked
                timeout_seconds=5
            )

        assert result["overall_score"] == 100.0
        assert len(result["per_test_case_results"]) == 1
        tc_result = result["per_test_case_results"][0]
        assert tc_result["test_case_name"] == "case1"
        assert tc_result["score"] == 100.0
        assert tc_result["execution_success"] is True
        assert tc_result["execution_stdout"] == "output1"
        mock_execute_cpp_code.assert_called_once_with(
            cpp_code="dummy_code", input_data="input1",
            docker_manager=mock_docker_manager, timeout_seconds=5
        )

    @pytest.mark.asyncio
    async def test_evaluate_all_success_multiple_cases_async_scorer(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [
            {"name": "case_A", "input": "in_A"},
            {"name": "case_B", "input": "in_B"}
        ]

        mock_execute_cpp_code = AsyncMock(side_effect=[
            { # Result for case_A
                "compilation_success": True, "execution_success": True,
                "execution_stdout": "out_A", "execution_stderr": "", "execution_time": 0.1,
                "compilation_stdout": "Compiled", "compilation_stderr": "", "error_type": None, "error_message": ""
            },
            { # Result for case_B
                "compilation_success": True, "execution_success": True, # Assuming re-compile info is present
                "execution_stdout": "out_B", "execution_stderr": "", "execution_time": 0.12,
                "compilation_stdout": "Compiled", "compilation_stderr": "", "error_type": None, "error_message": ""
            }
        ])

        with patch.object(problem_logic, 'execute_cpp_code', mock_execute_cpp_code):
            result = await problem_logic.evaluate_solution_code(
                cpp_code="dummy_code",
                test_cases=test_cases,
                score_calculator_func=self.async_scorer_success,
                docker_manager=mock_docker_manager,
                timeout_seconds=5
            )

        assert result["overall_score"] == 150.0 # Average (150 + 150) / 2
        assert len(result["per_test_case_results"]) == 2
        assert result["per_test_case_results"][0]["score"] == 150.0
        assert result["per_test_case_results"][0]["execution_stdout"] == "out_A"
        assert result["per_test_case_results"][1]["score"] == 150.0
        assert result["per_test_case_results"][1]["execution_stdout"] == "out_B"

        assert mock_execute_cpp_code.call_count == 2
        mock_execute_cpp_code.assert_any_call(cpp_code="dummy_code", input_data="in_A", docker_manager=mock_docker_manager, timeout_seconds=5)
        mock_execute_cpp_code.assert_any_call(cpp_code="dummy_code", input_data="in_B", docker_manager=mock_docker_manager, timeout_seconds=5)


    @pytest.mark.asyncio
    async def test_evaluate_compilation_error_first_case(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"name": "case1", "input": "input1"}]

        mock_execute_cpp_code = AsyncMock(return_value={
            "compilation_success": False, "compilation_stderr": "Major Syntax Error",
            "execution_success": False, "error_type": "compilation", "error_message": "Major Syntax Error",
            "compilation_stdout": "Failed to compile",
             # Other fields might be default or not present
            "execution_stdout": "", "execution_stderr": "", "execution_time": 0.0,
        })

        with patch.object(problem_logic, 'execute_cpp_code', mock_execute_cpp_code):
            result = await problem_logic.evaluate_solution_code(
                cpp_code="faulty_code",
                test_cases=test_cases,
                score_calculator_func=self.sync_scorer_success, # Scorer won't be called
                docker_manager=mock_docker_manager,
                timeout_seconds=5
            )

        assert result["overall_score"] == 0.0 # As per ProblemLogic.evaluate_solution_code
        assert len(result["per_test_case_results"]) == 1
        tc_result = result["per_test_case_results"][0]
        assert tc_result["test_case_name"] == "compilation_check" # evaluate_solution_code specific
        assert tc_result["compilation_success"] is False
        assert tc_result["compilation_stderr"] == "Major Syntax Error"
        assert tc_result["score"] == 0.0
        assert tc_result["error_type"] == "compilation"
        mock_execute_cpp_code.assert_called_once()


    @pytest.mark.asyncio
    async def test_evaluate_runtime_error_one_case(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [
            {"name": "case_ok", "input": "in_ok"},
            {"name": "case_fail", "input": "in_fail"},
            {"name": "case_good_again", "input": "in_good"}
        ]

        mock_execute_cpp_code = AsyncMock(side_effect=[
            { # case_ok
                "compilation_success": True, "execution_success": True, "execution_stdout": "out_ok",
                "error_type": None, "compilation_stdout": "Compiled"
            },
            { # case_fail
                "compilation_success": True, "execution_success": False, "execution_stderr": "Runtime Segfault",
                "error_type": "runtime", "error_message": "Runtime Segfault", "compilation_stdout": "Compiled"
            },
            { # case_good_again
                "compilation_success": True, "execution_success": True, "execution_stdout": "out_good",
                "error_type": None, "compilation_stdout": "Compiled"
            }
        ])

        with patch.object(problem_logic, 'execute_cpp_code', mock_execute_cpp_code):
            result = await problem_logic.evaluate_solution_code(
                cpp_code="some_code",
                test_cases=test_cases,
                score_calculator_func=self.sync_scorer_success, # Returns 100.0
                docker_manager=mock_docker_manager,
                timeout_seconds=5
            )

        assert result["overall_score"] == (100.0 + 0.0 + 100.0) / 3 # Average
        assert len(result["per_test_case_results"]) == 3

        assert result["per_test_case_results"][0]["score"] == 100.0
        assert result["per_test_case_results"][0]["error_type"] is None

        assert result["per_test_case_results"][1]["score"] == 0.0
        assert result["per_test_case_results"][1]["execution_success"] is False
        assert result["per_test_case_results"][1]["error_type"] == "runtime"
        assert result["per_test_case_results"][1]["error_message"] == "Runtime Segfault"

        assert result["per_test_case_results"][2]["score"] == 100.0
        assert result["per_test_case_results"][2]["error_type"] is None
        assert mock_execute_cpp_code.call_count == 3

    @pytest.mark.asyncio
    async def test_evaluate_score_calculation_error_sync_scorer(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"name": "case_score_fail", "input": "in_score_fail"}]

        mock_execute_cpp_code = AsyncMock(return_value={
            "compilation_success": True, "execution_success": True, "execution_stdout": "output_ok",
            "error_type": None, "compilation_stdout": "Compiled"
        })

        with patch.object(problem_logic, 'execute_cpp_code', mock_execute_cpp_code):
            result = await problem_logic.evaluate_solution_code(
                cpp_code="some_code",
                test_cases=test_cases,
                score_calculator_func=self.sync_scorer_exception, # Raises ValueError
                docker_manager=mock_docker_manager,
                timeout_seconds=5
            )

        assert result["overall_score"] == 0.0
        assert len(result["per_test_case_results"]) == 1
        tc_result = result["per_test_case_results"][0]
        assert tc_result["score"] == 0.0
        assert tc_result["score_calculation_error"] == "Exception in score_calculator_func: Scorer failed"

    @pytest.mark.asyncio
    async def test_evaluate_score_calculation_error_tuple_scorer(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"name": "case_score_tuple_err", "input": "in_tuple_err"}]

        mock_execute_cpp_code = AsyncMock(return_value={
            "compilation_success": True, "execution_success": True, "execution_stdout": "output_ok",
             "error_type": None, "compilation_stdout": "Compiled"
        })

        with patch.object(problem_logic, 'execute_cpp_code', mock_execute_cpp_code):
            result = await problem_logic.evaluate_solution_code(
                cpp_code="some_code",
                test_cases=test_cases,
                score_calculator_func=self.async_scorer_error_tuple, # Returns (50.0, "Minor issue")
                docker_manager=mock_docker_manager,
                timeout_seconds=5
            )

        assert result["overall_score"] == 50.0
        assert len(result["per_test_case_results"]) == 1
        tc_result = result["per_test_case_results"][0]
        assert tc_result["score"] == 50.0
        assert tc_result["score_calculation_error"] == "Minor issue in scoring"

    @pytest.mark.asyncio
    async def test_evaluate_no_test_cases(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = []

        mock_execute_cpp_code = AsyncMock() # Should not be called

        with patch.object(problem_logic, 'execute_cpp_code', mock_execute_cpp_code):
            result = await problem_logic.evaluate_solution_code(
                cpp_code="some_code",
                test_cases=test_cases,
                score_calculator_func=self.sync_scorer_success,
                docker_manager=mock_docker_manager,
                timeout_seconds=5
            )

        assert result["overall_score"] == 0.0
        assert len(result["per_test_case_results"]) == 0
        mock_execute_cpp_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_score_calculation_error_tuple_scorer_fixed_score(self, problem_logic: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"name": "case_score_tuple_err_fixed", "input": "in_tuple_err_fixed"}]

        mock_execute_cpp_code = AsyncMock(return_value={
            "compilation_success": True, "execution_success": True, "execution_stdout": "output_ok",
             "error_type": None, "compilation_stdout": "Compiled"
        })

        with patch.object(problem_logic, 'execute_cpp_code', mock_execute_cpp_code):
            result = await problem_logic.evaluate_solution_code(
                cpp_code="some_code",
                test_cases=test_cases,
                score_calculator_func=self.sync_scorer_error_tuple_fixed_score, # Returns (75.0, "Error details...")
                docker_manager=mock_docker_manager,
                timeout_seconds=5
            )

        assert result["overall_score"] == 75.0
        assert len(result["per_test_case_results"]) == 1
        tc_result = result["per_test_case_results"][0]
        assert tc_result["score"] == 75.0
        assert tc_result["score_calculation_error"] == "Error details for 75 score"

# Pytest needs to be able to find the file, and the imports must be correct
# relative to the project structure when pytest is run.
# For example, if tests are run from project root, and ahc_agent is a top-level package:
# from ahc_agent.core.problem_logic import ProblemLogic
# from ahc_agent.core.docker_manager import DockerManager
# from ahc_agent.utils.llm import LLMClient
# The above imports assume such a structure.
# If the test file is moved or the structure is different, imports might need adjustment.
# For now, this assumes pytest can resolve "ahc_agent".
# The use of tempfile and os is because the method being tested (execute_cpp_code) uses them internally.
# While these are not directly part of the test logic for mocking, they are dependencies of the tested code.
# The actual creation/deletion of temp files/dirs is part of execute_cpp_code's responsibility.
# Our tests for execute_cpp_code mock out the DockerManager calls, so the interactions with the
# filesystem (writing main.cpp to a temp dir) are implicitly part of what's being run.
# We don't assert on these file operations directly but on the results derived from them.
# This is acceptable for unit testing these methods.

# One final check on `mock_docker_manager.compile_cpp` return for success in `TestProblemLogicExecuteCppCode`:
# `execute_cpp_code` expects `compilation_result.get("success")` and uses `os.path.join(temp_dir, executable_name)`
# so `executable_path` in the mock isn't strictly necessary for `execute_cpp_code` itself, but good for completeness.
# `execute_cpp_code` determines the path itself based on `temp_dir` and `executable_name`.
# The mock for `compile_cpp` in `test_execute_success` currently doesn't provide `executable_path`,
# which is fine as `execute_cpp_code` doesn't use it from the *return* of `compile_cpp`.
# It *sets* `result["executable_path"]` based on its own `temp_dir`.

# The `run_cpp` mock for timeout in `test_execute_timeout_error` should return `status: "timeout"`
# as this is what `execute_cpp_code` checks for: `elif execution_result.get("status") == "timeout":`
# The prompt's initial suggestion of `{"error": "TimeoutExpired"}` is different. I've used `status: "timeout"`.
# Similarly for runtime error, `status: "runtime_error"` is used.

# For `TestProblemLogicEvaluateSolutionCode`, I've chosen to mock `problem_logic.execute_cpp_code` directly.
# This makes the tests for `evaluate_solution_code` more focused unit tests.
# If I were to mock `docker_manager.compile_cpp` and `docker_manager.run_cpp` instead,
# the tests for `evaluate_solution_code` would also be testing much of `execute_cpp_code`'s logic,
# making them more like integration tests between these two methods.
# Given the prompt asks for unit tests for *both* methods, this separation seems appropriate.
# The `docker_manager` is still passed to `evaluate_solution_code` as it's part of its signature and
# might be used for other things or passed down, even if `execute_cpp_code` is mocked.
# The `mock_execute_cpp_code.assert_called_once_with(...)` checks ensure that `evaluate_solution_code`
# calls `execute_cpp_code` with the correct parameters.

# One small adjustment in `test_execute_success`: `mock_docker_manager.run_cpp.return_value`
# should use `status: "success"` not `success: True` to align with what `execute_cpp_code` expects
# from `run_cpp` when determining status (it checks `execution_result.get("status") == "success"`).
# I've updated this in the generated code block.
# Similarly for `test_execute_runtime_error`, it should be `status: "runtime_error"` or similar that
# `execute_cpp_code` then translates to `error_type="runtime"`.
# `execute_cpp_code` sets `error_type="runtime"` if `status` is not "success" and not "timeout".
# So, a `status: "generic_error"` or `status: "runtime_error"` from `run_cpp` mock would work.
# I've used `status: "runtime_error"` for clarity in the mock.
# And `status: "timeout"` for timeout scenario.
# These details are important for `execute_cpp_code` to correctly interpret the mock `run_cpp` results.
# The `compilation_success` key in `execute_cpp_code` comes from `compilation_result.get("success")`, so that's fine.
# The `executable_path` in the `compile_cpp` mock for `TestProblemLogicExecuteCppCode`
# is not used by `execute_cpp_code` as it forms its own path. So, it's okay to omit or keep.
# The test `test_execute_success` checks `isinstance(result["executable_path"], str)` which is correct.
# The current ProblemLogic.execute_cpp_code uses `container_workdir=temp_dir` for `run_cpp`, and
# `executable_path=executable_name` (e.g. "a.out"). This means `run_cpp` is expected to find "a.out"
# within that mounted `temp_dir`. This detail is abstracted away by the mocks.

# The `test_evaluate_compilation_error_first_case` checks for `tc_result["test_case_name"] == "compilation_check"`.
# This is a specific behavior of `evaluate_solution_code` when compilation fails on the first attempt.
# This seems correct.
# The overall score is also correctly asserted as 0.0 in this case.

# The `test_evaluate_all_success_multiple_cases_async_scorer` asserts `overall_score == 150.0`.
# Since the scorer returns 150.0 for each of the two cases, the average is (150+150)/2 = 150. This is correct.

# All seems consistent with the problem description and the code for ProblemLogic.

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from ahc_agent.core.problem_logic import ProblemLogic
from ahc_agent.core.docker_manager import DockerManager
from ahc_agent.utils.llm import LLMClient


@pytest.fixture
def mock_llm_client():
    return MagicMock(spec=LLMClient)

@pytest.fixture
def mock_docker_manager():
    manager = MagicMock(spec=DockerManager)
    manager.compile_cpp = AsyncMock()
    manager.run_cpp = AsyncMock()
    return manager

@pytest.fixture
def problem_logic_instance(mock_llm_client, mock_docker_manager): # Renamed to avoid conflict with module name
    # ProblemLogic might try to access config on LLMClient, so ensure it's there
    mock_llm_client.config = {}
    return ProblemLogic(llm_client=mock_llm_client, config={})


class TestProblemLogicExecuteCppCode:
    @pytest.mark.asyncio
    async def test_execute_success(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        mock_docker_manager.compile_cpp.return_value = {
            "success": True,
            "stdout": "Compiled successfully",
            "stderr": "",
        }
        mock_docker_manager.run_cpp.return_value = {
            "status": "success",
            "stdout": "Hello World",
            "stderr": "",
            "execution_time": 0.05,
        }

        cpp_code = "#include <iostream>\nint main() { std::cout << \"Hello World\" << std::endl; return 0; }"
        input_data = "test_input"
        timeout = 10

        result = await problem_logic_instance.execute_cpp_code(cpp_code, input_data, mock_docker_manager, timeout)

        assert result["compilation_success"] is True
        assert result["compilation_stdout"] == "Compiled successfully"
        assert result["execution_success"] is True
        assert result["execution_stdout"] == "Hello World"
        assert result["execution_time"] == 0.05
        assert result["error_type"] is None
        mock_docker_manager.compile_cpp.assert_called_once()
        # The path to source_file_path for compile_cpp will be dynamic due to tempfile
        # We can assert that it was called with a path ending in 'main.cpp'
        args, kwargs = mock_docker_manager.compile_cpp.call_args
        assert kwargs['source_file_path'].endswith("main.cpp")
        assert kwargs['output_name'] == "a.out"

        mock_docker_manager.run_cpp.assert_called_once()
        args_run, kwargs_run = mock_docker_manager.run_cpp.call_args
        assert kwargs_run['executable_path'] == "a.out" # Name in container
        assert kwargs_run['input_data'] == input_data
        assert kwargs_run['timeout_seconds'] == timeout
        assert os.path.isdir(kwargs_run['container_workdir']) # temp_dir from host

        assert isinstance(result["executable_path"], str) # Path on host
        assert result["executable_path"].endswith("a.out")


    @pytest.mark.asyncio
    async def test_execute_compilation_error(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        mock_docker_manager.compile_cpp.return_value = {
            "success": False,
            "stdout": "Compilation failed",
            "stderr": "Syntax Error on line 1",
        }

        cpp_code = "invalid code"
        input_data = "test_input"
        timeout = 10

        result = await problem_logic_instance.execute_cpp_code(cpp_code, input_data, mock_docker_manager, timeout)

        assert result["compilation_success"] is False
        assert result["compilation_stderr"] == "Syntax Error on line 1"
        assert result["execution_success"] is False
        assert result["error_type"] == "compilation"
        assert result["error_message"] == "Syntax Error on line 1"
        mock_docker_manager.compile_cpp.assert_called_once()
        mock_docker_manager.run_cpp.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        mock_docker_manager.compile_cpp.return_value = {"success": True, "stdout": "", "stderr": ""}
        mock_docker_manager.run_cpp.return_value = {
            "status": "runtime_error",
            "stdout": "",
            "stderr": "Segmentation fault",
            "execution_time": 0.02,
        }

        cpp_code = "valid code"
        input_data = "trigger_runtime_error"
        timeout = 10

        result = await problem_logic_instance.execute_cpp_code(cpp_code, input_data, mock_docker_manager, timeout)

        assert result["compilation_success"] is True
        assert result["execution_success"] is False
        assert result["execution_stderr"] == "Segmentation fault"
        assert result["error_type"] == "runtime"
        assert result["error_message"] == "Segmentation fault" # This comes from execution_result.get("stderr", ...)
        mock_docker_manager.run_cpp.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_timeout_error(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        mock_docker_manager.compile_cpp.return_value = {"success": True, "stdout": "", "stderr": ""}
        mock_docker_manager.run_cpp.return_value = {
            "status": "timeout",
            "stdout": "Partial output",
            "stderr": "",
            "execution_time": 10.0,
        }
        cpp_code = "long running code"
        input_data = "test_input"
        timeout = 10

        result = await problem_logic_instance.execute_cpp_code(cpp_code, input_data, mock_docker_manager, timeout)

        assert result["compilation_success"] is True
        assert result["execution_success"] is False
        assert result["error_type"] == "timeout"
        assert result["error_message"] == "Execution timed out."
        assert result["execution_stdout"] == "Partial output"
        mock_docker_manager.run_cpp.assert_called_once()

class TestProblemLogicEvaluateSolutionCode:

    def sync_scorer_success(self, input_str, output_str): return 100.0
    async def async_scorer_success(self, input_str, output_str): await asyncio.sleep(0); return 150.0
    def sync_scorer_exception(self, input_str, output_str): raise ValueError("Scorer failed")
    async def async_scorer_error_tuple(self, input_str, output_str): await asyncio.sleep(0); return 50.0, "Minor issue in scoring"
    def sync_scorer_error_tuple_fixed_score(self, input_str, output_str): return 75.0, "Error details for 75 score"


    @pytest.mark.asyncio
    async def test_evaluate_all_success_single_case_sync_scorer(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"name": "case1", "input": "input1"}]

        mock_exec_code = AsyncMock(return_value={
            "compilation_success": True, "execution_success": True,
            "execution_stdout": "output1", "execution_stderr": "", "execution_time": 0.1,
            "compilation_stdout": "Compiled", "compilation_stderr": "", "error_type": None, "error_message": "",
            "executable_path": "/tmp/a.out"
        })

        with patch.object(problem_logic_instance, 'execute_cpp_code', mock_exec_code):
            result = await problem_logic_instance.evaluate_solution_code(
                "dummy_code", test_cases, self.sync_scorer_success, mock_docker_manager, 5
            )

        assert result["overall_score"] == 100.0
        assert len(result["per_test_case_results"]) == 1
        tc_res = result["per_test_case_results"][0]
        assert tc_res["test_case_name"] == "case1"
        assert tc_res["score"] == 100.0
        assert tc_res["execution_success"] is True
        mock_exec_code.assert_called_once_with(
            cpp_code="dummy_code", input_data="input1",
            docker_manager=mock_docker_manager, timeout_seconds=5
        )

    @pytest.mark.asyncio
    async def test_evaluate_all_success_multiple_cases_async_scorer(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"name": "A", "input": "inA"}, {"name": "B", "input": "inB"}]

        mock_exec_code = AsyncMock(side_effect=[
            {"compilation_success": True, "execution_success": True, "execution_stdout": "outA", "executable_path": "/tmp/a.out"},
            {"compilation_success": True, "execution_success": True, "execution_stdout": "outB", "executable_path": "/tmp/a.out"}
        ])

        with patch.object(problem_logic_instance, 'execute_cpp_code', mock_exec_code):
            result = await problem_logic_instance.evaluate_solution_code(
                "dummy_code", test_cases, self.async_scorer_success, mock_docker_manager, 5
            )

        assert result["overall_score"] == 150.0
        assert len(result["per_test_case_results"]) == 2
        assert result["per_test_case_results"][0]["score"] == 150.0
        assert result["per_test_case_results"][1]["score"] == 150.0
        assert mock_exec_code.call_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_compilation_error_first_case(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"name": "case1", "input": "input1"}]

        mock_exec_code = AsyncMock(return_value={
            "compilation_success": False, "compilation_stderr": "Syntax Error",
            "error_type": "compilation", "error_message": "Syntax Error",
            "execution_success": False, # ensure this is also false
            "executable_path": None
        })

        with patch.object(problem_logic_instance, 'execute_cpp_code', mock_exec_code):
            result = await problem_logic_instance.evaluate_solution_code(
                "faulty_code", test_cases, self.sync_scorer_success, mock_docker_manager, 5
            )

        assert result["overall_score"] == 0.0
        assert len(result["per_test_case_results"]) == 1
        tc_res = result["per_test_case_results"][0]
        assert tc_res["test_case_name"] == "compilation_check"
        assert tc_res["compilation_success"] is False
        assert tc_res["score"] == 0.0
        assert tc_res["error_type"] == "compilation"
        mock_exec_code.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_runtime_error_one_case(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"input": "in_ok"}, {"input": "in_fail"}, {"input": "in_good"}]

        mock_exec_code = AsyncMock(side_effect=[
            {"compilation_success": True, "execution_success": True, "execution_stdout": "out_ok", "executable_path": "/tmp/a.out"},
            {"compilation_success": True, "execution_success": False, "error_type": "runtime", "error_message": "Segfault", "executable_path": "/tmp/a.out"},
            {"compilation_success": True, "execution_success": True, "execution_stdout": "out_good", "executable_path": "/tmp/a.out"}
        ])

        with patch.object(problem_logic_instance, 'execute_cpp_code', mock_exec_code):
            result = await problem_logic_instance.evaluate_solution_code(
                "code", test_cases, self.sync_scorer_success, mock_docker_manager, 5 # score 100
            )

        assert result["overall_score"] == (100.0 + 0.0 + 100.0) / 3
        assert result["per_test_case_results"][1]["score"] == 0.0
        assert result["per_test_case_results"][1]["error_type"] == "runtime"

    @pytest.mark.asyncio
    async def test_evaluate_score_calculation_error_sync_scorer(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"input": "in_score_fail"}]
        mock_exec_code = AsyncMock(return_value={"compilation_success": True, "execution_success": True, "execution_stdout": "out", "executable_path": "/tmp/a.out"})

        with patch.object(problem_logic_instance, 'execute_cpp_code', mock_exec_code):
            result = await problem_logic_instance.evaluate_solution_code(
                "code", test_cases, self.sync_scorer_exception, mock_docker_manager, 5
            )

        assert result["overall_score"] == 0.0
        assert result["per_test_case_results"][0]["score"] == 0.0
        assert "Scorer failed" in result["per_test_case_results"][0]["score_calculation_error"]

    @pytest.mark.asyncio
    async def test_evaluate_score_calculation_error_tuple_scorer(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"input": "in_tuple_err"}]
        mock_exec_code = AsyncMock(return_value={"compilation_success": True, "execution_success": True, "execution_stdout": "out", "executable_path": "/tmp/a.out"})

        with patch.object(problem_logic_instance, 'execute_cpp_code', mock_exec_code):
            result = await problem_logic_instance.evaluate_solution_code(
                "code", test_cases, self.async_scorer_error_tuple, mock_docker_manager, 5
            )

        assert result["overall_score"] == 50.0
        assert result["per_test_case_results"][0]["score"] == 50.0
        assert result["per_test_case_results"][0]["score_calculation_error"] == "Minor issue in scoring"

    @pytest.mark.asyncio
    async def test_evaluate_no_test_cases(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        mock_exec_code = AsyncMock()
        with patch.object(problem_logic_instance, 'execute_cpp_code', mock_exec_code):
            result = await problem_logic_instance.evaluate_solution_code(
                "code", [], self.sync_scorer_success, mock_docker_manager, 5
            )
        assert result["overall_score"] == 0.0
        assert len(result["per_test_case_results"]) == 0
        mock_exec_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_score_calculation_error_tuple_scorer_fixed_score(self, problem_logic_instance: ProblemLogic, mock_docker_manager: MagicMock):
        test_cases = [{"input": "in_tuple_err_fixed"}]
        mock_exec_code = AsyncMock(return_value={"compilation_success": True, "execution_success": True, "execution_stdout": "out", "executable_path": "/tmp/a.out"})

        with patch.object(problem_logic_instance, 'execute_cpp_code', mock_exec_code):
            result = await problem_logic_instance.evaluate_solution_code(
                "code", test_cases, self.sync_scorer_error_tuple_fixed_score, mock_docker_manager, 5
            )

        assert result["overall_score"] == 75.0
        assert result["per_test_case_results"][0]["score"] == 75.0
        assert result["per_test_case_results"][0]["score_calculation_error"] == "Error details for 75 score"

```
