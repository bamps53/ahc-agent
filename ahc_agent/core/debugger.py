"""
Implementation debugger module for AHCAgent.

This module provides functionality for debugging and testing C++ implementations.
"""

import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Union

from ahc_agent.utils.docker_manager import DockerManager
from ahc_agent.utils.file_io import ensure_directory, write_file
from ahc_agent.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class ImplementationDebugger:
    """
    Debugger for C++ implementations.
    """

    def __init__(self, llm_client: LLMClient, docker_manager: DockerManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the implementation debugger.

        Args:
            llm_client: LLM client for code analysis and debugging
            docker_manager: Docker manager for code compilation and execution
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.docker_manager = docker_manager
        self.config = config or {}

        # Get C++ compilation settings from config or use defaults
        self.cpp_compiler = self.config.get("cpp_compiler", "g++")
        self.cpp_flags = self.config.get("cpp_flags", "-std=c++17 -O2 -Wall")
        self.execution_timeout = self.config.get("execution_timeout", 10)

        logger.info("Initialized implementation debugger")

    async def compile_solution(self, code: str, work_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Compile a C++ solution.

        Args:
            code: C++ code string.
            work_dir: Working directory. If None, a temporary directory will be created.

        Returns:
            Path to the compiled executable if successful, None otherwise.
        """
        logger.info("Compiling C++ solution...")
        work_dir_path = Path(ensure_directory(work_dir) if work_dir else tempfile.mkdtemp(prefix="ahc_compile_"))
        source_file = "solution.cpp"
        source_path = work_dir_path / source_file
        executable_name = "solution_executable"
        executable_path = work_dir_path / executable_name

        try:
            write_file(source_path, code)
            compile_result = self.docker_manager.compile_cpp(source_file=source_file, work_dir=str(work_dir_path), output_filename=executable_name)

            if compile_result["success"]:
                logger.info(f"Compilation successful. Executable at: {executable_path}")
                return executable_path
            logger.error(f"Compilation failed: {compile_result['stderr']}")
            # Attempt to fix compilation errors once
            fixed_code = await self._fix_compilation_errors(code, compile_result["stderr"])
            if fixed_code != code:
                logger.info("Attempting to compile with fixed code...")
                write_file(source_path, fixed_code)
                compile_result = self.docker_manager.compile_cpp(
                    source_file=source_file, work_dir=str(work_dir_path), output_filename=executable_name
                )
                if compile_result["success"]:
                    logger.info(f"Compilation successful with fixed code. Executable at: {executable_path}")
                    # It might be good to also return the fixed_code if successful
                    return executable_path
                logger.error(f"Compilation still failed with fixed code: {compile_result['stderr']}")
                return None
            return None
        except Exception as e:
            logger.error(f"Error during compilation process: {type(e).__name__} - {e!s}")
            return None

    def run_test_case(self, executable_path: Path, input_data: str) -> Dict[str, Any]:
        """
        Run a compiled C++ solution with the given input.

        Args:
            executable_path: Path to the compiled executable.
            input_data: Test input string.

        Returns:
            Dictionary with execution results (stdout, stderr, execution_time, success).
        """
        logger.info(f"Running test case with executable: {executable_path}")
        work_dir_path = executable_path.parent
        input_file = "input.txt"
        input_file_path = work_dir_path / input_file
        # output_file variable was unused

        try:
            write_file(input_file_path, input_data)
            run_result = self.docker_manager.run_cpp(
                executable_name=executable_path.name,  # run_cpp expects just the name relative to work_dir
                work_dir=str(work_dir_path),
                input_filename=input_file,
                timeout_seconds=self.execution_timeout,
            )
            return {
                "stdout": run_result["stdout"],
                "stderr": run_result["stderr"],
                "execution_time": run_result["execution_time"],
                "success": run_result["success"] and not run_result["timeout"],
                "timeout": run_result["timeout"],
            }
        except Exception as e:
            logger.error(f"Error running test case: {type(e).__name__} - {e!s}")
            return {
                "stdout": "",
                "stderr": f"Error in run_test_case: {type(e).__name__} - {e!s}",
                "execution_time": 0,
                "success": False,
                "timeout": False,
            }

    async def compile_and_test(self, code: str, test_input: str, work_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Compile and test a C++ implementation. (Combines compile_solution and run_test_case)

        Args:
            code: C++ code
            test_input: Test input data
            work_dir: Working directory. If None, a temporary directory will be created.

        Returns:
            Dictionary with test results, including compilation and execution status.
        """
        logger.info("Attempting to compile and test C++ implementation...")

        # Determine working directory, creating a temporary one if not provided
        # This temp directory will be used for compilation and execution artifacts.
        # It's important that compile_solution and run_test_case use the same work_dir if called sequentially.
        # For compile_and_test, we manage a single work_dir for both operations.
        current_work_dir = Path(ensure_directory(work_dir) if work_dir else tempfile.mkdtemp(prefix="ahc_ct_"))

        executable_path = await self.compile_solution(code, work_dir=current_work_dir)

        if not executable_path:
            # Compilation failed (and optional fix attempt also failed)
            # compile_solution logs the details, so we just return a failure structure.
            # If a fixed code was attempted and failed, it's not directly returned here,
            # but the original code is what's considered to have failed compilation.
            return {
                "success": False,
                "compilation_success": False,
                "compilation_errors": "Compilation failed. Check logs for details.",  # More specific errors logged by compile_solution
                "execution_success": False,
                "execution_output": "",
                "execution_errors": "",
                "execution_time": None,
                "fixed_code": code,  # The code that failed to compile
                "executable_path": None,
            }

        # Compilation successful, now run the test case
        run_result = self.run_test_case(executable_path, test_input)

        # If execution failed and it wasn't a timeout, attempt to fix runtime errors
        # This part is simplified: we won't re-compile here. A more robust fix might require it.
        # For now, if a runtime fix is generated, it's more of a suggestion.
        fixed_code_runtime = code  # Default to original or compilation-fixed code
        if not run_result["success"] and not run_result["timeout"] and run_result["stderr"]:
            logger.info("Execution failed, attempting to fix runtime error...")
            fixed_code_runtime_attempt = await self._fix_runtime_errors(code, run_result["stderr"], test_input)
            if fixed_code_runtime_attempt != code:
                logger.info("Runtime fix generated. Note: This version is not re-compiled/re-run by compile_and_test.")
                # In a more complex scenario, you might re-compile and re-run here.
                # For now, we just note that a fix was suggested.
                fixed_code_runtime = fixed_code_runtime_attempt
            else:
                logger.info("LLM did not suggest a change for the runtime error.")

        # Clean up: DockerManager might create a.out or other executables.
        # The specific executable_path from compile_solution is the one we care about.
        # Temporary directory cleanup is handled by the caller or OS if not specified.
        return {
            "success": run_result["success"],
            "compilation_success": True,
            "compilation_errors": "",  # Already handled by compile_solution
            "execution_success": run_result["success"],
            "execution_output": run_result["stdout"],
            "execution_errors": run_result["stderr"],
            "execution_time": run_result["execution_time"],
            "fixed_code": fixed_code_runtime,  # This could be original, compilation-fixed, or runtime-fixed suggestion
            "executable_path": str(executable_path),  # Path to the executable used
        }

    async def evaluate_solution(self, code: str, test_cases: List[Dict[str, Any]], work_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a solution on multiple test cases.

        Args:
            code: C++ code
            test_cases: List of test cases, each with "input" and "expected_output" (optional)
            work_dir: Working directory

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating solution on {len(test_cases)} test cases")

        # Ensure work directory exists
        work_dir_path = ensure_directory(work_dir) if work_dir else tempfile.mkdtemp(prefix="ahc_eval_")

        test_results = []
        overall_success = True
        total_score = 0
        fixed_code_applied_globally = None

        try:
            # Initial compilation attempt
            source_file = "solution.cpp"
            source_path = os.path.join(work_dir_path, source_file)
            write_file(source_path, code)

            compile_result = self.docker_manager.compile_cpp(source_file, work_dir_path)

            if not compile_result["success"]:
                logger.error(f"Initial compilation failed: {compile_result['stderr']}")
                compilation_errors = compile_result["stderr"]
                fixed_code_comp = await self._fix_compilation_errors(code, compilation_errors)
                if fixed_code_comp != code:
                    logger.info("Attempting to use fixed code for all test cases after initial compilation failure.")
                    write_file(source_path, fixed_code_comp)
                    compile_result = self.docker_manager.compile_cpp(source_file, work_dir_path)
                    if compile_result["success"]:
                        logger.info("Compilation successful with fixed code.")
                        code = fixed_code_comp
                        fixed_code_applied_globally = code
                    else:
                        logger.error(f"Compilation still failed with fixed code: {compile_result['stderr']}")
                        return {
                            "success": False,
                            "compilation_success": False,
                            "compilation_errors": compile_result["stderr"],
                            "test_results": [],
                            "total_score": 0,
                            "average_score": 0,
                            "fixed_code": fixed_code_comp,
                        }
                else:
                    # No fix was generated or applied
                    return {
                        "success": False,
                        "compilation_success": False,
                        "compilation_errors": compilation_errors,
                        "test_results": [],
                        "total_score": 0,
                        "average_score": 0,
                        "fixed_code": None,
                    }

            executable_file = "solution"

            for i, test_case in enumerate(test_cases):
                test_input = test_case["input"]
                input_file = f"input_{i}.txt"
                input_path = os.path.join(work_dir_path, input_file)
                write_file(input_path, test_input)

                execute_result = self.docker_manager.run_executable(executable_file, work_dir_path, input_file, timeout=self.execution_timeout)

                current_code_used_for_this_test_case = code
                fixed_in_this_iteration = False

                if not execute_result["success"]:
                    logger.warning(f"Execution failed for test case {i}: {execute_result['stderr']}")
                    runtime_errors = execute_result["stderr"]
                    # Attempt to fix runtime error based on this specific test case
                    fixed_code_rt = await self._fix_runtime_errors(current_code_used_for_this_test_case, runtime_errors, test_input)

                    if fixed_code_rt != current_code_used_for_this_test_case:
                        logger.info(f"Runtime fix generated for test case {i}. Re-compiling and re-testing.")
                        # Write new fixed code, compile, and run again for this test case
                        write_file(source_path, fixed_code_rt)
                        temp_compile_result = self.docker_manager.compile_cpp(source_file, work_dir_path)

                        if temp_compile_result["success"]:
                            execute_result = self.docker_manager.run_executable(
                                executable_file, work_dir_path, input_file, timeout=self.execution_timeout
                            )
                            if execute_result["success"]:
                                logger.info(f"Execution successful with runtime-fixed code for test case {i}.")
                                # Decide if this fix should be applied globally or just for this test
                                # For simplicity, let's assume we might want to update the global 'code' variable
                                # if a fix works. This could be made more sophisticated.
                                code = fixed_code_rt
                                fixed_code_applied_globally = code
                                current_code_used_for_this_test_case = code
                                fixed_in_this_iteration = True
                            else:
                                logger.warning(f"Execution still failed for test case {i} with runtime-fixed code.")
                        else:
                            logger.error(f"Compilation failed for runtime-fixed code on test case {i}: {temp_compile_result['stderr']}")
                            # If compilation of a fix fails, revert to original error for this test case
                            # and continue with the original code for subsequent tests unless a global fix was already applied.
                            # This 'execute_result' will be the one from the failed execution before the fix attempt.
                    # If no fix was generated, execute_result remains the failed one.

                # Scoring logic (simplified, adapt as needed)
                score = 0
                if execute_result["success"]:
                    # Placeholder for actual scoring logic
                    # For example, compare execute_result["stdout"] with test_case["expected_output"]
                    # This example assumes a simple score of 1 for success, 0 for failure.
                    score = test_case.get("score_if_correct", 1)
                    total_score += score
                else:
                    overall_success = False

                test_results.append(
                    {
                        "test_case_id": i,
                        "success": execute_result["success"],
                        "output": execute_result["stdout"],
                        "errors": execute_result["stderr"],
                        "execution_time": execute_result["execution_time"],
                        "score": score,
                        "fixed_this_iteration": fixed_in_this_iteration,
                        "code_used": current_code_used_for_this_test_case,
                    }
                )

            # Calculate average score
            average_score = total_score / len(test_cases) if test_cases else 0

            logger.info(f"Evaluation complete: {len(test_results)} test cases, total score = {total_score}, average score = {average_score}")

            return {
                "success": overall_success,
                "compilation_success": True,
                "compilation_errors": "",
                "test_results": test_results,
                "total_score": total_score,
                "average_score": average_score,
                "fixed_code": fixed_code_applied_globally,  # The last successfully applied fix
            }

        except (ValueError, RuntimeError, TypeError, OSError) as e:
            logger.error(f"Error in evaluate_solution: {e!s}")
            return {
                "success": False,
                "compilation_success": False,
                "compilation_errors": str(e),
                "test_results": [],
                "total_score": 0,
                "average_score": 0,
                "fixed_code": None,
            }

    async def _fix_compilation_errors(self, code: str, compilation_errors: str) -> str:
        """
        Fix compilation errors using LLM.

        Args:
            code: Original C++ code
            compilation_errors: Compilation error messages

        Returns:
            Fixed C++ code
        """
        logger.info("Attempting to fix compilation errors")

        prompt = f"""
        You are an expert C++ programmer. Please fix the compilation errors in the following code:

        ```cpp
        {code}
        ```

        Compilation errors:
        ```
        {compilation_errors}
        ```

        Please provide the fixed code. Make minimal changes to fix the errors.
        Return only the fixed C++ code without any explanations.
        ```cpp
        // Your fixed code here
        ```
        """

        try:
            response = await self.llm_client.generate(prompt)

            # Extract code from response
            return self._extract_code(response)

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error fixing compilation errors: {e!s}")
            return code

    async def _fix_runtime_errors(self, code: str, runtime_errors: str, test_input: str) -> str:
        """
        Fix runtime errors using LLM.

        Args:
            code: Original C++ code
            runtime_errors: Runtime error messages
            test_input: Test input data

        Returns:
            Fixed C++ code
        """
        logger.info("Attempting to fix runtime errors")

        prompt = f"""
        You are an expert C++ programmer. Please fix the runtime errors in the following code:

        ```cpp
        {code}
        ```

        Runtime errors:
        ```
        {runtime_errors}
        ```

        Test input:
        ```
        {test_input}
        ```

        Please provide the fixed code. Make minimal changes to fix the errors.
        Return only the fixed C++ code without any explanations.
        ```cpp
        // Your fixed code here
        ```
        """

        try:
            response = await self.llm_client.generate(prompt)

            # Extract code from response
            return self._extract_code(response)

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error fixing runtime errors: {e!s}")
            return code

    def _extract_code(self, response: str) -> str:
        """
        Extract code from LLM response.

        Args:
            response: LLM response

        Returns:
            Extracted code
        """
        # Try to extract code between ```cpp and ``` markers

        code_match = re.search(r"```cpp\s*(.*?)\s*```", response, re.DOTALL)

        if code_match:
            return code_match.group(1).strip()

        # If no markers found, return the entire response
        return response.strip()
