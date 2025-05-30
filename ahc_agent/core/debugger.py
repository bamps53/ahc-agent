"""
Implementation debugger module for AHCAgent.

This module provides functionality for debugging and testing C++ implementations.
"""

import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

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

    async def compile_and_test(self, code: str, test_input: str, work_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile and test a C++ implementation.

        Args:
            code: C++ code
            test_input: Test input data
            work_dir: Working directory

        Returns:
            Dictionary with test results
        """
        logger.info("Compiling and testing C++ implementation")

        # Ensure work directory exists
        work_dir_path = ensure_directory(work_dir) if work_dir else tempfile.mkdtemp(prefix="ahc_debug_")

        try:
            # Write code to file
            source_file = "solution.cpp"
            source_path = os.path.join(work_dir_path, source_file)
            write_file(source_path, code)

            # Write test input to file
            input_file = "input.txt"
            input_path = os.path.join(work_dir_path, input_file)
            write_file(input_path, test_input)

            # Compile code
            compile_result = self.docker_manager.compile_cpp(source_file, work_dir_path)

            if not compile_result["success"]:
                logger.error(f"Compilation failed: {compile_result['stderr']}")

                # Get compilation errors
                compilation_errors = compile_result["stderr"]

                # Fix compilation errors using LLM
                fixed_code = await self._fix_compilation_errors(code, compilation_errors)

                # Try compiling again with fixed code
                if fixed_code != code:
                    logger.info("Trying to compile with fixed code")

                    # Write fixed code to file
                    write_file(source_path, fixed_code)

                    # Compile fixed code
                    compile_result = self.docker_manager.compile_cpp(source_file, work_dir_path)

                    if compile_result["success"]:
                        logger.info("Compilation successful with fixed code")
                        code = fixed_code
                    else:
                        logger.error(f"Compilation still failed with fixed code: {compile_result['stderr']}")
                        return {
                            "success": False,
                            "compilation_success": False,
                            "compilation_errors": compile_result["stderr"],
                            "execution_success": False,
                            "execution_output": "",
                            "execution_errors": "",
                            "execution_time": None,
                            "fixed_code": fixed_code,
                        }
                else:
                    return {
                        "success": False,
                        "compilation_success": False,
                        "compilation_errors": compilation_errors,
                        "execution_success": False,
                        "execution_output": "",
                        "execution_errors": "",
                        "execution_time": None,
                        "fixed_code": None,
                    }

            # Execute code
            executable_file = "solution"
            execute_result = self.docker_manager.run_executable(executable_file, work_dir_path, input_file, timeout=self.execution_timeout)

            if not execute_result["success"]:
                logger.error(f"Execution failed: {execute_result['stderr']}")

                # Get runtime errors
                runtime_errors = execute_result["stderr"]

                # Fix runtime errors using LLM
                fixed_code_runtime = await self._fix_runtime_errors(code, runtime_errors, test_input)

                # Try executing again with fixed code
                if fixed_code_runtime != code:
                    logger.info("Trying to execute with fixed code (runtime error fix)")

                    # Write fixed code to file
                    write_file(source_path, fixed_code_runtime)

                    # Re-compile fixed code
                    compile_result_rerun = self.docker_manager.compile_cpp(source_file, work_dir_path)

                    if compile_result_rerun["success"]:
                        # Execute fixed code
                        execute_result = self.docker_manager.run_executable(
                            executable_file, work_dir_path, input_file, timeout=self.execution_timeout
                        )
                        if execute_result["success"]:
                            logger.info("Execution successful with fixed code")
                            code = fixed_code_runtime
                        else:
                            logger.error(f"Execution still failed with fixed code: {execute_result['stderr']}")
                            # Fall through to return current execute_result
                    else:
                        logger.error(f"Compilation failed for runtime-fixed code: {compile_result_rerun['stderr']}")
                        # Return info about this new compilation failure
                        return {
                            "success": False,
                            "compilation_success": False,
                            "compilation_errors": compile_result_rerun["stderr"],
                            "execution_success": False,
                            "execution_output": "",
                            "execution_errors": "",
                            "execution_time": None,
                            "fixed_code": fixed_code_runtime,
                        }

            return {
                "success": execute_result["success"],
                "compilation_success": True,
                "compilation_errors": "",
                "execution_success": execute_result["success"],
                "execution_output": execute_result["stdout"],
                "execution_errors": execute_result["stderr"],
                "execution_time": execute_result["execution_time"],
                "fixed_code": code,
            }

        except (ValueError, RuntimeError, TypeError, OSError) as e:
            logger.error(f"Error in compile_and_test: {e!s}")
            return {
                "success": False,
                "compilation_success": False,
                "compilation_errors": str(e),
                "execution_success": False,
                "execution_output": "",
                "execution_errors": "",
                "execution_time": None,
                "fixed_code": None,
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
