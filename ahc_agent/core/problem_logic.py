"""
Problem logic module for AHCAgent.

This module provides functionality for handling AtCoder Heuristic Contest problem-specific logic.
"""

import json
import logging
import asyncio
import os
import random
import tempfile
from typing import Any, Callable, Dict, List, Optional

from ahc_agent.core.docker_manager import DockerManager
from ahc_agent.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class ProblemLogic:
    """
    Logic for handling AtCoder Heuristic Contest problems.
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the problem logic.

        Args:
            llm_client: LLM client.
            config: Configuration dictionary (optional).
        """
        self.llm_client = llm_client
        self.config = config or {}

        logger.info("Initialized problem logic")

    async def _extract_io_format(self, problem_text: str) -> Dict[str, Any]:
        """
        Extract I/O format from problem text using LLM.
        """
        prompt = f"""
            Extract the input and output format from the problem text.
            Problem: {problem_text}

            Output JSON format:
            {{
              "input_format": {{ "<parameter_name>": "<description>" }},
              "output_format": {{ "<value_name>": "<description>" }}
            }}

            Return only the JSON object.
            """
        try:
            return await self.llm_client.generate_json(prompt)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error extracting I/O format: {type(e).__name__} - {e!s}")
            return {"input_format": {}, "output_format": {}}

    async def _extract_constraints(self, problem_text: str) -> Dict[str, Any]:
        """
        Extract constraints from problem text using LLM.
        """
        prompt = f"""
            Extract the constraints from the problem text.
            Problem: {problem_text}

            Output JSON format:
            {{
              "<constraint_name>": "<description>"
            }}

            Return only the JSON object.
            """
        try:
            return await self.llm_client.generate_json(prompt)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error extracting constraints: {type(e).__name__} - {e!s}")
            return {}

    async def _extract_scoring_rules(self, problem_text: str) -> Dict[str, Any]:
        """
        Extract scoring rules from problem text using LLM.
        """
        prompt = f"""
            Extract the scoring rules from the problem text.
            Problem: {problem_text}

            Output JSON format:
            {{
              "objective": "<minimize|maximize>",
              "formula": "<description of scoring formula>",
              "score_range": "<typical score range, e.g., 0 to 1,000,000>"
            }}

            Return only the JSON object.
            """
        try:
            return await self.llm_client.generate_json(prompt)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error extracting scoring rules: {type(e).__name__} - {e!s}")
            return {"objective": "unknown", "formula": "", "score": ""}

    async def generate_test_cases(self, problem_info: Dict[str, Any], num_cases: int = 5) -> List[Dict[str, Any]]:
        """
        Generate test cases for a problem.
        """
        test_cases = []
        for i in range(num_cases):
            try:
                test_case_json = await self._generate_test_case(problem_info, i)
                if test_case_json:
                    test_cases.append(test_case_json)
            except Exception as e:
                logger.error(f"Error generating and processing test case {i + 1}: {type(e).__name__} - {e!s}")

        logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases

    async def _generate_test_case(self, problem_info: Dict[str, Any], case_index: int) -> Optional[Dict[str, Any]]:
        """
        Generate a single test case.
        """
        prompt = (
            f"Problem Information:\n{json.dumps(problem_info, indent=2)}\n\n"
            f"Constraints:\n{json.dumps(problem_info.get('constraints', {}), indent=2)}\n\n"
            f"Generate a valid test case that satisfies all constraints. For test case {case_index + 1}, \n"
            f"make it {('simple and small' if case_index == 0 else 'complex and challenging')}.\n\n"
            f"Return only the test case input data in the exact format required by the problem, without any explanations."
        )
        try:
            input_data = await self.llm_client.generate(prompt)
            return {"id": f"case_{case_index + 1}", "input": input_data}
        except Exception as e:
            logger.error(f"Error generating test case data: {type(e).__name__} - {e!s}")
            return None

    async def create_score_calculator(self, problem_info: Dict[str, Any]) -> Callable[[str, str], float]:
        """
        Create a score calculator function for a problem.
        """
        scoring_rules = problem_info.get("scoring", {})
        score_formula = scoring_rules.get("formula", "")

        prompt = (
            f"Create a Python function `calculate_score(input_data_str, output_data_str)` \n"
            f"that implements the scoring logic: {score_formula}.\n"
            f"Return only the function code.\n"
            f"Input and output are strings.\n"
        )
        try:
            calculator_code = await self.llm_client.generate(prompt)
            namespace = {}
            exec(calculator_code, namespace)  # noqa: S102

            if "calculate_score" in namespace:
                calculator_func = namespace["calculate_score"]

                def safe_calculator(input_data: str, output_data: str) -> float:
                    try:
                        return calculator_func(input_data, output_data)
                    except Exception as e:
                        logger.error(f"Error calculating score: {type(e).__name__} - {e!s}")
                        return 0.0

                return safe_calculator
            logger.error("Failed to extract calculate_score function")
            return lambda _input_data, _output_data: 0.0

        except Exception as e:
            logger.error(f"Error creating score calculator: {type(e).__name__} - {e!s}")
            return lambda _input_data, _output_data: 0.0

    async def generate_initial_solution(self, problem_info: Dict[str, Any]) -> str:
        """
        Generate an initial solution for a problem.
        """
        prompt = f"""
            Generate a basic C++ solution for the problem described by: {json.dumps(problem_info, indent=2)}.
            Return only the C++ code.
            """
        try:
            return await self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"Error generating initial solution: {type(e).__name__} - {e!s}")
            return self._get_basic_template()

    def _get_basic_template(self) -> str:
        """
        Get a basic C++ template.
        """
        return """\
#include <iostream>
#include <vector>
#include <string>
// Add other necessary headers

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    // Read input
    // Implement logic
    // Print output
    return 0;
}
        """

    async def mutate_solution(
        self,
        solution_code: str,
        problem_info: Dict[str, Any],
        mutation_type: str = "small",
        feedback: Optional[str] = None,
    ) -> str:
        """
        Mutate a solution code using LLM.
        """
        feedback_section = f"\nFeedback on previous version:\n{feedback}\n" if feedback else ""
        mutation_type_description = (
            "small changes"
            if mutation_type == "small"
            else "significant changes"
            if mutation_type == "large"
            else "focused improvements"
            if mutation_type == "focused"
            else "random changes"
        )
        prompt = (
            f"Problem: {json.dumps(problem_info.get('title', 'N/A'), indent=2)}\n\n"
            f"Current C++ Solution:\n```cpp\n{solution_code}\n```\n\n"
            f"Please create a mutated version of the solution that:\n"
            f"1. Maintains the basic structure\n"
            f"2. Introduces {mutation_type_description}\n"
            f"3. Aims to improve the score\n{feedback_section}"
            f"Return only the mutated C++ code without any explanations.\n```cpp\n// Your mutated solution here\n```"
        )
        try:
            return await self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"Error mutating solution: {type(e).__name__} - {e!s}")
            return solution_code

    async def crossover_solutions(
        self,
        solution1_code: str,
        solution2_code: str,
        problem_info: Dict[str, Any],
        score1: Optional[float] = None,
        score2: Optional[float] = None,
    ) -> str:
        """
        Perform crossover between two solutions using LLM.
        """
        prompt = (
            f"Problem: {json.dumps(problem_info.get('title', 'N/A'), indent=2)}\n\n"
            f"Solution 1 (Score: {score1 if score1 is not None else 'N/A'}):\n```cpp\n{solution1_code}\n```\n\n"
            f"Solution 2 (Score: {score2 if score2 is not None else 'N/A'}):\n```cpp\n{solution2_code}\n```\n\n"
            f"Combine the best aspects of Solution 1 and Solution 2 to create a new, potentially better solution.\n"
            f"Return only the new C++ code without any explanations.\n```cpp\n// Your combined solution here\n```"
        )
        try:
            return await self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"Error performing crossover: {type(e).__name__} - {e!s}")
            if score1 is not None and score2 is not None:
                return solution1_code if score1 >= score2 else solution2_code
            return random.choice([solution1_code, solution2_code])  # noqa: S311

    async def execute_cpp_code(
        self,
        cpp_code: str,
        input_data: str,
        docker_manager: DockerManager,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        """
        Compiles and runs C++ code using Docker, then returns execution results.
        """
        result: Dict[str, Any] = {
            "compilation_success": False,
            "compilation_stdout": "",
            "compilation_stderr": "",
            "executable_path": None,
            "execution_success": False,
            "execution_stdout": "",
            "execution_stderr": "",
            "execution_time": 0.0,
            "error_type": None,  # "compilation", "timeout", "runtime"
            "error_message": "",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            source_file_path = os.path.join(temp_dir, "main.cpp")
            executable_name = "a.out"  # Standard name for compiled output
            executable_path_in_container = f"/app/{executable_name}" # Assuming Docker context

            try:
                with open(source_file_path, "w", encoding="utf-8") as f:
                    f.write(cpp_code)

                # --- Compilation Step ---
                compilation_result = await docker_manager.compile_cpp(
                    source_file_path=source_file_path,
                    output_name=executable_name, # This will be placed in temp_dir
                    container_workdir="/app" # compile_cpp should handle placing it in temp_dir/output_name
                )

                result["compilation_stdout"] = compilation_result.get("stdout", "")
                result["compilation_stderr"] = compilation_result.get("stderr", "")

                if compilation_result.get("success"):
                    result["compilation_success"] = True
                    # The actual path to the executable on the host for potential inspection,
                    # though it's run inside the container.
                    result["executable_path"] = os.path.join(temp_dir, executable_name)

                    # --- Execution Step (only if compilation succeeded) ---
                    execution_result = await docker_manager.run_cpp(
                        executable_path=executable_name, # Name of the executable within the container's workdir
                        input_data=input_data,
                        timeout_seconds=timeout_seconds,
                        container_workdir=temp_dir # Mount temp_dir to /app in container
                    )

                    result["execution_stdout"] = execution_result.get("stdout", "")
                    result["execution_stderr"] = execution_result.get("stderr", "")
                    result["execution_time"] = execution_result.get("execution_time", 0.0)

                    if execution_result.get("status") == "success":
                        result["execution_success"] = True
                    elif execution_result.get("status") == "timeout":
                        result["execution_success"] = False
                        result["error_type"] = "timeout"
                        result["error_message"] = "Execution timed out."
                    else: # runtime error or other
                        result["execution_success"] = False
                        result["error_type"] = "runtime"
                        result["error_message"] = execution_result.get("stderr", "Runtime error occurred.")
                else:
                    result["compilation_success"] = False
                    result["error_type"] = "compilation"
                    result["error_message"] = compilation_result.get("stderr", "Compilation failed.")
                    # Execution is skipped if compilation fails

            except Exception as e:
                logger.error(f"Error during C++ code execution: {type(e).__name__} - {e!s}")
                result["error_type"] = "system" # Or a more specific type if identifiable
                result["error_message"] = f"An unexpected error occurred: {str(e)}"
                # Ensure execution_success remains False if an error occurs before execution
                if result["error_type"] == "system" and not result["compilation_success"]:
                     result["execution_success"] = False # Should already be false

            finally:
                # TemporaryDirectory is cleaned up automatically upon exiting the 'with' block.
                pass

        return result

    async def evaluate_solution_code(
        self,
        cpp_code: str,
        test_cases: List[Dict[str, str]],
        score_calculator_func: Callable[[str, str], Any],
        docker_manager: DockerManager,
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        """
        Evaluates C++ code against multiple test cases and calculates an overall score.
        """
        total_score: float = 0.0
        per_test_case_results: List[Dict[str, Any]] = []

        # As per current execute_cpp_code, compilation happens on each call.
        # A dedicated "compile_once" step isn't strictly necessary with the current design
        # of execute_cpp_code, as it recompiles each time.
        # However, we can do a preliminary check with the first test case or a dummy one
        # to catch compilation errors early, though it will recompile again.
        # For simplicity and adherence to "call execute_cpp_code for each test case" implied
        # by its current structure, we'll proceed without a separate pre-compilation check here.
        # If test_cases is empty, this loop won't run, and score will be 0.

        if not test_cases:
            return {"overall_score": 0.0, "per_test_case_results": []}

        for i, test_case_spec in enumerate(test_cases):
            test_case_name = test_case_spec.get("name", f"test_{i+1}")
            input_data = test_case_spec["input"] # Assuming "input" key is always present

            current_test_result: Dict[str, Any] = {
                "test_case_name": test_case_name,
                "score": 0.0, # Default score
                "score_calculation_error": None,
            }

            # Execute the code for the current test case
            # This will compile and run the code.
            exec_result = await self.execute_cpp_code(
                cpp_code=cpp_code,
                input_data=input_data,
                docker_manager=docker_manager,
                timeout_seconds=timeout_seconds,
            )

            # Merge execution results into current_test_result
            current_test_result.update(exec_result)

            if exec_result.get("compilation_success") is False:
                # If any test case fails compilation (e.g. if execute_cpp_code changes to allow this per call)
                # or if we decide to make a single upfront compilation check that fails.
                # For now, if compilation fails for one, it likely failed for all (since it's the same code).
                # We can stop early or collect all compilation failures.
                # The prompt implies one initial compilation check. Let's adjust for that.
                #
                # Re-thinking: The prompt wants one initial compilation.
                # If execute_cpp_code recompiles each time, this is tricky.
                # Let's simulate the "initial compilation" by checking the first execution's compile status.
                # If this first one fails compilation, we assume it's a general code issue.
                if i == 0 and not exec_result.get("compilation_success"):
                    # This is the first attempt, and compilation failed.
                    # Populate a single result indicating compilation failure for the whole batch.
                    per_test_case_results.append({
                        "test_case_name": "compilation_check",
                        "compilation_success": False,
                        "compilation_stdout": exec_result.get("compilation_stdout"),
                        "compilation_stderr": exec_result.get("compilation_stderr"),
                        "execution_success": False,
                        "score": 0.0,
                        "error_type": "compilation",
                        "error_message": exec_result.get("error_message", "Compilation failed"),
                    })
                    return {"overall_score": 0.0, "per_test_case_results": per_test_case_results}

            # If compilation for this specific call failed (e.g., if execute_cpp_code could have variable success)
            if not exec_result.get("compilation_success"):
                current_test_result["score"] = 0.0
                # error_type and error_message are already set by execute_cpp_code
                per_test_case_results.append(current_test_result)
                continue # Move to the next test case, this one cannot be scored.

            if exec_result.get("execution_success"):
                try:
                    if asyncio.iscoroutinefunction(score_calculator_func):
                        score_val = await score_calculator_func(input_data, exec_result.get("execution_stdout", ""))
                    else:
                        score_val = score_calculator_func(input_data, exec_result.get("execution_stdout", ""))

                    if isinstance(score_val, tuple) and len(score_val) == 2:
                        current_test_result["score"] = float(score_val[0])
                        if score_val[1]: # Error message present
                            current_test_result["score_calculation_error"] = str(score_val[1])
                    else:
                        current_test_result["score"] = float(score_val)

                except Exception as e:
                    logger.error(f"Error during score calculation for {test_case_name}: {type(e).__name__} - {e!s}")
                    current_test_result["score"] = 0.0
                    current_test_result["score_calculation_error"] = f"Exception in score_calculator_func: {str(e)}"

                total_score += current_test_result["score"]
            else:
                # Execution failed (timeout or runtime error)
                current_test_result["score"] = 0.0
                # error_type and error_message are already set by execute_cpp_code

            per_test_case_results.append(current_test_result)

        overall_score = total_score / len(test_cases) if test_cases else 0.0
        return {"overall_score": overall_score, "per_test_case_results": per_test_case_results}
