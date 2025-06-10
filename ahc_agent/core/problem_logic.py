"""
Problem logic module for AHCAgent.

This module provides functionality for handling
AtCoder Heuristic Contest problem-specific logic.
"""

import importlib.util
import json
import logging
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional

from ahc_agent.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class ProblemLogic:
    """
    Logic for handling AtCoder Heuristic Contest problems.
    """

    def __init__(self, llm_client: LLMClient, workspace_dir: Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the problem logic.

        Args:
            llm_client: LLM client.
            workspace_dir: Workspace directory.
            config: Configuration dictionary (optional).
        """
        self.llm_client = llm_client
        self.workspace_dir = workspace_dir
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
        problem_statement = problem_info.get("problem_statement", "")
        input_format = problem_info.get("input_format", {})
        output_format = problem_info.get("output_format", {})
        constraints = problem_info.get("constraints", {})

        prompt = (
            f"Create a Python function `calculate_score(input_data_str, output_data_str)` "
            f"that calculates the score for an AtCoder Heuristic Contest problem.\n\n"
            f"Problem statement:\n{problem_statement}\n\n"
            f"Input format:\n{json.dumps(input_format, indent=2)}\n\n"
            f"Output format:\n{json.dumps(output_format, indent=2)}\n\n"
            f"Constraints:\n{json.dumps(constraints, indent=2)}\n\n"
            f"Scoring logic:\n{score_formula}\n\n"
            f"Instructions:\n"
            f"1. The function must take two string parameters: input_data_str and output_data_str\n"
            f"2. Parse both strings according to the input/output formats\n"
            f"3. Calculate and return the score as a float\n"
            f"4. Handle any validation of the output format and constraints\n"
            f"5. Return 0.0 for invalid outputs\n\n"
            f"Return your code wrapped in a ```python``` code block.\n"
        )
        score_calculator_file_path = self.workspace_dir / "score_calculator.py"
        try:
            calculator_code = await self.llm_client.generate(prompt)

            # 必ず```python```コードブロックがあることを前提に処理
            # パターン1: ```python\n..コード..\n```
            if "```python" in calculator_code and "```" in calculator_code.split("```python", 1)[1]:
                # ```pythonの後のコードを抽出
                code_part = calculator_code.split("```python", 1)[1]
                # コードの後の```を削除
                calculator_code = code_part.split("```", 1)[0].strip()
            # パターン2: ```\n..コード..\n```
            elif "```" in calculator_code and "```" in calculator_code.split("```", 2)[2]:
                # 最初の```の後のコードを抽出
                code_part = calculator_code.split("```", 2)[1]
                calculator_code = code_part.strip()
            # 先頭の空行やインデントを削除
            calculator_code = calculator_code.strip()

            # Ensure the directory exists
            score_calculator_file_path.parent.mkdir(parents=True, exist_ok=True)
            score_calculator_file_path.write_text(calculator_code)

            spec = importlib.util.spec_from_file_location("score_calculator_module", score_calculator_file_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec for {score_calculator_file_path}")
                return lambda _input_data, _output_data: 0.0

            score_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(score_module)

            if hasattr(score_module, "calculate_score") and callable(score_module.calculate_score):
                calculator_func = score_module.calculate_score

                def safe_calculator(input_data: str, output_data: str) -> float:
                    try:
                        return calculator_func(input_data, output_data)
                    except Exception as e:
                        logger.error(f"Error in loaded score_calculator.py: {type(e).__name__} - {e!s}")
                        return 0.0

                return safe_calculator

            logger.error(f"'calculate_score' function not found or not callable in {score_calculator_file_path}")
            return lambda _input_data, _output_data: 0.0

        except SyntaxError as e:
            logger.error(f"Syntax error in generated score calculator code ({score_calculator_file_path}): {e.filename}:{e.lineno}: {e.msg}")
            # Save the erroneous code for debugging if it hasn't been saved yet
            if not score_calculator_file_path.exists():
                try:
                    score_calculator_file_path.write_text(calculator_code)  # type: ignore
                    logger.info(f"Saved erroneous calculator code to {score_calculator_file_path}")
                except Exception as save_err:
                    logger.error(f"Could not save erroneous calculator code: {save_err}")
            return lambda _input_data, _output_data: 0.0
        except Exception as e:
            logger.error(f"Error creating score calculator with {score_calculator_file_path}: {type(e).__name__} - {e!s}")
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
