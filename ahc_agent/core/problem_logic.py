"""
Problem logic module for AHCAgent CLI.

This module provides functionality for handling AtCoder Heuristic Contest problem-specific logic.
"""

import json
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional

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

    async def parse_problem_statement(self, problem_text: str) -> Dict[str, Any]:
        """
        Parse a problem statement to extract key information.

        Args:
            problem_text: Problem statement text

        Returns:
            Dictionary with parsed problem information
        """
        logger.info("Parsing problem statement")

        # Extract problem components using regex patterns
        title = self._extract_title(problem_text)
        time_limit = self._extract_time_limit(problem_text)
        memory_limit = self._extract_memory_limit(problem_text)

        # Extract input/output format using LLM
        io_format = await self._extract_io_format(problem_text)

        # Extract constraints using LLM
        constraints = await self._extract_constraints(problem_text)

        # Extract scoring rules using LLM
        scoring_rules = await self._extract_scoring_rules(problem_text)

        # Combine results
        parsed_info = {
            "title": title,
            "time_limit": time_limit,
            "memory_limit": memory_limit,
            "input_format": io_format.get("input_format", {}),
            "output_format": io_format.get("output_format", {}),
            "constraints": constraints,
            "scoring_rules": scoring_rules,
            "raw_text": problem_text,
        }

        logger.info("Problem statement parsing complete")
        logger.debug(f"Parsed info: {parsed_info}")

        return parsed_info

    def _extract_title(self, problem_text: str) -> str:
        """
        Extract problem title from problem text.

        Args:
            problem_text: Problem statement text

        Returns:
            Problem title
        """
        # Try to find title using regex
        title_match = re.search(r"^#\s+(.+?)$", problem_text, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()

        # Fallback: use first line
        lines = problem_text.strip().split("\n")
        if lines:
            return lines[0].strip()

        return "Unknown Problem"

    def _extract_time_limit(self, problem_text: str) -> Optional[str]:
        """
        Extract time limit from problem text.

        Args:
            problem_text: Problem statement text

        Returns:
            Time limit or None if not found
        """
        # Try to find time limit using regex
        time_limit_match = re.search(r"[Tt]ime [Ll]imit:?\s*(\d+(?:\.\d+)?)\s*(?:sec|seconds)", problem_text)
        if time_limit_match:
            return time_limit_match.group(1)

        return None

    def _extract_memory_limit(self, problem_text: str) -> Optional[str]:
        """
        Extract memory limit from problem text.

        Args:
            problem_text: Problem statement text

        Returns:
            Memory limit or None if not found
        """
        # Try to find memory limit using regex
        memory_limit_match = re.search(r"[Mm]emory [Ll]imit:?\s*(\d+)\s*(?:MB)", problem_text)
        if memory_limit_match:
            return memory_limit_match.group(1)

        return None

    async def analyze_problem(self, problem_text: str) -> Dict[str, Any]:
        """
        Analyze problem text to extract key information using LLM.
        """
        prompt = f"""
            Extract I/O format, constraints, and scoring rules from the problem text.
            Problem: {problem_text}

            Output JSON format:
            {{
              "io_format": {{ "input": "<description>", "output": "<description>" }},
              "constraints": {{ "<name>": "<description>" }},
              "scoring": {{ "objective": "<minimize|maximize>", "formula": "<description>" }}
            }}

            Return only the JSON object.
            """
        try:
            return await self.llm_client.generate_json(prompt)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error analyzing problem: {type(e).__name__} - {e!s}")
            return {}

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
