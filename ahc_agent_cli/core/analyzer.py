"""
Problem analyzer module for AHCAgent CLI.

This module provides functionality for analyzing AtCoder Heuristic Contest problems.
"""

import logging
from typing import Any, Dict, Optional

from ahc_agent_cli.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class ProblemAnalyzer:
    """
    Analyzer for AtCoder Heuristic Contest problems.
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the problem analyzer.

        Args:
            llm_client: LLM client for problem analysis
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}

        logger.info("Initialized problem analyzer")

    async def analyze(self, problem_text: str) -> Dict[str, Any]:
        """
        Analyze a problem statement.

        Args:
            problem_text: Problem statement text

        Returns:
            Dictionary with structured problem analysis
        """
        logger.info("Analyzing problem statement")

        # Extract problem components
        components = await self._extract_problem_components(problem_text)

        # Extract constraints
        constraints = await self._extract_constraints(problem_text, components)

        # Extract input format
        input_format = await self._extract_input_format(problem_text, components)

        # Extract output format
        output_format = await self._extract_output_format(problem_text, components)

        # Extract scoring rules
        scoring_rules = await self._extract_scoring_rules(problem_text, components)

        # Extract problem characteristics
        characteristics = await self._extract_characteristics(problem_text, components)

        # Combine results
        analysis = {
            "title": components.get("title", "Unknown"),
            "description": components.get("description", ""),
            "constraints": constraints,
            "input_format": input_format,
            "output_format": output_format,
            "scoring_rules": scoring_rules,
            "characteristics": characteristics,
            "raw_text": problem_text,
        }

        logger.info("Problem analysis complete")
        logger.debug(f"Analysis result: {analysis}")

        return analysis

    async def _extract_problem_components(self, problem_text: str) -> Dict[str, Any]:
        """
        Extract basic problem components.

        Args:
            problem_text: Problem statement text

        Returns:
            Dictionary with problem components
        """
        prompt = f"""
        You are an expert at analyzing AtCoder Heuristic Contest problems.

        Please analyze the following problem statement and extract the key components:

        ```
        {problem_text}
        ```

        Extract and return the following components in JSON format:
        - title: The title or name of the problem
        - description: A brief description of the problem
        - constraints: The constraints of the problem
        - input_format: The input format description
        - output_format: The output format description
        - scoring: The scoring rules

        Return only the JSON object without any explanations.
        """

        try:
            return await self.llm_client.generate_json(prompt)

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error extracting problem components: {e!s}")
            # Return minimal structure if extraction fails
            return {
                "title": "Unknown",
                "description": problem_text[:200] + "...",
                "constraints": "",
                "input_format": "",
                "output_format": "",
                "scoring": "",
            }

    async def _extract_constraints(self, problem_text: str, components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract problem constraints.

        Args:
            problem_text: Problem statement text
            components: Problem components

        Returns:
            Dictionary with constraints
        """
        prompt = f"""
        You are an expert at analyzing AtCoder Heuristic Contest problems.

        Please analyze the following problem constraints and extract them in a structured format:

        ```
        {components.get("constraints", "")}
        ```

        Full problem text for reference:
        ```
        {problem_text}
        ```

        Extract all variables and their constraints in JSON format. For each variable, include:
        - min: Minimum value
        - max: Maximum value
        - description: Description of the variable

        For example:
        ```json
        {{
            "N": {{"min": 100, "max": 100, "description": "Number of employees"}},
            "L": {{"min": 500000, "max": 500000, "description": "Number of weeks"}},
            "T_i": {{"min": 0, "max": 10000, "description": "Target cleaning duties"}}
        }}
        ```

        Return only the JSON object without any explanations.
        """

        try:
            return await self.llm_client.generate_json(prompt)

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error extracting constraints: {e!s}")
            return {"time_limit_seconds": 0, "memory_limit_mb": 0, "variables": []}

    async def _extract_input_format(self, problem_text: str, components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract input format.

        Args:
            problem_text: Problem statement text
            components: Problem components

        Returns:
            Dictionary with input format
        """
        prompt = f"""
        You are an expert at analyzing AtCoder Heuristic Contest problems.

        Please analyze the following input format description and extract it in a structured format:

        ```
        {components.get("input_format", "")}
        ```

        Full problem text for reference:
        ```
        {problem_text}
        ```

        Extract the input format in JSON format with the following structure:
        ```json
        {{
            "line_count": 2,
            "lines": [
                {{"line_number": 1, "content": ["N", "L"], "types": ["integer", "integer"]}},
                {{"line_number": 2, "content": ["T_i"], "types": ["integer"], "count": "N"}}
            ]
        }}
        ```

        Return only the JSON object without any explanations.
        """

        try:
            return await self.llm_client.generate_json(prompt)

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error extracting input format: {e!s}")
            return {"line_count": 0, "lines": []}

    async def _extract_output_format(self, problem_text: str, components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract output format.

        Args:
            problem_text: Problem statement text
            components: Problem components

        Returns:
            Dictionary with output format
        """
        prompt = f"""
        You are an expert at analyzing AtCoder Heuristic Contest problems.

        Please analyze the following output format description and extract it in a structured format:

        ```
        {components.get("output_format", "")}
        ```

        Full problem text for reference:
        ```
        {problem_text}
        ```

        Extract the output format in JSON format with the following structure:
        ```json
        {{
            "line_count": "N",
            "lines": [
                {{"line_number": "1 to N", "content": ["a_i", "b_i"], "types": ["integer", "integer"]}}
            ]
        }}
        ```

        Return only the JSON object without any explanations.
        """

        try:
            return await self.llm_client.generate_json(prompt)

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error extracting output format: {e!s}")
            return {"line_count": 0, "lines": []}

    async def _extract_scoring_rules(self, problem_text: str, components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract scoring rules.

        Args:
            problem_text: Problem statement text
            components: Problem components

        Returns:
            Dictionary with scoring rules
        """
        prompt = f"""
        You are an expert at analyzing AtCoder Heuristic Contest problems.

        Please analyze the following scoring rules and extract them in a structured format:

        ```
        {components.get("scoring", "")}
        ```

        Full problem text for reference:
        ```
        {problem_text}
        ```

        Extract the scoring rules in JSON format with the following structure:
        ```json
        {{
            "objective": "minimize",
            "formula": "E = |t_0 - T_0| + |t_1 - T_1| + ... + |t_{{N - 1}} - T_{{N - 1}}|", # noqa: F821
            "score": "10^6 - E",
            "max_score": 1000000
        }}
        ```

        Return only the JSON object without any explanations.
        """

        try:
            return await self.llm_client.generate_json(prompt)

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error extracting scoring rules: {e!s}")
            return {"objective": "unknown", "formula": "", "score": ""}

    async def _extract_characteristics(self, problem_text: str, components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract problem characteristics.

        Args:
            problem_text: Problem statement text
            components: Problem components

        Returns:
            Dictionary with problem characteristics
        """
        prompt = f"""
        You are an expert at analyzing AtCoder Heuristic Contest problems.

        Please analyze the following problem and extract its characteristics:

        Title: {components.get("title", "Unknown")}
        Description: {components.get("description", "")}
        Constraints: {components.get("constraints", "")}
        Scoring: {components.get("scoring", "")}

        Full problem text for reference:
        ```
        {problem_text}
        ```

        Extract the problem characteristics in JSON format with the following structure:
        ```json
        {{
            "problem_type": "assignment",
            "effective_algorithms": ["simulated annealing", "genetic algorithm"],
            "time_complexity": "O(L) for simulation",
            "memory_complexity": "O(N) for state representation",
            "optimization_targets": ["minimize absolute difference between actual and target duties"],
            "potential_pitfalls": ["getting stuck in local optima", "uneven distribution"]
        }}
        ```

        Return only the JSON object without any explanations.
        """

        try:
            return await self.llm_client.generate_json(prompt)

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"Error extracting problem characteristics: {e!s}")
            return {"problem_type": "unknown", "effective_algorithms": []}
