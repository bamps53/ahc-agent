"""
Problem logic module for AHCAgent CLI.

This module provides functionality for handling AtCoder Heuristic Contest problem-specific logic.
"""

import os
import re
import json
import logging
import random
from typing import Dict, Any, Optional, List, Tuple, Callable

from ..utils.llm import LLMClient
from ..utils.file_io import read_file, write_file, ensure_directory

logger = logging.getLogger(__name__)

class ProblemLogic:
    """
    Logic for handling AtCoder Heuristic Contest problems.
    """
    
    def __init__(self, llm_client: LLMClient, config: Dict[str, Any] = None):
        """
        Initialize the problem logic.
        
        Args:
            llm_client: LLM client for problem-specific logic
            config: Configuration dictionary
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
            "raw_text": problem_text
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
        title_match = re.search(r'^#\s+(.+?)$', problem_text, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()
        
        # Fallback: use first line
        lines = problem_text.strip().split('\n')
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
        time_limit_match = re.search(r'[Tt]ime [Ll]imit:?\s*(\d+(?:\.\d+)?)\s*(?:sec|seconds)', problem_text)
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
        memory_limit_match = re.search(r'[Mm]emory [Ll]imit:?\s*(\d+)\s*(?:MB)', problem_text)
        if memory_limit_match:
            return memory_limit_match.group(1)
        
        return None
    
    async def _extract_io_format(self, problem_text: str) -> Dict[str, Any]:
        """
        Extract input and output format from problem text using LLM.
        
        Args:
            problem_text: Problem statement text
            
        Returns:
            Dictionary with input and output format
        """
        prompt = f"""
        You are an expert at analyzing AtCoder Heuristic Contest problems.
        
        Please analyze the following problem statement and extract the input and output format:
        
        ```
        {problem_text}
        ```
        
        Extract and return the following in JSON format:
        - input_format: Detailed description of the input format, including variable names, types, and constraints
        - output_format: Detailed description of the output format, including variable names, types, and constraints
        
        For each format, include:
        - line_count: Number of lines in the input/output
        - lines: Array of line descriptions, each with:
          - line_number: Line number or range (e.g., "1", "2-N")
          - content: Array of variables on this line
          - types: Array of types for each variable
          - count: Number of such lines (if applicable)
        
        Return only the JSON object without any explanations.
        """
        
        try:
            result = await self.llm_client.generate_json(prompt)
            return result
        
        except Exception as e:
            logger.error(f"Error extracting I/O format: {str(e)}")
            return {
                "input_format": {},
                "output_format": {}
            }
    
    async def _extract_constraints(self, problem_text: str) -> Dict[str, Any]:
        """
        Extract constraints from problem text using LLM.
        
        Args:
            problem_text: Problem statement text
            
        Returns:
            Dictionary with constraints
        """
        prompt = f"""
        You are an expert at analyzing AtCoder Heuristic Contest problems.
        
        Please analyze the following problem statement and extract the constraints:
        
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
            result = await self.llm_client.generate_json(prompt)
            return result
        
        except Exception as e:
            logger.error(f"Error extracting constraints: {str(e)}")
            return {}
    
    async def _extract_scoring_rules(self, problem_text: str) -> Dict[str, Any]:
        """
        Extract scoring rules from problem text using LLM.
        
        Args:
            problem_text: Problem statement text
            
        Returns:
            Dictionary with scoring rules
        """
        prompt = f"""
        You are an expert at analyzing AtCoder Heuristic Contest problems.
        
        Please analyze the following problem statement and extract the scoring rules:
        
        ```
        {problem_text}
        ```
        
        Extract the scoring rules in JSON format with the following structure:
        ```json
        {{
            "objective": "minimize",
            "formula": "E = |t_0 - T_0| + |t_1 - T_1| + ... + |t_{{N-1}} - T_{{N-1}}|",
            "score": "10^6 - E",
            "max_score": 1000000
        }}
        ```
        
        Return only the JSON object without any explanations.
        """
        
        try:
            result = await self.llm_client.generate_json(prompt)
            return result
        
        except Exception as e:
            logger.error(f"Error extracting scoring rules: {str(e)}")
            return {"objective": "unknown", "formula": "", "score": ""}
    
    async def generate_test_cases(self, problem_info: Dict[str, Any], num_cases: int = 5) -> List[Dict[str, Any]]:
        """
        Generate test cases for a problem.
        
        Args:
            problem_info: Problem information from parse_problem_statement
            num_cases: Number of test cases to generate
            
        Returns:
            List of test cases, each with input data
        """
        logger.info(f"Generating {num_cases} test cases")
        
        test_cases = []
        
        for i in range(num_cases):
            logger.info(f"Generating test case {i+1}/{num_cases}")
            
            try:
                # Generate test case using LLM
                test_case = await self._generate_test_case(problem_info, i)
                
                if test_case:
                    test_cases.append(test_case)
            
            except Exception as e:
                logger.error(f"Error generating test case {i+1}: {str(e)}")
        
        logger.info(f"Generated {len(test_cases)} test cases")
        
        return test_cases
    
    async def _generate_test_case(self, problem_info: Dict[str, Any], case_index: int) -> Optional[Dict[str, Any]]:
        """
        Generate a single test case.
        
        Args:
            problem_info: Problem information from parse_problem_statement
            case_index: Test case index
            
        Returns:
            Dictionary with test case data or None if generation failed
        """
        # Prepare prompt for test case generation
        prompt = f"""
        You are an expert at generating test cases for AtCoder Heuristic Contest problems.
        
        Please generate a test case for the following problem:
        
        Title: {problem_info.get('title', 'Unknown')}
        
        Input Format:
        {json.dumps(problem_info.get('input_format', {}), indent=2)}
        
        Constraints:
        {json.dumps(problem_info.get('constraints', {}), indent=2)}
        
        Generate a valid test case that satisfies all constraints. For test case {case_index+1}, make it {"simple and small" if case_index == 0 else "complex and challenging"}.
        
        Return only the test case input data in the exact format required by the problem, without any explanations.
        """
        
        try:
            # Generate test case
            input_data = await self.llm_client.generate(prompt)
            
            # Clean up input data
            input_data = input_data.strip()
            
            # Create test case object
            test_case = {
                "id": f"test_{case_index+1}",
                "input": input_data,
                "description": f"{'Simple' if case_index == 0 else 'Complex'} test case {case_index+1}"
            }
            
            return test_case
        
        except Exception as e:
            logger.error(f"Error generating test case: {str(e)}")
            return None
    
    async def create_score_calculator(self, problem_info: Dict[str, Any]) -> Callable[[str, str], float]:
        """
        Create a score calculator function for a problem.
        
        Args:
            problem_info: Problem information from parse_problem_statement
            
        Returns:
            Function that calculates score for a solution output given an input
        """
        logger.info("Creating score calculator")
        
        # Prepare prompt for score calculator generation
        prompt = f"""
        You are an expert at implementing scoring functions for AtCoder Heuristic Contest problems.
        
        Please implement a Python function to calculate the score for the following problem:
        
        Title: {problem_info.get('title', 'Unknown')}
        
        Input Format:
        {json.dumps(problem_info.get('input_format', {}), indent=2)}
        
        Output Format:
        {json.dumps(problem_info.get('output_format', {}), indent=2)}
        
        Scoring Rules:
        {json.dumps(problem_info.get('scoring_rules', {}), indent=2)}
        
        Implement a Python function with the following signature:
        ```python
        def calculate_score(input_data: str, output_data: str) -> float:
            '''
            Calculate the score for a solution.
            
            Args:
                input_data: Problem input data
                output_data: Solution output data
                
            Returns:
                Score (higher is better)
            '''
            # Your implementation here
            pass
        ```
        
        The function should:
        1. Parse the input and output data
        2. Validate the output format
        3. Calculate the score according to the scoring rules
        4. Return the score as a float (higher is better)
        
        Return only the Python function without any explanations.
        ```python
        def calculate_score(input_data: str, output_data: str) -> float:
            # Your implementation here
        ```
        """
        
        try:
            # Generate score calculator code
            calculator_code = await self.llm_client.generate(prompt)
            
            # Extract function from code
            function_match = re.search(r'def calculate_score\(.*?\).*?:.*?(?=```|$)', calculator_code, re.DOTALL)
            if function_match:
                calculator_code = function_match.group(0)
            
            # Create function object
            namespace = {}
            exec(calculator_code, namespace)
            
            if "calculate_score" in namespace:
                calculator_func = namespace["calculate_score"]
                
                # Wrap function to handle exceptions
                def safe_calculator(input_data: str, output_data: str) -> float:
                    try:
                        return calculator_func(input_data, output_data)
                    except Exception as e:
                        logger.error(f"Error calculating score: {str(e)}")
                        return 0.0
                
                return safe_calculator
            else:
                logger.error("Failed to extract calculate_score function")
                
                # Return default calculator
                return lambda input_data, output_data: 0.0
        
        except Exception as e:
            logger.error(f"Error creating score calculator: {str(e)}")
            
            # Return default calculator
            return lambda input_data, output_data: 0.0
    
    async def generate_initial_solution(self, problem_info: Dict[str, Any]) -> str:
        """
        Generate an initial solution for a problem.
        
        Args:
            problem_info: Problem information from parse_problem_statement
            
        Returns:
            Initial solution code
        """
        logger.info("Generating initial solution")
        
        # Prepare prompt for initial solution generation
        prompt = f"""
        You are an expert C++ programmer solving an AtCoder Heuristic Contest problem.
        
        Please generate an initial solution for the following problem:
        
        Title: {problem_info.get('title', 'Unknown')}
        
        Input Format:
        {json.dumps(problem_info.get('input_format', {}), indent=2)}
        
        Output Format:
        {json.dumps(problem_info.get('output_format', {}), indent=2)}
        
        Constraints:
        {json.dumps(problem_info.get('constraints', {}), indent=2)}
        
        Scoring Rules:
        {json.dumps(problem_info.get('scoring_rules', {}), indent=2)}
        
        Generate a complete C++ solution that:
        1. Correctly parses the input
        2. Implements a simple but valid algorithm
        3. Generates output in the required format
        
        The solution should be a good starting point for further optimization.
        
        Return only the C++ code without any explanations.
        ```cpp
        // Your solution here
        ```
        """
        
        try:
            # Generate initial solution
            solution_code = await self.llm_client.generate(prompt)
            
            # Extract code from response
            code_match = re.search(r'```cpp\s*(.*?)\s*```', solution_code, re.DOTALL)
            if code_match:
                solution_code = code_match.group(1).strip()
            
            return solution_code
        
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            
            # Return basic template
            return self._get_basic_template()
    
    def _get_basic_template(self) -> str:
        """
        Get a basic C++ template.
        
        Returns:
            Basic C++ template code
        """
        return """
        #include <iostream>
        #include <vector>
        #include <string>
        #include <algorithm>
        #include <cmath>
        #include <random>
        #include <chrono>
        
        using namespace std;
        
        int main() {
            // TODO: Parse input
            
            // TODO: Implement algorithm
            
            // TODO: Generate output
            
            return 0;
        }
        """.strip()
    
    async def mutate_solution(self, solution_code: str, problem_info: Dict[str, Any], score: Optional[float] = None, mutation_type: str = "random") -> str:
        """
        Mutate a solution.
        
        Args:
            solution_code: Original solution code
            problem_info: Problem information from parse_problem_statement
            score: Current solution score (optional)
            mutation_type: Type of mutation (random, small, large, focused)
            
        Returns:
            Mutated solution code
        """
        logger.info(f"Mutating solution with mutation type: {mutation_type}")
        
        # Prepare prompt for mutation
        prompt = f"""
        You are an expert C++ programmer solving an AtCoder Heuristic Contest problem.
        
        Please mutate the following solution:
        
        ```cpp
        {solution_code}
        ```
        
        Problem Title: {problem_info.get('title', 'Unknown')}
        
        Scoring Rules:
        {json.dumps(problem_info.get('scoring_rules', {}), indent=2)}
        
        Current Score: {score if score is not None else 'Unknown'}
        
        Mutation Type: {mutation_type}
        
        Please create a mutated version of the solution that:
        1. Maintains the basic structure
        2. Introduces {"small changes" if mutation_type == "small" else "significant changes" if mutation_type == "large" else "focused improvements" if mutation_type == "focused" else "random changes"}
        3. Aims to improve the score
        
        Return only the mutated C++ code without any explanations.
        ```cpp
        // Your mutated solution here
        ```
        """
        
        try:
            # Generate mutated solution
            mutated_code = await self.llm_client.generate(prompt)
            
            # Extract code from response
            code_match = re.search(r'```cpp\s*(.*?)\s*```', mutated_code, re.DOTALL)
            if code_match:
                mutated_code = code_match.group(1).strip()
            
            return mutated_code
        
        except Exception as e:
            logger.error(f"Error mutating solution: {str(e)}")
            return solution_code
    
    async def crossover_solutions(self, solution1_code: str, solution2_code: str, problem_info: Dict[str, Any], score1: Optional[float] = None, score2: Optional[float] = None) -> str:
        """
        Perform crossover between two solutions.
        
        Args:
            solution1_code: First solution code
            solution2_code: Second solution code
            problem_info: Problem information from parse_problem_statement
            score1: First solution score (optional)
            score2: Second solution score (optional)
            
        Returns:
            Crossover solution code
        """
        logger.info("Performing crossover between two solutions")
        
        # Prepare prompt for crossover
        prompt = f"""
        You are an expert C++ programmer solving an AtCoder Heuristic Contest problem.
        
        Please perform a crossover between the following two solutions:
        
        Solution 1 (Score: {score1 if score1 is not None else 'Unknown'}):
        ```cpp
        {solution1_code}
        ```
        
        Solution 2 (Score: {score2 if score2 is not None else 'Unknown'}):
        ```cpp
        {solution2_code}
        ```
        
        Problem Title: {problem_info.get('title', 'Unknown')}
        
        Scoring Rules:
        {json.dumps(problem_info.get('scoring_rules', {}), indent=2)}
        
        Please create a new solution by combining the best parts of both parent solutions.
        The new solution should:
        1. Inherit strengths from both parents
        2. Avoid weaknesses from both parents
        3. Be a coherent and functional solution
        
        Return only the combined C++ code without any explanations.
        ```cpp
        // Your crossover solution here
        ```
        """
        
        try:
            # Generate crossover solution
            crossover_code = await self.llm_client.generate(prompt)
            
            # Extract code from response
            code_match = re.search(r'```cpp\s*(.*?)\s*```', crossover_code, re.DOTALL)
            if code_match:
                crossover_code = code_match.group(1).strip()
            
            return crossover_code
        
        except Exception as e:
            logger.error(f"Error performing crossover: {str(e)}")
            
            # Return the better solution or a random one if scores are unknown
            if score1 is not None and score2 is not None:
                return solution1_code if score1 >= score2 else solution2_code
            else:
                return random.choice([solution1_code, solution2_code])
