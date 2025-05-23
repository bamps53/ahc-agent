"""
Evolutionary engine module for AHCAgent CLI.

This module provides functionality for evolutionary algorithm-based solution optimization.
"""

import os
import json
import random
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Callable

from ..utils.llm import LLMClient
from ..utils.file_io import read_file, write_file, ensure_directory

logger = logging.getLogger(__name__)

class EvolutionaryEngine:
    """
    Engine for evolutionary algorithm-based solution optimization.
    """
    
    def __init__(self, llm_client: LLMClient, config: Dict[str, Any] = None):
        """
        Initialize the evolutionary engine.
        
        Args:
            llm_client: LLM client for code generation and mutation
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        # Get evolution parameters from config or use defaults
        self.max_generations = self.config.get("max_generations", 30)
        self.population_size = self.config.get("population_size", 10)
        self.time_limit_seconds = self.config.get("time_limit_seconds", 1800)
        self.score_plateau_generations = self.config.get("score_plateau_generations", 5)
        
        # Initialize population and history
        self.population = []
        self.history = []
        self.best_solution = None
        self.best_score = float('-inf')
        self.generation = 0
        
        logger.info("Initialized evolutionary engine")
        logger.debug(f"Evolution parameters: max_generations={self.max_generations}, population_size={self.population_size}, time_limit_seconds={self.time_limit_seconds}")
    
    async def evolve(self, 
                    problem_analysis: Dict[str, Any], 
                    solution_strategy: Dict[str, Any], 
                    initial_solution: Optional[str] = None,
                    evaluate_solution: Callable[[str], Tuple[float, Dict[str, Any]]] = None,
                    workspace_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the evolutionary process.
        
        Args:
            problem_analysis: Problem analysis from ProblemAnalyzer
            solution_strategy: Solution strategy from SolutionStrategist
            initial_solution: Initial solution code (optional)
            evaluate_solution: Function to evaluate a solution (code -> (score, details))
            workspace_dir: Directory for storing solutions and logs
            
        Returns:
            Dictionary with evolution results
        """
        logger.info("Starting evolutionary process")
        
        # Ensure workspace directory exists
        if workspace_dir:
            workspace_dir = ensure_directory(workspace_dir)
        else:
            workspace_dir = ensure_directory(os.path.join(os.getcwd(), "ahc_workspace"))
        
        # Initialize evolution log
        evolution_log = {
            "start_time": time.time(),
            "generations": [],
            "best_solution": None,
            "best_score": float('-inf'),
            "parameters": {
                "max_generations": self.max_generations,
                "population_size": self.population_size,
                "time_limit_seconds": self.time_limit_seconds,
                "score_plateau_generations": self.score_plateau_generations
            }
        }
        
        try:
            # Initialize population
            await self._initialize_population(problem_analysis, solution_strategy, initial_solution, workspace_dir)
            
            # Evaluate initial population
            await self._evaluate_population(evaluate_solution)
            
            # Log initial generation
            self._log_generation(evolution_log)
            
            # Main evolution loop
            start_time = time.time()
            plateau_count = 0
            prev_best_score = self.best_score
            
            for generation in range(1, self.max_generations + 1):
                self.generation = generation
                
                # Check time limit
                if time.time() - start_time > self.time_limit_seconds:
                    logger.info(f"Time limit reached after {generation} generations")
                    break
                
                # Create next generation
                await self._create_next_generation(problem_analysis, solution_strategy, workspace_dir)
                
                # Evaluate new population
                await self._evaluate_population(evaluate_solution)
                
                # Log generation
                self._log_generation(evolution_log)
                
                # Check for improvement
                if self.best_score > prev_best_score:
                    plateau_count = 0
                    prev_best_score = self.best_score
                else:
                    plateau_count += 1
                
                # Check for plateau
                if plateau_count >= self.score_plateau_generations:
                    logger.info(f"Score plateau reached after {generation} generations")
                    break
                
                logger.info(f"Generation {generation}: Best score = {self.best_score}")
            
            # Finalize evolution log
            evolution_log["end_time"] = time.time()
            evolution_log["duration"] = evolution_log["end_time"] - evolution_log["start_time"]
            evolution_log["generations_completed"] = self.generation
            evolution_log["best_solution"] = self.best_solution
            evolution_log["best_score"] = self.best_score
            
            # Save evolution log
            log_path = os.path.join(workspace_dir, "evolution_log.json")
            with open(log_path, "w") as f:
                json.dump(evolution_log, f, indent=2)
            
            logger.info(f"Evolutionary process completed: {self.generation} generations, best score = {self.best_score}")
            logger.info(f"Evolution log saved to {log_path}")
            
            return {
                "best_solution": self.best_solution,
                "best_score": self.best_score,
                "generations_completed": self.generation,
                "evolution_log": evolution_log
            }
        
        except Exception as e:
            logger.error(f"Error in evolutionary process: {str(e)}")
            raise
    
    async def _initialize_population(self, 
                                    problem_analysis: Dict[str, Any], 
                                    solution_strategy: Dict[str, Any], 
                                    initial_solution: Optional[str] = None,
                                    workspace_dir: Optional[str] = None) -> None:
        """
        Initialize the population.
        
        Args:
            problem_analysis: Problem analysis from ProblemAnalyzer
            solution_strategy: Solution strategy from SolutionStrategist
            initial_solution: Initial solution code (optional)
            workspace_dir: Directory for storing solutions
        """
        logger.info(f"Initializing population with size {self.population_size}")
        
        self.population = []
        
        # Add initial solution if provided
        if initial_solution:
            self.population.append({
                "code": initial_solution,
                "score": None,
                "evaluation_details": None,
                "generation": 0,
                "parent_ids": [],
                "id": 0
            })
        
        # Generate remaining solutions
        tasks = []
        for i in range(len(self.population), self.population_size):
            tasks.append(self._generate_initial_solution(problem_analysis, solution_strategy, i, workspace_dir))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Add generated solutions to population
        for result in results:
            if result:
                self.population.append(result)
        
        logger.info(f"Population initialized with {len(self.population)} solutions")
    
    async def _generate_initial_solution(self, 
                                        problem_analysis: Dict[str, Any], 
                                        solution_strategy: Dict[str, Any], 
                                        solution_id: int,
                                        workspace_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an initial solution.
        
        Args:
            problem_analysis: Problem analysis from ProblemAnalyzer
            solution_strategy: Solution strategy from SolutionStrategist
            solution_id: Solution ID
            workspace_dir: Directory for storing solutions
            
        Returns:
            Dictionary with solution details
        """
        try:
            # Prepare prompt for initial solution generation
            prompt = f"""
            You are an expert C++ programmer solving an AtCoder Heuristic Contest problem.
            
            Problem Title: {problem_analysis.get('title', 'Unknown')}
            
            Problem Description:
            {problem_analysis.get('description', '')}
            
            Input Format:
            {problem_analysis.get('input_format', {})}
            
            Output Format:
            {problem_analysis.get('output_format', {})}
            
            Constraints:
            {problem_analysis.get('constraints', {})}
            
            Scoring Rules:
            {problem_analysis.get('scoring_rules', {})}
            
            Solution Strategy:
            {solution_strategy.get('high_level_strategy', {})}
            
            Algorithm Selection:
            {solution_strategy.get('algorithm_selection', {})}
            
            Data Structures:
            {solution_strategy.get('data_structures', {})}
            
            Implementation Plan:
            {solution_strategy.get('implementation_plan', {})}
            
            Please generate a complete C++ solution for this problem. The solution should:
            1. Parse the input correctly
            2. Implement the selected algorithm
            3. Generate output in the required format
            4. Be optimized for performance
            5. Include appropriate comments
            
            Generate a solution with some randomness or variation compared to other solutions.
            
            Return only the C++ code without any explanations.
            ```cpp
            // Your solution here
            ```
            """
            
            # Generate solution
            response = await self.llm_client.generate(prompt)
            
            # Extract code from response
            code = self._extract_code(response)
            
            # Save solution to file if workspace_dir is provided
            if workspace_dir:
                solution_dir = os.path.join(workspace_dir, "solutions", f"gen_{self.generation}")
                os.makedirs(solution_dir, exist_ok=True)
                
                solution_path = os.path.join(solution_dir, f"solution_{solution_id}.cpp")
                with open(solution_path, "w") as f:
                    f.write(code)
            
            # Create solution object
            solution = {
                "code": code,
                "score": None,
                "evaluation_details": None,
                "generation": self.generation,
                "parent_ids": [],
                "id": solution_id
            }
            
            return solution
        
        except Exception as e:
            logger.error(f"Error generating initial solution {solution_id}: {str(e)}")
            return None
    
    async def _evaluate_population(self, evaluate_solution: Callable[[str], Tuple[float, Dict[str, Any]]]) -> None:
        """
        Evaluate the population.
        
        Args:
            evaluate_solution: Function to evaluate a solution (code -> (score, details))
        """
        if not evaluate_solution:
            logger.error("No evaluation function provided")
            return
        
        logger.info(f"Evaluating population with {len(self.population)} solutions")
        
        # Evaluate each solution
        for i, solution in enumerate(self.population):
            if solution["score"] is None:  # Only evaluate if not already evaluated
                try:
                    score, details = evaluate_solution(solution["code"])
                    
                    solution["score"] = score
                    solution["evaluation_details"] = details
                    
                    # Update best solution if better
                    if score > self.best_score:
                        self.best_solution = solution["code"]
                        self.best_score = score
                        logger.info(f"New best solution found: score = {score}")
                
                except Exception as e:
                    logger.error(f"Error evaluating solution {i}: {str(e)}")
                    solution["score"] = float('-inf')
                    solution["evaluation_details"] = {"error": str(e)}
        
        # Sort population by score (descending)
        self.population.sort(key=lambda x: x["score"] if x["score"] is not None else float('-inf'), reverse=True)
        
        logger.info(f"Population evaluated: best score = {self.best_score}")
    
    async def _create_next_generation(self, 
                                     problem_analysis: Dict[str, Any], 
                                     solution_strategy: Dict[str, Any],
                                     workspace_dir: Optional[str] = None) -> None:
        """
        Create the next generation.
        
        Args:
            problem_analysis: Problem analysis from ProblemAnalyzer
            solution_strategy: Solution strategy from SolutionStrategist
            workspace_dir: Directory for storing solutions
        """
        logger.info(f"Creating generation {self.generation}")
        
        # Add current population to history
        self.history.extend(self.population)
        
        # Select parents for next generation
        parents = self._select_parents()
        
        # Create new population
        new_population = []
        
        # Elitism: Keep the best solution
        if self.population:
            best_solution = self.population[0].copy()
            best_solution["generation"] = self.generation
            new_population.append(best_solution)
        
        # Generate remaining solutions
        tasks = []
        for i in range(len(new_population), self.population_size):
            # Randomly choose between mutation and crossover
            if random.random() < 0.7 and len(parents) >= 2:  # 70% chance of crossover if enough parents
                parent1, parent2 = random.sample(parents, 2)
                tasks.append(self._crossover(parent1, parent2, problem_analysis, solution_strategy, i, workspace_dir))
            else:
                parent = random.choice(parents)
                tasks.append(self._mutate(parent, problem_analysis, solution_strategy, i, workspace_dir))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Add generated solutions to new population
        for result in results:
            if result:
                new_population.append(result)
        
        # Update population
        self.population = new_population
        
        logger.info(f"Generation {self.generation} created with {len(self.population)} solutions")
    
    def _select_parents(self) -> List[Dict[str, Any]]:
        """
        Select parents for the next generation.
        
        Returns:
            List of parent solutions
        """
        # Use tournament selection
        parents = []
        
        # Number of parents to select (half of population size)
        num_parents = max(2, self.population_size // 2)
        
        # Tournament size
        tournament_size = max(2, len(self.population) // 3)
        
        for _ in range(num_parents):
            # Select random candidates for tournament
            candidates = random.sample(self.population, min(tournament_size, len(self.population)))
            
            # Select the best candidate
            best_candidate = max(candidates, key=lambda x: x["score"] if x["score"] is not None else float('-inf'))
            
            parents.append(best_candidate)
        
        return parents
    
    async def _mutate(self, 
                     parent: Dict[str, Any], 
                     problem_analysis: Dict[str, Any], 
                     solution_strategy: Dict[str, Any],
                     solution_id: int,
                     workspace_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Mutate a solution.
        
        Args:
            parent: Parent solution
            problem_analysis: Problem analysis from ProblemAnalyzer
            solution_strategy: Solution strategy from SolutionStrategist
            solution_id: Solution ID
            workspace_dir: Directory for storing solutions
            
        Returns:
            Dictionary with mutated solution details
        """
        try:
            # Prepare prompt for mutation
            prompt = f"""
            You are an expert C++ programmer solving an AtCoder Heuristic Contest problem.
            
            You need to mutate the following solution to create a new variant:
            
            ```cpp
            {parent["code"]}
            ```
            
            Problem Title: {problem_analysis.get('title', 'Unknown')}
            
            Scoring Rules:
            {problem_analysis.get('scoring_rules', {})}
            
            Current Solution Score: {parent["score"]}
            
            Evaluation Details:
            {parent["evaluation_details"]}
            
            Please mutate the solution to improve it. Consider:
            1. Fixing any bugs or issues
            2. Optimizing the algorithm
            3. Adjusting parameters
            4. Trying different heuristics
            5. Introducing randomness in a controlled way
            
            Make meaningful changes that could improve the solution, but don't completely rewrite it.
            
            Return only the mutated C++ code without any explanations.
            ```cpp
            // Your mutated solution here
            ```
            """
            
            # Generate mutated solution
            response = await self.llm_client.generate(prompt)
            
            # Extract code from response
            code = self._extract_code(response)
            
            # Save solution to file if workspace_dir is provided
            if workspace_dir:
                solution_dir = os.path.join(workspace_dir, "solutions", f"gen_{self.generation}")
                os.makedirs(solution_dir, exist_ok=True)
                
                solution_path = os.path.join(solution_dir, f"solution_{solution_id}.cpp")
                with open(solution_path, "w") as f:
                    f.write(code)
            
            # Create solution object
            solution = {
                "code": code,
                "score": None,
                "evaluation_details": None,
                "generation": self.generation,
                "parent_ids": [parent["id"]],
                "id": solution_id,
                "mutation_type": "standard"
            }
            
            return solution
        
        except Exception as e:
            logger.error(f"Error mutating solution {solution_id}: {str(e)}")
            return None
    
    async def _crossover(self, 
                        parent1: Dict[str, Any], 
                        parent2: Dict[str, Any],
                        problem_analysis: Dict[str, Any], 
                        solution_strategy: Dict[str, Any],
                        solution_id: int,
                        workspace_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform crossover between two solutions.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            problem_analysis: Problem analysis from ProblemAnalyzer
            solution_strategy: Solution strategy from SolutionStrategist
            solution_id: Solution ID
            workspace_dir: Directory for storing solutions
            
        Returns:
            Dictionary with crossover solution details
        """
        try:
            # Prepare prompt for crossover
            prompt = f"""
            You are an expert C++ programmer solving an AtCoder Heuristic Contest problem.
            
            You need to perform a crossover between the following two solutions to create a new solution:
            
            Parent 1 (Score: {parent1["score"]}):
            ```cpp
            {parent1["code"]}
            ```
            
            Parent 2 (Score: {parent2["score"]}):
            ```cpp
            {parent2["code"]}
            ```
            
            Problem Title: {problem_analysis.get('title', 'Unknown')}
            
            Scoring Rules:
            {problem_analysis.get('scoring_rules', {})}
            
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
            
            # Generate crossover solution
            response = await self.llm_client.generate(prompt)
            
            # Extract code from response
            code = self._extract_code(response)
            
            # Save solution to file if workspace_dir is provided
            if workspace_dir:
                solution_dir = os.path.join(workspace_dir, "solutions", f"gen_{self.generation}")
                os.makedirs(solution_dir, exist_ok=True)
                
                solution_path = os.path.join(solution_dir, f"solution_{solution_id}.cpp")
                with open(solution_path, "w") as f:
                    f.write(code)
            
            # Create solution object
            solution = {
                "code": code,
                "score": None,
                "evaluation_details": None,
                "generation": self.generation,
                "parent_ids": [parent1["id"], parent2["id"]],
                "id": solution_id,
                "crossover_type": "standard"
            }
            
            return solution
        
        except Exception as e:
            logger.error(f"Error performing crossover for solution {solution_id}: {str(e)}")
            return None
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from LLM response.
        
        Args:
            response: LLM response
            
        Returns:
            Extracted code
        """
        # Try to extract code between ```cpp and ``` markers
        import re
        code_match = re.search(r'```cpp\s*(.*?)\s*```', response, re.DOTALL)
        
        if code_match:
            return code_match.group(1).strip()
        
        # If no markers found, return the entire response
        return response.strip()
    
    def _log_generation(self, evolution_log: Dict[str, Any]) -> None:
        """
        Log the current generation.
        
        Args:
            evolution_log: Evolution log dictionary
        """
        # Create generation log
        generation_log = {
            "generation": self.generation,
            "timestamp": time.time(),
            "population_size": len(self.population),
            "best_score": self.best_score,
            "average_score": sum(s["score"] for s in self.population if s["score"] is not None) / len(self.population) if self.population else 0,
            "scores": [s["score"] for s in self.population],
            "solution_ids": [s["id"] for s in self.population]
        }
        
        # Add to evolution log
        evolution_log["generations"].append(generation_log)
        
        # Update best solution and score
        if self.best_score > evolution_log.get("best_score", float('-inf')):
            evolution_log["best_solution"] = self.best_solution
            evolution_log["best_score"] = self.best_score
