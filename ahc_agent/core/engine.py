"""
Evolutionary engine module for AHCAgent.

This module provides functionality for evolutionary algorithm-based solution optimization.
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ahc_agent.utils.file_io import ensure_directory
from ahc_agent.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class EvolutionaryEngine:
    """
    Engine for evolutionary algorithm-based solution optimization.
    """

    def __init__(self, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
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
        self.best_score = float("-inf")
        self.generation = 0

        logger.info("Initialized evolutionary engine")
        logger.debug(
            f"Evolution parameters: max_generations={self.max_generations}, "
            f"population_size={self.population_size}, "
            f"time_limit_seconds={self.time_limit_seconds}"
        )

    async def evolve(
        self,
        problem_analysis: Dict[str, Any],
        solution_strategy: Dict[str, Any],
        initial_solution: Optional[str] = None,
        evaluate_solution: Optional[Callable[[str], Tuple[float, Dict[str, Any]]]] = None,
        workspace_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        workspace_dir = ensure_directory(workspace_dir) if workspace_dir else ensure_directory(os.path.join(os.getcwd(), "workspace"))

        # Initialize evolution log
        evolution_log = {
            "start_time": time.time(),
            "generations": [],
            "best_solution": None,
            "best_score": float("-inf"),
            "parameters": {
                "max_generations": self.max_generations,
                "population_size": self.population_size,
                "time_limit_seconds": self.time_limit_seconds,
                "score_plateau_generations": self.score_plateau_generations,
            },
        }

        try:
            # Initialize population
            self.population = []
            if initial_solution:
                self.population.append(
                    {
                        "code": initial_solution,
                        "score": None,
                        "evaluation_details": None,
                        "generation": 0,
                        "parent_ids": [],
                        "id": 0,
                    }
                )

            # Generate initial solutions if population is smaller than target size
            num_solutions_to_generate = self.population_size - len(self.population)
            if num_solutions_to_generate > 0:
                generated_solutions = await asyncio.gather(
                    *[
                        self._generate_initial_solution(
                            problem_analysis,
                            solution_strategy,
                            i + len(self.population),
                            workspace_dir,
                        )
                        for i in range(num_solutions_to_generate)
                    ]
                )
                self.population.extend([sol for sol in generated_solutions if sol])

            logger.info(f"Initialized population with {len(self.population)} solutions")

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
                "evolution_log": evolution_log,
            }

        except Exception as e:
            logger.error(f"Error in evolutionary process: {e!s}")
            raise

    async def _generate_initial_solution(
        self,
        problem_analysis: Dict[str, Any],
        solution_strategy: Dict[str, Any],
        solution_id: int,
        workspace_dir: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
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

            Problem Title: {problem_analysis.get("title", "Unknown")}

            Problem Description:
            {problem_analysis.get("description", "")}

            Input Format:
            {problem_analysis.get("input_format", {})}

            Output Format:
            {problem_analysis.get("output_format", {})}

            Constraints:
            {problem_analysis.get("constraints", {})}

            Scoring Rules:
            {problem_analysis.get("scoring_rules", {})}

            Solution Strategy:
            {solution_strategy.get("high_level_strategy", {})}

            Algorithm Selection:
            {solution_strategy.get("algorithm_selection", {})}

            Data Structures:
            {solution_strategy.get("data_structures", {})}

            Implementation Plan:
            {solution_strategy.get("implementation_plan", {})}

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
            return {
                "code": code,
                "score": None,
                "evaluation_details": None,
                "generation": 0,
                "parent_ids": [],
                "id": solution_id,
            }

        except (RuntimeError, ValueError) as e:
            logger.error(f"Error generating initial solution {solution_id}: {e!s}")
            return None

    async def _evaluate_population(self, evaluate_solution: Optional[Callable[[str], Tuple[float, Dict[str, Any]]]]) -> None:
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
        for solution in self.population:
            if solution["score"] is None:  # Only evaluate if not already evaluated
                try:
                    # 修正: awaitを追加して非同期関数を正しく呼び出す
                    score, details = await evaluate_solution(solution["code"])

                    solution["score"] = score
                    solution["evaluation_details"] = details

                    # Update best solution if better
                    if score > self.best_score:
                        self.best_solution = solution["code"]
                        self.best_score = score
                        logger.info(f"New best solution found: score = {score}")

                except (RuntimeError, ValueError) as e:  # Catch specific errors from evaluate_solution
                    logger.error(f"Error evaluating solution {solution['id']} in generation {self.generation}: {e!s}")
                    solution["score"] = float("-inf")  # Assign a very low score on error
                    solution["evaluation_details"] = {"error": str(e)}

        # Sort population by score (descending)
        self.population.sort(key=lambda x: x["score"] if x["score"] is not None else float("-inf"), reverse=True)

        logger.info(f"Population evaluated: best score = {self.best_score}")

    async def _create_next_generation(
        self, problem_analysis: Dict[str, Any], solution_strategy: Dict[str, Any], workspace_dir: Optional[str] = None
    ) -> None:
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

        # Select parents
        parents = self._select_parents()

        # Create offspring through mutation and crossover
        offspring = []
        solution_id_counter = max(s["id"] for s in self.population) + 1 if self.population else 0

        # Elitism: carry over best solutions
        num_elites = int(self.population_size * self.config.get("elitism_ratio", 0.1))
        elites = sorted(self.population, key=lambda s: s.get("score", float("-inf")), reverse=True)[:num_elites]
        offspring.extend(elites)

        mutation_tasks = []
        crossover_tasks = []

        while len(offspring) < self.population_size:
            if random.random() < self.config.get("crossover_probability", 0.7) and len(parents) >= 2:  # noqa: S311
                parent1, parent2 = random.sample(parents, 2)
                crossover_tasks.append(
                    self._crossover(
                        parent1,
                        parent2,
                        problem_analysis,
                        solution_id_counter,
                        workspace_dir,
                    )
                )
                solution_id_counter += 1
            elif parents:  # Ensure there's at least one parent for mutation
                parent = random.choice(parents)  # noqa: S311
                mutation_tasks.append(self._mutate(parent, problem_analysis, solution_id_counter, workspace_dir))
                solution_id_counter += 1
            else:  # Fallback if no parents (e.g. very small initial population failed to evaluate)
                logger.warning("No parents available for crossover or mutation, attempting to generate new initial solution.")
                # Attempt to generate a new initial solution as a fallback
                # This assumes _generate_initial_solution can be called here
                new_initial_solution = await self._generate_initial_solution(problem_analysis, solution_strategy, solution_id_counter, workspace_dir)
                if new_initial_solution:
                    offspring.append(new_initial_solution)
                solution_id_counter += 1

        mutated_solutions = await asyncio.gather(*mutation_tasks)
        crossover_solutions = await asyncio.gather(*crossover_tasks)

        offspring.extend([sol for sol in mutated_solutions if sol])
        offspring.extend([sol for sol in crossover_solutions if sol])

        # Trim offspring to population size if it exceeds due to parallel generation
        self.population = offspring[: self.population_size]

    def _select_parents(self) -> List[Dict[str, Any]]:
        """
        Select parents for the next generation.
        """
        # Tournament selection
        selected_parents = []
        tournament_size = self.config.get("tournament_size", 3)
        for _ in range(self.population_size):  # Select enough parents to generate a new population
            tournament_contenders = random.sample(self.population, tournament_size)
            winner = max(tournament_contenders, key=lambda s: s.get("score", float("-inf")))
            selected_parents.append(winner)
        return selected_parents

    async def _mutate(
        self,
        parent: Dict[str, Any],
        problem_analysis: Dict[str, Any],
        solution_id: int,
        workspace_dir: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Mutate a solution.

        Args:
            parent: Parent solution
            problem_analysis: Problem analysis from ProblemAnalyzer
            solution_id: Solution ID
            workspace_dir: Directory for storing solutions

        Returns:
            Dictionary with mutated solution details
        """
        try:
            # Prepare prompt for mutation
            eval_details = parent.get("evaluation_details")
            error_message = ""
            if isinstance(eval_details, dict):
                # Check if this looks like a compilation failure result from DockerManager
                # "original_stderr" implies it came from compile_cpp
                if eval_details.get("success") is False and "original_stderr" in eval_details:
                    error_message = f"Compilation failed. Error messages:\n{eval_details.get('stderr', 'No stderr provided.')}"
                else:
                    error_message = str(eval_details) # Keep current behavior for other types of evaluation details
            elif eval_details is not None:
                error_message = str(eval_details)
            else:
                error_message = "No evaluation details provided."

            prompt = f"""
            You are an expert C++ programmer solving an AtCoder Heuristic Contest problem.

            You need to mutate the following solution to create a new variant:

            ```cpp
            {parent["code"]}
            ```

            Problem Title: {problem_analysis.get("title", "Unknown")}

            Scoring Rules:
            {problem_analysis.get("scoring_rules", {})}

            Current Solution Score: {parent["score"]}

            Evaluation Details:
            {error_message}

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

            if "Compilation failed. Error messages:" in error_message: # Check if we formatted it as a compilation error
                logger.info(f"Attempting mutation for a solution that failed compilation. LLM Prompt:\n{prompt}")

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
            return {
                "code": code,
                "score": None,
                "evaluation_details": None,
                "generation": self.generation,
                "parent_ids": [parent["id"]],
                "id": solution_id,
                "mutation_type": "standard",
            }

        except (RuntimeError, ValueError) as e:
            logger.error(f"Error mutating solution {solution_id} from parent {parent['id']}: {e!s}")
            return None

    async def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        problem_analysis: Dict[str, Any],
        solution_id: int,
        workspace_dir: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Perform crossover between two solutions.

        Args:
            parent1: First parent solution
            parent2: Second parent solution
            problem_analysis: Problem analysis from ProblemAnalyzer
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

            Problem Title: {problem_analysis.get("title", "Unknown")}

            Scoring Rules:
            {problem_analysis.get("scoring_rules", {})}

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
            return {
                "code": code,
                "score": None,
                "evaluation_details": None,
                "generation": self.generation,
                "parent_ids": [parent1["id"], parent2["id"]],
                "id": solution_id,
                "crossover_type": "standard",
            }

        except (RuntimeError, ValueError) as e:
            logger.error(f"Error performing crossover for solution {solution_id}: {e!s}")
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

        code_match = re.search(r"```cpp\s*(.*?)\s*```", response, re.DOTALL)

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
            "average_score": (
                sum(s["score"] for s in self.population if s["score"] is not None) / len(self.population)
                if self.population and len(self.population) > 0 and any(s["score"] is not None for s in self.population)
                else 0
            ),
            "scores": [s["score"] for s in self.population],
            "solution_ids": [s["id"] for s in self.population],
        }

        # Add to evolution log
        evolution_log["generations"].append(generation_log)

        # Update best solution and score
        if self.best_score > evolution_log.get("best_score", float("-inf")):
            evolution_log["best_solution"] = self.best_solution
            evolution_log["best_score"] = self.best_score
