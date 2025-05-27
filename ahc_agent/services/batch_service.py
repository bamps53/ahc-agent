import asyncio
import json
import logging
import os
from pathlib import Path
import time
from typing import Optional

import yaml

from ahc_agent.config import Config

# Core module imports
from ahc_agent.core.analyzer import ProblemAnalyzer
from ahc_agent.core.debugger import ImplementationDebugger
from ahc_agent.core.engine import EvolutionaryEngine
from ahc_agent.core.knowledge import KnowledgeBase
from ahc_agent.core.problem_logic import ProblemLogic
from ahc_agent.core.strategist import SolutionStrategist
from ahc_agent.utils.docker_manager import DockerManager

# Util imports
from ahc_agent.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class BatchService:
    def __init__(self, llm_client: LLMClient, docker_manager: DockerManager, config: Config):
        self.llm_client = llm_client
        self.docker_manager = docker_manager
        self.config = config

    @staticmethod
    def _format_duration(seconds) -> str:
        if seconds is None:  # Check for None explicitly
            return "Unknown"
        try:
            s = int(seconds)
        except (TypeError, ValueError):
            return "Invalid duration"

        if s == 0:
            return "0s"

        minutes, s = divmod(s, 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {s}s"
        if minutes > 0:
            return f"{minutes}m {s}s"
        return f"{s}s"

    @staticmethod
    def _set_nested_dict(d: dict, keys: list[str], value) -> dict:
        """
        Set a value in a nested dictionary.
        """
        if not keys:
            # This case should ideally not be reached if keys list is properly managed.
            # If it is, it means we're trying to set a value at the current dict level without a key.
            # This could be an error or imply replacing the dict, which is not typical for this function.
            # For robustness, one might raise an error or handle as per specific needs.
            # Given the original context, it's assumed keys will not be empty.
            return d

        key = keys[0]
        if len(keys) == 1:
            d[key] = value
            return d

        if key not in d or not isinstance(d[key], dict):
            d[key] = {}

        # Recursively call _set_nested_dict for the next level
        # No need to use BatchService._set_nested_dict explicitly here,
        # as it's a static method being called from within the class context.
        # However, for clarity or if called from outside where `_set_nested_dict` might be ambiguous,
        # `BatchService._set_nested_dict` could be used.
        # Python's scoping rules will find the static method directly.
        BatchService._set_nested_dict(d[key], keys[1:], value)
        return d

    async def _evaluate_solution_for_experiment(
        self,
        code: str,
        current_test_cases: list,
        current_score_calculator: callable,
        implementation_debugger: ImplementationDebugger,
    ) -> tuple[float, dict]:
        total_score = 0.0  # Initialize as float for avg_score calculation
        details = {}

        if not current_test_cases:
            logger.warning("No test cases provided for evaluation in experiment.")
            return 0.0, {"warning": "No test cases provided"}

        for i, test_case in enumerate(current_test_cases):
            # Ensure test_case is a dict and has 'input'
            if not isinstance(test_case, dict) or "input" not in test_case:
                logger.error(f"Invalid test case format or missing 'input' for test case {i + 1}.")
                details[test_case.get("name", f"test_{i + 1}")] = {
                    "error": "Invalid test case format or missing 'input'",
                    "score": 0,
                }
                continue

            result = await implementation_debugger.compile_and_test(code, test_case["input"])
            test_name = test_case.get("name", f"test_{i + 1}")

            if result["success"]:
                # current_score_calculator could be async or sync
                if asyncio.iscoroutinefunction(current_score_calculator):
                    score = await current_score_calculator(test_case["input"], result["execution_output"])
                else:
                    score = current_score_calculator(test_case["input"], result["execution_output"])
                total_score += score
                details[test_name] = {"score": score, "execution_time": result["execution_time"]}
            else:
                details[test_name] = {
                    "error": result["compilation_errors"] or result["execution_errors"],
                    "score": 0,  # Explicitly set score to 0 on error
                }

        avg_score = total_score / len(current_test_cases) if current_test_cases else 0.0
        return avg_score, details

    async def _run_single_experiment_service(
        self, experiment_id: str, problem_config: dict, parameter_set_config: dict, experiment_dir_path: Path
    ) -> dict:
        logger.info(f"Starting experiment: {experiment_id} in {experiment_dir_path}")

        exp_config_dict = self.config.export()  # Get a dictionary copy of the main config

        # Override with parameter_set_config
        for key, value in parameter_set_config.items():
            if key != "name":  # 'name' is metadata for the parameter set itself
                # Use the static method _set_nested_dict for clarity and correctness
                BatchService._set_nested_dict(exp_config_dict, key.split("."), value)

        # Determine if LLMClient/DockerManager need to be re-instantiated based on exp_config_dict
        # For now, assume self.llm_client and self.docker_manager are used,
        # and core modules take their respective sub-configurations from exp_config_dict.
        # If LLM/Docker settings (e.g. API keys, image names) can change per experiment,
        # new instances would be needed:
        # llm_client_for_exp = LLMClient(exp_config_dict.get("llm", {}))
        # docker_manager_for_exp = DockerManager(exp_config_dict.get("docker", {}))
        # For this implementation, we'll use the service-level clients and pass sub-configs.

        problem_text_path = Path(problem_config["path"])
        if not problem_text_path.is_file():  # More specific check than exists()
            logger.error(f"Problem file {problem_text_path} not found or is not a file for experiment {experiment_id}.")
            return {
                "experiment_id": experiment_id,
                "problem_name": problem_config.get("name"),
                "parameter_set_name": parameter_set_config.get("name"),
                "best_score": float("-inf"),
                "generations": 0,
                "duration": 0,
                "session_id": None,
                "error": f"Problem file not found: {problem_text_path}",
            }

        problem_id_from_path = problem_text_path.stem
        # KnowledgeBase per experiment, in its own directory
        knowledge_base = KnowledgeBase(str(experiment_dir_path), problem_id=problem_id_from_path)

        # Instantiate core modules with potentially experiment-specific configurations
        problem_analyzer = ProblemAnalyzer(self.llm_client, exp_config_dict.get("analyzer", {}))
        solution_strategist = SolutionStrategist(self.llm_client, exp_config_dict.get("strategist", {}))
        evolutionary_engine = EvolutionaryEngine(self.llm_client, exp_config_dict.get("evolution", {}))
        implementation_debugger = ImplementationDebugger(self.llm_client, self.docker_manager, exp_config_dict.get("debugger", {}))
        problem_logic = ProblemLogic(self.llm_client, exp_config_dict.get("problem_logic", {}))

        try:
            with open(problem_text_path, encoding="utf-8") as f:
                problem_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read problem file {problem_text_path} for experiment {experiment_id}: {e}")
            return {
                "experiment_id": experiment_id,
                "problem_name": problem_config.get("name"),
                "parameter_set_name": parameter_set_config.get("name"),
                "best_score": float("-inf"),
                "generations": 0,
                "duration": 0,
                "session_id": None,
                "error": f"Failed to read problem file: {e}",
            }

        session_id = knowledge_base.create_session(
            problem_config.get("name", problem_id_from_path),  # Use problem_id_from_path as fallback name
            {"problem_text": problem_text, "experiment_id": experiment_id, "problem_id": problem_id_from_path},
        )

        problem_analysis = await problem_analyzer.analyze(problem_text)
        knowledge_base.save_problem_analysis(session_id, problem_analysis)

        solution_strategy = await solution_strategist.develop_strategy(problem_analysis)
        knowledge_base.save_solution_strategy(session_id, solution_strategy)

        initial_solution = await problem_logic.generate_initial_solution(problem_analysis)

        num_test_cases = exp_config_dict.get("problem_logic", {}).get("test_cases_count", 3)
        test_cases = await problem_logic.generate_test_cases(problem_analysis, num_test_cases)
        score_calculator = await problem_logic.create_score_calculator(problem_analysis)

        start_time = time.time()

        session_workspace_dir = experiment_dir_path / "sessions" / session_id
        os.makedirs(session_workspace_dir, exist_ok=True)

        def eval_func_for_engine(code):
            return self._evaluate_solution_for_experiment(code, test_cases, score_calculator, implementation_debugger)

        result = await evolutionary_engine.evolve(
            problem_analysis, solution_strategy, initial_solution, eval_func_for_engine, str(session_workspace_dir)
        )
        duration = time.time() - start_time

        experiment_result_data = {
            "experiment_id": experiment_id,
            "problem_name": problem_config.get("name"),
            "parameter_set_name": parameter_set_config.get("name"),
            "best_score": result.get("best_score", float("-inf")),  # Ensure default for safety
            "generations": result.get("generations_completed", 0),
            "duration": duration,
            "session_id": session_id,
            "error": None,  # Explicitly set error to None on success
        }

        result_file_path = experiment_dir_path / "result.json"
        try:
            with open(result_file_path, "w", encoding="utf-8") as f:
                json.dump(experiment_result_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write result JSON for experiment {experiment_id} to {result_file_path}: {e}")
            # Optionally, update experiment_result_data to indicate this error
            experiment_result_data["error"] = f"Failed to write result.json: {e}"

        result_save_status = result_file_path if "error" not in experiment_result_data else "Error saving result file"
        logger.info(f"Experiment {experiment_id} completed. Results saved to {result_save_status}")
        return experiment_result_data

    async def run_batch_experiments_service(
        self, batch_config_path: str, output_dir_override: Optional[str] = None, parallel_override: Optional[int] = None
    ) -> list[dict]:
        logger.info(f"Running batch experiments from config: {batch_config_path}")
        try:
            with open(batch_config_path, encoding="utf-8") as f:
                batch_cfg = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load batch configuration from {batch_config_path}: {e}")
            return [{"error": f"Failed to load batch configuration: {e}"}]  # Return list with error dict

        # Determine output directory
        if output_dir_override:
            output_dir = Path(output_dir_override).resolve()
        else:
            default_output_dir_str = self.config.get("batch.output_dir", str(Path.home() / "ahc_batch_results"))
            output_dir_str = batch_cfg.get("common", {}).get("output_dir", default_output_dir_str)
            output_dir = Path(output_dir_str).expanduser().resolve()

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create batch output directory {output_dir}: {e}")
            return [{"error": f"Failed to create batch output directory: {e}"}]

        logger.info(f"Batch output directory: {output_dir}")

        # Determine parallelism
        if parallel_override is not None:
            parallel = parallel_override
        else:
            parallel = batch_cfg.get("common", {}).get("parallel", self.config.get("batch.parallel", 1))

        if not isinstance(parallel, int) or parallel <= 0:
            logger.warning(f"Invalid 'parallel' value ({parallel}), defaulting to 1.")
            parallel = 1
        logger.info(f"Parallel executions: {parallel}")

        problems_map = {p["name"]: p for p in batch_cfg.get("problems", [])}
        param_sets_map = {p["name"]: p for p in batch_cfg.get("parameter_sets", [])}
        experiments_config = batch_cfg.get("experiments", [])

        if not experiments_config:
            logger.warning("No experiments found in batch configuration.")
            return []

        tasks = []
        for exp_conf in experiments_config:
            problem_name = exp_conf.get("problem")
            param_set_name = exp_conf.get("parameter_set")
            repeats = exp_conf.get("repeats", 1)

            if problem_name not in problems_map:
                logger.error(f"Problem '{problem_name}' not found in problems configuration. Skipping experiment setup for: {exp_conf}")
                continue
            if param_set_name not in param_sets_map:
                logger.error(f"Parameter set '{param_set_name}' not found in parameter_sets configuration. Skipping experiment setup for: {exp_conf}")
                continue

            problem_details = problems_map[problem_name]
            param_set_details = param_sets_map[param_set_name]

            for i in range(repeats):
                exp_id = f"{problem_name}_{param_set_name}_repeat{i + 1}"
                exp_dir = output_dir / exp_id
                try:
                    os.makedirs(exp_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"Failed to create experiment directory {exp_dir} for {exp_id}: {e}. Skipping this experiment.")
                    continue

                tasks.append(self._run_single_experiment_service(exp_id, problem_details, param_set_details, exp_dir))

        if not tasks:
            logger.warning("No valid experiments to run after processing configuration.")
            return []

        all_results = []
        semaphore = asyncio.Semaphore(parallel)

        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        semaphore_tasks = [run_with_semaphore(task) for task in tasks]

        logger.info(f"Starting {len(semaphore_tasks)} experiments with concurrency limit {parallel}...")

        for i, future in enumerate(asyncio.as_completed(semaphore_tasks), 1):
            try:
                result = await future
                all_results.append(result)
            except Exception as e:  # Catch errors from _run_single_experiment_service if they weren't caught internally
                logger.error(f"An experiment task failed with an unhandled exception: {e}")
                # Potentially log which task by adding more context if possible, or add a placeholder result
                all_results.append({"error": f"Unhandled exception in experiment: {e}", "experiment_id": "unknown"})
            last_completed = all_results[-1].get("experiment_id", "unknown") if all_results else "N/A"
            logger.info(f"Completed {i}/{len(semaphore_tasks)} experiments. Last completed: {last_completed}")

        logger.info("All experiments completed.")
        logger.info("\n=== Batch Experiment Results Summary ===")
        for res in all_results:
            if res.get("error"):
                logger.warning(f"  Experiment: {res.get('experiment_id', 'unknown_id')}, Error: {res.get('error')}")
            else:
                exp_id = res.get("experiment_id")
                problem = res.get("problem_name")
                params = res.get("parameter_set_name")
                score = res.get("best_score")
                duration = self._format_duration(res.get("duration"))
                logger.info(f"  Experiment: {exp_id}, Problem: {problem}, Params: {params}, Score: {score}, Duration: {duration}")

        summary_path = output_dir / "summary.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Batch summary written to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to write batch summary JSON to {summary_path}: {e}")

        return all_results
