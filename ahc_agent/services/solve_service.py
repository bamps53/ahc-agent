import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Tuple # Added Tuple

import questionary
from questionary import Choice

from ahc_agent.config import Config
from ahc_agent.core.analyzer import ProblemAnalyzer
# ImplementationDebugger removed as it's no longer directly used in _evaluate_solution_wrapper
from ahc_agent.core.engine import EvolutionaryEngine
from ahc_agent.core.problem_logic import ProblemLogic # Already here, good for type hint
from ahc_agent.core.strategist import SolutionStrategist
from ahc_agent.core.workspace_store import WorkspaceStore
from ahc_agent.utils.docker_manager import DockerManager # Already here, good for type hint
from ahc_agent.utils.file_io import ensure_directory
from ahc_agent.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class SolveService:
    def __init__(
        self,
        llm_client: LLMClient,
        docker_manager: DockerManager,
        config: Config,
        workspace_store: WorkspaceStore,
    ):
        self.llm_client = llm_client
        self.docker_manager = docker_manager
        self.config = config
        self.workspace_store = workspace_store

        # ワークスペースディレクトリのパスを取得
        self.workspace_dir = self.config.get("workspace.base_dir", str(Path.cwd()))

        # LLMクライアントにワークスペースディレクトリを渡すための辞書
        self.llm_kwargs = {"workspace_dir": self.workspace_dir}

        # 各コアモジュールにLLMクライアントとワークスペースディレクトリを渡す
        self.problem_analyzer = ProblemAnalyzer(self.llm_client, self.config.get("analyzer"))
        self.solution_strategist = SolutionStrategist(self.llm_client, self.config.get("strategist"))
        self.evolutionary_engine = EvolutionaryEngine(self.llm_client, self.config.get("evolution"))
        # self.implementation_debugger is still a member of the class, just not used in _evaluate_solution_wrapper
        self.implementation_debugger = ImplementationDebugger(self.llm_client, self.docker_manager, self.config.get("debugger"))
        self.problem_logic = ProblemLogic(self.llm_client, self.config.get("problem_logic"))

        # 各コアモジュールのLLMクライアントにワークスペースディレクトリを設定
        self._set_workspace_for_core_modules()

    def _set_workspace_for_core_modules(self):
        """各コアモジュールのLLMクライアントにワークスペースディレクトリを設定する"""
        # 各モジュールのLLMクライアントにワークスペースディレクトリを設定
        modules = [self.problem_analyzer, self.solution_strategist, self.evolutionary_engine, self.implementation_debugger, self.problem_logic]

        for module in modules:
            if hasattr(module, "llm_client") and module.llm_client is not None:
                # LLMクライアントにワークスペースディレクトリを設定
                module.llm_client.set_workspace_dir(self.workspace_dir)

    async def run_analyze_step(self, problem_text: str) -> Dict[str, Any] | None:
        logger.info("Running Analyze Step...")
        if not problem_text:
            logger.error("Problem text not provided for analysis.")
            # In interactive mode, we might rely on problem_text being loaded by a previous step
            # or directly via run_solve. For now, let's assume it's passed if called directly.
            problem_text_from_store = self.workspace_store.load_problem_text()
            if not problem_text_from_store:
                logger.error("Problem text not found in workspace store either.")
                return None
            problem_text = problem_text_from_store
            logger.info("Using problem text from workspace store.")

        problem_analysis_data = await self.problem_analyzer.analyze(problem_text)
        self.workspace_store.save_problem_analysis(problem_analysis_data)
        logger.info("Problem analysis complete and saved.")
        title = problem_analysis_data.get("title", "Unknown")
        logger.info(f"Problem Title: {title}")
        return problem_analysis_data

    async def run_strategy_step(self) -> Dict[str, Any] | None:
        logger.info("Running Strategy Step...")
        problem_analysis_data = self.workspace_store.load_problem_analysis()
        if not problem_analysis_data:
            logger.error("Problem analysis not found. Please run the analyze step first.")
            return None

        solution_strategy_data = await self.solution_strategist.develop_strategy(problem_analysis_data)
        self.workspace_store.save_solution_strategy(solution_strategy_data)
        logger.info("Solution strategy development complete and saved.")
        return solution_strategy_data

    async def run_testcases_step(self, load_from_tools: bool, num_to_generate: int = 3) -> Dict[str, Any] | None:
        logger.info("Running Test Cases Step...")
        problem_analysis_data = self.workspace_store.load_problem_analysis()
        if not problem_analysis_data:
            logger.error("Problem analysis not found. Please run the analyze step first.")
            return None

        test_cases = []
        if load_from_tools:
            tools_in_dir = Path(self.config.get("workspace.base_dir")) / "tools" / "in"
            if tools_in_dir.exists() and tools_in_dir.is_dir():
                for test_file in sorted(tools_in_dir.glob("*.txt")):
                    with open(test_file) as f:
                        test_cases.append({"name": test_file.name, "input": f.read()})
                if test_cases:
                    logger.info(f"Loaded {len(test_cases)} test cases from tools/in/.")
                else:
                    logger.info("No test cases found in tools/in/. Will attempt to generate.")
            else:
                logger.info(f"Directory not found: {tools_in_dir}. Will attempt to generate test cases.")

        if not test_cases:
            logger.info(f"Generating {num_to_generate} test cases...")
            test_cases = await self.problem_logic.generate_test_cases(problem_analysis_data, num_to_generate)
            logger.info(f"Generated {len(test_cases)} test cases.")

        if not test_cases:
            logger.error("Failed to load or generate test cases.")
            return None

        # For now, we don't save test cases or score calculator to WorkspaceStore directly in this step,
        # as they are often specific to a run or might be generated on the fly.
        # They will be returned and used by subsequent steps if needed.
        logger.info("Creating score calculator...")
        score_calculator = await self.problem_logic.create_score_calculator(problem_analysis_data)
        logger.info("Score calculator created.")

        return {"test_cases": test_cases, "score_calculator": score_calculator}

    async def run_initial_solution_step(self) -> str | None:
        logger.info("Running Initial Solution Step...")
        problem_analysis_data = self.workspace_store.load_problem_analysis()
        if not problem_analysis_data:
            logger.error("Problem analysis not found. Please run the analyze step first.")
            return None

        # Note: The original run_solve method also checks for best_solution_from_kb
        # and uses that as initial_solution_code if available.
        # This granular step will focus on generating a new initial solution.
        # The interactive mode or a higher-level orchestrator can decide if to use an existing one.

        initial_code = await self.problem_logic.generate_initial_solution(problem_analysis_data)
        if initial_code:
            self.workspace_store.save_solution("initial", {"code": initial_code, "score": 0, "generation": 0, "details": None}) # Add details: None
            logger.info("Initial solution generated and saved to Knowledge Base.")
            return initial_code

        logger.error("Failed to generate initial solution.")
        return None

    async def run_evolve_step(
        self,
        test_cases: list,
        score_calculator: Any,  # Callable
        max_generations: int,
        population_size: int,
        time_limit_seconds: int,
        initial_code_override: str | None = None,
    ) -> Dict[str, Any] | None:
        logger.info("Running Evolve Step...")

        problem_analysis_data = self.workspace_store.load_problem_analysis()
        if not problem_analysis_data:
            logger.error("Problem analysis not found. Please run the analyze step first.")
            return None

        solution_strategy_data = self.workspace_store.load_solution_strategy()
        if not solution_strategy_data:
            logger.error("Solution strategy not found. Please run the strategy step first.")
            return None

        if not test_cases or not score_calculator:
            logger.error("Test cases or score calculator not provided for evolution. Please run the testcases step first.")
            return None

        initial_code_for_evolution = initial_code_override
        if not initial_code_for_evolution:
            current_best_sol_kb = self.workspace_store.get_best_solution()
            if current_best_sol_kb and current_best_sol_kb.get("code"):
                initial_code_for_evolution = current_best_sol_kb["code"]
                logger.info(f"Using best known solution from KB (score: {current_best_sol_kb.get('score', 'N/A')}) as starting point.")
            else:
                initial_code = self.workspace_store.load_solution_code("initial")
                initial_sol_from_kb_dict = None
                if initial_code:
                    metadata = self.workspace_store.load_solution_metadata("initial") # metadata may contain 'details'
                    initial_sol_from_kb_dict = {"code": initial_code}
                    if metadata: # Ensure metadata is not None before updating
                        initial_sol_from_kb_dict.update(metadata)


                if initial_sol_from_kb_dict and initial_sol_from_kb_dict.get("code"):
                    initial_code_for_evolution = initial_sol_from_kb_dict["code"]
                    logger.info("Using initial solution from knowledge base (loaded code and meta).")
                else:
                    logger.info("Generating new initial solution for evolution as no prior solution was found or provided...")
                    initial_code_for_evolution = await self.problem_logic.generate_initial_solution(problem_analysis_data)
                    if not initial_code_for_evolution:
                        logger.error("Failed to generate an initial solution for evolution.")
                        return None
                    # Add details:None for consistency
                    self.workspace_store.save_solution("initial_for_evolve", {"code": initial_code_for_evolution, "score": 0, "generation": 0, "details": None})
                    logger.info("New initial solution generated and saved for evolution.")

        self.evolutionary_engine.max_generations = max_generations
        self.evolutionary_engine.population_size = population_size
        self.evolutionary_engine.time_limit_seconds = time_limit_seconds

        async def eval_func_for_engine(code):
            # Using a placeholder for timeout_seconds as per instruction
            # problem_logic.default_test_timeout (e.g., 5 seconds)
            # TODO: Get this from config: self.config.get("problem_logic.default_test_timeout", 5)
            return await self._evaluate_solution_wrapper(code, test_cases, score_calculator, timeout_seconds=5)

        logger.info(
            f"Starting evolutionary process with max_generations={max_generations}, "
            f"population_size={population_size}, time_limit_seconds={time_limit_seconds}s."
        )

        workspace_dir = self.workspace_store.get_workspace_dir()
        generations_dir = workspace_dir / "generations"  # Ensure this path is consistent with EvolutionaryEngine expectations
        ensure_directory(generations_dir)

        result = await self.evolutionary_engine.evolve(
            problem_analysis_data,
            solution_strategy_data,
            initial_code_for_evolution,
            eval_func_for_engine,
            str(generations_dir),  # Pass as string if the engine expects a path string
        )

        self.workspace_store.save_evolution_log(result["evolution_log"])
        self.workspace_store.save_solution(
            "best",
            {
                "code": result["best_solution_code"],
                "score": result["best_score"],
                "generation": result["generations_completed"],
                "details": result.get("best_solution_details") # Use .get for safety
            },
        )

        logger.info(f"Evolution complete: {result['generations_completed']} generations completed.")
        logger.info(f"Best score: {result['best_score']}")
        best_solution_path = self.workspace_store.solutions_dir / "best.cpp"  # Assuming C++
        logger.info(f"Best solution saved to {best_solution_path}")

        return result

    async def _evaluate_solution_wrapper(
        self,
        code_to_evaluate: str,
        test_cases: list,
        score_calculator_func: Any, # Callable[[str, str], Any]
        timeout_seconds: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Wrapper to call problem_logic.evaluate_solution_code and format results.
        """
        if not self.docker_manager:
            logger.error("DockerManager is not initialized in SolveService. Cannot evaluate solution.")
            return float("-inf"), {"error": "DockerManager not available", "per_test_case_results": []}

        evaluation_result = await self.problem_logic.evaluate_solution_code(
            cpp_code=code_to_evaluate,
            test_cases=test_cases,
            score_calculator_func=score_calculator_func,
            docker_manager=self.docker_manager,
            timeout_seconds=timeout_seconds
        )

        overall_score = evaluation_result.get("overall_score", 0.0)
        # Store the detailed per_test_case_results directly.
        # Engine's _mutate method will need to parse this structure.
        details = {"per_test_case_results": evaluation_result.get("per_test_case_results", [])}

        # Check for compilation failure from the first test case result (as per evaluate_solution_code logic)
        # The evaluate_solution_code already handles setting a 0 score if compilation fails for the first test case,
        # but here we want to signal a more catastrophic failure to the evolutionary engine.
        per_test_results = details["per_test_case_results"]
        if per_test_results: # Ensure list is not empty
            first_case_result = per_test_results[0]
            if first_case_result.get("test_case_name") == "compilation_check" and \
               first_case_result.get("compilation_success") is False:
                overall_score = float("-inf") # Signal catastrophic failure for compilation error
                details["error"] = "Compilation failed" # Top-level error for quick check
                details["compilation_stderr"] = first_case_result.get("compilation_stderr", "")
            # Also, if overall_score is already very low due to multiple failures, ensure it's -inf
            # This might not be strictly necessary if evaluate_solution_code handles it, but good for emphasis.
            elif overall_score == 0.0 and any(not r.get("compilation_success", True) for r in per_test_results):
                 # If any case had a compile error (though evaluate_solution_code aims for one check)
                 # This scenario is less likely with the current evaluate_solution_code logic
                 # but provides a fallback.
                 comp_failed_result = next((r for r in per_test_results if not r.get("compilation_success", True)), None)
                 if comp_failed_result:
                    overall_score = float("-inf")
                    details["error"] = "Compilation failed during one of the test cases"
                    details["compilation_stderr"] = comp_failed_result.get("compilation_stderr", "")


        return overall_score, details

    async def run_solve(
        self,
        problem_text: str,
        interactive: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        contest_id_from_config = self.config.get("contest_id")
        if not contest_id_from_config:
            base_dir = self.config.get("workspace.base_dir", ".")
            contest_id_from_config = Path(base_dir).name
            logger.warning(
                f"'contest_id' not found in config, using workspace name "
                f"'{contest_id_from_config}' as problem_id for WorkspaceStore (within SolveService)."
            )

        logger.info(f"Starting solve for problem {contest_id_from_config}")

        # インタラクティブモードが指定されている場合は、run_interactive_solveを呼び出す
        if interactive:
            # Pass problem_text to run_interactive_solve if it's available
            await self.run_interactive_solve(problem_text_initial=problem_text)
            return {"interactive": True}

        problem_analysis_data = self.workspace_store.load_problem_analysis()
        if problem_analysis_data:
            logger.info("Loaded existing problem analysis")
        else:
            logger.info("Analyzing problem...")
            problem_analysis_data = await self.problem_analyzer.analyze(problem_text)
            self.workspace_store.save_problem_analysis(problem_analysis_data)
            logger.info("Problem analysis completed")

        solution_strategy = self.workspace_store.load_solution_strategy()
        if not solution_strategy:
            solution_strategy = await self.solution_strategist.develop_strategy(problem_analysis_data)
            self.workspace_store.save_solution_strategy(solution_strategy)

        best_solution_from_kb = self.workspace_store.get_best_solution()
        initial_solution_code = (
            best_solution_from_kb.get("code") if best_solution_from_kb else await self.problem_logic.generate_initial_solution(problem_analysis_data)
        )

        tools_in_dir = Path(self.config.get("workspace.base_dir")) / "tools" / "in"
        current_test_cases = []
        if tools_in_dir.exists() and tools_in_dir.is_dir():
            for test_file in sorted(tools_in_dir.glob("*.txt")):
                with open(test_file) as f:
                    current_test_cases.append({"name": test_file.name, "input": f.read()})

        if not current_test_cases:
            logger.info("No test cases found in tools/in/. Generating fallback test cases...")
            current_test_cases = await self.problem_logic.generate_test_cases(problem_analysis_data, 3)
        else:
            logger.info(f"Loaded {len(current_test_cases)} test cases from tools/in/")

        current_score_calculator = await self.problem_logic.create_score_calculator(problem_analysis_data)

        logger.info("Running evolutionary process...")
        workspace_dir = self.workspace_store.get_workspace_dir()
        workspace_dir.mkdir(parents=True, exist_ok=True)

        async def eval_func_for_engine(code):
            # Using a placeholder for timeout_seconds as per instruction
            # problem_logic.default_test_timeout (e.g., 5 seconds)
            # TODO: Get this from config: self.config.get("problem_logic.default_test_timeout", 5)
            return await self._evaluate_solution_wrapper(code, current_test_cases, current_score_calculator, timeout_seconds=5)

        result = await self.evolutionary_engine.evolve(
            problem_analysis_data,
            solution_strategy,
            initial_solution_code,
            eval_func_for_engine,
            str(workspace_dir),
        )

        self.workspace_store.save_evolution_log(result["evolution_log"])
        self.workspace_store.save_solution(
            "best",
            {
                "code": result["best_solution_code"],
                "score": result["best_score"],
                "generation": result["generations_completed"],
                "details": result.get("best_solution_details") # Use .get for safety
            },
        )

        logger.info(f"Evolution complete: {result['generations_completed']} generations completed.")
        logger.info(f"Best score: {result['best_score']}")
        best_solution_path = self.workspace_store.solutions_dir / "best.cpp"  # Assuming C++
        logger.info(f"Best solution saved to {best_solution_path}")

        return result

    async def run_interactive_solve(self, problem_text_initial: str | None = None) -> None:
        logger.info("Starting interactive solve")

        # Attempt to load existing data, but don't require it to start interactive mode
        problem_analysis_data = self.workspace_store.load_problem_analysis()
        solution_strategy_data = self.workspace_store.load_solution_strategy()

        # These will be populated by the respective steps
        interactive_test_cases = None
        interactive_score_calculator = None
        # This can be populated by initial solution step or by evolve step's internal logic
        current_initial_code: str | None = None

        running = True
        while running:
            print("\n=== AHCAgent Interactive Mode (Service) ===")
            print(f"Problem: {self.workspace_store.problem_id}")

            choices = [
                Choice("Analyze the problem", "analyze"),
                Choice("Develop solution strategy", "strategy"),
                Choice("Generate/load test cases", "testcases"),
                Choice("Generate initial solution", "initial"),
                Choice("Run evolutionary process", "evolve"),
                Choice("Show current status", "status"),
                Choice("Show help", "help"),
                Choice("Exit interactive mode", "exit"),
            ]

            # Dynamically enable/disable choices based on available data
            if not problem_analysis_data:
                for choice_key in ["strategy", "testcases", "initial", "evolve"]:
                    for choice_obj in choices:
                        if choice_obj.value == choice_key:
                            choice_obj.disabled = "Run 'Analyze' first"
            if not solution_strategy_data and "evolve" in [c.value for c in choices if not c.disabled]:
                for choice_obj in choices:
                    if choice_obj.value == "evolve":
                        choice_obj.disabled = "Run 'Strategy' first"
            if not interactive_test_cases and "evolve" in [c.value for c in choices if not c.disabled]:
                for choice_obj in choices:
                    if choice_obj.value == "evolve":
                        choice_obj.disabled = "Run 'Testcases' first"

            command = await questionary.select(
                "Select an operation:",
                choices=choices,
            ).ask_async()

            if command is None:  # Handle potential None return from ask_async if interrupted
                continue

            if command == "exit":
                running = False
            elif command == "help":
                print("\nAvailable commands:")
                # (Help text remains largely the same)
                print("  analyze - Analyze the problem")
                print("  strategy - Develop solution strategy")
                print("  testcases - Generate/load test cases for interactive use")
                print("  initial - Generate initial solution")
                print("  evolve - Run evolutionary process")
                print("  status - Show current status")
                print("  help - Show this help")
                print("  exit - Exit interactive mode")
            elif command == "status":
                print("\n=== Current Status ===")
                print(f"Problem: {self.workspace_store.problem_id}")
                print(f"Problem Analysis: {'Loaded/Complete' if problem_analysis_data else 'Not loaded/generated'}")
                print(f"Solution Strategy: {'Loaded/Complete' if solution_strategy_data else 'Not loaded/generated'}")
                print(f"Interactive Test Cases: {len(interactive_test_cases) if interactive_test_cases else 'Not generated/loaded'}")
                print(f"Score Calculator: {'Created' if interactive_score_calculator else 'Not created'}")
                print(f"Current Initial Code for Evolve: {'Set' if current_initial_code else 'Not set'}")
                best_sol = self.workspace_store.get_best_solution()
                if best_sol:
                    print(f"Best Score in KB: {best_sol.get('score', 'Unknown')}")

            elif command == "analyze":
                problem_text_to_analyze = problem_text_initial  # Use text passed to run_solve if available
                if not problem_text_to_analyze:
                    problem_text_to_analyze = self.workspace_store.load_problem_text()

                if problem_text_to_analyze is None:
                    logger.error("Problem text not found. Please ensure it's in workspace or provide via run_solve.")
                    # Allow user to input problem text manually if not found
                    problem_text_to_analyze = await questionary.text("Problem text not found. Please paste the problem text here:").ask_async()
                    if not problem_text_to_analyze:
                        logger.error("No problem text provided. Cannot analyze.")
                        continue
                    # Save the manually entered problem text
                    self.workspace_store.save_problem_text(problem_text_to_analyze)

                analysis_result = await self.run_analyze_step(problem_text_to_analyze)
                if analysis_result:
                    problem_analysis_data = analysis_result
                else:
                    logger.error("Analysis step failed.")

            elif command == "strategy":
                if not problem_analysis_data:
                    logger.error("Cannot develop strategy without problem analysis. Run 'analyze' first.")
                    continue
                strategy_result = await self.run_strategy_step()
                if strategy_result:
                    solution_strategy_data = strategy_result
                else:
                    logger.error("Strategy step failed.")

            elif command == "testcases":
                if not problem_analysis_data:
                    logger.error("Cannot generate test cases without problem analysis. Run 'analyze' first.")
                    continue

                load_option_str = await questionary.select(
                    "Select test case acquisition method:",
                    choices=[
                        Choice("Load test cases from 'tools/in/'", "load"),
                        Choice("Generate new test cases", "generate"),
                    ],
                ).ask_async()
                load_from_tools = load_option_str == "load"

                num_to_generate = 3
                if not load_from_tools:
                    num_cases_str = await questionary.text("Enter the number of test cases to generate [default: 3]:", default="3").ask_async()
                    num_to_generate = int(num_cases_str) if num_cases_str and num_cases_str.isdigit() else 3

                testcase_result = await self.run_testcases_step(load_from_tools=load_from_tools, num_to_generate=num_to_generate)
                if testcase_result:
                    interactive_test_cases = testcase_result["test_cases"]
                    interactive_score_calculator = testcase_result["score_calculator"]
                else:
                    logger.error("Testcases step failed.")

            elif command == "initial":
                if not problem_analysis_data:
                    logger.error("Cannot generate initial solution without problem analysis. Run 'analyze' first.")
                    continue
                initial_code_result = await self.run_initial_solution_step()
                if initial_code_result:
                    current_initial_code = initial_code_result  # Store for potential use in evolve
                    show_solution = await questionary.confirm("Show initial solution?", default=False).ask_async()
                    if show_solution:
                        print("\n=== Initial Solution ===")
                        print(current_initial_code)
                else:
                    logger.error("Initial solution step failed.")

            elif command == "evolve":
                if not problem_analysis_data or not solution_strategy_data or not interactive_test_cases or not interactive_score_calculator:
                    logger.error(
                        "Missing prerequisites for evolution: problem analysis, strategy, test cases, or score calculator. Please run previous steps."
                    )
                    continue

                # Configuration for evolution (similar to old logic, but uses run_evolve_step)
                temp_engine_config = self.config.get("evolution").copy()
                max_gens = int(
                    await questionary.text(
                        f"Maximum generations [default: {temp_engine_config.get('max_generations', 30)}]:",
                        default=str(temp_engine_config.get("max_generations", 30)),
                    ).ask_async()
                    or temp_engine_config.get("max_generations", 30)
                )
                pop_size = int(
                    await questionary.text(
                        f"Population size [default: {temp_engine_config.get('population_size', 10)}]:",
                        default=str(temp_engine_config.get("population_size", 10)),
                    ).ask_async()
                    or temp_engine_config.get("population_size", 10)
                )
                time_limit = int(
                    await questionary.text(
                        f"Time limit (seconds) [default: {temp_engine_config.get('time_limit_seconds', 1800)}]:",
                        default=str(temp_engine_config.get("time_limit_seconds", 1800)),
                    ).ask_async()
                    or temp_engine_config.get("time_limit_seconds", 1800)
                )

                # Allow overriding the initial code for evolution
                code_to_evolve_from = current_initial_code  # Default to one from 'initial' step

                use_best_kb = await questionary.confirm(
                    "Use best solution from Knowledge Base as starting point (if available)?", default=True
                ).ask_async()
                if use_best_kb:
                    kb_best_sol = self.workspace_store.get_best_solution()
                    if kb_best_sol and kb_best_sol.get("code"):
                        code_to_evolve_from = kb_best_sol["code"]
                        logger.info(f"Overriding with best solution from KB (Score: {kb_best_sol.get('score')}).")

                if not code_to_evolve_from:  # If still no code (e.g. initial step skipped, KB empty)
                    logger.info("No specific initial code set, run_evolve_step will attempt to find/generate one.")

                evolution_result = await self.run_evolve_step(
                    test_cases=interactive_test_cases,
                    score_calculator=interactive_score_calculator,
                    max_generations=max_gens,
                    population_size=pop_size,
                    time_limit_seconds=time_limit,
                    initial_code_override=code_to_evolve_from,
                )

                if evolution_result:
                    # Update current_initial_code to the best from this evolution run for subsequent evolves
                    current_initial_code = evolution_result["best_solution_code"] # Use best_solution_code
                    show_best = await questionary.confirm("Show best solution from this evolution run?", default=False).ask_async()
                    if show_best:
                        print("\n=== Best Solution (Current Evolution Run) ===")
                        print(evolution_result["best_solution_code"]) # Use best_solution_code
                else:
                    logger.error("Evolution step failed.")
            else:
                print(f"Unknown command: {command}")
