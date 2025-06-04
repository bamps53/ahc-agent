import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

import questionary
from questionary import Choice

from ahc_agent.config import Config
from ahc_agent.core.analyzer import ProblemAnalyzer
from ahc_agent.core.debugger import ImplementationDebugger
from ahc_agent.core.engine import EvolutionaryEngine
from ahc_agent.core.problem_logic import ProblemLogic
from ahc_agent.core.strategist import SolutionStrategist
from ahc_agent.core.workspace_store import WorkspaceStore
from ahc_agent.utils.docker_manager import DockerManager
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

    async def _evaluate_solution_wrapper(
        self,
        code_to_evaluate: str,
        test_cases: list,
        score_calculator_func,
        implementation_debugger_instance: ImplementationDebugger,
    ):
        total_score = 0
        details = {}

        if not test_cases:
            logger.warning("No test cases provided for evaluation.")
            return 0, {"warning": "No test cases provided"}

        for i, test_case in enumerate(test_cases):
            if "input" not in test_case:
                logger.error(f"Test case {i} is missing 'input' field.")
                details[test_case.get("name", f"test_{i + 1}")] = {"error": "Missing 'input' field", "score": 0}
                continue

            result = await implementation_debugger_instance.compile_and_test(code_to_evaluate, test_case["input"])
            test_name = test_case.get("name", f"test_{i + 1}")

            if result["success"]:
                if asyncio.iscoroutinefunction(score_calculator_func):
                    current_score = await score_calculator_func(test_case["input"], result["execution_output"])
                else:
                    current_score = score_calculator_func(test_case["input"], result["execution_output"])
                total_score += current_score
                details[test_name] = {"score": current_score, "execution_time": result["execution_time"]}
            else:
                details[test_name] = {
                    "error": result["compilation_errors"] or result["execution_errors"],
                    "score": 0,
                }

        avg_score = total_score / len(test_cases) if test_cases else 0
        return avg_score, details

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
            await self.run_interactive_solve()
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
            return await self._evaluate_solution_wrapper(code, current_test_cases, current_score_calculator, self.implementation_debugger)

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
            {"code": result["best_solution"], "score": result["best_score"], "generation": result["generations_completed"]},
        )

        logger.info(f"Evolution complete: {result['generations_completed']} generations")
        logger.info(f"Best score: {result['best_score']}")
        best_solution_path = workspace_dir / "solutions" / "best.cpp"
        logger.info(f"Best solution saved to {best_solution_path}")

        return {
            "problem_id": self.workspace_store.problem_id,
            "initial_solution": initial_solution_code,
            "best_solution": result["best_solution"],
            "logs": result["evolution_log"],
        }

    async def run_interactive_solve(self) -> None:
        logger.info("Starting interactive solve")

        analysis = self.workspace_store.load_problem_analysis()
        if analysis is None:
            logger.error(f"Problem analysis not found for problem {self.workspace_store.problem_id}")
            return

        strategy = self.workspace_store.load_solution_strategy()
        if strategy is None:
            logger.error(f"Solution strategy not found for problem {self.workspace_store.problem_id}")
            return

        running = True
        problem_analysis_data = analysis
        solution_strategy_data = strategy
        interactive_test_cases = None
        interactive_score_calculator = None

        while running:
            print("\n=== AHCAgent Interactive Mode (Service) ===")
            print(f"Problem: {self.workspace_store.problem_id}")
            # Display cursor-selectable menu
            command = await questionary.select(
                "Select an operation:",
                choices=[
                    Choice("Analyze the problem", "analyze"),
                    Choice("Develop solution strategy", "strategy"),
                    Choice("Generate/load test cases", "testcases"),
                    Choice("Generate initial solution", "initial"),
                    Choice("Run evolutionary process", "evolve"),
                    Choice("Show current status", "status"),
                    Choice("Show help", "help"),
                    Choice("Exit interactive mode", "exit"),
                ],
            ).ask_async()

            if command == "exit":
                running = False
            elif command == "help":
                print("\nAvailable commands:")
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
                print(f"Problem Analysis: {'Loaded/Complete' if problem_analysis_data else 'Not loaded'}")
                print(f"Solution Strategy: {'Loaded/Complete' if solution_strategy_data else 'Not loaded'}")
                print(f"Interactive Test Cases: {'Generated' if interactive_test_cases else 'Not generated'}")
                best_sol = self.workspace_store.get_best_solution()
                if best_sol:
                    print(f"Best Score in KB: {best_sol.get('score', 'Unknown')}")

            elif command == "analyze":
                print("\nAnalyzing problem...")
                problem_text = self.workspace_store.load_problem_text()
                if problem_text is None:
                    print("Problem text not found. Please load it first.")
                    continue
                problem_analysis_data = await self.problem_analyzer.analyze(problem_text)
                self.workspace_store.save_problem_analysis(problem_analysis_data)
                print("Problem analysis complete and saved.")
                print(f"Title: {problem_analysis_data.get('title', 'Unknown')}")

            elif command == "strategy":
                print("\nDeveloping solution strategy...")
                solution_strategy_data = await self.solution_strategist.develop_strategy(problem_analysis_data)
                self.workspace_store.save_solution_strategy(solution_strategy_data)
                print("Solution strategy development complete and saved.")

            elif command == "testcases":
                # Test case loading option with cursor selection
                load_option = await questionary.select(
                    "Select test case acquisition method:",
                    choices=[
                        Choice("Load test cases from 'tools/in/'", "load"),
                        Choice("Generate new test cases", "generate"),
                    ],
                ).ask_async()

                if load_option == "load":
                    tools_in_dir = Path(self.config.get("workspace.base_dir")) / "tools" / "in"
                    if tools_in_dir.exists() and tools_in_dir.is_dir():
                        interactive_test_cases = []
                        for test_file in sorted(tools_in_dir.glob("*.txt")):
                            with open(test_file) as f:
                                interactive_test_cases.append({"name": test_file.name, "input": f.read()})
                        if interactive_test_cases:
                            print(f"Loaded {len(interactive_test_cases)} test cases from tools/in/.")
                        else:
                            print("No test cases found in tools/in/.")
                    else:
                        print(f"Directory not found: {tools_in_dir}")

                if not interactive_test_cases or load_option == "generate":
                    # Number of test cases with cursor selection
                    num_cases_str = await questionary.text("Enter the number of test cases to generate [default: 3]:", default="3").ask_async()
                    num_cases = int(num_cases_str) if num_cases_str.isdigit() else 3
                    print(f"\nGenerating {num_cases} test cases...")
                    interactive_test_cases = await self.problem_logic.generate_test_cases(problem_analysis_data, num_cases)
                    print(f"Generated {len(interactive_test_cases)} test cases.")

                if interactive_test_cases:
                    print("Creating score calculator...")
                    interactive_score_calculator = await self.problem_logic.create_score_calculator(problem_analysis_data)
                    print("Score calculator created.")
                else:
                    print("No test cases available to create score calculator.")

            elif command == "initial":
                print("\nGenerating initial solution...")
                initial_code = await self.problem_logic.generate_initial_solution(problem_analysis_data)
                self.workspace_store.save_solution("initial", {"code": initial_code, "score": 0, "generation": 0})
                print("Initial solution generated and saved to Knowledge Base.")

                # Show initial solution option with cursor selection
                show_solution = await questionary.confirm("Show initial solution?", default=False).ask_async()
                if show_solution:
                    print("\n=== Initial Solution ===")
                    print(initial_code)

            elif command == "evolve":
                current_best_sol_kb = self.workspace_store.get_best_solution()
                initial_code_for_evolution = ""
                if current_best_sol_kb and current_best_sol_kb.get("code"):
                    initial_code_for_evolution = current_best_sol_kb["code"]
                    print(f"Using best known solution from KB (score: {current_best_sol_kb.get('score', 'N/A')}) as starting point.")
                else:
                    initial_sol_from_kb = self.workspace_store.get_solution("initial")
                    if initial_sol_from_kb and initial_sol_from_kb.get("code"):
                        initial_code_for_evolution = initial_sol_from_kb["code"]
                        print("Using saved initial solution from KB as starting point.")
                    else:
                        print("Generating new initial solution for evolution...")
                        initial_code_for_evolution = await self.problem_logic.generate_initial_solution(problem_analysis_data)
                        self.workspace_store.save_solution("initial_for_evolve", {"code": initial_code_for_evolution})
                        print("Initial solution generated and saved.")

                temp_engine_config = self.config.get("evolution").copy()

                # Maximum generations with cursor selection
                default_gens = temp_engine_config.get("max_generations", 30)
                max_gens_str = await questionary.text(f"Maximum generations [default: {default_gens}]:", default=str(default_gens)).ask_async()
                temp_engine_config["max_generations"] = int(max_gens_str) if max_gens_str.isdigit() else default_gens

                # Population size with cursor selection
                default_pop = temp_engine_config.get("population_size", 10)
                pop_size_str = await questionary.text(f"Population size [default: {default_pop}]:", default=str(default_pop)).ask_async()
                temp_engine_config["population_size"] = int(pop_size_str) if pop_size_str.isdigit() else default_pop

                # Time limit with cursor selection
                default_time_limit = temp_engine_config.get("time_limit_seconds", 1800)
                time_limit_str = await questionary.text(
                    f"Time limit (seconds) [default: {default_time_limit}]:", default=str(default_time_limit)
                ).ask_async()
                temp_engine_config["time_limit_seconds"] = int(time_limit_str) if time_limit_str.isdigit() else default_time_limit

                self.evolutionary_engine.max_generations = temp_engine_config["max_generations"]
                self.evolutionary_engine.population_size = temp_engine_config["population_size"]
                self.evolutionary_engine.time_limit_seconds = temp_engine_config["time_limit_seconds"]

                async def eval_func(code):
                    return await self._evaluate_solution_wrapper(
                        code, interactive_test_cases, interactive_score_calculator, self.implementation_debugger
                    )

                print("\nRunning evolutionary process...")
                generations_dir = self.workspace_store.solutions_dir / "generations"
                ensure_directory(generations_dir)

                evolution_result = await self.evolutionary_engine.evolve(
                    problem_analysis_data,
                    solution_strategy_data,
                    initial_code_for_evolution,
                    eval_func,
                    str(generations_dir),
                )
                self.workspace_store.save_evolution_log(evolution_result["evolution_log"])
                self.workspace_store.save_solution(
                    "best",
                    {
                        "code": evolution_result["best_solution"],
                        "score": evolution_result["best_score"],
                        "generation": evolution_result["generations_completed"],
                    },
                )
                print(f"\nEvolution complete: {evolution_result['generations_completed']} generations")
                print(f"Best score: {evolution_result['best_score']}")

                # Show best solution option with cursor selection
                show_best = await questionary.confirm("Show best solution?", default=False).ask_async()
                if show_best:
                    print("\n=== Best Solution ===")
                    print(evolution_result["best_solution"])
            else:
                print(f"Unknown command: {command}")
