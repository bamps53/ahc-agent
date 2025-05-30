import asyncio
import logging
from pathlib import Path
from typing import Optional

from ahc_agent.config import Config
from ahc_agent.core.analyzer import ProblemAnalyzer
from ahc_agent.core.debugger import ImplementationDebugger
from ahc_agent.core.engine import EvolutionaryEngine
from ahc_agent.core.knowledge import KnowledgeBase
from ahc_agent.core.problem_logic import ProblemLogic
from ahc_agent.core.strategist import SolutionStrategist
from ahc_agent.utils.docker_manager import DockerManager
from ahc_agent.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class SolveService:
    def __init__(
        self,
        llm_client: LLMClient,
        docker_manager: DockerManager,
        config: Config,
        knowledge_base: KnowledgeBase,
    ):
        self.llm_client = llm_client
        self.docker_manager = docker_manager
        self.config = config
        self.knowledge_base = knowledge_base
        # Initialize core modules that don't depend on session-specific data here
        # Or, if their config is static, they can be initialized here.
        # For now, keeping them initialized within methods if they need specific config sections
        # that might be dynamically altered or are closely tied to the method's context.
        self.problem_analyzer = ProblemAnalyzer(self.llm_client, self.config.get("analyzer"))
        self.solution_strategist = SolutionStrategist(self.llm_client, self.config.get("strategist"))
        self.evolutionary_engine = EvolutionaryEngine(self.llm_client, self.config.get("evolution"))
        self.implementation_debugger = ImplementationDebugger(self.llm_client, self.docker_manager, self.config.get("debugger"))
        self.problem_logic = ProblemLogic(self.llm_client, self.config.get("problem_logic"))

    async def _evaluate_solution_wrapper(
        self,
        code_to_evaluate: str,
        test_cases: list,
        score_calculator_func,
        implementation_debugger_instance: ImplementationDebugger,
    ):
        """
        A helper method to evaluate a solution against a set of test cases.
        Dependencies (test_cases, score_calculator_func, implementation_debugger_instance) are passed as arguments.
        """
        total_score = 0
        details = {}

        if not test_cases:
            logger.warning("No test cases provided for evaluation.")
            return 0, {"warning": "No test cases provided"}

        for i, test_case in enumerate(test_cases):
            # Ensure test_case has 'input' and 'name' (optional)
            if "input" not in test_case:
                logger.error(f"Test case {i} is missing 'input' field.")
                details[test_case.get("name", f"test_{i + 1}")] = {"error": "Missing 'input' field", "score": 0}
                continue

            result = await implementation_debugger_instance.compile_and_test(code_to_evaluate, test_case["input"])
            test_name = test_case.get("name", f"test_{i + 1}")

            if result["success"]:
                # score_calculator_func could be async or sync
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

    async def run_solve_session(self, problem_text: str, session_id: Optional[str] = None, interactive: bool = False):
        """
        Solve a problem asynchronously.
        """
        contest_id_from_config = self.config.get("contest_id")
        if not contest_id_from_config:
            base_dir = self.config.get("workspace.base_dir", ".")
            contest_id_from_config = Path(base_dir).name
            logger.warning(
                f"'contest_id' not found in config, using workspace name "
                f"'{contest_id_from_config}' as problem_id for KnowledgeBase (within SolveService)."
            )

        if session_id:
            session = self.knowledge_base.get_session(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return
            logger.info(f"Resuming session {session_id}")
            # Potentially load problem_text from session if not provided for resumed sessions
            if not problem_text and "problem_text" in session.get("metadata", {}):
                problem_text = session["metadata"]["problem_text"]
            elif not problem_text:
                logger.error(f"No problem text provided or found in session {session_id} for resumption.")
                return

        else:
            parsed_info = await self.problem_logic.parse_problem_statement(problem_text)
            session_id = self.knowledge_base.create_session(
                parsed_info.get("title", contest_id_from_config or "unknown"),  # Use contest_id as fallback title
                {"problem_text": problem_text, "parsed_info": parsed_info, "problem_id": contest_id_from_config},
            )
            logger.info(f"Created session {session_id}")

        if interactive:
            # Note: run_interactive_session uses its own instances of core modules if needed,
            # or could be refactored to use self.problem_analyzer etc.
            await self.run_interactive_session(session_id)
            return

        logger.info("Analyzing problem...")
        problem_analysis = self.knowledge_base.get_problem_analysis(session_id)
        if not problem_analysis:
            problem_analysis = await self.problem_analyzer.analyze(problem_text)
            self.knowledge_base.save_problem_analysis(session_id, problem_analysis)

        logger.info("Developing solution strategy...")
        solution_strategy = self.knowledge_base.get_solution_strategy(session_id)
        if not solution_strategy:
            solution_strategy = await self.solution_strategist.develop_strategy(problem_analysis)
            self.knowledge_base.save_solution_strategy(session_id, solution_strategy)

        logger.info("Generating initial solution...")
        best_solution_from_kb = self.knowledge_base.get_best_solution(session_id)  # Renamed to avoid conflict
        initial_solution_code = (
            best_solution_from_kb.get("code") if best_solution_from_kb else await self.problem_logic.generate_initial_solution(problem_analysis)
        )

        logger.info("Loading test cases from tools/in/...")
        tools_in_dir = Path(self.config.get("workspace.base_dir")) / "tools" / "in"
        current_test_cases = []  # Renamed
        if tools_in_dir.exists() and tools_in_dir.is_dir():
            for test_file in sorted(tools_in_dir.glob("*.txt")):
                with open(test_file) as f:
                    current_test_cases.append({"name": test_file.name, "input": f.read()})

        if not current_test_cases:
            logger.info("No test cases found in tools/in/. Generating fallback test cases...")
            current_test_cases = await self.problem_logic.generate_test_cases(problem_analysis, 3)
        else:
            logger.info(f"Loaded {len(current_test_cases)} test cases from tools/in/")

        current_score_calculator = await self.problem_logic.create_score_calculator(problem_analysis)  # Renamed

        logger.info("Running evolutionary process...")
        # KnowledgeBaseのセッションディレクトリを使用する
        session_dir = self.knowledge_base.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Use the wrapper for evaluation
        def eval_func_for_engine(code):
            return self._evaluate_solution_wrapper(code, current_test_cases, current_score_calculator, self.implementation_debugger)

        result = await self.evolutionary_engine.evolve(
            problem_analysis,
            solution_strategy,
            initial_solution_code,
            eval_func_for_engine,
            str(session_dir),
        )

        self.knowledge_base.save_evolution_log(session_id, result["evolution_log"])
        self.knowledge_base.save_solution(
            session_id,
            "best",
            {"code": result["best_solution"], "score": result["best_score"], "generation": result["generations_completed"]},
        )

        logger.info(f"Evolution complete: {result['generations_completed']} generations")
        logger.info(f"Best score: {result['best_score']}")
        best_solution_path = session_dir / "solutions" / "best.cpp"
        logger.info(f"Best solution saved to {best_solution_path}")

    async def run_interactive_session(self, session_id: str):
        """
        Interactive problem solving.
        Core modules (problem_analyzer, etc.) are taken from self.
        """
        session = self.knowledge_base.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return

        # problem_text should be part of the session's metadata
        problem_text = session.get("metadata", {}).get("problem_text")
        if not problem_text:
            logger.error(f"Problem text not found in session {session_id}")
            # Attempt to load from problem_id if available
            problem_id = session.get("problem_id") or session.get("metadata", {}).get("problem_id")
            if problem_id:
                # This assumes a way to get problem_text from problem_id, which is not directly available here.
                # For now, we'll just log an error. A more robust solution might involve
                # a method in KnowledgeBase or ProblemLogic to fetch problem_text by ID if stored.
                logger.error(f"Problem text could not be loaded for problem_id: {problem_id}")
            return

        # workspace_dir_str = self.config.get("workspace.base_dir") # Not directly used, paths derived from session_id and config

        running = True
        current_step = "init"
        # These will store the actual data, not the modules themselves
        problem_analysis_data = self.knowledge_base.get_problem_analysis(session_id)
        solution_strategy_data = self.knowledge_base.get_solution_strategy(session_id)
        interactive_test_cases = None  # Loaded or generated on demand
        interactive_score_calculator = None  # Created on demand

        while running:
            if current_step == "init":
                print("\n=== AHCAgent Interactive Mode (Service) ===")
                print(f"Session: {session_id}")
                problem_id_display = session.get("problem_id") or session.get("metadata", {}).get("problem_id", "Unknown")
                print(f"Problem: {problem_id_display}")
                print("\nAvailable commands:")
                print("  analyze - Analyze the problem")
                print("  strategy - Develop solution strategy")
                print("  testcases - Generate/load test cases for interactive use")
                print("  initial - Generate initial solution")
                print("  evolve - Run evolutionary process")
                print("  status - Show current status")
                print("  help - Show this help")
                print("  exit - Exit interactive mode")
                current_step = "command"

            elif current_step == "command":
                command = input("\nEnter command: ").strip().lower()

                if command == "exit":
                    running = False
                elif command == "help":
                    current_step = "init"
                elif command == "status":
                    print("\n=== Current Status ===")
                    print(f"Session: {session_id}")
                    problem_id_display = session.get("problem_id") or session.get("metadata", {}).get("problem_id", "Unknown")
                    print(f"Problem: {problem_id_display}")
                    pa_exists = problem_analysis_data is not None  # Check if data is loaded
                    print(f"Problem Analysis: {'Loaded/Complete' if pa_exists else 'Not started/loaded'}")
                    ss_exists = solution_strategy_data is not None
                    print(f"Solution Strategy: {'Loaded/Complete' if ss_exists else 'Not started/loaded'}")
                    print(f"Interactive Test Cases: {'Generated' if interactive_test_cases else 'Not generated'}")
                    best_sol = self.knowledge_base.get_best_solution(session_id)
                    if best_sol:
                        print(f"Best Score in KB: {best_sol.get('score', 'Unknown')}")

                elif command == "analyze":
                    print("\nAnalyzing problem...")
                    if not problem_analysis_data:  # Analyze only if not already loaded
                        problem_analysis_data = await self.problem_analyzer.analyze(problem_text)
                        self.knowledge_base.save_problem_analysis(session_id, problem_analysis_data)
                        print("Problem analysis complete and saved.")
                    else:
                        print("Problem analysis already loaded.")
                    print(f"Title: {problem_analysis_data.get('title', 'Unknown')}")
                    # ... (add more output as needed)

                elif command == "strategy":
                    if not problem_analysis_data:
                        problem_analysis_data = self.knowledge_base.get_problem_analysis(session_id)  # Ensure it's loaded
                    if not problem_analysis_data:
                        print("Please analyze the problem first (run 'analyze')")
                        continue
                    print("\nDeveloping solution strategy...")
                    if not solution_strategy_data:  # Develop only if not already loaded
                        solution_strategy_data = await self.solution_strategist.develop_strategy(problem_analysis_data)
                        self.knowledge_base.save_solution_strategy(session_id, solution_strategy_data)
                        print("Solution strategy development complete and saved.")
                    else:
                        print("Solution strategy already loaded.")
                    # ... (add more output)

                elif command == "testcases":
                    if not problem_analysis_data:
                        problem_analysis_data = self.knowledge_base.get_problem_analysis(session_id)
                    if not problem_analysis_data:
                        print("Please analyze the problem first (run 'analyze')")
                        continue

                    # Option to load from tools/in or generate
                    load_option = input("Load test cases from 'tools/in/'? (y/N, default N will generate): ").lower()
                    if load_option == "y":
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

                    if not interactive_test_cases:  # If not loaded or load failed/skipped
                        num_cases_str = input("Number of test cases to generate [default: 3]: ")
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
                    if not problem_analysis_data:
                        problem_analysis_data = self.knowledge_base.get_problem_analysis(session_id)
                    if not problem_analysis_data:
                        print("Please analyze the problem first (run 'analyze')")
                        continue
                    print("\nGenerating initial solution...")
                    initial_code = await self.problem_logic.generate_initial_solution(problem_analysis_data)
                    self.knowledge_base.save_solution(session_id, "initial", {"code": initial_code, "score": 0, "generation": 0})
                    print("Initial solution generated and saved to Knowledge Base.")
                    if input("Show initial solution? [y/N]: ").lower() == "y":
                        print("\n=== Initial Solution ===")
                        print(initial_code)

                elif command == "evolve":
                    if not problem_analysis_data:
                        problem_analysis_data = self.knowledge_base.get_problem_analysis(session_id)
                    if not solution_strategy_data:
                        solution_strategy_data = self.knowledge_base.get_solution_strategy(session_id)

                    if not problem_analysis_data:
                        print("Please analyze the problem first.")
                        continue
                    if not solution_strategy_data:
                        print("Please develop solution strategy first.")
                        continue
                    if not interactive_test_cases:
                        print("Please generate/load test cases first (run 'testcases').")
                        continue
                    if not interactive_score_calculator:
                        print("Score calculator not available. Please run 'testcases' first.")
                        continue

                    current_best_sol_kb = self.knowledge_base.get_best_solution(session_id)
                    initial_code_for_evolution = ""
                    if current_best_sol_kb and current_best_sol_kb.get("code"):
                        initial_code_for_evolution = current_best_sol_kb["code"]
                        print(f"Using best known solution from KB (score: {current_best_sol_kb.get('score', 'N/A')}) as starting point.")
                    else:
                        initial_sol_from_kb = self.knowledge_base.get_solution(session_id, "initial")
                        if initial_sol_from_kb and initial_sol_from_kb.get("code"):
                            initial_code_for_evolution = initial_sol_from_kb["code"]
                            print("Using saved initial solution from KB as starting point.")
                        else:
                            print("Generating new initial solution for evolution...")
                            initial_code_for_evolution = await self.problem_logic.generate_initial_solution(problem_analysis_data)
                            self.knowledge_base.save_solution(session_id, "initial_for_evolve", {"code": initial_code_for_evolution})  # Save it
                            print("Initial solution generated and saved.")

                    # Use self.evolutionary_engine and configure it
                    temp_engine_config = self.config.get("evolution").copy()  # Get a copy to modify for this run

                    default_gens = temp_engine_config.get("max_generations", 30)
                    max_gens_str = input(f"Maximum generations [default: {default_gens}]: ")
                    temp_engine_config["max_generations"] = int(max_gens_str) if max_gens_str.isdigit() else default_gens

                    default_pop = temp_engine_config.get("population_size", 10)
                    pop_size_str = input(f"Population size [default: {default_pop}]: ")
                    temp_engine_config["population_size"] = int(pop_size_str) if pop_size_str.isdigit() else default_pop

                    default_time_limit = temp_engine_config.get("time_limit_seconds", 1800)
                    time_limit_str = input(f"Time limit (seconds) [default: {default_time_limit}]: ")
                    temp_engine_config["time_limit_seconds"] = int(time_limit_str) if time_limit_str.isdigit() else default_time_limit

                    # Create a temporary engine instance with these settings or update self.evolutionary_engine if its design allows
                    # For simplicity, let's assume self.evolutionary_engine can be reconfigured or its parameters are passed to evolve()
                    # The current EvolutionaryEngine takes these as __init__ params, so we'd ideally pass them to evolve or re-init.
                    # Let's assume its evolve method can take these as overrides, or we update its attributes directly (if safe).
                    # For this refactor, we'll update attributes of self.evolutionary_engine.
                    # This is not ideal if multiple sessions run concurrently with different settings on the same service instance.
                    # A better approach might be to pass evolution params directly to engine.evolve() if the engine supports it.
                    # Or, create a new engine instance for this interactive evolution run.

                    # For now, let's update self.evolutionary_engine directly for simplicity:
                    self.evolutionary_engine.max_generations = temp_engine_config["max_generations"]
                    self.evolutionary_engine.population_size = temp_engine_config["population_size"]
                    self.evolutionary_engine.time_limit_seconds = temp_engine_config["time_limit_seconds"]

                    def eval_func(code):
                        return self._evaluate_solution_wrapper(
                            code, interactive_test_cases, interactive_score_calculator, self.implementation_debugger
                        )

                    print("\nRunning evolutionary process...")
                    # KnowledgeBaseのセッションディレクトリを使用する
                    session_dir_path = self.knowledge_base.get_session_dir(session_id)
                    session_dir_path.mkdir(parents=True, exist_ok=True)

                    evolution_result = await self.evolutionary_engine.evolve(
                        problem_analysis_data,
                        solution_strategy_data,
                        initial_code_for_evolution,
                        eval_func,
                        str(session_dir_path),
                    )
                    self.knowledge_base.save_evolution_log(session_id, evolution_result["evolution_log"])
                    self.knowledge_base.save_solution(
                        session_id,
                        "best",
                        {
                            "code": evolution_result["best_solution"],
                            "score": evolution_result["best_score"],
                            "generation": evolution_result["generations_completed"],
                        },
                    )
                    print(f"\nEvolution complete: {evolution_result['generations_completed']} generations")
                    print(f"Best score: {evolution_result['best_score']}")
                    if input("Show best solution? [y/N]: ").lower() == "y":
                        print("\n=== Best Solution ===")
                        print(evolution_result["best_solution"])
                else:
                    print(f"Unknown command: {command}")
        # End of run_interactive_session
