"""
CLI entrypoint module for AHCAgent CLI.

This module provides the main CLI entrypoint for the AHCAgent CLI tool.
"""

# Standard library imports
import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Optional

# Third-party imports
import click
import yaml

# Local application/library specific imports
from .config import Config
from .core.analyzer import ProblemAnalyzer
from .core.debugger import ImplementationDebugger
from .core.engine import EvolutionaryEngine
from .core.session_store import SessionManager, SessionStore
from .core.heuristic_knowledge_base import HeuristicKnowledgeBase
from .core.problem_logic import ProblemLogic
from .core.strategist import SolutionStrategist
from .utils.docker_manager import DockerManager
from .utils.llm import LLMClient
from .utils.logging import setup_logging
from .utils.scraper import scrape_and_setup_problem

# Set up logger
logger = logging.getLogger(__name__)


# Create Click group
@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Minimize output")
@click.option("--no-docker", is_flag=True, help="Disable Docker usage")
@click.pass_context
def cli(ctx, config, verbose, quiet, no_docker):
    """
    AHCAgent CLI - A tool for solving AtCoder Heuristic Contest problems.
    """
    # Set up logging
    log_level = "DEBUG" if verbose else "ERROR" if quiet else "INFO"
    setup_logging(level=log_level)

    # Load configuration
    cfg = Config(config)

    # Override configuration with command-line options
    if no_docker:
        cfg.set("docker.enabled", False)

    # Store configuration in context
    ctx.obj = {"config": cfg}

    logger.debug("CLI initialized")


@cli.command()
@click.option("--template", "-t", type=str, help="Template to use")
@click.option("--docker-image", "-i", type=str, help="Docker image to use")
@click.option(
    "--workspace",
    "-w",
    type=click.Path(),
    default=None,
    help=("Directory to create the project in. If not set, creates a directory named CONTEST_ID in the current location."),
)
@click.argument("contest_id", type=str, required=True)
@click.pass_context
def init(ctx, template, docker_image, workspace, contest_id):
    """
    Initialize a new AHC project.
    Optionally, provide a CONTEST_ID (e.g., ahc030) to scrape the problem statement.
    """
    config = ctx.obj["config"]

    # Override configuration with command-line options
    if template:
        config.set("template", template)

    if docker_image:
        config.set("docker.image", docker_image)

    project_dir = Path(workspace).resolve() if workspace else Path.cwd() / contest_id

    try:
        # project_dir がファイルの場合、mkdirは FileExistsError を送出する
        project_dir.mkdir(parents=True, exist_ok=False)  # exist_ok=False に変更
        display_path = project_dir
        with contextlib.suppress(ValueError):
            display_path = project_dir.relative_to(Path.cwd())
        click.echo(f"Initialized AHC project in ./{display_path}")

    except FileExistsError:  # FileExistsError を明示的にキャッチ
        click.echo(
            f"Error creating project directory: '{project_dir}' already exists and is a file or non-empty directory.", err=True
        )
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Error creating project directory '{project_dir}': {e}", err=True)
        ctx.exit(1)

    project_specific_config_data = {
        "contest_id": contest_id,
        "template": template if template else config.get("template", "default"),
        "docker_image": docker_image if docker_image else config.get("docker.image", "ubuntu:latest"),
    }

    project_config_file_path = project_dir / "ahc_config.yaml"
    try:
        with open(project_config_file_path, "w") as f:
            yaml.dump(project_specific_config_data, f, default_flow_style=False)
        click.echo(f"Project configuration saved to {project_config_file_path}")
    except Exception as e:
        click.echo(f"Error saving project configuration to '{project_config_file_path}': {e}", err=True)
        return

    problem_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
    click.echo(f"Attempting to scrape problem from {problem_url}...")
    try:
        scrape_and_setup_problem(problem_url, str(project_dir))
        click.echo(f"Problem scraped and set up successfully in '{project_dir}'.")
    except Exception as e:
        click.echo(f"Error during scraping: {e}", err=True)


@cli.command()
@click.argument("workspace", type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option("--session-id", "-s", type=str, help="Session ID for resuming")
@click.option("--time-limit", "-t", type=int, help="Time limit in seconds")
@click.option("--generations", "-g", type=int, help="Maximum number of generations")
@click.option("--population-size", "-p", type=int, help="Population size")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.pass_context
def solve(ctx, workspace, session_id, time_limit, generations, population_size, interactive):
    """
    Solve a problem in the specified workspace.
    The workspace must contain 'problem.md' and 'ahc_config.yaml'.
    """
    workspace_path = Path(workspace)
    problem_file_path = workspace_path / "problem.md"
    config_file_path = workspace_path / "ahc_config.yaml"

    if not problem_file_path.exists():
        click.echo(f"Error: 'problem.md' not found in workspace '{workspace_path}'.", err=True)
        ctx.exit(1)
    if not config_file_path.exists():
        click.echo(f"Error: 'ahc_config.yaml' not found in workspace '{workspace_path}'.", err=True)
        ctx.exit(1)

    # Load configuration specifically from the workspace's ahc_config.yaml
    # This overrides any globally passed config from the main cli context for this command's scope.
    try:
        config = Config(str(config_file_path))
        # Ensure workspace.base_dir in the loaded config points to the provided workspace
        # This is important if the ahc_config.yaml itself has a different workspace.base_dir
        config.set("workspace.base_dir", str(workspace_path))
    except Exception as e:
        click.echo(f"Error loading config from '{config_file_path}': {e}", err=True)
        ctx.exit(1)

    click.echo(f"Solving problem in workspace: {workspace_path}")
    click.echo(f"Using config: {config.config_file_path}")

    # Override loaded configuration with command-line options
    if time_limit is not None:
        config.set("evolution.time_limit_seconds", time_limit)

    if generations is not None:
        config.set("evolution.max_generations", generations)

    if population_size is not None:
        config.set("evolution.population_size", population_size)

    # Read problem file
    try:
        with open(problem_file_path) as f:
            problem_text = f.read()
    except Exception as e:
        click.echo(f"Error reading problem file '{problem_file_path}': {e}", err=True)
        ctx.exit(1)

    # Run solver
    problem_dir_path = Path(workspace)
    sm_workspace_dir = str(problem_dir_path.parent)
    sm_problem_id = problem_dir_path.name # This is the contest_id

    session_manager = SessionManager(sm_workspace_dir, sm_problem_id)

    if session_id:
        session_store = session_manager.get_session_store(session_id)
        if not session_store:
            click.echo(f"Error: Session {session_id} not found in {problem_dir_path}.", err=True)
            ctx.exit(1)
        click.echo(f"Resuming session {session_id}")
    else:
        # Create new session
        # problem_logic is needed here to parse problem_text first for metadata
        llm_client_temp = LLMClient(config.get("llm")) # Temporary for initial parsing
        problem_logic_temp = ProblemLogic(llm_client_temp, config.get("problem_logic"))
        parsed_info = asyncio.run(problem_logic_temp.parse_problem_statement(problem_text)) # Run async parse

        initial_session_metadata = {
            "problem_title": parsed_info.get("title", "unknown"),
            "parsed_info": parsed_info,
            "status": "initialized",
        }
        session_store = session_manager.create_session(initial_metadata=initial_session_metadata)
        session_id = session_store.session_id # Get the newly created session_id
        # Save problem_text as a generic file in the session dir for interactive mode to potentially access
        session_store.save_generic_file_data("problem_text.md", problem_text)
        click.echo(f"Created new session {session_id}")

    # Initialize HeuristicKnowledgeBases
    problem_kb_path = problem_dir_path / "knowledge" / "kb"
    problem_heuristic_kb = HeuristicKnowledgeBase(str(problem_kb_path))

    global_hkb_path_str = config.get("heuristic_knowledge_base.global_path", os.path.expanduser("~/.ahc_agent_heuristic_knowledge"))
    global_heuristic_kb = HeuristicKnowledgeBase(global_hkb_path_str)
    
    # Pass session_store and HKBs to solver
    asyncio.run(_solve_problem(config, problem_text, session_store, problem_heuristic_kb, global_heuristic_kb, interactive))


async def _solve_problem(
    config: Config,
    problem_text: str,
    session_store: SessionStore,
    problem_heuristic_kb: HeuristicKnowledgeBase,
    global_heuristic_kb: HeuristicKnowledgeBase,
    interactive: bool = False,
):
    """
    Solve a problem asynchronously.
    """
    # Initialize clients and modules
    llm_client = LLMClient(config.get("llm"))
    docker_manager = DockerManager(config.get("docker"))
    
    # ProblemAnalyzer might use problem_heuristic_kb for problem_instance.json etc.
    problem_analyzer = ProblemAnalyzer(llm_client, config.get("analyzer"), problem_heuristic_kb=problem_heuristic_kb)
    # SolutionStrategist uses global_heuristic_kb and potentially problem_heuristic_kb
    solution_strategist = SolutionStrategist(
        llm_client, 
        config.get("strategist"), 
        global_heuristic_kb=global_heuristic_kb,
        problem_heuristic_kb=problem_heuristic_kb
    )
    evolutionary_engine = EvolutionaryEngine(llm_client, config.get("evolution"))
    implementation_debugger = ImplementationDebugger(llm_client, docker_manager, config.get("debugger"))
    # ProblemLogic might use problem_heuristic_kb for problem_instance.json, score calculation logic etc.
    problem_logic = ProblemLogic(llm_client, config.get("problem_logic"), problem_heuristic_kb=problem_heuristic_kb)

    # Interactive mode
    if interactive:
        # Ensure problem_text is available for interactive mode, might load from session_store if not passed directly
        # For _solve_problem, problem_text is passed. For _interactive_solve, it might need to load it.
        await _interactive_solve(
            config, # Pass config
            session_store,
            problem_heuristic_kb,
            global_heuristic_kb,
            problem_analyzer, # Pass initialized components
            solution_strategist,
            evolutionary_engine,
            implementation_debugger,
            problem_logic,
        )
        return

    # Non-interactive mode
    click.echo("Analyzing problem...")
    problem_analysis = session_store.get_problem_analysis()
    if not problem_analysis:
        problem_analysis = await problem_analyzer.analyze(problem_text)
        session_store.save_problem_analysis(problem_analysis)
        session_store.update_session_metadata({"status": "analysis_complete"})

    click.echo("Developing solution strategy...")
    solution_strategy = session_store.get_solution_strategy()
    if not solution_strategy:
        solution_strategy = await solution_strategist.develop_strategy(problem_analysis)
        session_store.save_solution_strategy(solution_strategy)
        session_store.update_session_metadata({"status": "strategy_complete"})

    click.echo("Generating initial solution...")
    # Check for existing solutions first; if not, generate.
    # For non-interactive, we'd usually evolve, so an explicit "initial" named solution might not be the one we pick up.
    # get_best_solution or a similar logic might be used. For now, let's assume we try to get *any* solution.
    existing_solutions = session_store.list_solutions()
    initial_solution_code = None
    if existing_solutions:
        # Simplistic: take the latest one if multiple exist without a clear "best" marker yet
        # Or, if a "best_solution" method exists that doesn't rely on score only (e.g. latest)
        best_solution_artifact = session_store.get_best_solution() # Assumes get_best_solution can return non-scored or latest
        if best_solution_artifact and "code" in best_solution_artifact:
            initial_solution_code = best_solution_artifact["code"]
            click.echo(f"Starting from existing solution: {best_solution_artifact.get('solution_id', 'unknown')}")

    if not initial_solution_code:
        initial_solution_code = await problem_logic.generate_initial_solution(problem_analysis)
        # Save this explicitly generated initial solution
        session_store.save_solution(
            solution_id="initial_generated", 
            solution_data={"code": initial_solution_code, "score": None, "generation": 0, "source": "generation"},
            code=initial_solution_code
        )
        session_store.update_session_metadata({"status": "initial_solution_generated"})
        click.echo("Generated and saved initial solution.")
    
    # Generate test cases from tools/in/*.txt
    click.echo("Loading test cases from tools/in/...")
    tools_in_dir = Path(config.get("workspace.base_dir")) / "tools" / "in"
    test_cases = []
    if tools_in_dir.exists() and tools_in_dir.is_dir():
        for test_file in sorted(tools_in_dir.glob("*.txt")):
            with open(test_file) as f:
                test_cases.append({"name": test_file.name, "input": f.read()})

    if not test_cases:
        click.echo("No test cases found in tools/in/. Generating fallback test cases...")
        # Fallback to generating test cases if none are found in tools/in/
        # Ensure this returns list of dicts with 'input' and 'name'
        test_cases = await problem_logic.generate_test_cases(problem_analysis, 3)
    else:
        click.echo(f"Loaded {len(test_cases)} test cases from tools/in/")

    # Create score calculator
    score_calculator = await problem_logic.create_score_calculator(problem_analysis)

    # Define evaluation function
    def evaluate_solution(code, current_test_cases, current_score_calculator):
        total_score = 0
        details = {}

        for test_case in current_test_cases:  # No need for i
            result = asyncio.run(implementation_debugger.compile_and_test(code, test_case["input"]))
            test_name = test_case.get("name", f"test_{current_test_cases.index(test_case) + 1}")  # Use name if available

            if result["success"]:
                score = current_score_calculator(test_case["input"], result["execution_output"])
                total_score += score
                details[test_name] = {"score": score, "execution_time": result["execution_time"]}
            else:
                details[test_name] = {
                    "error": result["compilation_errors"] or result["execution_errors"],
                    "score": 0,
                }

        avg_score = total_score / len(current_test_cases) if current_test_cases else 0
        return avg_score, details

    click.echo("Loading test cases from tools/in/...")
    # Assuming workspace.base_dir in config is the problem directory
    problem_dir_path_str = config.get("workspace.base_dir")
    tools_in_dir = Path(problem_dir_path_str) / "tools" / "in"
    test_cases = []
    if tools_in_dir.exists() and tools_in_dir.is_dir():
        for test_file in sorted(tools_in_dir.glob("*.txt")):
            with open(test_file) as f:
                test_cases.append({"name": test_file.name, "input": f.read()})

    if not test_cases:
        click.echo("No test cases found in tools/in/. Generating fallback test cases...")
        test_cases = await problem_logic.generate_test_cases(problem_analysis, 3) # problem_analysis should be loaded
    else:
        click.echo(f"Loaded {len(test_cases)} test cases from tools/in/")

    score_calculator = await problem_logic.create_score_calculator(problem_analysis)

    async def evaluate_solution_async_direct(code, current_test_cases, current_score_calculator, debugger_instance):
        # This is the core async evaluation logic
        total_score = 0
        details = {}
        for test_case in current_test_cases:
            result = await debugger_instance.compile_and_test(code, test_case["input"])
            test_name = test_case.get("name", f"test_{current_test_cases.index(test_case) + 1}")
            if result["success"]:
                score = current_score_calculator(test_case["input"], result["execution_output"])
                total_score += score
                details[test_name] = {"score": score, "execution_time": result["execution_time"]}
            else:
                details[test_name] = {
                    "error": result["compilation_errors"] or result["execution_errors"],
                    "score": 0,
                }
        avg_score = total_score / len(current_test_cases) if current_test_cases else 0
        return avg_score, details
    
    # This function will be passed to the possibly synchronous EvolutionaryEngine.
    # It needs to run the async evaluate_solution_async_direct.
    # It should create its own event loop if one isn't running, or use the existing one carefully.
    # To avoid asyncio.run() if a loop is already running (common in tests that call asyncio.run(_solve_problem))
    # we check if a loop is running. If so, we schedule the async task and wait for it.
    # If no loop is running, we use asyncio.run(). This is complex.

    # A simpler, more common pattern if EvolutionaryEngine is purely synchronous:
    # The evaluate_solution function provided to it MUST be synchronous.
    # So, this synchronous function will call asyncio.run() on the async logic.
    # The RuntimeError in tests occurs because the test itself runs _solve_problem using asyncio.run(),
    # leading to nested asyncio.run() calls.
    # The fix is to ensure that if an event loop is already running, we don't create a new one with asyncio.run().
    
    def sync_evaluate_wrapper(code_to_eval):
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_running():
            # If a loop is running, we cannot use asyncio.run().
            # We need to create a task and run it to completion in the current loop.
            # This is tricky because we need to block until this specific task is done.
            # This typically requires the outer function (evolve) to be async.
            # Given the constraints, if 'evolve' is synchronous, this path will be problematic.
            # A common pattern for this is to use a thread to run the async code
            # or use something like `nest_asyncio` if truly necessary (but it's a hack).

            # For now, let's assume that if _solve_problem is called via asyncio.run (like in tests),
            # then the engine's evaluate_solution will be called in a context where a new asyncio.run()
            # will indeed cause nesting.
            # The previous approach of making evaluate_solution_async and passing it directly
            # assumes the engine can await it. If it can't, this wrapper is needed.
            # The RuntimeError implies the engine *is* synchronous and the lambda tries to call asyncio.run
            # while _solve_problem is already under asyncio.run.

            # The change to make evaluate_solution_async and pass it to the engine
            # (assuming the engine awaits it) is the cleaner path.
            # The failure in test_solve_command_uses_tools_in_files suggests that
            # the EvolutionaryEngine.evolve (or its mock) is not awaiting the eval_func.
            # The test's mock_evolve *does* await it. This means the actual
            # evaluate_solution_async was correct, but the lambda was not.
            # Let's stick to the async version and ensure the lambda passes the coroutine.
            # The error must be somewhere else if this is the case.

            # Reverting to the direct async call, assuming the engine will await it.
            # The problem was likely how the lambda was defined or called.
            # The lambda `eval_func_for_engine` returns a coroutine.
            # The test `test_solve_command_uses_tools_in_files` has a mock_evolve that correctly `await`s.
            # The RuntimeError implies that the actual `implementation_debugger.compile_and_test`
            # or something it calls is trying to `asyncio.run` internally when it shouldn't.
            # This shouldn't be the case if `compile_and_test` is `async def`.

            # Let's simplify and assume the engine handles awaiting the coroutine.
            # The lambda should just call the async function, returning the coroutine.
            # The error "RuntimeError: asyncio.run() cannot be called from a running event loop"
            # must be from the `asyncio.run` that was previously inside `evaluate_solution`.
            # Making `evaluate_solution_async` and passing it correctly should have fixed it.
            # The test failure log points to line 386 in cli.py:
            # result = asyncio.run(implementation_debugger.compile_and_test(code, test_case["input"]))
            # This line was inside the *synchronous* evaluate_solution.
            # The fix is to make evaluate_solution async and await that call.
            # The previous diff *did* this by renaming to evaluate_solution_async and awaiting.
            # The issue might be that the lambda itself needs to be async or the engine isn't awaiting.
            # If the engine is synchronous, then evaluate_solution must be synchronous and use asyncio.run().
            # This leads to the conflict if _solve_problem is ALREADY run with asyncio.run().

            # The most direct way to fix the test failure is to ensure the `evaluate_solution`
            # passed to the synchronous `evolve` function does not call `asyncio.run` if a loop is running.
            # This is what `problem_logic.evaluate_solution_code` probably should do.
            # Let's assume `problem_logic.evaluate_solution_code` handles this.
            # The lambda should then call this (potentially async) problem_logic method.

            # Re-evaluating: The error occurs because the `evaluate_solution` *inside* `_solve_problem`
            # (the one defined at line 386 in the error log) calls `asyncio.run`.
            # This function needs to be `async def` and `await` the call.
            # The lambda passed to `evolve` should then be:
            # `lambda code: asyncio.run(evaluate_solution_async_direct(code, test_cases, score_calculator, implementation_debugger))`
            # if `evolve` is synchronous. But this is the source of the problem in tests.
            #
            # If `evolve` can take an async function, then:
            # `eval_func = evaluate_solution_async_direct` (pass the function itself)
            # And `evolve` will do `await eval_func(...)`
            # Or `eval_func = lambda code: evaluate_solution_async_direct(code, ...)` and `evolve` awaits.

            # The issue is the test `test_solve_command_uses_tools_in_files` calls `asyncio.run(_solve_problem)`.
            # `_solve_problem` calls `evolve`. `evolve` calls `evaluate_solution`.
            # If `evaluate_solution` calls `asyncio.run`, we have nested `asyncio.run`.
            # So, `evaluate_solution` should be `async def`.
            # The `evolve` method (or its mock in the test) should `await` it.
            # The test's `mock_evolve` *does* `await eval_func(ic)`.
            # The `eval_func` passed to it is `lambda code: evaluate_solution_async_direct(...)`.
            # So, the lambda returns a coroutine, and `mock_evolve` awaits it. This is correct.
            # The error log points to line 386 of cli.py, which was the `asyncio.run` call
            # *inside the old synchronous evaluate_solution*.
            # This means the file was not correctly patched in the previous step.
            # The fix is to ensure `evaluate_solution_async_direct` is used and awaited.
            # The lambda `eval_func_for_engine` should create the coroutine.
            pass # This complex comment block is to re-affirm the previous fix was likely correct but misapplied.
                 # The core idea is: the function called by `evolve` should be `async` and `await` internally,
                 # and `evolve` itself (or its mock) should `await` that function.

    eval_func_for_engine = lambda code: evaluate_solution_async_direct(
        code, test_cases, score_calculator, implementation_debugger
    )

    click.echo("Running evolutionary process...")
    evolution_artifacts_dir = os.path.join(session_store.session_dir, "evolution_artifacts")
    os.makedirs(evolution_artifacts_dir, exist_ok=True)

    result = await evolutionary_engine.evolve(
        problem_analysis,
        solution_strategy,
        initial_solution_code, # Use the determined initial_solution_code
        eval_func_for_engine, # Pass the new evaluation function wrapper
        evolution_artifacts_dir, # Output dir for engine artifacts
    )

    session_store.save_evolution_log(result["evolution_log"])
    session_store.update_session_metadata({"status": "evolution_complete"})

    best_solution_id = f"best_gen_{result['generations_completed']}"
    session_store.save_solution(
        solution_id=best_solution_id,
        solution_data={"code": result["best_solution"], "score": result["best_score"], "generation": result["generations_completed"]},
        code=result["best_solution"]
    )
    session_store.update_session_metadata({"last_solution_id": best_solution_id, "best_score": result["best_score"]})

    click.echo(f"Evolution complete: {result['generations_completed']} generations")
    click.echo(f"Best score: {result['best_score']}")
    # Best solution path is now managed by SessionStore, e.g. session_store.session_dir/solutions/best_gen_X.cpp
    click.echo(f"Best solution saved in session {session_store.session_id} with ID {best_solution_id}")


async def _interactive_solve(
    config: Config,
    session_store: SessionStore,
    problem_heuristic_kb: HeuristicKnowledgeBase,
    global_heuristic_kb: HeuristicKnowledgeBase,
    problem_analyzer: ProblemAnalyzer, # Pass components directly
    solution_strategist: SolutionStrategist,
    evolutionary_engine: EvolutionaryEngine,
    implementation_debugger: ImplementationDebugger,
    problem_logic: ProblemLogic,
):
    """
    Interactive problem solving.
    """
    session_id = session_store.session_id
    session_metadata = session_store.get_session_metadata()
    if not session_metadata:
        click.echo(f"Could not load metadata for session {session_id}")
        return

    # Try to load problem_text from session store if not directly available
    # (though in current flow, _solve_problem passes it if it calls _interactive_solve)
    problem_text = session_store.get_generic_file_data("problem_text.md")
    if not problem_text:
        # Fallback: try to get from initial metadata if stored there (older sessions might)
        problem_text = session_metadata.get("parsed_info", {}).get("problem_text") # Check if it was stored this way
        if not problem_text:
             click.echo("Problem text not found in session store. Analyze command might fail if text is required again.")
             # problem_text will be None, commands needing it must handle this

    # Interactive loop
    running = True
    current_step = "init"

    # State variables (loaded from session_store or live)
    current_problem_analysis = session_store.get_problem_analysis()
    current_solution_strategy = session_store.get_solution_strategy()
    current_test_cases = None # Test cases are usually generated on demand in interactive
    current_score_calculator = None # Derived from problem_analysis
    evolution_result = None # Stores result of an evolution run

    while running:
        if current_step == "init":
            click.echo("\n=== AHCAgent Interactive Mode ===")
            click.echo(f"Session: {session_id}")
            click.echo(f"Problem ID: {session_store.problem_id}") # From SessionStore
            click.echo("\nAvailable commands:")
            click.echo("  analyze - Analyze the problem")
            click.echo("  strategy - Develop solution strategy")
            click.echo("  testcases - Generate test cases (and score calculator)")
            click.echo("  initial - Generate initial solution")
            click.echo("  evolve - Run evolutionary process")
            click.echo("  status - Show current status")
            click.echo("  list_solutions - List saved solutions")
            click.echo("  help - Show this help")
            click.echo("  exit - Exit interactive mode")
            current_step = "command"

        elif current_step == "command":
            command = click.prompt("\nEnter command", type=str).strip().lower()

            if command == "exit":
                running = False
            elif command == "help":
                current_step = "init"
            elif command == "status":
                click.echo("\n=== Current Status ===")
                session_metadata = session_store.get_session_metadata() # Refresh
                click.echo(f"Session: {session_id}")
                click.echo(f"Problem ID: {session_store.problem_id}")
                click.echo(f"Status: {session_metadata.get('status', 'Unknown')}")
                pa_status = "Complete" if current_problem_analysis or session_store.get_problem_analysis() else "Not started"
                click.echo(f"Problem Analysis: {pa_status}")
                ss_status = "Complete" if current_solution_strategy or session_store.get_solution_strategy() else "Not started"
                click.echo(f"Solution Strategy: {ss_status}")
                click.echo(f"Test Cases: {'Generated' if current_test_cases else 'Not generated'}")
                best_sol = session_store.get_best_solution()
                if best_sol:
                    click.echo(f"Best Score: {best_sol.get('score', 'Unknown')} (ID: {best_sol.get('solution_id', 'N/A')})")

            elif command == "analyze":
                click.echo("\nAnalyzing problem...")
                if not problem_text:
                    click.echo("Error: Problem text is not available to perform analysis.", err=True)
                    continue
                current_problem_analysis = await problem_analyzer.analyze(problem_text)
                session_store.save_problem_analysis(current_problem_analysis)
                session_store.update_session_metadata({"status": "analysis_complete"})
                click.echo("Problem analysis complete and saved.")
                click.echo(f"Title: {current_problem_analysis.get('title', 'Unknown')}")

            elif command == "strategy":
                if not current_problem_analysis:
                    current_problem_analysis = session_store.get_problem_analysis()
                    if not current_problem_analysis:
                        click.echo("Please analyze the problem first ('analyze' command).")
                        continue
                click.echo("\nDeveloping solution strategy...")
                current_solution_strategy = await solution_strategist.develop_strategy(current_problem_analysis)
                session_store.save_solution_strategy(current_solution_strategy)
                session_store.update_session_metadata({"status": "strategy_complete"})
                click.echo("Solution strategy developed and saved.")
                click.echo(f"Approach: {current_solution_strategy.get('high_level_strategy', {}).get('approach', 'Unknown')}")

            elif command == "testcases":
                if not current_problem_analysis:
                    current_problem_analysis = session_store.get_problem_analysis()
                    if not current_problem_analysis:
                        click.echo("Please analyze the problem first ('analyze' command).")
                        continue
                num_cases = click.prompt("Number of test cases to generate", type=int, default=3)
                click.echo(f"\nGenerating {num_cases} test cases...")
                current_test_cases = await problem_logic.generate_test_cases(current_problem_analysis, num_cases)
                click.echo(f"Generated {len(current_test_cases)} test cases.")
                click.echo("Creating score calculator...")
                current_score_calculator = await problem_logic.create_score_calculator(current_problem_analysis)
                click.echo("Score calculator created.")

            elif command == "initial":
                if not current_problem_analysis:
                    current_problem_analysis = session_store.get_problem_analysis()
                    if not current_problem_analysis:
                        click.echo("Please analyze the problem first ('analyze' command).")
                        continue
                click.echo("\nGenerating initial solution...")
                initial_code = await problem_logic.generate_initial_solution(current_problem_analysis)
                solution_id = f"initial_interactive_{int(time.time())}"
                session_store.save_solution(
                    solution_id=solution_id,
                    solution_data={"code": initial_code, "score": None, "generation": 0, "source": "interactive_initial"},
                    code=initial_code
                )
                session_store.update_session_metadata({"status": "initial_solution_generated", "last_solution_id": solution_id})
                click.echo(f"Initial solution generated and saved with ID: {solution_id}")
                if click.confirm("Show initial solution?"):
                    click.echo(initial_code)
            
            elif command == "list_solutions":
                solutions = session_store.list_solutions()
                if not solutions:
                    click.echo("No solutions found for this session.")
                    continue
                click.echo("\n=== Saved Solutions ===")
                for sol_data in solutions:
                    sid = sol_data.get('solution_id', 'Unknown ID')
                    score = sol_data.get('score', 'N/A')
                    gen = sol_data.get('generation', 'N/A')
                    source = sol_data.get('source', 'N/A')
                    click.echo(f"  ID: {sid}, Score: {score}, Gen: {gen}, Source: {source}, Has Code: {'code' in sol_data}")


            elif command == "evolve":
                # Ensure all prerequisites are met
                if not current_problem_analysis: current_problem_analysis = session_store.get_problem_analysis()
                if not current_solution_strategy: current_solution_strategy = session_store.get_solution_strategy()
                if not current_problem_analysis: click.echo("Please analyze problem first."); continue
                if not current_solution_strategy: click.echo("Please develop strategy first."); continue
                if not current_test_cases: click.echo("Please generate test cases first."); continue
                if not current_score_calculator: click.echo("Score calculator missing (generate test cases)."); continue

                # Determine initial code for evolution
                initial_code_to_evolve = None
                use_best_existing = click.confirm("Use best existing solution as starting point (if available)?", default=True)
                if use_best_existing:
                    best_sol = session_store.get_best_solution()
                    if best_sol and "code" in best_sol:
                        initial_code_to_evolve = best_sol["code"]
                        click.echo(f"Starting evolution from best solution (ID: {best_sol.get('solution_id')}, Score: {best_sol.get('score')}).")
                
                if not initial_code_to_evolve:
                    # Fallback: use last generated initial, or generate new
                    # This part could be more sophisticated, e.g. prompting for a specific solution ID
                    click.echo("No existing best solution to start from, or chose not to use it.")
                    last_initial_sols = [s for s in session_store.list_solutions() if s.get("source", "").startswith("initial")]
                    if last_initial_sols:
                         # Simplistic: take the most recent initial solution
                        initial_code_to_evolve = last_initial_sols[-1]["code"] # Assumes code is loaded by list_solutions
                        click.echo(f"Using most recent initial solution (ID: {last_initial_sols[-1].get('solution_id')}) as starting point.")
                    else:
                        click.echo("No prior initial solution found. Generating a new one...")
                        initial_code_to_evolve = await problem_logic.generate_initial_solution(current_problem_analysis)
                        temp_id = f"temp_initial_for_evolve_{int(time.time())}"
                        session_store.save_solution(temp_id, {"code": initial_code_to_evolve, "source": "temp_for_evolve"}, code=initial_code_to_evolve)
                        click.echo(f"Generated new initial solution (ID: {temp_id}) for evolution.")


                if not initial_code_to_evolve:
                    click.echo("Could not determine initial code for evolution. Aborting evolve step.")
                    continue

                # Configure evolution parameters
                max_generations = click.prompt("Max generations", type=int, default=config.get("evolution.max_generations", 10)) # Shorter for interactive
                population_size = click.prompt("Population size", type=int, default=config.get("evolution.population_size", 5)) # Smaller for interactive
                time_limit_interactive = click.prompt("Time limit (seconds)", type=int, default=config.get("evolution.time_limit_seconds", 300)) # Shorter

                # Update config for this run (does not save to file)
                current_evo_config = config.get("evolution").copy()
                current_evo_config["max_generations"] = max_generations
                current_evo_config["population_size"] = population_size
                current_evo_config["time_limit_seconds"] = time_limit_interactive
                
                # Re-initialize engine with interactive settings (or update its properties if mutable)
                interactive_engine = EvolutionaryEngine(evolutionary_engine.llm_client, current_evo_config)


                async def evaluate_solution_interactive(code): # Closure for interactive use, now async
                    # Simplified error handling for interactive use
                    avg_score, details = 0, {}
                    try:
                    # If problem_logic.evaluate_solution_code is async, it needs to be awaited
                    # If it's sync but calls async debugger, that's the issue.
                    # Assuming evaluate_solution_code is now async or handles async debugger calls correctly
                        avg_score, details = await problem_logic.evaluate_solution_code( # Ensure this is awaited
                            code, current_test_cases, current_score_calculator, implementation_debugger
                        )
                    except Exception as e:
                        click.echo(f"Error during evaluation: {e}", err=True)
                    return avg_score, details

                click.echo("\nRunning evolutionary process (interactive settings)...")
                evolution_artifacts_dir = os.path.join(session_store.session_dir, "evolution_artifacts_interactive")
                os.makedirs(evolution_artifacts_dir, exist_ok=True)

                evolution_result = await interactive_engine.evolve(
                    current_problem_analysis,
                    current_solution_strategy,
                    initial_code_to_evolve,
                    evaluate_solution_interactive,
                    evolution_artifacts_dir,
                )

                session_store.save_evolution_log(evolution_result["evolution_log"]) # Save log under default name
                session_store.update_session_metadata({"status": "evolution_interactive_complete"})

                best_solution_id = f"best_interactive_gen_{evolution_result['generations_completed']}_{int(time.time())}"
                session_store.save_solution(
                    solution_id=best_solution_id,
                    solution_data={
                        "code": evolution_result["best_solution"],
                        "score": evolution_result["best_score"],
                        "generation": evolution_result["generations_completed"],
                        "source": "interactive_evolve"
                    },
                    code=evolution_result["best_solution"]
                )
                session_store.update_session_metadata({"last_solution_id": best_solution_id, "best_score": evolution_result["best_score"]})

                click.echo(f"\nEvolution complete: {evolution_result['generations_completed']} generations")
                click.echo(f"Best score from this run: {evolution_result['best_score']}")
                if click.confirm("Show best solution from this run?"):
                    click.echo(evolution_result["best_solution"])
            else:
                click.echo(f"Unknown command: {command}")


@cli.command()
@click.argument("session_id", required=False)
@click.option("--watch", "-w", is_flag=True, help="Watch status updates")
@click.pass_context
def status(ctx, session_id, watch):
    """
    Show session status.
    If WORKSPACE (problem directory) is not specified via a global option or config,
    it defaults to the current directory if it looks like a problem workspace.
    """
    config_obj = ctx.obj["config"] # Renamed to avoid conflict with 'config' function
    
    # Determine problem directory for SessionManager context
    # This assumes 'workspace.base_dir' in config is the problem directory.
    problem_dir_str = config_obj.get("workspace.base_dir")
    if not problem_dir_str:
        # Fallback or error if not defined. For now, let's assume it's defined.
        # Or, it could default to current working directory if it contains ahc_config.yaml
        if (Path.cwd() / "ahc_config.yaml").exists():
            problem_dir_str = str(Path.cwd())
            click.echo(f"Warning: workspace.base_dir not set in main config, using current directory {problem_dir_str} as problem context.", err=True)
        else:
            click.echo("Error: Problem directory context not established. Set workspace.base_dir in config or run from a problem directory.", err=True)
            ctx.exit(1)
            
    problem_dir_path = Path(problem_dir_str).resolve()
    sm_workspace_dir = str(problem_dir_path.parent)
    sm_problem_id = problem_dir_path.name
    
    try:
        session_manager = SessionManager(sm_workspace_dir, sm_problem_id)
    except Exception as e:
        click.echo(f"Error initializing SessionManager for {problem_dir_path}: {e}", err=True)
        ctx.exit(1)

    if session_id:
        session_store = session_manager.get_session_store(session_id)
        if not session_store:
            click.echo(f"Session {session_id} not found in problem {sm_problem_id}")
            return

        _show_session_status(session_store) # Pass SessionStore instance

        if watch:
            click.echo("\nWatching for updates (Ctrl+C to stop)...")
            try:
                while True:
                    time.sleep(5) # time was already imported
                    # Re-fetch session_store in case it was deleted or its path became invalid (though unlikely)
                    refetched_store = session_manager.get_session_store(session_id)
                    if refetched_store:
                        _show_session_status(refetched_store)
                    else:
                        click.echo(f"\nSession {session_id} no longer found. Stopping watch.")
                        break
            except KeyboardInterrupt:
                click.echo("\nStopped watching")
    else:
        sessions_metadata_list = session_manager.list_sessions()
        if not sessions_metadata_list:
            click.echo(f"No sessions found for problem {sm_problem_id}")
            return

        click.echo(f"Found {len(sessions_metadata_list)} sessions for problem {sm_problem_id}:")
        for session_meta in sessions_metadata_list:
            click.echo(f"\nSession ID: {session_meta.get('session_id')}")
            # problem_id should be consistent (sm_problem_id) but if stored, display that
            click.echo(f"Problem: {session_meta.get('problem_id', sm_problem_id)}") 
            click.echo(f"Created: {_format_timestamp(session_meta.get('created_at'))}")
            click.echo(f"Status: {session_meta.get('status', 'Unknown')}")
            click.echo(f"Last Updated: {_format_timestamp(session_meta.get('updated_at'))}")


def _show_session_status(session_store: SessionStore):
    """
    Show detailed session status using a SessionStore instance.
    """
    session_metadata = session_store.get_session_metadata()
    if not session_metadata:
        click.echo(f"Could not retrieve metadata for session {session_store.session_id}.")
        return

    session_id = session_store.session_id # From store

    click.echo("\n=== Session Status ===")
    click.echo(f"Session ID: {session_id}")
    click.echo(f"Problem ID: {session_store.problem_id}") # From store
    click.echo(f"Created: {_format_timestamp(session_metadata.get('created_at'))}")
    click.echo(f"Updated: {_format_timestamp(session_metadata.get('updated_at'))}")
    click.echo(f"Status: {session_metadata.get('status', 'Unknown')}")

    problem_analysis = session_store.get_problem_analysis()
    click.echo(f"Problem Analysis: {'Complete' if problem_analysis else 'Not started'}")

    solution_strategy = session_store.get_solution_strategy()
    click.echo(f"Solution Strategy: {'Complete' if solution_strategy else 'Not started'}")

    evolution_log = session_store.get_evolution_log()
    if evolution_log: # Check if log exists
        # Assuming evolution_log is a dict as saved by SessionStore
        click.echo(f"Evolution: Complete ({evolution_log.get('generations_completed', 0)} generations)")
        # Best score might be in evolution_log or session_metadata
        best_score_from_log = evolution_log.get('best_score')
        best_score_from_meta = session_metadata.get('best_score')
        display_score = best_score_from_meta if best_score_from_meta is not None else best_score_from_log
        click.echo(f"Best Score (Overall Session): {display_score if display_score is not None else 'Unknown'}")
        click.echo(f"Duration (Last Evolution): {_format_duration(evolution_log.get('duration_seconds', 0))}") # Assuming duration_seconds
    else:
        click.echo("Evolution: Log not found or not started")

    best_solution = session_store.get_best_solution()
    if best_solution:
        click.echo(f"Best Solution Artifact: Available (ID: {best_solution.get('solution_id', 'N/A')}, Score: {best_solution.get('score', 'Unknown')})")
    else:
        click.echo("Best Solution Artifact: Not found")


def _format_timestamp(timestamp):
    """
    Format a timestamp.
    """
    if not timestamp:
        return "Unknown"
    import datetime # Keep import here as it's only used by this function
    try:
        return datetime.datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(timestamp) # If not a valid number, return as is


def _format_duration(seconds: Optional[float]):
    """
    Format a duration in seconds.
    """
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "Unknown"

    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


@cli.command()
@click.argument("session_id")
@click.pass_context
def stop(ctx, session_id):
    """
    Mark a session as 'stopped'.
    """
    config_obj = ctx.obj["config"]
    problem_dir_str = config_obj.get("workspace.base_dir")
    if not problem_dir_str: # Basic validation
        click.echo("Error: Problem directory context (workspace.base_dir) not set.", err=True)
        ctx.exit(1)

    problem_dir_path = Path(problem_dir_str).resolve()
    sm_workspace_dir = str(problem_dir_path.parent)
    sm_problem_id = problem_dir_path.name
    
    session_manager = SessionManager(sm_workspace_dir, sm_problem_id)
    session_store = session_manager.get_session_store(session_id)

    if not session_store:
        click.echo(f"Session {session_id} not found in problem {sm_problem_id}")
        return

    if session_store.update_session_metadata({"status": "stopped"}):
        click.echo(f"Session {session_id} marked as stopped.")
    else:
        click.echo(f"Error marking session {session_id} as stopped.", err=True)


@cli.command()
@click.argument("session_id")
@click.option("--output", "-o", type=click.Path(), help="Output file path for the solution code")
@click.pass_context
def submit(ctx, session_id, output):
    """
    Retrieve the best solution code from a session and print or save it.
    """
    config_obj = ctx.obj["config"]
    problem_dir_str = config_obj.get("workspace.base_dir")
    if not problem_dir_str:
        click.echo("Error: Problem directory context (workspace.base_dir) not set.", err=True)
        ctx.exit(1)

    problem_dir_path = Path(problem_dir_str).resolve()
    sm_workspace_dir = str(problem_dir_path.parent)
    sm_problem_id = problem_dir_path.name

    session_manager = SessionManager(sm_workspace_dir, sm_problem_id)
    session_store = session_manager.get_session_store(session_id)

    if not session_store:
        click.echo(f"Session {session_id} not found in problem {sm_problem_id}")
        return

    best_solution_artifact = session_store.get_best_solution()
    if not best_solution_artifact:
        click.echo(f"No best solution found for session {session_id}")
        return

    solution_code = best_solution_artifact.get("code") # get_best_solution should load code
    if not solution_code:
        click.echo(f"No code found in the best solution artifact for session {session_id} (ID: {best_solution_artifact.get('solution_id')})")
        return

    if output:
        try:
            with open(output, "w", encoding="utf-8") as f:
                f.write(solution_code)
            click.echo(f"Best solution code from session {session_id} written to {output}")
        except Exception as e:
            click.echo(f"Error writing solution code to '{output}': {e}", err=True)
    else:
        click.echo("\n=== Best Solution Code ===")
        click.echo(solution_code)

    click.echo(f"Source Solution ID: {best_solution_artifact.get('solution_id', 'N/A')}")
    click.echo(f"Score: {best_solution_artifact.get('score', 'Unknown')}")


@cli.command()
@click.argument("batch_config", type=click.Path(exists=True))
@click.option("--parallel", "-p", type=int, help="Number of parallel executions")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.pass_context
def batch(ctx, batch_config, parallel, output_dir):
    """
    Run batch processing.
    """
    config = ctx.obj["config"]

    # Override configuration with command-line options
    if parallel:
        config.set("batch.parallel", parallel)

    if output_dir:
        config.set("batch.output_dir", output_dir)

    # Load batch configuration
    with open(batch_config) as f:
        batch_cfg = yaml.safe_load(f)

    # Get output directory
    if not output_dir:
        output_dir = batch_cfg.get("common", {}).get("workspace", os.path.join(os.getcwd(), "ahc_batch"))

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    click.echo(f"Batch processing configuration loaded from {batch_config}")
    click.echo(f"Output directory: {output_dir}")

    # Get number of parallel executions
    if not parallel:
        parallel = batch_cfg.get("common", {}).get("parallel", 1)

    click.echo(f"Parallel executions: {parallel}")

    # Get problems
    problems = batch_cfg.get("problems", [])
    if not problems:
        click.echo("No problems found in batch configuration")
        return

    click.echo(f"Found {len(problems)} problems")

    # Get parameter sets
    parameter_sets = batch_cfg.get("parameter_sets", [])
    if not parameter_sets:
        click.echo("No parameter sets found in batch configuration")
        return

    click.echo(f"Found {len(parameter_sets)} parameter sets")

    # Get experiments
    experiments = batch_cfg.get("experiments", [])
    if not experiments:
        click.echo("No experiments found in batch configuration")
        return

    click.echo(f"Found {len(experiments)} experiments")

    # Run experiments
    asyncio.run(_run_batch_experiments(config, batch_cfg, output_dir, parallel))


async def _run_batch_experiments(config, batch_cfg, output_dir, parallel):
    """
    Run batch experiments asynchronously.
    """
    # Get problems
    problems = {p["name"]: p for p in batch_cfg.get("problems", [])}

    # Get parameter sets
    parameter_sets = {p["name"]: p for p in batch_cfg.get("parameter_sets", [])}

    # Get experiments
    experiments = batch_cfg.get("experiments", [])

    # Create experiment tasks
    tasks = []
    for _, experiment in enumerate(experiments):
        problem_name = experiment.get("problem")
        parameter_set_name = experiment.get("parameter_set")
        repeats = experiment.get("repeats", 1)

        if problem_name not in problems:
            click.echo(f"Problem {problem_name} not found, skipping experiment")
            continue

        if parameter_set_name not in parameter_sets:
            click.echo(f"Parameter set {parameter_set_name} not found, skipping experiment")
            continue

        problem = problems[problem_name]
        parameter_set = parameter_sets[parameter_set_name]

        for j in range(repeats):
            experiment_id = f"{problem_name}_{parameter_set_name}_{j + 1}"
            experiment_dir = os.path.join(output_dir, experiment_id)
            os.makedirs(experiment_dir, exist_ok=True)

            # Create experiment task
            task = _run_experiment(config, experiment_id, problem, parameter_set, experiment_dir)

            tasks.append(task)

    # Run experiments in parallel
    click.echo(f"Running {len(tasks)} experiments in parallel batches of {parallel}")

    results = []
    for i in range(0, len(tasks), parallel):
        batch_tasks = tasks[i : i + parallel]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

        click.echo(f"Completed {min(i + parallel, len(tasks))}/{len(tasks)} experiments")

    click.echo("All experiments completed")

    # Summarize results
    click.echo("\n=== Experiment Results ===")

    for result in results:
        click.echo(f"\nExperiment: {result['experiment_id']}")
        click.echo(f"Problem: {result['problem_name']}")
        click.echo(f"Parameter Set: {result['parameter_set_name']}")
        click.echo(f"Best Score: {result['best_score']}")
        click.echo(f"Generations: {result['generations']}")
        click.echo(f"Duration: {_format_duration(result['duration'])}")

    # Write summary to file
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"\nSummary written to {summary_path}")


async def _run_experiment(config, experiment_id, problem, parameter_set, experiment_dir):
    """
    Run a single experiment.
    """
    # Create experiment-specific config
    experiment_config = config.export()

    # Override with parameter set
    for key, value in parameter_set.items():
        if key != "name":
            if "." in key:
                experiment_config = _set_nested_dict(experiment_config, key.split("."), value)
            else:
                experiment_config[key] = value

    # Initialize clients and modules
    llm_client = LLMClient(experiment_config.get("llm"))
    docker_manager = DockerManager(experiment_config.get("docker"))
    
    # experiment_dir is the root for this specific experiment, acting as "sm_workspace_dir"
    # problem_id here is derived from the problem file name for this experiment
    problem_file_name = os.path.splitext(os.path.basename(problem["path"]))[0] 
    # For batch, each experiment is effectively a unique "problem" context for SessionManager
    # The session_manager's workspace_dir will be experiment_dir itself, and problem_id will be problem_file_name.
    # This means sessions will be under experiment_dir/problem_file_name/knowledge/sessions/
    session_manager = SessionManager(workspace_dir=experiment_dir, problem_id=problem_file_name)

    # Heuristic KBs for batch - they might be less relevant or need careful scoping.
    # For simplicity, let's assume batch doesn't heavily rely on pre-existing problem-specific KB for now,
    # or it would need to be configured per problem in the batch_cfg.
    # Global HKB could still be used.
    global_hkb_path_str_batch = experiment_config.get("heuristic_knowledge_base", {}).get("global_path", os.path.expanduser("~/.ahc_agent_heuristic_knowledge"))
    global_heuristic_kb_batch = HeuristicKnowledgeBase(global_hkb_path_str_batch)
    # Problem specific HKB for batch experiment - this path would be relative to experiment_dir
    problem_hkb_batch_path = Path(experiment_dir) / problem_file_name / "knowledge" / "kb"
    problem_heuristic_kb_batch = HeuristicKnowledgeBase(str(problem_hkb_batch_path))

    problem_analyzer = ProblemAnalyzer(llm_client, experiment_config.get("analyzer"), problem_heuristic_kb=problem_heuristic_kb_batch)
    solution_strategist = SolutionStrategist(
        llm_client, 
        experiment_config.get("strategist"), 
        global_heuristic_kb=global_heuristic_kb_batch, 
        problem_heuristic_kb=problem_heuristic_kb_batch
    )
    evolutionary_engine = EvolutionaryEngine(llm_client, experiment_config.get("evolution"))
    implementation_debugger = ImplementationDebugger(llm_client, docker_manager, experiment_config.get("debugger"))
    problem_logic = ProblemLogic(llm_client, experiment_config.get("problem_logic"), problem_heuristic_kb=problem_heuristic_kb_batch)

    # Parse problem statement
    problem_path = problem.get("path")
    with open(problem_path, encoding='utf-8') as f: # Added encoding
        problem_text = f.read()

    # Create session for this specific experiment run
    initial_session_metadata_batch = {
        "experiment_id": experiment_id,
        "problem_name": problem.get("name", "unknown"),
        "parameter_set_name": parameter_set.get("name"),
        "status": "batch_initialized",
    }
    session_store = session_manager.create_session(initial_metadata=initial_session_metadata_batch)
    # Save problem_text to the session specific storage
    session_store.save_generic_file_data("problem_text.md", problem_text)


    # Analyze problem
    problem_analysis = await problem_analyzer.analyze(problem_text)
    session_store.save_problem_analysis(problem_analysis)
    session_store.update_session_metadata({"status": "batch_analysis_complete"})

    # Develop solution strategy
    solution_strategy = await solution_strategist.develop_strategy(problem_analysis)
    session_store.save_solution_strategy(solution_strategy)
    session_store.update_session_metadata({"status": "batch_strategy_complete"})
    
    # Generate initial solution
    initial_solution_code = await problem_logic.generate_initial_solution(problem_analysis)
    session_store.save_solution(
        solution_id="initial_batch_generated",
        solution_data={"code": initial_solution_code, "score": None, "generation": 0, "source": "batch_generation"},
        code=initial_solution_code
    )
    session_store.update_session_metadata({"status": "batch_initial_solution_generated"})

    # Generate test cases
    test_cases = await problem_logic.generate_test_cases(problem_analysis, 3)

    # Create score calculator
    score_calculator = await problem_logic.create_score_calculator(problem_analysis)

    # Define evaluation function
    def evaluate_solution(code, current_test_cases, current_score_calculator):
        total_score = 0
        details = {}

        for i, test_case in enumerate(current_test_cases):
            result = asyncio.run(implementation_debugger.compile_and_test(code, test_case["input"]))

            if result["success"]:
                score = current_score_calculator(test_case["input"], result["execution_output"])
                total_score += score
                details[f"test_{i + 1}"] = {"score": score, "execution_time": result["execution_time"]}
            else:
                details[f"test_{i + 1}"] = {
                    "error": result["compilation_errors"] or result["execution_errors"],
                    "score": 0,
                }

        avg_score = total_score / len(current_test_cases) if current_test_cases else 0
        return avg_score, details

    # Run evolutionary process
    start_time = time.time()
    
    evolution_artifacts_dir_batch = os.path.join(session_store.session_dir, "evolution_artifacts_batch")
    os.makedirs(evolution_artifacts_dir_batch, exist_ok=True)

    result = await evolutionary_engine.evolve(
        problem_analysis,
        solution_strategy,
        initial_solution_code, # Use the generated initial_solution_code
        lambda code: evaluate_solution(code, test_cases, score_calculator), # evaluate_solution needs to be defined or adapted
        evolution_artifacts_dir_batch,
    )

    duration = time.time() - start_time

    # Save evolution log and best solution to the session store
    session_store.save_evolution_log(result["evolution_log"])
    batch_best_solution_id = f"best_batch_gen_{result['generations_completed']}"
    session_store.save_solution(
        solution_id=batch_best_solution_id,
        solution_data={"code": result["best_solution"], "score": result["best_score"], "generation": result["generations_completed"]},
        code=result["best_solution"]
    )
    session_store.update_session_metadata({
        "status": "batch_evolution_complete", 
        "last_solution_id": batch_best_solution_id, 
        "best_score": result["best_score"]
    })

    # Save experiment results
    result_file = os.path.join(experiment_dir, "result.json")
    with open(result_file, "w") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "problem_name": problem.get("name"),
                "parameter_set_name": parameter_set.get("name"),
                "best_score": result["best_score"],
                "generations": result["generations_completed"],
                "duration": duration,
                "session_id": session_id,
            },
            f,
            indent=4,
        )

    logger.info(f"Experiment {experiment_id} completed. Results saved to {result_file}")
    return {
        "experiment_id": experiment_id,
        "problem_name": problem.get("name"),
        "parameter_set_name": parameter_set.get("name"),
        "best_score": result["best_score"], # This is from the evolution result
        "generations": result["generations_completed"],
        "duration": duration,
        "session_id": session_store.session_id, # Use session_id from the store
    }


def _set_nested_dict(d, keys, value):
    """
    Set a value in a nested dictionary.
    """
    if len(keys) == 1:
        d[keys[0]] = value
        return d

    if keys[0] not in d:
        d[keys[0]] = {}

    d[keys[0]] = _set_nested_dict(d[keys[0]], keys[1:], value)
    return d


@cli.group()
@click.pass_context
def config(_ctx):
    """
    Manage configuration.
    """


@config.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx, key):
    """
    Get a configuration value.
    """
    config = ctx.obj["config"]
    value = config.get(key)

    if value is None:
        click.echo(f"Configuration key '{key}' not found")
    else:
        click.echo(f"{key} = {value}")


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """
    Set a configuration value.
    """
    config = ctx.obj["config"]

    # Convert value to appropriate type
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.isdigit():
        value = int(value)
    elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
        value = float(value)

    config.set(key, value)

    click.echo(f"Set {key} = {value}")


@config.command("export")
@click.argument("path", type=click.Path())
@click.pass_context
def config_export(ctx, path):
    """
    Export configuration to a file.
    """
    config = ctx.obj["config"]
    config.save(path)

    click.echo(f"Configuration exported to {path}")


@config.command("import")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def config_import(ctx, path):
    """
    Import configuration from a file.
    """
    config = ctx.obj["config"]

    with open(path) as f:
        imported_config = yaml.safe_load(f)

    config.import_config(imported_config)

    click.echo(f"Configuration imported from {path}")


@cli.group()
@click.pass_context
def docker(_ctx):
    """
    Manage Docker environment.
    """


@docker.command("setup")
@click.pass_context
def docker_setup(ctx):
    """
    Set up Docker environment.
    """
    config = ctx.obj["config"]

    # Initialize Docker manager
    docker_manager = DockerManager(config.get("docker"))

    # Pull Docker image
    click.echo(f"Pulling Docker image: {docker_manager.image}")
    success = docker_manager.pull_image()

    if success:
        click.echo("Docker image pulled successfully")
    else:
        click.echo("Failed to pull Docker image")


@docker.command("status")
@click.pass_context
def docker_status(ctx):
    """
    Show Docker environment status.
    """
    config = ctx.obj["config"]

    # Initialize Docker manager
    docker_manager = DockerManager(config.get("docker"))

    # Check Docker
    try:
        docker_manager.check_docker_availability()
        click.echo("Docker is available")

        # Run test command
        result = docker_manager.run_command("echo 'Docker test successful'", os.getcwd())

        if result["success"]:
            click.echo("Docker test successful")
        else:
            click.echo(f"Docker test failed: {result['stderr']}")

    except RuntimeError as e:
        logger.error(f"Docker availability check failed. Type: {type(e).__name__}, Error: {e}")
        click.echo(f"Docker is not available: {e!s}")


@docker.command("cleanup")
@click.pass_context
def docker_cleanup(ctx):
    """
    Clean up Docker environment.
    """
    config = ctx.obj["config"]

    # Initialize Docker manager
    docker_manager = DockerManager(config.get("docker"))

    # Clean up
    click.echo("Cleaning up Docker resources")
    success = docker_manager.cleanup()

    if success:
        click.echo("Docker resources cleaned up successfully")
    else:
        click.echo("Failed to clean up Docker resources")


def main():
    """
    Main entry point.
    """
    cli(obj={})


if __name__ == "__main__":
    main()
