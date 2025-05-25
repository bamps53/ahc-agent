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

# Third-party imports
import click
import yaml

# Local application/library specific imports
from .config import Config
from .core.analyzer import ProblemAnalyzer
from .core.debugger import ImplementationDebugger
from .core.engine import EvolutionaryEngine
from .core.knowledge import KnowledgeBase
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
    asyncio.run(_solve_problem(config, problem_text, session_id, interactive))


async def _solve_problem(config, problem_text, session_id=None, interactive=False):
    """
    Solve a problem asynchronously.
    """
    # Initialize clients and modules
    llm_client = LLMClient(config.get("llm"))
    docker_manager = DockerManager(config.get("docker"))
    contest_id_from_config = config.get("contest_id")
    if not contest_id_from_config:
        # 設定ファイルに contest_id がない場合、ワークスペース名から推測
        contest_id_from_config = Path(config.get("workspace.base_dir")).name
        click.echo(
            f"Warning: 'contest_id' not found in config, using workspace name "
            f"'{contest_id_from_config}' as problem_id for KnowledgeBase.",
            err=True,
        )

    knowledge_base = KnowledgeBase(config.get("workspace.base_dir"), problem_id=contest_id_from_config)

    problem_analyzer = ProblemAnalyzer(llm_client, config.get("analyzer"))
    solution_strategist = SolutionStrategist(llm_client, config.get("strategist"))
    evolutionary_engine = EvolutionaryEngine(llm_client, config.get("evolution"))
    implementation_debugger = ImplementationDebugger(llm_client, docker_manager, config.get("debugger"))
    problem_logic = ProblemLogic(llm_client, config.get("problem_logic"))

    # Create or get session
    if session_id:
        session = knowledge_base.get_session(session_id)
        if not session:
            click.echo(f"Session {session_id} not found")
            return

        click.echo(f"Resuming session {session_id}")
    else:
        # Parse problem statement
        parsed_info = await problem_logic.parse_problem_statement(problem_text)

        # Create session
        session_id = knowledge_base.create_session(
            parsed_info.get("title", "unknown"), {"problem_text": problem_text, "parsed_info": parsed_info}
        )

        click.echo(f"Created session {session_id}")

    # Interactive mode
    if interactive:
        await _interactive_solve(
            session_id,
            config,
            knowledge_base,
            problem_analyzer,
            solution_strategist,
            evolutionary_engine,
            implementation_debugger,
            problem_logic,
        )
        return

    # Non-interactive mode
    click.echo("Analyzing problem...")

    # Get problem analysis
    problem_analysis = knowledge_base.get_problem_analysis(session_id)
    if not problem_analysis:
        # Analyze problem
        problem_analysis = await problem_analyzer.analyze(problem_text)

        # Save problem analysis
        knowledge_base.save_problem_analysis(session_id, problem_analysis)

    click.echo("Developing solution strategy...")

    # Get solution strategy
    solution_strategy = knowledge_base.get_solution_strategy(session_id)
    if not solution_strategy:
        # Develop solution strategy
        solution_strategy = await solution_strategist.develop_strategy(problem_analysis)

        # Save solution strategy
        knowledge_base.save_solution_strategy(session_id, solution_strategy)

    click.echo("Generating initial solution...")

    # Get best solution or generate initial solution
    best_solution = knowledge_base.get_best_solution(session_id)
    initial_solution = (
        best_solution.get("code") if best_solution else await problem_logic.generate_initial_solution(problem_analysis)
    )

    # Generate test cases
    click.echo("Generating test cases...")
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
    click.echo("Running evolutionary process...")
    result = await evolutionary_engine.evolve(
        problem_analysis,
        solution_strategy,
        initial_solution,
        lambda code: evaluate_solution(code, test_cases, score_calculator),
        os.path.join(config.get("workspace.base_dir"), "sessions", session_id),
    )

    # Save evolution log
    knowledge_base.save_evolution_log(session_id, result["evolution_log"])

    # Save best solution
    knowledge_base.save_solution(
        session_id,
        "best",
        {"code": result["best_solution"], "score": result["best_score"], "generation": result["generations_completed"]},
    )

    click.echo(f"Evolution complete: {result['generations_completed']} generations")
    click.echo(f"Best score: {result['best_score']}")
    best_solution_path = os.path.join(config.get("workspace.base_dir"), "sessions", session_id, "solutions", "best.cpp")
    click.echo(f"Best solution saved to {best_solution_path}")


async def _interactive_solve(
    session_id,
    config,
    knowledge_base,
    problem_analyzer,
    solution_strategist,
    evolutionary_engine,
    implementation_debugger,
    problem_logic,
):
    """
    Interactive problem solving.
    """
    # Get session
    session = knowledge_base.get_session(session_id)
    if not session:
        click.echo(f"Session {session_id} not found")
        return

    # Get problem text
    problem_text = session.get("problem_text")
    if not problem_text:
        click.echo("Problem text not found in session")
        return

    # Get workspace directory
    workspace_dir = config.get("workspace.base_dir")
    workspace_dir = os.path.expanduser(workspace_dir)

    # Interactive loop
    running = True
    current_step = "init"

    # State variables
    problem_analysis = None
    solution_strategy = None
    test_cases = None
    score_calculator = None
    evolution_result = None

    while running:
        if current_step == "init":
            click.echo("\n=== AHCAgent Interactive Mode ===")
            click.echo(f"Session: {session_id}")
            click.echo(f"Problem: {session.get('problem_id', 'Unknown')}")
            click.echo("\nAvailable commands:")
            click.echo("  analyze - Analyze the problem")
            click.echo("  strategy - Develop solution strategy")
            click.echo("  testcases - Generate test cases")
            click.echo("  initial - Generate initial solution")
            click.echo("  evolve - Run evolutionary process")
            click.echo("  status - Show current status")
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
                click.echo(f"Session: {session_id}")
                click.echo(f"Problem: {session.get('problem_id', 'Unknown')}")
                problem_analysis_exists = problem_analysis or knowledge_base.get_problem_analysis(session_id)
                problem_analysis_status = "Complete" if problem_analysis_exists else "Not started"
                click.echo(f"Problem Analysis: {problem_analysis_status}")
                solution_strategy_exists = solution_strategy or knowledge_base.get_solution_strategy(session_id)
                solution_strategy_status = "Complete" if solution_strategy_exists else "Not started"
                click.echo(f"Solution Strategy: {solution_strategy_status}")
                click.echo(f"Test Cases: {'Generated' if test_cases else 'Not generated'}")

                # Show best solution if available
                best_solution = knowledge_base.get_best_solution(session_id)
                if best_solution:
                    click.echo(f"Best Score: {best_solution.get('score', 'Unknown')}")

            elif command == "analyze":
                click.echo("\nAnalyzing problem...")

                # Get problem analysis from knowledge base or analyze
                problem_analysis = knowledge_base.get_problem_analysis(session_id)
                if not problem_analysis:
                    problem_analysis = await problem_analyzer.analyze(problem_text)
                    knowledge_base.save_problem_analysis(session_id, problem_analysis)

                click.echo("Problem analysis complete")
                click.echo(f"Title: {problem_analysis.get('title', 'Unknown')}")
                click.echo(f"Characteristics: {problem_analysis.get('characteristics', {}).get('problem_type', 'Unknown')}")

                # Show effective algorithms
                effective_algorithms = problem_analysis.get("characteristics", {}).get("effective_algorithms", [])
                if effective_algorithms:
                    click.echo("Effective Algorithms:")
                    for algo in effective_algorithms:
                        click.echo(f"  - {algo}")

            elif command == "strategy":
                if not problem_analysis:
                    problem_analysis = knowledge_base.get_problem_analysis(session_id)
                    if not problem_analysis:
                        click.echo("Please analyze the problem first")
                        continue

                click.echo("\nDeveloping solution strategy...")

                # Get solution strategy from knowledge base or develop
                solution_strategy = knowledge_base.get_solution_strategy(session_id)
                if not solution_strategy:
                    solution_strategy = await solution_strategist.develop_strategy(problem_analysis)
                    knowledge_base.save_solution_strategy(session_id, solution_strategy)

                click.echo("Solution strategy development complete")
                click.echo(f"Approach: {solution_strategy.get('high_level_strategy', {}).get('approach', 'Unknown')}")

                # Show key insights
                key_insights = solution_strategy.get("high_level_strategy", {}).get("key_insights", [])
                if key_insights:
                    click.echo("Key Insights:")
                    for insight in key_insights:
                        click.echo(f"  - {insight}")

                # Show main algorithm
                main_algo = solution_strategy.get("algorithm_selection", {}).get("main_algorithm", {})
                if main_algo:
                    click.echo(f"Main Algorithm: {main_algo.get('name', 'Unknown')}")
                    click.echo(f"Suitability: {main_algo.get('suitability', 'Unknown')}")

            elif command == "testcases":
                if not problem_analysis:
                    problem_analysis = knowledge_base.get_problem_analysis(session_id)
                    if not problem_analysis:
                        click.echo("Please analyze the problem first")
                        continue

                num_cases = click.prompt("Number of test cases to generate", type=int, default=3)

                click.echo(f"\nGenerating {num_cases} test cases...")
                test_cases = await problem_logic.generate_test_cases(problem_analysis, num_cases)

                click.echo(f"Generated {len(test_cases)} test cases")

                # Create score calculator
                click.echo("Creating score calculator...")
                score_calculator = await problem_logic.create_score_calculator(problem_analysis)

                click.echo("Score calculator created")

            elif command == "initial":
                if not problem_analysis:
                    problem_analysis = knowledge_base.get_problem_analysis(session_id)
                    if not problem_analysis:
                        click.echo("Please analyze the problem first")
                        continue

                click.echo("\nGenerating initial solution...")
                initial_solution = await problem_logic.generate_initial_solution(problem_analysis)

                # Save initial solution
                knowledge_base.save_solution(
                    session_id,
                    "initial",
                    {"code": initial_solution, "score": 0, "generation": 0},
                )

                click.echo("Initial solution generated and saved")

                # Show initial solution
                if click.confirm("Show initial solution?"):
                    click.echo("\n=== Initial Solution ===")
                    click.echo(initial_solution)

            elif command == "evolve":
                if not problem_analysis:
                    problem_analysis = knowledge_base.get_problem_analysis(session_id)
                    if not problem_analysis:
                        click.echo("Please analyze the problem first")
                        continue

                if not solution_strategy:
                    solution_strategy = knowledge_base.get_solution_strategy(session_id)
                    if not solution_strategy:
                        click.echo("Please develop solution strategy first")
                        continue

                if not test_cases:
                    num_cases = click.prompt("Number of test cases to generate", type=int, default=3)
                    click.echo(f"\nGenerating {num_cases} test cases...")
                    test_cases = await problem_logic.generate_test_cases(problem_analysis, num_cases)
                    click.echo(f"Generated {len(test_cases)} test cases")

                if not score_calculator:
                    click.echo("Creating score calculator...")
                    score_calculator = await problem_logic.create_score_calculator(problem_analysis)
                    click.echo("Score calculator created")

                # Get best solution or initial solution
                best_solution = knowledge_base.get_best_solution(session_id)
                if best_solution:
                    initial_code = best_solution.get("code")
                    click.echo(f"Using best solution (score: {best_solution.get('score', 'Unknown')}) as starting point")
                else:
                    # Get initial solution or generate
                    initial_solution = knowledge_base.get_solution(session_id, "initial")
                    if initial_solution:
                        initial_code = initial_solution.get("code")
                        click.echo("Using saved initial solution as starting point")
                    else:
                        click.echo("Generating initial solution...")
                        initial_code = await problem_logic.generate_initial_solution(problem_analysis)
                        click.echo("Initial solution generated")

                # Configure evolution parameters
                max_generations = click.prompt(
                    "Maximum generations", type=int, default=config.get("evolution.max_generations", 30)
                )
                population_size = click.prompt("Population size", type=int, default=config.get("evolution.population_size", 10))
                time_limit = click.prompt(
                    "Time limit (seconds)", type=int, default=config.get("evolution.time_limit_seconds", 1800)
                )

                # Update configuration
                config.set("evolution.max_generations", max_generations)
                config.set("evolution.population_size", population_size)
                config.set("evolution.time_limit_seconds", time_limit)

                # Configure evolutionary engine
                evolutionary_engine.max_generations = max_generations
                evolutionary_engine.population_size = population_size
                evolutionary_engine.time_limit_seconds = time_limit

                # Define evaluation function
                def evaluate_solution(code):
                    total_score = 0
                    details = {}

                    for i, test_case in enumerate(test_cases):
                        result = asyncio.run(implementation_debugger.compile_and_test(code, test_case["input"]))

                        if result["success"]:
                            score = score_calculator(test_case["input"], result["execution_output"])
                            total_score += score
                            details[f"test_{i + 1}"] = {"score": score, "execution_time": result["execution_time"]}
                        else:
                            details[f"test_{i + 1}"] = {
                                "error": result["compilation_errors"] or result["execution_errors"],
                                "score": 0,
                            }

                    avg_score = total_score / len(test_cases) if test_cases else 0
                    return avg_score, details

                # Run evolutionary process
                click.echo("\nRunning evolutionary process...")
                click.echo(f"Max generations: {max_generations}")
                click.echo(f"Population size: {population_size}")
                click.echo(f"Time limit: {time_limit} seconds")

                evolution_result = await evolutionary_engine.evolve(
                    problem_analysis,
                    solution_strategy,
                    initial_code,
                    evaluate_solution,
                    os.path.join(config.get("workspace.base_dir"), "sessions", session_id),
                )

                # Save evolution log
                knowledge_base.save_evolution_log(session_id, evolution_result["evolution_log"])

                # Save best solution
                knowledge_base.save_solution(
                    session_id,
                    "best",
                    {
                        "code": evolution_result["best_solution"],
                        "score": evolution_result["best_score"],
                        "generation": evolution_result["generations_completed"],
                    },
                )

                click.echo(f"\nEvolution complete: {evolution_result['generations_completed']} generations")
                click.echo(f"Best score: {evolution_result['best_score']}")

                # Show best solution
                if click.confirm("Show best solution?"):
                    click.echo("\n=== Best Solution ===")
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
    """
    config = ctx.obj["config"]

    # Get workspace directory
    workspace_dir = config.get("workspace.base_dir")
    workspace_dir = os.path.expanduser(workspace_dir)

    # Initialize knowledge base
    knowledge_base = KnowledgeBase(workspace_dir)

    if session_id:
        # Show single session status
        session = knowledge_base.get_session(session_id)
        if not session:
            click.echo(f"Session {session_id} not found")
            return

        _show_session_status(session, knowledge_base)

        if watch:
            click.echo("\nWatching for updates (Ctrl+C to stop)...")
            try:
                while True:
                    import time

                    time.sleep(5)

                    session = knowledge_base.get_session(session_id)
                    if session:
                        _show_session_status(session, knowledge_base)
            except KeyboardInterrupt:
                click.echo("\nStopped watching")
    else:
        # List all sessions
        sessions = knowledge_base.list_sessions()

        if not sessions:
            click.echo("No sessions found")
            return

        click.echo(f"Found {len(sessions)} sessions:")

        for session in sessions:
            click.echo(f"\nSession ID: {session.get('session_id')}")
            click.echo(f"Problem: {session.get('problem_id', 'Unknown')}")
            click.echo(f"Created: {_format_timestamp(session.get('created_at'))}")
            click.echo(f"Status: {session.get('status', 'Unknown')}")


def _show_session_status(session, knowledge_base):
    """
    Show detailed session status.
    """
    session_id = session.get("session_id")

    click.echo("\n=== Session Status ===")
    click.echo(f"Session ID: {session_id}")
    click.echo(f"Problem: {session.get('problem_id', 'Unknown')}")
    click.echo(f"Created: {_format_timestamp(session.get('created_at'))}")
    click.echo(f"Updated: {_format_timestamp(session.get('updated_at'))}")
    click.echo(f"Status: {session.get('status', 'Unknown')}")

    # Check for problem analysis
    problem_analysis = knowledge_base.get_problem_analysis(session_id)
    click.echo(f"Problem Analysis: {'Complete' if problem_analysis else 'Not started'}")

    # Check for solution strategy
    solution_strategy = knowledge_base.get_solution_strategy(session_id)
    click.echo(f"Solution Strategy: {'Complete' if solution_strategy else 'Not started'}")

    # Check for evolution log
    evolution_log = knowledge_base.get_evolution_log(session_id)
    if evolution_log:
        click.echo(f"Evolution: Complete ({evolution_log.get('generations_completed', 0)} generations)")
        click.echo(f"Best Score: {evolution_log.get('best_score', 'Unknown')}")
        click.echo(f"Duration: {_format_duration(evolution_log.get('duration', 0))}")
    else:
        click.echo("Evolution: Not started")

    # Check for best solution
    best_solution = knowledge_base.get_best_solution(session_id)
    if best_solution:
        click.echo(f"Best Solution: Available (score: {best_solution.get('score', 'Unknown')})")


def _format_timestamp(timestamp):
    """
    Format a timestamp.
    """
    if not timestamp:
        return "Unknown"

    import datetime

    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds):
    """
    Format a duration in seconds.
    """
    if not seconds:
        return "Unknown"

    minutes, seconds = divmod(int(seconds), 60)
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
    Stop a running session.
    """
    config = ctx.obj["config"]

    # Initialize knowledge base
    knowledge_base = KnowledgeBase(config.get("workspace.base_dir"))

    # Get session
    session = knowledge_base.get_session(session_id)
    if not session:
        click.echo(f"Session {session_id} not found")
        return

    # Update session status
    knowledge_base.update_session(session_id, {"status": "stopped"})

    click.echo(f"Session {session_id} stopped")


@cli.command()
@click.argument("session_id")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def submit(ctx, session_id, output):
    """
    Submit the best solution from a session.
    """
    config = ctx.obj["config"]

    # Initialize knowledge base
    knowledge_base = KnowledgeBase(config.get("workspace.base_dir"))

    # Get session
    session = knowledge_base.get_session(session_id)
    if not session:
        click.echo(f"Session {session_id} not found")
        return

    # Get best solution
    best_solution = knowledge_base.get_best_solution(session_id)
    if not best_solution:
        click.echo(f"No solution found for session {session_id}")
        return

    # Get solution code
    solution_code = best_solution.get("code")
    if not solution_code:
        click.echo(f"No code found in best solution for session {session_id}")
        return

    # Write to output file or print
    if output:
        with open(output, "w") as f:
            f.write(solution_code)
        click.echo(f"Best solution written to {output}")
    else:
        click.echo("\n=== Best Solution ===")
        click.echo(solution_code)

    click.echo(f"Score: {best_solution.get('score', 'Unknown')}")


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
    problem_id = os.path.splitext(os.path.basename(problem["path"]))[0]
    knowledge_base = KnowledgeBase(experiment_dir, problem_id=problem_id)
    problem_analyzer = ProblemAnalyzer(llm_client, experiment_config.get("analyzer"))
    solution_strategist = SolutionStrategist(llm_client, experiment_config.get("strategist"))
    evolutionary_engine = EvolutionaryEngine(llm_client, experiment_config.get("evolution"))
    implementation_debugger = ImplementationDebugger(llm_client, docker_manager, experiment_config.get("debugger"))
    problem_logic = ProblemLogic(llm_client, experiment_config.get("problem_logic"))

    # Parse problem statement
    problem_path = problem.get("path")
    with open(problem_path) as f:
        problem_text = f.read()

    # Create session
    session_id = knowledge_base.create_session(
        problem.get("name", "unknown"), {"problem_text": problem_text, "experiment_id": experiment_id}
    )

    # Analyze problem
    problem_analysis = await problem_analyzer.analyze(problem_text)
    knowledge_base.save_problem_analysis(session_id, problem_analysis)

    # Develop solution strategy
    solution_strategy = await solution_strategist.develop_strategy(problem_analysis)
    knowledge_base.save_solution_strategy(session_id, solution_strategy)

    # Generate initial solution
    initial_solution = await problem_logic.generate_initial_solution(problem_analysis)

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

    result = await evolutionary_engine.evolve(
        problem_analysis,
        solution_strategy,
        initial_solution,
        lambda code: evaluate_solution(code, test_cases, score_calculator),
        os.path.join(experiment_dir, "sessions", session_id),
    )

    duration = time.time() - start_time

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
        "best_score": result["best_score"],
        "generations": result["generations_completed"],
        "duration": duration,
        "session_id": session_id,
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
