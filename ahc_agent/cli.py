import asyncio
import logging
from pathlib import Path

import click

from .config import Config
from .core.workspace_store import WorkspaceStore
from .services.init_service import InitService
from .services.solve_service import SolveService
from .utils.docker_manager import DockerManager
from .utils.llm import LLMClient
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--quiet", "-q", is_flag=True, help="Minimize output.")
@click.pass_context
def cli(ctx, verbose, quiet):
    """
    AHCAgent - A tool for solving AtCoder Heuristic Contest problems.
    """
    log_level = "DEBUG" if verbose else "ERROR" if quiet else "INFO"
    setup_logging(level=log_level)

    ctx.obj = {}


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--workspace",
    "-w",
    type=click.Path(),
    default=None,
    help="Directory to create the project in. If not set, creates a directory named CONTEST_ID in the current location.",
)
@click.option(
    "--html",
    "html_file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to a local HTML file containing the problem statement.",
)
@click.argument("contest_id", type=str, required=True)
@click.pass_context
def init(ctx, workspace, html_file, contest_id):
    """
    Initialize a new AHC project.
    Scrapes problem statement for CONTEST_ID (e.g., ahc030).
    """
    init_service = InitService()

    try:
        problem_url = None
        if not html_file and contest_id:
            problem_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"

        project_info = init_service.initialize_project(contest_id=contest_id, workspace=workspace, html_file=html_file, url=problem_url)
        click.echo(f"Project for contest '{project_info['contest_id']}' initialized successfully in '{project_info['project_dir']}'.")
        click.echo(f"Config file created at: {project_info['config_file_path']}")
    except RuntimeError as e:
        click.secho(f"Error during project initialization: {e}", fg="red")
        ctx.exit(1)
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg="red")
        logger.exception("Unexpected error in init command.")  # Log stack trace for unexpected errors
        ctx.exit(1)


@cli.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("workspace", type=click.Path(file_okay=False, dir_okay=True, resolve_path=True))
@click.option("--interactive", "-i", is_flag=True, help="Enable interactive mode for solving (if no subcommand is given).")
@click.pass_context
def solve(ctx, workspace, interactive):
    """
    Solve a problem or run specific solution steps in the workspace.
    The workspace must contain 'problem.md' and 'config.yaml'.
    If called without subcommands, runs the full solve process.
    """
    workspace_path = Path(workspace)
    problem_file_path = workspace_path / "problem.md"
    config_file_path = workspace_path / "config.yaml"

    # This check applies if 'solve' is called directly or if a subcommand is called.
    # For subcommands, we might need to adjust if problem.md is not always required.
    if not workspace_path.exists():
        click.secho(f"Error: Workspace directory '{workspace_path}' not found.", fg="red")
        ctx.exit(1)

    if not problem_file_path.exists():
        click.secho(f"Error: 'problem.md' not found in workspace '{workspace_path}'.", fg="red")
        ctx.exit(1)
    if not config_file_path.exists():
        click.secho(f"Error: 'config.yaml' not found in workspace '{workspace_path}'.", fg="red")
        ctx.exit(1)

    try:
        ws_config = Config(str(config_file_path))
        ws_config.set("workspace.base_dir", str(workspace_path))  # Ensure base_dir is absolute path
    except Exception as e:
        click.secho(f"Error loading config from '{config_file_path}': {e}", fg="red")
        ctx.exit(1)

    try:
        with open(problem_file_path, encoding="utf-8") as f:
            problem_text = f.read()
    except Exception as e:
        click.secho(f"Error reading problem file '{problem_file_path}': {e}", fg="red")
        ctx.exit(1)

    problem_id_for_kb = ws_config.get("contest_id")
    if not problem_id_for_kb:
        problem_id_for_kb = workspace_path.name  # Fallback
        logger.warning(f"'contest_id' not found in {config_file_path}, using workspace name '{problem_id_for_kb}' as problem_id.")
        ws_config.set("contest_id", problem_id_for_kb)

    llm_client = LLMClient(ws_config.get("llm", {}))
    docker_manager = DockerManager(ws_config.get("docker", {}))
    workspace_store = WorkspaceStore(str(workspace_path), problem_id=problem_id_for_kb)
    solve_service = SolveService(llm_client, docker_manager, ws_config, workspace_store)

    ctx.obj["workspace_path"] = workspace_path
    ctx.obj["problem_text"] = problem_text
    ctx.obj["ws_config"] = ws_config
    ctx.obj["llm_client"] = llm_client
    ctx.obj["docker_manager"] = docker_manager
    ctx.obj["workspace_store"] = workspace_store
    ctx.obj["solve_service"] = solve_service
    ctx.obj["interactive_flag"] = interactive  # Store interactive flag for subcommands if needed

    if ctx.invoked_subcommand is None:
        click.echo(f"Running full solve process in workspace: {workspace_path}")
        click.echo(f"Using config: {ws_config.config_file_path}")
        try:
            if interactive:
                # The run_interactive_solve method in SolveService now handles its own data loading
                # based on problem_text_initial.
                asyncio.run(solve_service.run_interactive_solve(problem_text_initial=problem_text))
            else:
                asyncio.run(solve_service.run_solve(problem_text=problem_text, interactive=False))
        except Exception as e:
            click.secho(f"An error occurred during the solve process: {e}", fg="red")
            logger.exception("Unexpected error in solve command (full run).")
            ctx.exit(1)


@solve.command(name="analyze", help="Run only the problem analysis step.")
@click.pass_context
def analyze_step(ctx):
    problem_text = ctx.obj["problem_text"]
    solve_service = ctx.obj["solve_service"]
    workspace_path = ctx.obj["workspace_path"]
    ws_store = ctx.obj["workspace_store"]  # type: WorkspaceStore

    click.echo(f"Running analysis for problem in workspace: {workspace_path}")
    try:
        analysis_result = asyncio.run(solve_service.run_analyze_step(problem_text=problem_text))
        if analysis_result:
            analysis_file_path = ws_store.get_problem_analysis_filepath()
            click.secho(f"Analysis complete. Results saved in '{analysis_file_path}'.", fg="green")
        else:
            click.secho("Analysis step did not return results or failed.", fg="yellow")
    except Exception as e:
        click.secho(f"An error occurred during the analysis step: {e}", fg="red")
        logger.exception(f"Analyze step error for {workspace_path}")
        ctx.exit(1)


@solve.command(name="strategy", help="Run only the solution strategy development step.")
@click.pass_context
def strategy_step(ctx):
    solve_service = ctx.obj["solve_service"]  # type: SolveService
    workspace_path = ctx.obj["workspace_path"]  # type: Path
    ws_store = ctx.obj["workspace_store"]  # type: WorkspaceStore

    analysis_file_path = ws_store.get_problem_analysis_filepath()
    if not analysis_file_path.exists():
        click.secho(
            f"Error: Problem analysis file ('{ws_store.get_problem_analysis_filepath().name}') not found in '{workspace_path}'. "
            "Please run the 'analyze' step first.",
            fg="red",
        )
        ctx.exit(1)

    click.echo(f"Running strategy development for problem in workspace: {workspace_path}")
    try:
        strategy_result = asyncio.run(solve_service.run_strategy_step())
        if strategy_result:
            strategy_file_path = ws_store.get_solution_strategy_filepath()
            click.secho(f"Strategy development complete. Results saved in '{strategy_file_path}'.", fg="green")
        else:
            click.secho("Strategy step did not return results or failed (possibly due to missing analysis).", fg="yellow")
    except Exception as e:
        click.secho(f"An error occurred during the strategy step: {e}", fg="red")
        logger.exception(f"Strategy step error for {workspace_path}")
        ctx.exit(1)


@solve.command(name="testcases", help="Generate or load test cases.")
@click.option("--load-tools", is_flag=True, default=False, help="Attempt to load test cases from 'tools/in/' directory first.")
@click.option("--force-generate", is_flag=True, default=False, help="Force generation of new test cases, ignoring --load-tools if also set.")
@click.option("--num-cases", type=int, default=3, show_default=True, help="Number of test cases to generate if generation occurs.")
@click.pass_context
def testcases_step(ctx, load_tools, force_generate, num_cases):
    solve_service = ctx.obj["solve_service"]  # type: SolveService
    workspace_path = ctx.obj["workspace_path"]  # type: Path
    ws_store = ctx.obj["workspace_store"]  # type: WorkspaceStore

    analysis_file_path = ws_store.get_problem_analysis_filepath()
    if not analysis_file_path.exists():
        click.secho(
            f"Error: Problem analysis file ('{ws_store.get_problem_analysis_filepath().name}') not found in '{workspace_path}'. "
            "Please run the 'analyze' step first.",
            fg="red",
        )
        ctx.exit(1)

    service_should_try_load = False
    if force_generate:
        service_should_try_load = False  # Generation is the primary goal
        click.echo(f"Forcing generation of {num_cases} new test cases for problem in workspace: {workspace_path}")
    elif load_tools:
        service_should_try_load = True
        click.echo(f"Attempting to load test cases from 'tools/in/' for problem in workspace: {workspace_path}")
        click.echo(f"If loading fails or no files are found, {num_cases} new test cases will be generated.")
    else:  # Default is to generate if neither flag is set explicitly to guide
        service_should_try_load = False  # Fallback to generation
        click.echo(f"Attempting to generate {num_cases} new test cases for problem in workspace: {workspace_path}")

    try:
        # run_testcases_step returns a dict with "test_cases" and "score_calculator"
        # These are not saved to WorkspaceStore by run_testcases_step itself.
        # For CLI invocation, we mainly report success/failure and count.
        test_data = asyncio.run(solve_service.run_testcases_step(load_from_tools=service_should_try_load, num_to_generate=num_cases))

        if test_data and test_data.get("test_cases"):
            click.secho(f"{len(test_data['test_cases'])} test cases are now prepared.", fg="green")
            if test_data.get("score_calculator"):
                click.secho("Score calculator also prepared.", fg="green")
            # Note: The actual test case data and calculator are not saved to files by this CLI command directly.
            # They are held in memory by SolveService if run_interactive_solve is orchestrating.
            # For individual CLI step, this merely confirms generation/loading.
        elif test_data and "test_cases" in test_data:  # test_cases might be an empty list
            click.secho("Test case step ran, but no test cases were loaded or generated.", fg="yellow")
        else:
            click.secho("Test cases step failed to produce results or encountered an issue.", fg="yellow")

    except Exception as e:
        click.secho(f"An error occurred during the test cases step: {e}", fg="red")
        logger.exception(f"Test cases step error for {workspace_path}")
        ctx.exit(1)


@solve.command(name="initial", help="Run only the initial solution generation step.")
@click.pass_context
def initial_step(ctx):
    solve_service = ctx.obj["solve_service"]  # type: SolveService
    workspace_path = ctx.obj["workspace_path"]  # type: Path
    ws_store = ctx.obj["workspace_store"]  # type: WorkspaceStore

    analysis_file_path = ws_store.get_problem_analysis_filepath()
    if not analysis_file_path.exists():
        click.secho(
            f"Error: Problem analysis file ('{ws_store.get_problem_analysis_filepath().name}') not found in '{workspace_path}'. "
            "Please run the 'analyze' step first.",
            fg="red",
        )
        ctx.exit(1)

    click.echo(f"Running initial solution generation for problem in workspace: {workspace_path}")
    try:
        initial_code_result = asyncio.run(solve_service.run_initial_solution_step())
        if initial_code_result:
            # run_initial_solution_step saves to KB. We can refer to that.
            # e.g. initial_solution_kb_entry = ws_store.get_solution("initial")
            click.secho("Initial solution generated and saved in the Knowledge Base.", fg="green")

            # Optional: Offer to show or save the code to a specific file
            # For now, let's just confirm it was generated and stored by the service.
            # If user wants the code, they can check the KB or it might be printed by the service's logger.
        else:
            click.secho("Initial solution generation failed or did not produce code.", fg="yellow")
    except Exception as e:
        click.secho(f"An error occurred during the initial solution generation step: {e}", fg="red")
        logger.exception(f"Initial solution step error for {workspace_path}")
        ctx.exit(1)


@solve.command(name="evolve", help="Run the evolutionary optimization process.")
@click.option("--generations", type=int, default=None, help="Number of generations. Overrides config.")
@click.option("--population", type=int, default=None, help="Population size. Overrides config.")
@click.option("--time-limit", type=int, default=None, help="Time limit in seconds. Overrides config.")
@click.option(
    "--initial-code-path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to a file with initial code for evolution (optional).",
)
@click.pass_context
def evolve_step(ctx, generations, population, time_limit, initial_code_path):
    solve_service = ctx.obj["solve_service"]  # type: SolveService
    workspace_path = ctx.obj["workspace_path"]  # type: Path
    ws_store = ctx.obj["workspace_store"]  # type: WorkspaceStore
    ws_config = ctx.obj["ws_config"]  # type: Config

    # Check prerequisites: analysis and strategy files must exist
    analysis_file_path = ws_store.get_problem_analysis_filepath()
    strategy_file_path = ws_store.get_solution_strategy_filepath()

    if not analysis_file_path.exists():
        click.secho(
            f"Error: Problem analysis file ('{ws_store.get_problem_analysis_filepath().name}') not found in '{workspace_path}'. "
            "Please run the 'analyze' step first.",
            fg="red",
        )
        ctx.exit(1)
    if not strategy_file_path.exists():
        click.secho(
            f"Error: Solution strategy file ('{ws_store.get_solution_strategy_filepath().name}') not found in '{workspace_path}'. "
            "Please run the 'strategy' step first.",
            fg="red",
        )
        ctx.exit(1)

    click.echo(f"Preparing for evolution process in workspace: {workspace_path}")

    # Prepare test cases and score calculator
    # For CLI, we'll default to trying to load from tools/in, then generating 3 if not found or if tools empty.
    # This makes the 'evolve' CLI command more usable standalone after 'analyze' and 'strategy'.
    click.echo("Ensuring test cases and score calculator are available...")
    test_data_result = asyncio.run(solve_service.run_testcases_step(load_from_tools=True, num_to_generate=3))

    if not test_data_result or not test_data_result.get("test_cases") or not test_data_result.get("score_calculator"):
        click.secho("Failed to prepare test cases or score calculator, which are essential for evolution. Aborting.", fg="red")
        ctx.exit(1)

    prepared_test_cases = test_data_result["test_cases"]
    prepared_score_calculator = test_data_result["score_calculator"]
    click.secho(f"Successfully prepared/loaded {len(prepared_test_cases)} test cases and a score calculator.", fg="cyan")

    initial_code_override = None
    if initial_code_path:
        try:
            with open(initial_code_path, encoding="utf-8") as f:
                initial_code_override = f.read()
            click.echo(f"Using initial code for evolution from: {initial_code_path}")
        except Exception as e:
            click.secho(f"Error reading initial code file '{initial_code_path}': {e}", fg="red")
            ctx.exit(1)

    # Evolution parameters: use CLI options if provided, otherwise SolveService will use its config defaults.
    # The service's run_evolve_step is designed to take these as direct int values.
    # We fetch from config here mainly for the click.echo status message.
    evo_config_defaults = ws_config.get("evolution", {})
    effective_generations = generations if generations is not None else evo_config_defaults.get("max_generations", 30)
    effective_population = population if population is not None else evo_config_defaults.get("population_size", 10)
    effective_time_limit = time_limit if time_limit is not None else evo_config_defaults.get("time_limit_seconds", 1800)

    click.echo(
        f"Starting evolutionary process with parameters: "
        f"Generations={effective_generations}, Population={effective_population}, Time Limit (s)={effective_time_limit}"
    )
    if initial_code_override:
        click.echo("An initial code override is being used.")
    else:
        click.echo("No initial code override; service will use best from KB, then initial from KB, or generate new.")

    try:
        evolution_result = asyncio.run(
            solve_service.run_evolve_step(
                test_cases=prepared_test_cases,
                score_calculator=prepared_score_calculator,
                max_generations=effective_generations,  # Pass effective_... which has CLI or config value
                population_size=effective_population,
                time_limit_seconds=effective_time_limit,
                initial_code_override=initial_code_override,
            )
        )
        if evolution_result and "best_solution" in evolution_result:  # Check key existence
            best_score = evolution_result.get("best_score", "N/A")
            gens_completed = evolution_result.get("generations_completed", "N/A")
            click.secho(f"Evolution complete. Best score achieved: {best_score} (completed {gens_completed} generations).", fg="green")
            # The service saves the best solution to KB and a file.
            click.secho(f"Best solution saved in Knowledge Base. Check '{ws_store.solutions_dir}' for output files.", fg="green")
        else:
            click.secho("Evolution step completed but did not return the expected result structure or failed.", fg="yellow")
    except Exception as e:
        click.secho(f"An error occurred during the evolution step: {e}", fg="red")
        logger.exception(f"Evolution step error for {workspace_path}")
        ctx.exit(1)


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
