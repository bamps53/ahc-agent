"""
CLI entrypoint module for AHCAgent CLI.

This module provides the main CLI entrypoint for the AHCAgent CLI tool.
"""

# Standard library imports
import asyncio
import logging
import os  # For Path.cwd() in init if workspace is not provided
from pathlib import Path

# Third-party imports
import click

# import yaml # No longer directly used in cli.py for ahc_config.yaml or batch_config.yaml
# Local application/library specific imports
from .config import Config
from .core.knowledge import KnowledgeBase  # Still needed for some commands to instantiate
from .services.batch_service import BatchService

# from .utils.scraper import scrape_and_setup_problem # Moved to InitService
# Import services
from .services.init_service import InitService
from .services.solve_service import SolveService
from .services.status_service import StatusService
from .services.submit_service import SubmitService
from .utils.docker_manager import DockerManager
from .utils.llm import LLMClient
from .utils.logging import setup_logging

# Set up logger
logger = logging.getLogger(__name__)


# Create Click group
@click.group()
@click.option("--config", "-c", "config_path_option", type=click.Path(exists=True), help="Path to configuration file.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--quiet", "-q", is_flag=True, help="Minimize output.")
@click.option("--no-docker", is_flag=True, help="Disable Docker usage.")
@click.pass_context
def cli(ctx, config_path_option, verbose, quiet, no_docker):
    """
    AHCAgent CLI - A tool for solving AtCoder Heuristic Contest problems.
    """
    log_level = "DEBUG" if verbose else "ERROR" if quiet else "INFO"
    setup_logging(level=log_level)

    cfg = Config(config_path_option)  # config_path_option can be None, Config handles it

    if no_docker:
        cfg.set("docker.enabled", False)

    llm_client = LLMClient(cfg.get("llm", {}))  # Pass LLM config section
    docker_manager = DockerManager(cfg.get("docker", {}))  # Pass Docker config section

    ctx.obj = {"config": cfg, "llm_client": llm_client, "docker_manager": docker_manager}
    logger.debug("CLI initialized with services.")


@cli.command()
@click.option("--template", "-t", type=str, help="Template to use for the project.")
@click.option("--docker-image", "-i", type=str, help="Docker image to specify in project config.")
@click.option(
    "--workspace",
    "-w",
    type=click.Path(),
    default=None,
    help="Directory to create the project in. If not set, creates a directory named CONTEST_ID in the current location.",
)
@click.argument("contest_id", type=str, required=True)
@click.pass_context
def init(ctx, template, docker_image, workspace, contest_id):
    """
    Initialize a new AHC project.
    Scrapes problem statement for CONTEST_ID (e.g., ahc030).
    """
    config = ctx.obj["config"]
    init_service = InitService(config)

    try:
        # The InitService now handles overriding config with template/docker_image options if provided
        project_info = init_service.initialize_project(
            contest_id=contest_id,
            template=template,
            docker_image=docker_image,
            workspace=workspace,
        )
        click.echo(f"Project for contest '{project_info['contest_id']}' initialized successfully in '{project_info['project_dir']}'.")
        click.echo(f"Config file created at: {project_info['config_file_path']}")
    except RuntimeError as e:
        click.secho(f"Error during project initialization: {e}", fg="red")
        ctx.exit(1)
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg="red")
        logger.exception("Unexpected error in init command.")  # Log stack trace for unexpected errors
        ctx.exit(1)


@cli.command()
@click.argument("workspace", type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option("--session-id", "-s", "session_id_option", type=str, help="Session ID for resuming a previous solve session.")
@click.option("--time-limit", "-t", type=int, help="Override time limit in seconds for evolution.")
@click.option("--generations", "-g", type=int, help="Override maximum number of generations for evolution.")
@click.option("--population-size", "-p", type=int, help="Override population size for evolution.")
@click.option("--interactive", "-i", is_flag=True, help="Enable interactive mode for solving.")
@click.pass_context
def solve(ctx, workspace, session_id_option, time_limit, generations, population_size, interactive):
    """
    Solve a problem in the specified workspace.
    The workspace must contain 'problem.md' and 'ahc_config.yaml'.
    """
    llm_client = ctx.obj["llm_client"]
    docker_manager = ctx.obj["docker_manager"]
    # Global config is in ctx.obj["config"], but for solve, we load workspace-specific config.

    workspace_path = Path(workspace)
    problem_file_path = workspace_path / "problem.md"
    config_file_path = workspace_path / "ahc_config.yaml"

    if not problem_file_path.exists():
        click.secho(f"Error: 'problem.md' not found in workspace '{workspace_path}'.", fg="red")
        ctx.exit(1)
    if not config_file_path.exists():
        click.secho(f"Error: 'ahc_config.yaml' not found in workspace '{workspace_path}'.", fg="red")
        ctx.exit(1)

    try:
        # Load workspace-specific config for this solve session
        # This config instance will be passed to SolveService
        ws_config = Config(str(config_file_path))
        ws_config.set("workspace.base_dir", str(workspace_path))  # Ensure base_dir points to the active workspace
    except Exception as e:
        click.secho(f"Error loading config from '{config_file_path}': {e}", fg="red")
        ctx.exit(1)

    click.echo(f"Solving problem in workspace: {workspace_path}")
    click.echo(f"Using config: {ws_config.config_file_path}")

    # Override workspace config with command-line options for evolution parameters
    if time_limit is not None:
        ws_config.set("evolution.time_limit_seconds", time_limit)
    if generations is not None:
        ws_config.set("evolution.max_generations", generations)
    if population_size is not None:
        ws_config.set("evolution.population_size", population_size)

    try:
        with open(problem_file_path, encoding="utf-8") as f:
            problem_text = f.read()
    except Exception as e:
        click.secho(f"Error reading problem file '{problem_file_path}': {e}", fg="red")
        ctx.exit(1)

    # Determine problem_id for KnowledgeBase: from ws_config, or fallback to workspace directory name
    # The contest_id in ahc_config.yaml should be the definitive problem_id for KB.
    problem_id_for_kb = ws_config.get("contest_id")
    if not problem_id_for_kb:
        problem_id_for_kb = workspace_path.name  # Fallback
        logger.warning(f"'contest_id' not found in {config_file_path}, using workspace name '{problem_id_for_kb}' as problem_id for KnowledgeBase.")
        ws_config.set("contest_id", problem_id_for_kb)  # Also update config for service if it relies on it

    try:
        knowledge_base = KnowledgeBase(str(workspace_path), problem_id=problem_id_for_kb)
        solve_service = SolveService(llm_client, docker_manager, ws_config, knowledge_base)

        # run_solve_session handles both non-interactive and dispatching to interactive
        asyncio.run(solve_service.run_solve_session(problem_text=problem_text, session_id=session_id_option, interactive=interactive))
    except Exception as e:
        click.secho(f"An error occurred during the solve process: {e}", fg="red")
        logger.exception("Unexpected error in solve command.")
        ctx.exit(1)


@cli.command()
@click.argument("session_id_arg", metavar="SESSION_ID", required=False)  # Renamed to avoid clash
@click.option(
    "--workspace",
    "-w",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    help="Workspace directory path. If not provided, uses current directory.",
)
@click.option("--watch", is_flag=True, help="Watch status updates continuously.")
@click.pass_context
def status(ctx, session_id_arg, workspace, watch):
    """
    Show session status. If SESSION_ID is not provided, lists all sessions
    within the specified workspace context.
    """
    global_config = ctx.obj["config"]  # Use the global config for status context

    # Determine workspace_dir for KnowledgeBase
    # If workspace is provided, use it
    # Otherwise, try to use current directory
    if workspace:
        workspace_path = Path(workspace).resolve()
    else:
        # Use current directory as workspace
        workspace_path = Path(os.getcwd()).resolve()
        logger.debug(f"No workspace provided, using current directory: {workspace_path}")

    if not workspace_path.is_dir():
        click.secho(f"Error: Workspace directory '{workspace_path}' does not exist or is not a directory.", fg="red")
        ctx.exit(1)

    # Check if the workspace has ahc_config.yaml to determine problem_id
    problem_specific_config_path = workspace_path / "ahc_config.yaml"
    problem_id_for_kb = workspace_path.name  # Default/fallback

    if problem_specific_config_path.exists():
        try:
            problem_cfg = Config(str(problem_specific_config_path))
            cfg_contest_id = problem_cfg.get("contest_id")
            if cfg_contest_id:
                problem_id_for_kb = cfg_contest_id
                logger.debug(f"Using contest_id '{problem_id_for_kb}' from ahc_config.yaml")
            else:
                logger.warning(f"'contest_id' not found in {problem_specific_config_path}, using directory name '{problem_id_for_kb}' as problem_id.")
        except Exception as e:
            logger.warning(
                f"Could not load {problem_specific_config_path} to determine contest_id, using directory name '{problem_id_for_kb}'. Error: {e}"
            )
    else:
        logger.warning(f"No ahc_config.yaml found in {workspace_path}, using directory name '{problem_id_for_kb}' as problem_id.")

    try:
        # KnowledgeBase needs workspace path and problem_id.
        knowledge_base = KnowledgeBase(str(workspace_path), problem_id=problem_id_for_kb)
        status_service = StatusService(global_config, knowledge_base)  # Pass global_config

        status_lines = status_service.get_status(session_id=session_id_arg, watch=watch)

        # get_status already logs, but we can echo the final output here if not in watch mode
        # or if it's a summary. The service returns lines, so we print them.
        if not watch or not session_id_arg:  # Avoid redundant printing if watch is active for a single session
            for line in status_lines:
                click.echo(line)

    except ValueError as e:  # Catch errors like session not found
        click.secho(str(e), fg="red")
        ctx.exit(1)
    except Exception as e:
        click.secho(f"An unexpected error occurred while fetching status: {e}", fg="red")
        logger.exception("Unexpected error in status command.")
        ctx.exit(1)


@cli.command()
@click.argument("session_id_arg", metavar="SESSION_ID")  # Renamed
@click.option(
    "--workspace",
    "-w",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    help="Workspace directory path. If not provided, uses current directory.",
)
@click.option("--output", "-o", "output_path_option", type=click.Path(), help="Output file path for the solution code.")
@click.pass_context
def submit(ctx, session_id_arg, workspace, output_path_option):
    """
    Submit the best solution from a session.
    """
    global_config = ctx.obj["config"]  # Use global config for submit context

    # Determine workspace_dir for KnowledgeBase
    # If workspace is provided, use it
    # Otherwise, try to use current directory
    if workspace:
        workspace_path = Path(workspace).resolve()
    else:
        # Use current directory as workspace
        workspace_path = Path(os.getcwd()).resolve()
        logger.debug(f"No workspace provided, using current directory: {workspace_path}")

    if not workspace_path.is_dir():
        click.secho(f"Error: Workspace directory '{workspace_path}' does not exist or is not a directory.", fg="red")
        ctx.exit(1)

    # Determine problem_id for KnowledgeBase
    # If ahc_config.yaml exists in this workspace_path, try to get contest_id from it.
    problem_specific_config_path = workspace_path / "ahc_config.yaml"
    problem_id_for_kb = workspace_path.name  # Default/fallback

    if problem_specific_config_path.exists():
        try:
            problem_cfg = Config(str(problem_specific_config_path))
            cfg_contest_id = problem_cfg.get("contest_id")
            if cfg_contest_id:
                problem_id_for_kb = cfg_contest_id
                logger.debug(f"Using contest_id '{problem_id_for_kb}' from ahc_config.yaml")
            else:
                logger.warning(f"'contest_id' not found in {problem_specific_config_path}, using directory name '{problem_id_for_kb}' as problem_id.")
        except Exception as e:
            logger.warning(
                f"Could not load {problem_specific_config_path} to determine contest_id, using directory name '{problem_id_for_kb}'. Error: {e}"
            )
    else:
        logger.warning(f"No ahc_config.yaml found in {workspace_path}, using directory name '{problem_id_for_kb}' as problem_id.")

    try:
        knowledge_base = KnowledgeBase(str(workspace_path), problem_id=problem_id_for_kb)
        submit_service = SubmitService(global_config, knowledge_base)  # Pass global_config

        details = submit_service.submit_solution(session_id=session_id_arg, output_path=output_path_option)

        if details["output_path"] != "logged_to_console":
            click.echo(f"Best solution for session {details['session_id']} (Score: {details['score']}) written to {details['output_path']}")
        else:
            click.echo(f"Best solution for session {details['session_id']} (Score: {details['score']}):")
            # The service logs the code, so we don't need to print it again from details['solution_code']
            # unless explicit console print of code is desired here.
            # For now, relying on service logging for the code itself if not written to file.
            click.echo("The solution code has been logged.")

    except ValueError as e:  # Specific errors from service (e.g., session/solution not found)
        click.secho(str(e), fg="red")
        ctx.exit(1)
    except RuntimeError as e:  # E.g., file writing error
        click.secho(str(e), fg="red")
        ctx.exit(1)
    except Exception as e:
        click.secho(f"An unexpected error occurred during submission: {e}", fg="red")
        logger.exception("Unexpected error in submit command.")
        ctx.exit(1)


@cli.command()
@click.argument("batch_config_path", metavar="BATCH_CONFIG_FILE", type=click.Path(exists=True, dir_okay=False))
@click.option("--parallel", "-p", "parallel_override", type=int, help="Number of parallel experiment executions (overrides batch config).")
@click.option(
    "--output-dir",
    "-o",
    "output_dir_override",
    type=click.Path(file_okay=False, writable=True),
    help="Output directory for batch results (overrides batch config).",
)
@click.pass_context
def batch(ctx, batch_config_path, parallel_override, output_dir_override):
    """
    Run batch processing of experiments defined in BATCH_CONFIG_FILE.
    """
    global_config = ctx.obj["config"]  # Global config can provide defaults for batch service
    llm_client = ctx.obj["llm_client"]
    docker_manager = ctx.obj["docker_manager"]

    batch_service = BatchService(llm_client, docker_manager, global_config)

    try:
        click.echo(f"Starting batch processing with config: {batch_config_path}")
        if output_dir_override:
            click.echo(f"Output directory override: {output_dir_override}")
        if parallel_override is not None:
            click.echo(f"Parallel executions override: {parallel_override}")

        results = asyncio.run(
            batch_service.run_batch_experiments_service(
                batch_config_path=batch_config_path,
                output_dir_override=output_dir_override,
                parallel_override=parallel_override,
            )
        )

        click.echo("\nBatch processing completed.")
        # Batch service already logs summary, but we can print a high-level summary here too.
        successful_experiments = sum(1 for r in results if not r.get("error"))
        failed_experiments = len(results) - successful_experiments
        click.echo(f"Total experiments processed: {len(results)}")
        click.echo(f"  Successful: {successful_experiments}")
        click.echo(f"  Failed: {failed_experiments}")
        if results:  # If there are any results (even errors)
            Path(results[0].get("experiment_dir_path", output_dir_override or ".")).parent / "summary.json"
            # Attempt to get the actual summary path from service if possible, or construct best guess
            # The service logs where summary.json is saved.
            click.echo("Detailed results and summary.json are typically in the batch output directory (see logs).")

    except Exception as e:
        click.secho(f"An error occurred during batch processing: {e}", fg="red")
        logger.exception("Unexpected error in batch command.")
        ctx.exit(1)


# Placeholder for 'stop' command if it's to be refactored or kept
@cli.command()
@click.argument("session_id")
@click.pass_context
def stop(ctx, session_id):
    """
    Stop a running session (Placeholder - TBD if part of a service or direct KB interaction).
    For now, this directly interacts with KnowledgeBase status update.
    """
    config = ctx.obj["config"]  # Global config
    workspace_dir_str = config.get("workspace.base_dir")
    if not workspace_dir_str:
        click.secho("Error: 'workspace.base_dir' not defined. Cannot determine context.", fg="red")
        ctx.exit(1)

    workspace_path = Path(workspace_dir_str).resolve()
    problem_id_for_kb = workspace_path.name  # Assuming base_dir is problem specific

    # Similar to 'submit', determine problem_id more robustly if possible
    problem_specific_config_path = workspace_path / "ahc_config.yaml"
    if problem_specific_config_path.exists():
        try:
            problem_cfg = Config(str(problem_specific_config_path))
            try:
                cfg_contest_id = problem_cfg.get("contest_id")
                if cfg_contest_id:
                    problem_id_for_kb = cfg_contest_id
            except Exception as e:
                logger.debug(f"Failed to load contest ID from config: {e}")  # Log exception instead of silent pass
        except Exception as e:
            logger.debug(f"Failed to load problem config: {e}")  # Log exception instead of silent pass

    try:
        knowledge_base = KnowledgeBase(str(workspace_path), problem_id=problem_id_for_kb)
        session = knowledge_base.get_session(session_id)
        if not session:
            click.secho(f"Session {session_id} not found in problem context '{problem_id_for_kb}'.", fg="red")
            ctx.exit(1)

        knowledge_base.update_session(session_id, {"status": "stopped"})
        click.echo(f"Session {session_id} marked as stopped.")
    except Exception as e:
        click.secho(f"Error stopping session {session_id}: {e}", fg="red")
        logger.exception(f"Error in stop command for session {session_id}.")
        ctx.exit(1)


# Commands like 'config' and 'docker' are more meta-operations or direct interactions
# with Config/DockerManager, not necessarily fitting into the problem-solving services.
# These can be refactored later if a pattern emerges (e.g., a ConfigManagementService).


@cli.group()
@click.pass_context
def config(ctx):  # Renamed variable to avoid conflict with 'config' module
    """Manage agent configuration."""


@config.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx, key):
    """Get a configuration value."""
    app_config = ctx.obj["config"]  # Renamed variable
    value = app_config.get(key)
    if value is None:
        click.echo(f"Configuration key '{key}' not found.")
    else:
        click.echo(f"{key} = {value}")


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """Set a configuration value (in memory, for current session)."""
    app_config = ctx.obj["config"]  # Renamed variable
    # Basic type conversion, similar to original
    if value.lower() == "true":
        value_to_set = True
    elif value.lower() == "false":
        value_to_set = False
    elif value.isdigit():
        value_to_set = int(value)
    elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
        value_to_set = float(value)
    else:
        value_to_set = value

    try:
        app_config.set(key, value_to_set)
        click.echo(f"Set {key} = {value_to_set} (in memory for this session).")
        # Note: This doesn't save to the config file unless explicitly done.
        # The original Config class might need a save method to persist these.
    except Exception as e:
        click.secho(f"Error setting configuration: {e}", fg="red")


# Docker command group remains, uses DockerManager from ctx.obj
@cli.group()
@click.pass_context
def docker(ctx):
    """Manage Docker environment for AHC."""


@docker.command("setup")
@click.pass_context
def docker_setup(ctx):
    """Set up Docker environment (e.g., pull image)."""
    docker_manager = ctx.obj["docker_manager"]
    click.echo(f"Attempting to pull Docker image: {docker_manager.image_name}")  # Corrected to image_name
    try:
        # DockerManager.pull_image() was synchronous in the original.
        # If it becomes async, this needs `asyncio.run()`. Assuming sync for now.
        success = docker_manager.pull_image()
        if success:
            click.echo("Docker image pulled successfully (or was already present).")
        else:
            # DockerManager pull_image should ideally raise error on failure or return detailed status
            click.secho("Failed to pull Docker image. Check logs for details.", fg="red")
    except Exception as e:
        click.secho(f"Error during Docker setup: {e}", fg="red")
        logger.exception("Error in docker setup command.")


@docker.command("status")
@click.pass_context
def docker_status(ctx):
    """Show Docker environment status."""
    docker_manager = ctx.obj["docker_manager"]
    try:
        docker_manager.check_docker_availability()  # This raises RuntimeError if unavailable
        click.echo("Docker is available.")

        # Test command execution
        # Assuming current directory is a safe workspace for test.
        # DockerManager.run_command() was synchronous.
        result = docker_manager.run_command("echo 'Docker test successful'", os.getcwd())
        if result and result.get("success"):  # Check if result is not None
            click.echo(f"Docker test command successful. Output: {result.get('stdout', '').strip()}")
        elif result:
            click.secho(f"Docker test command failed. Error: {result.get('stderr', 'Unknown error')}", fg="red")
        else:
            click.secho("Docker test command did not return expected result.", fg="red")

    except RuntimeError as e:
        click.secho(f"Docker is not available: {e}", fg="red")
    except Exception as e:
        click.secho(f"An unexpected error occurred while checking Docker status: {e}", fg="red")
        logger.exception("Error in docker status command.")


def main():
    """
    Main entry point.
    """
    cli(obj={})  # obj is initialized by Click context


if __name__ == "__main__":
    main()
