import asyncio
import logging
from pathlib import Path

import click

from .config import Config
from .core.knowledge import KnowledgeBase
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


@cli.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("workspace", type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option("--session-id", "-s", "session_id_option", type=str, help="Session ID for resuming a previous solve session.")
@click.option("--interactive", "-i", is_flag=True, help="Enable interactive mode for solving.")
@click.pass_context
def solve(ctx, workspace, session_id_option, interactive):
    """
    Solve a problem in the specified workspace.
    The workspace must contain 'problem.md' and 'config.yaml'.
    """
    workspace_path = Path(workspace)
    problem_file_path = workspace_path / "problem.md"
    config_file_path = workspace_path / "config.yaml"

    if not problem_file_path.exists():
        click.secho(f"Error: 'problem.md' not found in workspace '{workspace_path}'.", fg="red")
        ctx.exit(1)
    if not config_file_path.exists():
        click.secho(f"Error: 'config.yaml' not found in workspace '{workspace_path}'.", fg="red")
        ctx.exit(1)

    try:
        ws_config = Config(str(config_file_path))
        ws_config.set("workspace.base_dir", str(workspace_path))
    except Exception as e:
        click.secho(f"Error loading config from '{config_file_path}': {e}", fg="red")
        ctx.exit(1)

    llm_client = LLMClient(ws_config.get("llm", {}))
    docker_manager = DockerManager(ws_config.get("docker", {}))

    click.echo(f"Solving problem in workspace: {workspace_path}")
    click.echo(f"Using config: {ws_config.config_file_path}")

    try:
        with open(problem_file_path, encoding="utf-8") as f:
            problem_text = f.read()
    except Exception as e:
        click.secho(f"Error reading problem file '{problem_file_path}': {e}", fg="red")
        ctx.exit(1)

    problem_id_for_kb = ws_config.get("contest_id")
    if not problem_id_for_kb:
        problem_id_for_kb = workspace_path.name  # Fallback
        logger.warning(f"'contest_id' not found in {config_file_path}, using workspace name '{problem_id_for_kb}' as problem_id for KnowledgeBase.")
        ws_config.set("contest_id", problem_id_for_kb)  # Also update config for service if it relies on it

    try:
        knowledge_base = KnowledgeBase(str(workspace_path), problem_id=problem_id_for_kb)
        solve_service = SolveService(llm_client, docker_manager, ws_config, knowledge_base)
        asyncio.run(solve_service.run_solve_session(problem_text=problem_text, session_id=session_id_option, interactive=interactive))
    except Exception as e:
        click.secho(f"An error occurred during the solve process: {e}", fg="red")
        logger.exception("Unexpected error in solve command.")
        ctx.exit(1)


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
