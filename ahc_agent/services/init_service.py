import contextlib
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from ahc_agent.utils.scraper import scrape_and_setup_problem

logger = logging.getLogger(__name__)


class InitService:
    def __init__(self):
        pass

    def initialize_project(self, contest_id: str, workspace: Optional[str] = None) -> dict:
        """
        Initialize a new AHC project.
        Optionally, provide a CONTEST_ID (e.g., ahc030) to scrape the problem statement.
        """

        # Determine project directory
        project_dir = Path(workspace).resolve() if workspace else Path(os.getcwd()) / contest_id

        try:
            project_dir.mkdir(parents=True, exist_ok=False)
            display_path = project_dir
            with contextlib.suppress(ValueError):
                display_path = project_dir.relative_to(Path.cwd())
            logger.info(f"Initialized AHC project in ./{display_path}")

        except FileExistsError as err:
            err_msg = f"Error creating project directory: '{project_dir}' already exists and is a file or non-empty directory."
            logger.error(err_msg)
            raise RuntimeError(err_msg) from err
        except Exception as e:
            err_msg = f"Error creating project directory '{project_dir}': {e}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

        project_specific_config_data = {
            "contest_id": contest_id,
        }

        project_config_file_path = project_dir / "config.yaml"
        try:
            with open(project_config_file_path, "w") as f:
                yaml.dump(project_specific_config_data, f, default_flow_style=False)
            logger.info(f"Project configuration saved to {project_config_file_path}")
        except Exception as e:
            err_msg = f"Error saving project configuration to '{project_config_file_path}': {e}"
            logger.error(err_msg)
            # Clean up created directory if config saving fails? For now, no.
            raise RuntimeError(err_msg) from e

        # Scrape problem statement
        # It's important that contest_id is required for this service method.
        problem_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
        logger.info(f"Attempting to scrape problem from {problem_url}...")
        try:
            scrape_and_setup_problem(problem_url, str(project_dir))
            logger.info(f"Problem scraped and set up successfully in '{project_dir}'.")
        except Exception as e:
            # Non-fatal, project is initialized but scraping failed.
            logger.error(f"Error during scraping: {e}. Project initialized but problem scraping failed.")

        return {
            "project_dir": str(project_dir),
            "config_file_path": str(project_config_file_path),
            "contest_id": contest_id,
        }
