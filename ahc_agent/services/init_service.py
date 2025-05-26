import yaml
import contextlib
from pathlib import Path
import logging
import os # For Path.cwd()

from ..config import Config
from ..utils.scraper import scrape_and_setup_problem

logger = logging.getLogger(__name__)

class InitService:
    def __init__(self, config: Config):
        self.config = config

    def initialize_project(self, contest_id: str, template: str = None, docker_image: str = None, workspace: str = None) -> dict:
        """
        Initialize a new AHC project.
        Optionally, provide a CONTEST_ID (e.g., ahc030) to scrape the problem statement.
        """
        # Override configuration with command-line options if they are provided
        if template:
            self.config.set("template", template)
        
        # If docker_image is provided as an argument, it overrides the config.
        # If not, the config value (which might be a default) is used.
        effective_docker_image = docker_image if docker_image else self.config.get("docker.image", "ubuntu:latest")
        if docker_image: # ensure the config is updated if a specific image is passed
            self.config.set("docker.image", docker_image)


        # Determine project directory
        if workspace:
            project_dir = Path(workspace).resolve()
        else:
            project_dir = Path(os.getcwd()) / contest_id # Use os.getcwd() for current working directory

        try:
            project_dir.mkdir(parents=True, exist_ok=False)
            display_path = project_dir
            try:
                display_path = project_dir.relative_to(Path.cwd())
            except ValueError:
                pass # Not a subpath of cwd, use absolute path
            logger.info(f"Initialized AHC project in ./{display_path}")

        except FileExistsError:
            err_msg = f"Error creating project directory: '{project_dir}' already exists and is a file or non-empty directory."
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        except Exception as e:
            err_msg = f"Error creating project directory '{project_dir}': {e}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        # Determine template to use: argument > config > default
        effective_template = template if template else self.config.get("template", "default")

        project_specific_config_data = {
            "contest_id": contest_id,
            "template": effective_template,
            "docker_image": effective_docker_image, # Use the effective docker image
        }

        project_config_file_path = project_dir / "ahc_config.yaml"
        try:
            with open(project_config_file_path, "w") as f:
                yaml.dump(project_specific_config_data, f, default_flow_style=False)
            logger.info(f"Project configuration saved to {project_config_file_path}")
        except Exception as e:
            err_msg = f"Error saving project configuration to '{project_config_file_path}': {e}"
            logger.error(err_msg)
            # Clean up created directory if config saving fails? For now, no.
            raise RuntimeError(err_msg)
        
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
            # Depending on requirements, this could also raise an error or return a specific status.
            # For now, just logging and continuing.

        return {
            "project_dir": str(project_dir),
            "config_file_path": str(project_config_file_path),
            "contest_id": contest_id,
            "template": effective_template,
            "docker_image": effective_docker_image
        }
