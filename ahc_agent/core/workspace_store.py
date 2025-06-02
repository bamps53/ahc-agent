"""
Workspace store module for AHCAgent.

This module provides functionality for storing and retrieving workspace information.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ahc_agent.utils.file_io import ensure_directory, read_json, write_file, write_json

logger = logging.getLogger(__name__)


class WorkspaceStore:
    """
    Workspace store for storing and retrieving workspace information.
    """

    def __init__(self, workspace: str, problem_id: str):
        self.workspace_path = Path(workspace)
        self.problem_id = problem_id

        # Solutions directory
        self.solutions_dir = self.workspace_path / "solutions"
        ensure_directory(self.solutions_dir)

        # Logs directory
        self.logs_dir = self.workspace_path / "logs"
        ensure_directory(self.logs_dir)

        logger.info("Initialized workspace store")
        logger.debug(f"Workspace directory: {self.workspace_path}")

    def get_workspace_dir(self) -> Path:
        """Get the workspace directory path."""
        return self.workspace_path

    def save_problem_text(self, problem_text: str) -> bool:
        """Save problem text."""
        problem_path = self.workspace_path / "problem_text.md"
        try:
            with open(problem_path, "w") as f:
                f.write(problem_text)
        except OSError as e:
            logger.error(f"Error saving problem text: {e}")
            return False
        return True

    def load_problem_text(self) -> Optional[str]:
        """Load problem text."""
        problem_path = self.workspace_path / "problem_text.md"
        if not problem_path.exists():
            problem_path = self.workspace_path / "problem.md"  # 互換性のため
            if not problem_path.exists():
                logger.warning(f"Problem text not found at {problem_path}")
                return None

        try:
            with open(problem_path) as f:
                return f.read()
        except OSError as e:
            logger.error(f"Error loading problem text: {e}")
            return None

    def save_problem_analysis(self, analysis: Dict[str, Any]) -> bool:
        """
        Save problem analysis.

        Args:
            analysis: Problem analysis

        Returns:
            True if successful, False otherwise
        """
        analysis_path = self.workspace_path / "problem_analysis.json"
        try:
            write_json(analysis_path, analysis)
        except OSError as e:
            logger.error(f"Error saving problem analysis: {e}")
            return False

        logger.info(f"Saved problem analysis for problem {self.problem_id}")
        return True

    def load_problem_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Load problem analysis.

        Returns:
            Problem analysis or None if not found
        """
        analysis_path = self.workspace_path / "problem_analysis.json"
        if not analysis_path.exists():
            logger.warning(f"Problem analysis not found at {analysis_path}")
            return None

        try:
            return read_json(analysis_path)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading problem analysis: {e}")
            return None

    def get_problem_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Get problem analysis.

        Returns:
            Problem analysis or None if not found
        """
        return self.load_problem_analysis()

    def save_solution_strategy(self, strategy: Dict[str, Any]) -> bool:
        """
        Save solution strategy.

        Args:
            strategy: Solution strategy

        Returns:
            True if successful, False otherwise
        """
        strategy_path = self.workspace_path / "solution_strategy.json"
        try:
            write_json(strategy_path, strategy)
        except OSError as e:
            logger.error(f"Error saving solution strategy: {e}")
            return False

        logger.info(f"Saved solution strategy for problem {self.problem_id}")
        return True

    def load_solution_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Load solution strategy.

        Returns:
            Solution strategy or None if not found
        """
        strategy_path = self.workspace_path / "solution_strategy.json"
        if not strategy_path.exists():
            logger.warning(f"Solution strategy not found at {strategy_path}")
            return None

        try:
            return read_json(strategy_path)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading solution strategy: {e}")
            return None

    def get_solution_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Get solution strategy.

        Returns:
            Solution strategy or None if not found
        """
        return self.load_solution_strategy()

    def save_solution(self, solution_name: str, solution_code: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save a solution.

        Args:
            solution_name: Solution name (e.g., "initial", "best", "gen_X_id_Y")
            solution_code: Solution code
            metadata: Solution metadata

        Returns:
            True if successful, False otherwise
        """
        # Save solution code
        solution_path = self.solutions_dir / f"{solution_name}.cpp"
        try:
            write_file(solution_path, solution_code)
        except OSError as e:
            logger.error(f"Error saving solution {solution_name}: {e}")
            return False

        # Save solution metadata if provided
        if metadata:
            metadata_path = self.solutions_dir / f"{solution_name}_meta.json"
            try:
                write_json(metadata_path, metadata)
            except OSError as e:
                logger.error(f"Error saving solution metadata for {solution_name}: {e}")
                return False

        logger.info(f"Saved solution {solution_name} for problem {self.problem_id}")
        return True

    def load_solution_code(self, solution_name: str) -> Optional[str]:
        """
        Load solution code.

        Args:
            solution_name: Solution name

        Returns:
            Solution code or None if not found
        """
        solution_path = self.solutions_dir / f"{solution_name}.cpp"
        if not solution_path.exists():
            logger.warning(f"Solution {solution_name} not found at {solution_path}")
            return None

        try:
            with open(solution_path) as f:
                return f.read()
        except OSError as e:
            logger.error(f"Error loading solution {solution_name}: {e}")
            return None

    def load_solution_metadata(self, solution_name: str) -> Optional[Dict[str, Any]]:
        """
        Load solution metadata.

        Args:
            solution_name: Solution name

        Returns:
            Solution metadata or None if not found
        """
        metadata_path = self.solutions_dir / f"{solution_name}_meta.json"
        if not metadata_path.exists():
            logger.warning(f"Solution metadata for {solution_name} not found at {metadata_path}")
            return None

        try:
            return read_json(metadata_path)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading solution metadata for {solution_name}: {e}")
            return None

    def get_best_solution_code_and_meta(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the best solution code and metadata.

        Returns:
            Tuple of (solution_code, metadata) or None if not found
        """
        solution_code = self.load_solution_code("best")
        if not solution_code:
            logger.warning("Best solution code not found")
            return None

        metadata = self.load_solution_metadata("best")
        if not metadata:
            logger.warning("Best solution metadata not found")
            return None

        return solution_code, metadata

    def get_best_solution(self) -> Optional[Dict[str, Any]]:
        """
        Get the best solution with code and metadata.

        Returns:
            Dictionary with code and metadata or None if not found
        """
        result = self.get_best_solution_code_and_meta()
        if not result:
            return None

        solution_code, metadata = result
        return {"code": solution_code, **metadata}

    def save_evolution_log(self, evolution_log: Dict[str, Any]) -> bool:
        """
        Save evolution log.

        Args:
            evolution_log: Evolution log data

        Returns:
            True if successful, False otherwise
        """
        log_path = self.logs_dir / "evolution_log.json"
        try:
            write_json(log_path, evolution_log)
        except OSError as e:
            logger.error(f"Error saving evolution log: {e}")
            return False

        logger.info(f"Saved evolution log for problem {self.problem_id}")
        return True

    def load_evolution_log(self) -> Optional[Dict[str, Any]]:
        """
        Load evolution log.

        Returns:
            Evolution log data or None if not found
        """
        log_path = self.logs_dir / "evolution_log.json"
        if not log_path.exists():
            logger.warning(f"Evolution log not found at {log_path}")
            return None

        try:
            return read_json(log_path)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading evolution log: {e}")
            return None
