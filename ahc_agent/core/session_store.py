"""
Session store module for AHCAgent.

This module provides functionality for storing and retrieving session information.
"""

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

from ahc_agent.utils.file_io import ensure_directory, read_json, write_file, write_json

logger = logging.getLogger(__name__)


class SessionStore:
    """
    Session store for storing and retrieving session information.
    """

    def __init__(self, workspace: str, problem_id: str, _session_store: Optional[dict] = None):
        self.workspace = Path(workspace)
        self.problem_id = problem_id

        # Sessions directory
        self.sessions_dir = self.workspace / "sessions"
        ensure_directory(self.sessions_dir)

        # Solutions directory
        self.solutions_dir = self.workspace / "solutions"
        ensure_directory(self.solutions_dir)

        # Experiments directory
        self.experiments_dir = self.workspace / "experiments"
        ensure_directory(self.experiments_dir)

        logger.info("Initialized session store")
        logger.debug(f"Workspace directory: {self.workspace}")

    def get_session_dir(self, session_id: str) -> Path:
        """Get the directory path for a given session."""
        return self.sessions_dir / session_id

    def create_session(self, problem_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new session.

        Args:
            problem_id: Problem ID
            metadata: Session metadata

        Returns:
            Session ID
        """
        # Generate session ID
        import uuid

        session_id = str(uuid.uuid4())[:8]

        # Create session directory
        session_dir = self.get_session_dir(session_id)
        ensure_directory(session_dir)

        # Create session metadata
        session_metadata = {
            "session_id": session_id,
            "problem_id": problem_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": "created",
            "metadata": {},  # 明示的にmetadataフィールドを追加
        }
        if metadata:
            session_metadata["metadata"] = metadata  # metadataを正しく保存

        # Save session metadata
        metadata_path = session_dir / "metadata.json"
        try:
            write_json(metadata_path, session_metadata)
        except OSError as e:
            logger.error(f"Error saving session metadata: {e}")

        logger.info(f"Created session {session_id} for problem {problem_id}")

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata.

        Args:
            session_id: Session ID

        Returns:
            Session metadata or None if not found
        """
        # Check if session exists
        session_dir = self.get_session_dir(session_id)
        if not session_dir.exists():
            logger.warning(f"Session directory {session_dir} not found")
            return None

        # Load session metadata
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"Session {session_id} metadata not found")
            return None

        try:
            return read_json(metadata_path)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading session {session_id} metadata: {e}")
            return None

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session metadata.

        Args:
            session_id: Session ID
            updates: Metadata updates

        Returns:
            True if successful, False otherwise
        """
        # Get current metadata
        metadata = self.get_session(session_id)
        if not metadata:
            return False

        # Update metadata
        metadata.update(updates)
        metadata["updated_at"] = time.time()

        # Save updated metadata
        metadata_path = self.get_session_dir(session_id) / "metadata.json"
        try:
            write_json(metadata_path, metadata)
        except OSError as e:
            logger.error(f"Error updating session {session_id} metadata: {e}")
            return False

        logger.info(f"Updated session {session_id} metadata")
        return True

    def save_problem_analysis(self, session_id: str, analysis: Dict[str, Any]) -> bool:
        """
        Save problem analysis.

        Args:
            session_id: Session ID
            analysis: Problem analysis

        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        session_dir = self.get_session_dir(session_id)
        if not session_dir.exists():
            logger.warning(f"Session directory {session_dir} not found")
            return False

        # Save analysis
        analysis_path = session_dir / "problem_analysis.json"
        try:
            write_json(analysis_path, analysis)
        except OSError as e:
            logger.error(f"Error saving problem analysis for session {session_id}: {e}")
            return False

        logger.info(f"Saved problem analysis for session {session_id}")

        # Update session metadata
        self.update_session(session_id, {"has_problem_analysis": True})

        return True

    def load_problem_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load problem analysis for a given session."""
        session_dir = self.get_session_dir(session_id)
        analysis_path = session_dir / "problem_analysis.json"

        if not analysis_path.exists():
            logger.warning(f"Problem analysis {analysis_path} not found for session {session_id}")
            return None

        try:
            analysis = read_json(analysis_path)
            logger.info(f"Loaded problem analysis for session {session_id}")
            return analysis
        except OSError as e:
            logger.error(f"Error loading problem analysis for session {session_id}: {e}")
            return None

    def get_problem_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get problem analysis.

        Args:
            session_id: Session ID

        Returns:
            Problem analysis or None if not found
        """
        return self.load_problem_analysis(session_id)

    def save_solution_strategy(self, session_id: str, strategy: Dict[str, Any]) -> bool:
        """
        Save solution strategy.

        Args:
            session_id: Session ID
            strategy: Solution strategy

        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found")
            return False

        # Save strategy
        strategy_path = os.path.join(session_dir, "solution_strategy.json")
        try:
            write_json(strategy_path, strategy)
        except OSError as e:
            logger.error(f"Error saving solution strategy for session {session_id}: {e}")
            return False

        logger.info(f"Saved solution strategy for session {session_id}")

        # Update session metadata
        self.update_session(session_id, {"has_solution_strategy": True})

        return True

    def get_solution_strategy(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get solution strategy.

        Args:
            session_id: Session ID

        Returns:
            Solution strategy or None if not found
        """
        # Check if session exists
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found")
            return None

        # Load strategy
        strategy_path = os.path.join(session_dir, "solution_strategy.json")
        if not os.path.exists(strategy_path):
            logger.warning(f"Solution strategy for session {session_id} not found")
            return None

        try:
            return read_json(strategy_path)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading solution strategy for session {session_id}: {e}")
            return None

    def save_solution(self, session_id: str, solution_id: str, solution: Dict[str, Any]) -> bool:
        """
        Save a solution.

        Args:
            session_id: Session ID
            solution_id: Solution ID
            solution: Solution data

        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found")
            return False

        # Create solutions directory for session
        solutions_dir = os.path.join(session_dir, "solutions")
        ensure_directory(solutions_dir)

        # Save solution
        solution_path = os.path.join(solutions_dir, f"{solution_id}.json")
        try:
            write_json(solution_path, solution)
        except OSError as e:
            logger.error(f"Error saving solution {solution_id} for session {session_id}: {e}")
            return False

        # Save solution code separately
        if "code" in solution:
            code_path = os.path.join(solutions_dir, f"{solution_id}.cpp")
            try:
                write_file(code_path, solution["code"])
            except OSError as e:
                logger.error(f"Error saving solution {solution_id} code for session {session_id}: {e}")

        logger.info(f"Saved solution {solution_id} for session {session_id}")

        # Update session metadata
        self.update_session(session_id, {"last_solution_id": solution_id})

        return True

    def get_solution(self, session_id: str, solution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a solution.

        Args:
            session_id: Session ID
            solution_id: Solution ID

        Returns:
            Solution data or None if not found
        """
        # Check if session exists
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found")
            return None

        # Check if solutions directory exists
        solutions_dir = os.path.join(session_dir, "solutions")
        if not os.path.exists(solutions_dir):
            logger.warning(f"Solutions directory for session {session_id} not found")
            return None

        # Load solution
        solution_path = os.path.join(solutions_dir, f"{solution_id}.json")
        if not os.path.exists(solution_path):
            logger.warning(f"Solution {solution_id} for session {session_id} not found")
            return None

        try:
            return read_json(solution_path)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading solution {solution_id} for session {session_id}: {e}")
            return None

    def get_best_solution(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the best solution for a session.

        Args:
            session_id: Session ID

        Returns:
            Best solution data or None if not found
        """
        # Check if session exists
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found")
            return None

        # Check if solutions directory exists
        session_solutions_dir = os.path.join(session_dir, "solutions")
        if not os.path.exists(session_solutions_dir):
            logger.warning(f"Solutions directory for session {session_id} not found")
            return None

        # Get all solution files
        import glob

        solution_files = glob.glob(os.path.join(session_solutions_dir, "*.json"))
        if not solution_files:
            logger.info(f"No solutions found for session {session_id}")
            return None

        all_solutions_data = []
        for solution_file in solution_files:
            if solution_file.endswith(".json"):
                solution_path = os.path.join(session_solutions_dir, solution_file)
                try:
                    solution_data = read_json(solution_path)
                    if solution_data and isinstance(solution_data.get("score"), (int, float)):
                        all_solutions_data.append(solution_data)
                except (OSError, ValueError) as e:
                    logger.error(f"Error loading solution {solution_file}: {e}")

        if not all_solutions_data:
            logger.info(f"No valid solutions with scores found for session {session_id}")
            return None

        # Find the best solution by score
        return max(all_solutions_data, key=lambda s: s["score"])

    def save_evolution_log(self, session_id: str, evolution_log: Dict[str, Any]) -> bool:
        """
        Save evolution log.

        Args:
            session_id: Session ID
            evolution_log: Evolution log data

        Returns:
            True if successful, False otherwise
        """
        # Check if session exists
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found")
            return False

        # Save evolution log
        log_path = os.path.join(session_dir, "evolution_log.json")
        try:
            write_json(log_path, evolution_log)
        except OSError as e:
            logger.error(f"Error saving evolution log for session {session_id}: {e}")
            return False

        logger.info(f"Saved evolution log for session {session_id}")

        # Update session metadata
        self.update_session(session_id, {"has_evolution_log": True})

        return True

    def get_evolution_log(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get evolution log.

        Args:
            session_id: Session ID

        Returns:
            Evolution log data or None if not found
        """
        # Check if session exists
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found")
            return None

        # Load evolution log
        log_path = os.path.join(session_dir, "evolution_log.json")
        if not os.path.exists(log_path):
            logger.warning(f"Evolution log for session {session_id} not found")
            return None

        try:
            return read_json(log_path)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading evolution log for session {session_id}: {e}")
            return None
