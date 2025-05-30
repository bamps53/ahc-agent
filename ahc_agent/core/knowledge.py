"""
Knowledge base module for AHCAgent.

This module provides functionality for storing and retrieving knowledge about solutions.
"""

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from ahc_agent.utils.file_io import ensure_directory, read_file, read_json, write_file, write_json

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Knowledge base for storing and retrieving solution knowledge.
    """

    def __init__(self, workspace: str, problem_id: str, _knowledge_base: Optional[dict] = None):
        self.workspace = Path(workspace)
        self.problem_id = problem_id
        self.knowledge_dir = self.workspace / "knowledge"
        ensure_directory(self.knowledge_dir)

        # Knowledge base directory
        self.kb_dir = self.knowledge_dir / "kb"
        ensure_directory(self.kb_dir)

        # Sessions directory
        self.sessions_dir = self.knowledge_dir / "sessions"
        ensure_directory(self.sessions_dir)

        # Solutions directory
        self.solutions_dir = self.knowledge_dir / "solutions"
        ensure_directory(self.solutions_dir)

        # Experiments directory
        self.experiments_dir = self.knowledge_dir / "experiments"
        ensure_directory(self.experiments_dir)

        logger.info("Initialized knowledge base")
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

    def save_experiment(self, experiment_id: str, experiment_data: Dict[str, Any]) -> bool:
        """
        Save experiment data.

        Args:
            experiment_id: Experiment ID
            experiment_data: Experiment data

        Returns:
            True if successful, False otherwise
        """
        # Create experiment directory
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        ensure_directory(experiment_dir)

        # Save experiment data
        data_path = os.path.join(experiment_dir, "experiment_data.json")
        try:
            write_json(data_path, experiment_data)
        except OSError as e:
            logger.error(f"Error saving experiment data for experiment {experiment_id}: {e}")
            return False

        logger.info(f"Saved experiment data for experiment {experiment_id}")
        return True

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment data.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment data or None if not found
        """
        # Check if experiment exists
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        if not os.path.exists(experiment_dir):
            logger.warning(f"Experiment {experiment_id} not found")
            return None

        # Load experiment data
        data_path = os.path.join(experiment_dir, "experiment_data.json")
        if not os.path.exists(data_path):
            logger.warning(f"Experiment data for experiment {experiment_id} not found")
            return None

        try:
            return read_json(data_path)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading experiment data for experiment {experiment_id}: {e}")
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session metadata
        """
        # Check if sessions directory exists
        if not os.path.exists(self.sessions_dir):
            logger.warning("Sessions directory not found")
            return []

        # Get all session directories
        session_dirs = [d for d in os.listdir(self.sessions_dir) if os.path.isdir(os.path.join(self.sessions_dir, d))]

        # Load metadata for each session
        sessions = []
        for session_id in session_dirs:
            metadata = self.get_session(session_id)
            if metadata:
                sessions.append(metadata)

        return sessions

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.

        Returns:
            List of experiment data
        """
        # Check if experiments directory exists
        if not os.path.exists(self.experiments_dir):
            logger.warning("Experiments directory not found")
            return []

        # Get all experiment directories
        experiment_dirs = [d for d in os.listdir(self.experiments_dir) if os.path.isdir(os.path.join(self.experiments_dir, d))]

        # Load data for each experiment
        experiments = []
        for experiment_id in experiment_dirs:
            experiment_data = self.get_experiment(experiment_id)
            if experiment_data:
                experiments.append(experiment_data)

        return experiments

    def load_problem_instance(self) -> Optional[Dict[str, Any]]:
        """
        Load problem instance.

        Returns:
            Problem instance or None if not found
        """
        try:
            file_path = os.path.join(self.knowledge_dir, "problem_instance.json")
            if os.path.exists(file_path):
                return read_json(file_path)
            return None
        except (OSError, ValueError) as e:
            logger.error(f"Error loading problem instance: {e}")
            return None

    def save_problem_instance(self, problem_instance: Dict[str, Any]):
        """
        Save problem instance.

        Args:
            problem_instance: Problem instance
        """
        try:
            file_path = os.path.join(self.knowledge_dir, "problem_instance.json")
            write_json(file_path, problem_instance)
        except OSError as e:
            logger.error(f"Error saving problem instance: {e}")

    def load_analysis_result(self) -> Optional[Dict[str, Any]]:
        """
        Load analysis result.

        Returns:
            Analysis result or None if not found
        """
        try:
            file_path = os.path.join(self.knowledge_dir, "analysis_result.json")
            if os.path.exists(file_path):
                return read_json(file_path)
            return None
        except (OSError, ValueError) as e:
            logger.error(f"Error loading analysis result: {e}")
            return None

    def save_analysis_result(self, analysis_result: Dict[str, Any]):
        """
        Save analysis result.

        Args:
            analysis_result: Analysis result
        """
        try:
            file_path = os.path.join(self.knowledge_dir, "analysis_result.json")
            write_json(file_path, analysis_result)
        except OSError as e:
            logger.error(f"Error saving analysis result: {e}")

    def load_solution_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Load solution strategy.

        Returns:
            Solution strategy or None if not found
        """
        try:
            file_path = os.path.join(self.knowledge_dir, "solution_strategy.json")
            if os.path.exists(file_path):
                return read_json(file_path)
            return None
        except (OSError, ValueError) as e:
            logger.error(f"Error loading solution strategy: {e}")
            return None

    def load_evolution_log(self) -> Optional[Dict[str, Any]]:
        """
        Load evolution log.

        Returns:
            Evolution log or None if not found
        """
        try:
            file_path = os.path.join(self.knowledge_dir, "evolution_log.json")
            if os.path.exists(file_path):
                return read_json(file_path)
            return None
        except (OSError, ValueError) as e:
            logger.error(f"Error loading evolution log: {e}")
            return None

    def load_experiment_data(self) -> Optional[Dict[str, Any]]:
        """
        Load experiment data.

        Returns:
            Experiment data or None if not found
        """
        try:
            file_path = os.path.join(self.knowledge_dir, "experiment_data.json")
            if os.path.exists(file_path):
                return read_json(file_path)
            return None
        except (OSError, ValueError) as e:
            logger.error(f"Error loading experiment data: {e}")
            return None

    def load_debugger_output(self) -> Optional[Dict[str, Any]]:
        """
        Load debugger output.

        Returns:
            Debugger output or None if not found
        """
        try:
            file_path = os.path.join(self.knowledge_dir, "debugger_output.json")
            if os.path.exists(file_path):
                return read_json(file_path)
            return None
        except (OSError, ValueError) as e:
            logger.error(f"Error loading debugger output: {e}")
            return None

    def get_best_solution_code(self) -> Optional[str]:
        """
        Get the best solution code.

        Returns:
            Best solution code or None if not found
        """
        try:
            evolution_log = self.load_evolution_log()
            if evolution_log and evolution_log.get("generations"):
                best_generation = max(evolution_log["generations"], key=lambda g: g.get("best_fitness"))
                if best_generation.get("solutions"):
                    best_solution_in_gen = max(best_generation["solutions"], key=lambda s: s.get("fitness"))
                    return best_solution_in_gen.get("code")
            return None
        except (OSError, ValueError) as e:
            logger.error(f"Error getting best solution code: {e}")
            return None

    def get_all_solution_codes(self) -> List[str]:
        """
        Get all solution codes.

        Returns:
            List of solution codes
        """
        # Check if solutions directory exists
        solutions_dir = os.path.join(self.knowledge_dir, "solutions")
        if not os.path.exists(solutions_dir):
            logger.warning("Solutions directory not found")
            return []

        # Get all solution files
        import glob

        solution_files = glob.glob(os.path.join(solutions_dir, "*.json"))

        # Load all solutions
        solutions = []
        for solution_file in solution_files:
            try:
                solution = read_json(solution_file)

                # Load solution code if not in JSON
                if "code" not in solution:
                    solution_id = os.path.splitext(os.path.basename(solution_file))[0]
                    code_path = os.path.join(solutions_dir, f"{solution_id}.cpp")
                    if os.path.exists(code_path):
                        solution["code"] = read_file(code_path)

                solutions.append(solution)
            except (OSError, ValueError) as e:
                logger.error(f"Error loading solution from {solution_file}: {e}")

        # Extract solution codes
        return [solution.get("code") for solution in solutions if solution.get("code")]

    def get_all_solutions(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all solutions for a session.

        Args:
            session_id: Session ID

        Returns:
            List of solution data
        """
        # Check if session exists
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found")
            return []

        # Check if solutions directory exists
        solutions_dir = os.path.join(session_dir, "solutions")
        if not os.path.exists(solutions_dir):
            logger.warning(f"Solutions directory for session {session_id} not found")
            return []

        # Get all solution files
        import glob

        solution_files = glob.glob(os.path.join(solutions_dir, "*.json"))

        # Load all solutions
        solutions = []
        for solution_file in solution_files:
            try:
                solution = read_json(solution_file)

                # Load solution code if not in JSON
                if "code" not in solution:
                    solution_id = os.path.splitext(os.path.basename(solution_file))[0]
                    code_path = os.path.join(solutions_dir, f"{solution_id}.cpp")
                    if os.path.exists(code_path):
                        solution["code"] = read_file(code_path)

                solutions.append(solution)
            except (OSError, ValueError) as e:
                logger.error(f"Error loading solution from {solution_file}: {e}")

        return solutions
