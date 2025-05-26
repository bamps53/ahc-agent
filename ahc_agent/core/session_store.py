import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional

from ahc_agent.utils.file_io import ensure_directory, read_json, write_json, write_file, read_file

logger = logging.getLogger(__name__)

class SessionStore:
    def __init__(self, workspace_dir: str, problem_id: str, session_id: str):
        """
        Initializes a SessionStore for a specific session.

        Args:
            workspace_dir: The root directory of the workspace.
            problem_id: The ID of the problem.
            session_id: The ID of the session.
        """
        self.workspace_dir = workspace_dir
        self.problem_id = problem_id
        self.session_id = session_id
        # Path structure based on original KnowledgeBase: workspace_dir/problem_id/knowledge/sessions/session_id
        self.session_dir = os.path.join(
            self.workspace_dir,
            self.problem_id,
            "knowledge",
            "sessions",
            self.session_id,
        )
        ensure_directory(self.session_dir)
        logger.debug(f"SessionStore initialized for session {session_id} at {self.session_dir}")

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Loads session metadata."""
        metadata_path = os.path.join(self.session_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                return read_json(metadata_path)
            except Exception as e:
                logger.error(f"Error reading metadata for session {self.session_id} from {metadata_path}: {e}")
                return None
        return None

    def _save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Saves session metadata."""
        metadata_path = os.path.join(self.session_dir, "metadata.json")
        try:
            write_json(metadata_path, metadata)
            logger.debug(f"Saved metadata for session {self.session_id} to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata for session {self.session_id} to {metadata_path}: {e}")
            return False

    @staticmethod
    def create_new_session_id() -> str:
        return str(uuid.uuid4())

    def create_session_metadata(self, initial_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initializes and saves metadata for the current session.
        """
        metadata_path = os.path.join(self.session_dir, "metadata.json")
        if os.path.exists(metadata_path):
            logger.warning(f"Session metadata already exists for session {self.session_id} at {metadata_path}. Overwriting or updating might occur if called again.")
            # Depending on desired behavior, could return False or load existing.
            # For now, let's assume it's okay to proceed and potentially overwrite if _save_metadata is called.
            # However, a strict create should probably fail here.
            # Let's refine: this method should only create if not exists.
            return False # Indicate failure if metadata already exists, use update_session_metadata for updates.


        session_metadata = {
            "session_id": self.session_id,
            "problem_id": self.problem_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": "created",
        }
        if initial_metadata:
            session_metadata.update(initial_metadata)
        
        logger.info(f"Creating new session metadata for session {self.session_id}")
        return self._save_metadata(session_metadata)

    def get_session_metadata(self) -> Optional[Dict[str, Any]]:
        """Retrieves the metadata for the current session."""
        return self._load_metadata()

    def update_session_metadata(self, updates: Dict[str, Any]) -> bool:
        """Updates specific fields in the session metadata."""
        metadata = self._load_metadata()
        if not metadata:
            logger.error(f"Cannot update metadata for session {self.session_id}: metadata.json not found.")
            return False
        metadata.update(updates)
        metadata["updated_at"] = time.time()
        logger.info(f"Updating session metadata for session {self.session_id} with updates: {updates}")
        return self._save_metadata(metadata)

    def save_problem_analysis(self, analysis: Dict[str, Any]) -> bool:
        analysis_path = os.path.join(self.session_dir, "problem_analysis.json")
        try:
            write_json(analysis_path, analysis)
            self.update_session_metadata({"has_problem_analysis": True, "status": "analysis_complete"})
            logger.info(f"Saved problem analysis for session {self.session_id} to {analysis_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving problem analysis for session {self.session_id} to {analysis_path}: {e}")
            return False

    def get_problem_analysis(self) -> Optional[Dict[str, Any]]:
        analysis_path = os.path.join(self.session_dir, "problem_analysis.json")
        if os.path.exists(analysis_path):
            try:
                return read_json(analysis_path)
            except Exception as e:
                logger.error(f"Error reading problem analysis for session {self.session_id} from {analysis_path}: {e}")
                return None
        logger.debug(f"Problem analysis file not found for session {self.session_id} at {analysis_path}")
        return None

    def save_solution_strategy(self, strategy: Dict[str, Any]) -> bool:
        strategy_path = os.path.join(self.session_dir, "solution_strategy.json")
        try:
            write_json(strategy_path, strategy)
            self.update_session_metadata({"has_solution_strategy": True, "status": "strategy_complete"})
            logger.info(f"Saved solution strategy for session {self.session_id} to {strategy_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving solution strategy for session {self.session_id} to {strategy_path}: {e}")
            return False

    def get_solution_strategy(self) -> Optional[Dict[str, Any]]:
        strategy_path = os.path.join(self.session_dir, "solution_strategy.json")
        if os.path.exists(strategy_path):
            try:
                return read_json(strategy_path)
            except Exception as e:
                logger.error(f"Error reading solution strategy for session {self.session_id} from {strategy_path}: {e}")
                return None
        logger.debug(f"Solution strategy file not found for session {self.session_id} at {strategy_path}")
        return None

    def save_solution(self, solution_id: str, solution_data: Dict[str, Any], code: Optional[str] = None) -> bool:
        session_solutions_dir = os.path.join(self.session_dir, "solutions")
        ensure_directory(session_solutions_dir)
        
        solution_metadata_path = os.path.join(session_solutions_dir, f"{solution_id}.json")
        actual_solution_data = solution_data.copy() # Avoid modifying the input dict

        if code:
            actual_solution_data["code_filename"] = f"{solution_id}.cpp" # Store reference to code file

        try:
            write_json(solution_metadata_path, actual_solution_data)
            if code:
                code_path = os.path.join(session_solutions_dir, actual_solution_data["code_filename"])
                write_file(code_path, code)
            
            self.update_session_metadata({"last_solution_id": solution_id, "status": "solution_saved"})
            logger.info(f"Saved solution {solution_id} for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving solution {solution_id} for session {self.session_id}: {e}")
            return False

    def get_solution(self, solution_id: str) -> Optional[Dict[str, Any]]:
        session_solutions_dir = os.path.join(self.session_dir, "solutions")
        solution_metadata_path = os.path.join(session_solutions_dir, f"{solution_id}.json")
        
        if not os.path.exists(solution_metadata_path):
            logger.debug(f"Solution metadata file not found: {solution_metadata_path}")
            return None
            
        try:
            solution_data = read_json(solution_metadata_path)
            if "code_filename" in solution_data:
                code_path = os.path.join(session_solutions_dir, solution_data["code_filename"])
                if os.path.exists(code_path):
                    solution_data["code"] = read_file(code_path)
                else:
                    logger.warning(f"Code file {solution_data['code_filename']} referenced in {solution_id}.json not found for session {self.session_id}")
            elif "code" in solution_data:
                 pass # Code is directly in JSON
            return solution_data
        except Exception as e:
            logger.error(f"Error reading solution {solution_id} for session {self.session_id}: {e}")
            return None

    def list_solutions(self) -> List[Dict[str, Any]]:
        session_solutions_dir = os.path.join(self.session_dir, "solutions")
        solutions = []
        if os.path.exists(session_solutions_dir):
            for f_name in sorted(os.listdir(session_solutions_dir)): # Sort for some order
                if f_name.endswith(".json"):
                    solution_id = f_name[:-len(".json")] 
                    solution_data = self.get_solution(solution_id)
                    if solution_data:
                        solutions.append(solution_data)
        return solutions

    def get_best_solution(self) -> Optional[Dict[str, Any]]:
        all_solutions = self.list_solutions()
        if not all_solutions:
            return None
        
        valid_solutions = [s for s in all_solutions if isinstance(s.get("score"), (int, float))]
        if not valid_solutions:
            logger.warning(f"No solutions with valid scores found for session {self.session_id}")
            return None
            
        try:
            # Assumes higher score is better. This might need to be configurable per problem.
            return max(valid_solutions, key=lambda s: s["score"])
        except ValueError: 
            logger.error(f"ValueError when finding max score solution for session {self.session_id}")
            return None

    def save_evolution_log(self, evolution_log: Dict[str, Any]) -> bool:
        log_path = os.path.join(self.session_dir, "evolution_log.json")
        try:
            write_json(log_path, evolution_log)
            self.update_session_metadata({"has_evolution_log": True, "status": "evolution_logged"})
            logger.info(f"Saved evolution log for session {self.session_id} to {log_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving evolution log for session {self.session_id} to {log_path}: {e}")
            return False

    def get_evolution_log(self) -> Optional[Dict[str, Any]]:
        log_path = os.path.join(self.session_dir, "evolution_log.json")
        if os.path.exists(log_path):
            try:
                return read_json(log_path)
            except Exception as e:
                logger.error(f"Error reading evolution log for session {self.session_id} from {log_path}: {e}")
                return None
        logger.debug(f"Evolution log file not found for session {self.session_id} at {log_path}")
        return None
            
    def save_llm_interaction(self, interaction_id: str, interaction_data: Dict[str, Any]) -> bool:
        llm_logs_dir = os.path.join(self.session_dir, "llm_logs")
        ensure_directory(llm_logs_dir)
        interaction_file = os.path.join(llm_logs_dir, f"{interaction_id}.json")
        try:
            write_json(interaction_file, interaction_data)
            logger.debug(f"Saved LLM interaction {interaction_id} for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving LLM interaction {interaction_id} for session {self.session_id}: {e}")
            return False

    def get_llm_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        interaction_file = os.path.join(self.session_dir, "llm_logs", f"{interaction_id}.json")
        if os.path.exists(interaction_file):
            try:
                return read_json(interaction_file)
            except Exception as e:
                logger.error(f"Error reading LLM interaction {interaction_id} for session {self.session_id}: {e}")
                return None
        return None

    def list_llm_interactions(self) -> List[Dict[str, Any]]:
        llm_logs_dir = os.path.join(self.session_dir, "llm_logs")
        interactions = []
        if os.path.exists(llm_logs_dir):
            for f_name in sorted(os.listdir(llm_logs_dir)): # Sort for some order
                if f_name.endswith(".json"):
                    interaction_id = f_name[:-len(".json")]
                    interaction_data = self.get_llm_interaction(interaction_id)
                    if interaction_data:
                        interactions.append(interaction_data)
        return interactions

class SessionManager:
    def __init__(self, workspace_dir: str, problem_id: str):
        """
        Manages multiple sessions for a given problem.
        """
        self.workspace_dir = workspace_dir
        self.problem_id = problem_id
        self.sessions_base_dir = os.path.join(
            self.workspace_dir,
            self.problem_id,
            "knowledge", 
            "sessions"
        )
        ensure_directory(self.sessions_base_dir)
        logger.debug(f"SessionManager initialized for problem {problem_id} at {self.sessions_base_dir}")

    def create_session(self, initial_metadata: Optional[Dict[str, Any]] = None) -> SessionStore:
        """
        Creates a new session and returns a SessionStore instance for it.
        """
        session_id = SessionStore.create_new_session_id()
        store = SessionStore(self.workspace_dir, self.problem_id, session_id)
        
        # Prepare initial metadata, ensuring problem_id from SessionManager is included
        full_initial_metadata = {"problem_id": self.problem_id}
        if initial_metadata:
            full_initial_metadata.update(initial_metadata)
            
        if store.create_session_metadata(full_initial_metadata):
            logger.info(f"Successfully created new session {session_id} for problem {self.problem_id}")
        else:
            # This case should ideally not happen if session_id is truly unique and directory is new.
            # If create_session_metadata fails (e.g. due to disk issue or pre-existing identically named dir that wasn't a valid session)
            logger.error(f"Failed to create session metadata for new session {session_id}. Session may be unusable.")
            # Consider raising an exception here.
        return store

    def get_session_store(self, session_id: str) -> Optional[SessionStore]:
        """
        Returns a SessionStore instance for an existing session.
        Returns None if the session (specifically its metadata.json) does not exist.
        """
        session_dir_path = os.path.join(self.sessions_base_dir, session_id)
        metadata_path = os.path.join(session_dir_path, "metadata.json")
        
        if not os.path.exists(metadata_path): # Check for metadata.json as the source of truth
            logger.warning(f"Session metadata for session_id {session_id} not found at {metadata_path}. Cannot load session.")
            return None
        
        # Ensure the directory itself exists as well, though metadata check is primary
        if not os.path.isdir(session_dir_path):
             logger.warning(f"Session directory {session_dir_path} for session_id {session_id} not found or not a directory. Cannot load session.")
             return None

        logger.debug(f"Accessing existing session {session_id} for problem {self.problem_id}")
        return SessionStore(self.workspace_dir, self.problem_id, session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Lists metadata of all valid sessions for the problem.
        A session is considered valid if it has a readable metadata.json.
        """
        session_metadata_list = []
        if not os.path.exists(self.sessions_base_dir):
            logger.warning(f"Sessions base directory not found: {self.sessions_base_dir}")
            return []
            
        for session_id in sorted(os.listdir(self.sessions_base_dir)): # Sort for consistent order
            session_dir_path = os.path.join(self.sessions_base_dir, session_id)
            if os.path.isdir(session_dir_path):
                # Instantiate a temporary SessionStore to fetch metadata
                store = SessionStore(self.workspace_dir, self.problem_id, session_id)
                metadata = store.get_session_metadata()
                if metadata: # Only include if metadata was successfully read
                    session_metadata_list.append(metadata)
                else:
                    logger.debug(f"Skipping directory {session_id} as it does not contain valid session metadata.")
        return session_metadata_list
