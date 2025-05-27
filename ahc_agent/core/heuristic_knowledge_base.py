import os
import json
import logging
from typing import Any, Dict, List, Optional

from ahc_agent.utils.file_io import ensure_directory, read_json, write_json, write_file, read_file

logger = logging.getLogger(__name__)

class AlgorithmInfo:
    def __init__(self, name: str, description: str, filepath: str, metadata_path: str, code: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.name = name # This is often the algorithm's formal name from metadata
        self.description = description
        self.filepath = filepath # Path to the .cpp file or other code file
        self.metadata_path = metadata_path # Path to the .meta.json file
        self._code = code
        self._metadata = metadata
        self.file_basename = os.path.splitext(os.path.basename(filepath))[0] if filepath else name # e.g. "dijkstra" from "dijkstra.cpp"

    @property
    def code(self) -> Optional[str]:
        if self._code is None and self.filepath and os.path.exists(self.filepath):
            try:
                self._code = read_file(self.filepath)
            except Exception as e:
                logger.warning(f"Failed to read code for {self.name} from {self.filepath}: {e}")
                # self._code remains None
        return self._code

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        if self._metadata is None and self.metadata_path and os.path.exists(self.metadata_path):
            try:
                self._metadata = read_json(self.metadata_path)
            except Exception as e:
                logger.warning(f"Failed to read metadata for {self.name} from {self.metadata_path}: {e}")
                # self._metadata remains None
        return self._metadata

    def __repr__(self) -> str:
        return f"<AlgorithmInfo name='{self.name}' file='{self.filepath}'>"


class PastContestSolutionInfo:
    def __init__(self, contest_id: str, data: Dict[str, Any], filepath: str):
        self.contest_id = contest_id # e.g., "ahc001"
        self.data = data # The full JSON content for the contest
        self.filepath = filepath # Path to the contest's JSON file

    @property
    def title(self) -> Optional[str]:
        return self.data.get("title")

    @property
    def problem_summary(self) -> Optional[str]:
        return self.data.get("problem_summary")

    @property
    def tags(self) -> List[str]:
        return self.data.get("tags", [])

    @property
    def solution_approaches(self) -> List[Dict[str, Any]]:
        return self.data.get("solution_approaches", [])
        
    def __repr__(self) -> str:
        return f"<PastContestSolutionInfo contest_id='{self.contest_id}' title='{self.title}'>"


class HeuristicKnowledgeBase:
    def __init__(self, base_knowledge_dir: str):
        """
        Initializes HeuristicKnowledgeBase.

        Args:
            base_knowledge_dir: The root directory for storing/accessing heuristic knowledge.
                               Example: /workspace/ahc001/knowledge/kb/ (problem-specific general knowledge)
                               or ~/.ahc_agent_heuristic_knowledge/ (global general knowledge)
        """
        self.base_dir = base_knowledge_dir
        self.library_dir = os.path.join(self.base_dir, "library")
        self.past_contests_db_dir = os.path.join(self.base_dir, "past_contests_db")
        
        ensure_directory(self.base_dir) # Ensure base_dir itself exists
        ensure_directory(self.library_dir)
        ensure_directory(self.past_contests_db_dir)
        
        logger.info(f"HeuristicKnowledgeBase initialized with base directory: {self.base_dir}")
        logger.info(f"Algorithm library directory: {self.library_dir}")
        logger.info(f"Past contest DB directory: {self.past_contests_db_dir}")

    def add_algorithm(self, code_filename: str, metadata_filename: str, code_content: str, metadata_content: Dict[str, Any]) -> bool:
        """
        Adds a new algorithm to the library.
        Assumes filenames include extensions (e.g., dijkstra.cpp, dijkstra.meta.json).
        The 'name' of the algorithm is typically stored within the metadata_content.
        """
        code_filepath = os.path.join(self.library_dir, code_filename)
        metadata_filepath = os.path.join(self.library_dir, metadata_filename)

        try:
            write_file(code_filepath, code_content)
            write_json(metadata_filepath, metadata_content)
            algo_name = metadata_content.get("name", os.path.splitext(code_filename)[0])
            logger.info(f"Added algorithm '{algo_name}' to library: code='{code_filepath}', meta='{metadata_filepath}'")
            return True
        except Exception as e:
            logger.error(f"Error adding algorithm '{metadata_content.get('name', code_filename)}': {e}")
            # Consider cleanup of partial writes if necessary
            return False

    def get_algorithm(self, algorithm_file_basename: str) -> Optional[AlgorithmInfo]:
        """
        Retrieves a specific algorithm by its base filename (e.g., "dijkstra").
        """
        # Try to find matching code and metadata files by basename
        # This is a simple scan; a manifest or more structured lookup could be faster.
        potential_code_file = None
        potential_meta_file = None

        for ext in [".cpp", ".py", ".java"]: # Common extensions, can be configured
            if os.path.exists(os.path.join(self.library_dir, f"{algorithm_file_basename}{ext}")):
                potential_code_file = os.path.join(self.library_dir, f"{algorithm_file_basename}{ext}")
                break
        
        if os.path.exists(os.path.join(self.library_dir, f"{algorithm_file_basename}.meta.json")):
            potential_meta_file = os.path.join(self.library_dir, f"{algorithm_file_basename}.meta.json")

        if potential_code_file and potential_meta_file:
            try:
                meta_content = read_json(potential_meta_file)
                algo_name = meta_content.get("name", algorithm_file_basename)
                description = meta_content.get("description", "")
                
                return AlgorithmInfo(
                    name=algo_name, 
                    description=description,
                    filepath=potential_code_file,
                    metadata_path=potential_meta_file,
                    metadata=meta_content # Pre-load metadata as we've read it
                )
            except Exception as e:
                logger.error(f"Error reading metadata for algorithm '{algorithm_file_basename}': {e}")
                return None
        
        logger.debug(f"Algorithm '{algorithm_file_basename}' not found or missing parts in {self.library_dir}")
        return None

    def list_algorithms(self, tag: Optional[str] = None) -> List[AlgorithmInfo]:
        """Lists all available algorithms, optionally filtered by tag from their metadata."""
        algorithms = []
        if not os.path.exists(self.library_dir):
            logger.warning(f"Algorithm library directory not found: {self.library_dir}")
            return []

        for filename in sorted(os.listdir(self.library_dir)):
            if filename.endswith(".meta.json"):
                algo_file_basename = filename[:-len(".meta.json")]
                algo_info = self.get_algorithm(algo_file_basename) # This will load metadata
                if algo_info and algo_info.metadata: # algo_info.metadata should be populated by get_algorithm
                    if tag:
                        if tag in algo_info.metadata.get("tags", []):
                            algorithms.append(algo_info)
                    else:
                        algorithms.append(algo_info)
        return algorithms

    def search_algorithms(self, keyword: str) -> List[AlgorithmInfo]:
        """Searches algorithms by keyword in name, description, or tags (simple search)."""
        results = []
        kw_lower = keyword.lower()
        for algo_info in self.list_algorithms(): # list_algorithms ensures metadata is attempted to be loaded
            if algo_info.metadata: # Proceed only if metadata was successfully loaded
                name_match = kw_lower in algo_info.name.lower()
                desc_match = kw_lower in algo_info.description.lower()
                tag_match = any(kw_lower in t.lower() for t in algo_info.metadata.get("tags", []))
                
                if name_match or desc_match or tag_match:
                    results.append(algo_info)
        return results

    def add_past_contest_solution(self, contest_id: str, data: Dict[str, Any]) -> bool:
        """Adds or updates a past contest's solution information. contest_id is e.g. 'ahc001'."""
        filepath = os.path.join(self.past_contests_db_dir, f"{contest_id}.json")
        try:
            write_json(filepath, data)
            logger.info(f"Added/Updated past contest solution for '{contest_id}' at {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error adding past contest solution for '{contest_id}': {e}")
            return False

    def get_past_contest_solution(self, contest_id: str) -> Optional[PastContestSolutionInfo]:
        """Retrieves solution information for a specific past contest by contest_id (e.g. 'ahc001')."""
        filepath = os.path.join(self.past_contests_db_dir, f"{contest_id}.json")
        if os.path.exists(filepath):
            try:
                data = read_json(filepath)
                return PastContestSolutionInfo(contest_id=contest_id, data=data, filepath=filepath)
            except Exception as e:
                logger.error(f"Error reading past contest solution for '{contest_id}' from {filepath}: {e}")
                return None
        
        logger.debug(f"Past contest solution for '{contest_id}' not found at {filepath}")
        return None

    def list_past_contest_solutions(self, tag: Optional[str] = None) -> List[PastContestSolutionInfo]:
        """Lists all past contest solutions, optionally filtered by tag."""
        solutions = []
        if not os.path.exists(self.past_contests_db_dir):
            logger.warning(f"Past contest DB directory not found: {self.past_contests_db_dir}")
            return []
            
        for filename in sorted(os.listdir(self.past_contests_db_dir)):
            if filename.endswith(".json"):
                contest_id = filename[:-len(".json")]
                solution_info = self.get_past_contest_solution(contest_id)
                if solution_info:
                    if tag:
                        if tag in solution_info.tags: # Assumes tags are in PastContestSolutionInfo
                            solutions.append(solution_info)
                    else:
                        solutions.append(solution_info)
        return solutions

    def search_past_contests(self, keyword: Optional[str] = None, algorithm_used: Optional[str] = None, tag: Optional[str] = None) -> List[PastContestSolutionInfo]:
        """
        Searches past contest solutions based on keyword (in title, summary, key_ideas), algorithm used, or tag.
        """
        results = []
        for sol_info in self.list_past_contest_solutions(): # This loads all solutions, then filters. Could be optimized.
            match = True # Assume match until a criterion fails
            
            if tag is not None:
                if tag.lower() not in [t.lower() for t in sol_info.tags]:
                    match = False
            
            if keyword is not None and match:
                kw_lower = keyword.lower()
                # Check title, summary, and key ideas
                title_match = kw_lower in sol_info.title.lower() if sol_info.title else False
                summary_match = kw_lower in sol_info.problem_summary.lower() if sol_info.problem_summary else False
                
                ideas_match = False
                for approach in sol_info.solution_approaches:
                    if any(kw_lower in idea.lower() for idea in approach.get("key_ideas", [])):
                        ideas_match = True
                        break
                if not (title_match or summary_match or ideas_match):
                    match = False

            if algorithm_used is not None and match:
                algo_lower = algorithm_used.lower()
                found_algo_in_solution = False
                for approach in sol_info.solution_approaches:
                    if any(algo_lower in algo.lower() for algo in approach.get("algorithms_used", [])):
                        found_algo_in_solution = True
                        break
                if not found_algo_in_solution:
                    match = False
            
            if match:
                results.append(sol_info)
        return results

    # --- Generic Data Handling (from old KnowledgeBase, for problem-specific 'kb' data) ---
    # These methods operate directly under self.base_dir.
    # If self.base_dir is like 'workspace/problem_id/knowledge/kb/', they store problem-specific general files.
    
    def save_generic_json_data(self, data_key: str, data: Dict[str, Any]) -> bool:
        """
        Saves generic JSON data directly under base_dir (e.g., for problem_instance.json).
        data_key should include .json extension if desired, e.g., "problem_instance.json".
        """
        data_path = os.path.join(self.base_dir, data_key)
        try:
            write_json(data_path, data)
            logger.info(f"Saved generic JSON data '{data_key}' to {data_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving generic JSON data '{data_key}' to {data_path}: {e}")
            return False

    def get_generic_json_data(self, data_key: str) -> Optional[Dict[str, Any]]:
        """
        Loads generic JSON data by key from base_dir.
        data_key should include .json extension, e.g., "problem_instance.json".
        """
        data_path = os.path.join(self.base_dir, data_key)
        if os.path.exists(data_path):
            try:
                return read_json(data_path)
            except Exception as e:
                logger.error(f"Error loading generic JSON data '{data_key}' from {data_path}: {e}")
                return None
        logger.debug(f"Generic JSON data '{data_key}' not found at {data_path}")
        return None

    def save_generic_file_data(self, filename: str, content: str) -> bool:
        """Saves generic file data directly under base_dir (e.g., for a README in the kb)."""
        file_path = os.path.join(self.base_dir, filename)
        try:
            write_file(file_path, content)
            logger.info(f"Saved generic file data '{filename}' to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving generic file data '{filename}' to {file_path}: {e}")
            return False
            
    def get_generic_file_data(self, filename: str) -> Optional[str]:
        """Loads generic file data by filename from base_dir."""
        file_path = os.path.join(self.base_dir, filename)
        if os.path.exists(file_path):
            try:
                return read_file(file_path)
            except Exception as e:
                logger.error(f"Error loading generic file data '{filename}' from {file_path}: {e}")
                return None
        logger.debug(f"Generic file data '{filename}' not found at {file_path}")
        return None

    # --- Handling for existing problem-specific, non-session directories ---
    # These methods are for managing data in directories like 'experiments' or 'solutions'
    # that were at the root of the old KnowledgeBase's problem-specific knowledge dir.
    # They are included here assuming HeuristicKnowledgeBase might sometimes be
    # initialized with base_knowledge_dir = 'workspace/problem_id/knowledge/'.
    # Their long-term fit in a "Heuristic" KB is debatable if it's purely for general knowledge.

    def _get_problem_level_subdir_path(self, subdir_name: str) -> str:
        # Example: self.base_dir = /path/to/workspace/ahc001/knowledge/
        # This would return /path/to/workspace/ahc001/knowledge/experiments/
        # This assumes base_dir is already problem specific.
        # If HKB is global, these methods make less sense.
        path = os.path.join(self.base_dir, subdir_name)
        ensure_directory(path)
        return path

    def save_problem_level_data(self, subdir_name: str, data_id: str, data: Dict[str, Any]) -> bool:
        """
        Saves data into a subdirectory of base_dir (e.g., 'experiments', 'solutions_root').
        This is for data that is problem-specific but not session-specific.
        Example: save_problem_level_data('experiments', 'exp001', {...})
        will save to self.base_dir/experiments/exp001.json
        """
        target_dir = self._get_problem_level_subdir_path(subdir_name)
        data_path = os.path.join(target_dir, f"{data_id}.json")
        try:
            write_json(data_path, data)
            logger.info(f"Saved problem-level data for ID '{data_id}' in subdir '{subdir_name}' to {data_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving problem-level data for ID '{data_id}' in subdir '{subdir_name}': {e}")
            return False

    def get_problem_level_data(self, subdir_name: str, data_id: str) -> Optional[Dict[str, Any]]:
        target_dir = self._get_problem_level_subdir_path(subdir_name) # Ensures dir exists if we try to read from it
        data_path = os.path.join(target_dir, f"{data_id}.json")
        if os.path.exists(data_path):
            try:
                return read_json(data_path)
            except Exception as e:
                logger.error(f"Error loading problem-level data for ID '{data_id}' from subdir '{subdir_name}': {e}")
                return None
        logger.debug(f"Problem-level data for ID '{data_id}' in subdir '{subdir_name}' not found at {data_path}")
        return None

    def list_problem_level_data_ids(self, subdir_name: str) -> List[str]:
        target_dir = self._get_problem_level_subdir_path(subdir_name)
        ids = []
        if os.path.exists(target_dir):
            for filename in sorted(os.listdir(target_dir)):
                if filename.endswith(".json"):
                    ids.append(filename[:-len(".json")])
        return ids
