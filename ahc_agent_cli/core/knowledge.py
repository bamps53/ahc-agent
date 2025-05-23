"""
Knowledge base module for AHCAgent CLI.

This module provides functionality for storing and retrieving knowledge about solutions.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List

from ..utils.file_io import read_file, write_file, read_json, write_json, ensure_directory

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Knowledge base for storing and retrieving solution knowledge.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the knowledge base.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Get workspace directory from config or use default
        self.workspace_dir = self.config.get("workspace_dir")
        if not self.workspace_dir:
            self.workspace_dir = os.path.join(os.getcwd(), "ahc_workspace")
        
        # Ensure workspace directory exists
        self.workspace_dir = ensure_directory(self.workspace_dir)
        
        # Knowledge base directory
        self.kb_dir = os.path.join(self.workspace_dir, "knowledge_base")
        ensure_directory(self.kb_dir)
        
        # Sessions directory
        self.sessions_dir = os.path.join(self.workspace_dir, "sessions")
        ensure_directory(self.sessions_dir)
        
        # Solutions directory
        self.solutions_dir = os.path.join(self.workspace_dir, "solutions")
        ensure_directory(self.solutions_dir)
        
        # Experiments directory
        self.experiments_dir = os.path.join(self.workspace_dir, "experiments")
        ensure_directory(self.experiments_dir)
        
        logger.info("Initialized knowledge base")
        logger.debug(f"Workspace directory: {self.workspace_dir}")
    
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
        session_id = str(uuid.uuid4())
        
        # Create session directory
        session_dir = os.path.join(self.sessions_dir, session_id)
        ensure_directory(session_dir)
        
        # Create session metadata
        session_metadata = {
            "session_id": session_id,
            "problem_id": problem_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": "created",
            **metadata or {}
        }
        
        # Save session metadata
        metadata_path = os.path.join(session_dir, "metadata.json")
        write_json(metadata_path, session_metadata)
        
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
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session {session_id} not found")
            return None
        
        # Load session metadata
        metadata_path = os.path.join(session_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"Session {session_id} metadata not found")
            return None
        
        try:
            metadata = read_json(metadata_path)
            return metadata
        except Exception as e:
            logger.error(f"Error loading session {session_id} metadata: {str(e)}")
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
        metadata_path = os.path.join(self.sessions_dir, session_id, "metadata.json")
        try:
            write_json(metadata_path, metadata)
            logger.info(f"Updated session {session_id} metadata")
            return True
        except Exception as e:
            logger.error(f"Error updating session {session_id} metadata: {str(e)}")
            return False
    
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
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session {session_id} not found")
            return False
        
        # Save analysis
        analysis_path = os.path.join(session_dir, "problem_analysis.json")
        try:
            write_json(analysis_path, analysis)
            logger.info(f"Saved problem analysis for session {session_id}")
            
            # Update session metadata
            self.update_session(session_id, {"has_problem_analysis": True})
            
            return True
        except Exception as e:
            logger.error(f"Error saving problem analysis for session {session_id}: {str(e)}")
            return False
    
    def get_problem_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get problem analysis.
        
        Args:
            session_id: Session ID
            
        Returns:
            Problem analysis or None if not found
        """
        # Check if session exists
        session_dir = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session {session_id} not found")
            return None
        
        # Load analysis
        analysis_path = os.path.join(session_dir, "problem_analysis.json")
        if not os.path.exists(analysis_path):
            logger.warning(f"Problem analysis for session {session_id} not found")
            return None
        
        try:
            analysis = read_json(analysis_path)
            return analysis
        except Exception as e:
            logger.error(f"Error loading problem analysis for session {session_id}: {str(e)}")
            return None
    
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
            logger.warning(f"Session {session_id} not found")
            return False
        
        # Save strategy
        strategy_path = os.path.join(session_dir, "solution_strategy.json")
        try:
            write_json(strategy_path, strategy)
            logger.info(f"Saved solution strategy for session {session_id}")
            
            # Update session metadata
            self.update_session(session_id, {"has_solution_strategy": True})
            
            return True
        except Exception as e:
            logger.error(f"Error saving solution strategy for session {session_id}: {str(e)}")
            return False
    
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
            logger.warning(f"Session {session_id} not found")
            return None
        
        # Load strategy
        strategy_path = os.path.join(session_dir, "solution_strategy.json")
        if not os.path.exists(strategy_path):
            logger.warning(f"Solution strategy for session {session_id} not found")
            return None
        
        try:
            strategy = read_json(strategy_path)
            return strategy
        except Exception as e:
            logger.error(f"Error loading solution strategy for session {session_id}: {str(e)}")
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
            logger.warning(f"Session {session_id} not found")
            return False
        
        # Create solutions directory for session
        solutions_dir = os.path.join(session_dir, "solutions")
        ensure_directory(solutions_dir)
        
        # Save solution
        solution_path = os.path.join(solutions_dir, f"{solution_id}.json")
        try:
            write_json(solution_path, solution)
            
            # Save solution code separately
            if "code" in solution:
                code_path = os.path.join(solutions_dir, f"{solution_id}.cpp")
                write_file(code_path, solution["code"])
            
            logger.info(f"Saved solution {solution_id} for session {session_id}")
            
            # Update session metadata
            self.update_session(session_id, {"last_solution_id": solution_id})
            
            return True
        except Exception as e:
            logger.error(f"Error saving solution {solution_id} for session {session_id}: {str(e)}")
            return False
    
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
            logger.warning(f"Session {session_id} not found")
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
            solution = read_json(solution_path)
            
            # Load solution code if not in JSON
            if "code" not in solution:
                code_path = os.path.join(solutions_dir, f"{solution_id}.cpp")
                if os.path.exists(code_path):
                    solution["code"] = read_file(code_path)
            
            return solution
        except Exception as e:
            logger.error(f"Error loading solution {solution_id} for session {session_id}: {str(e)}")
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
            logger.warning(f"Session {session_id} not found")
            return None
        
        # Check if solutions directory exists
        solutions_dir = os.path.join(session_dir, "solutions")
        if not os.path.exists(solutions_dir):
            logger.warning(f"Solutions directory for session {session_id} not found")
            return None
        
        # Get all solution files
        import glob
        solution_files = glob.glob(os.path.join(solutions_dir, "*.json"))
        
        if not solution_files:
            logger.warning(f"No solutions found for session {session_id}")
            return None
        
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
            except Exception as e:
                logger.error(f"Error loading solution from {solution_file}: {str(e)}")
        
        if not solutions:
            logger.warning(f"No valid solutions found for session {session_id}")
            return None
        
        # Find best solution
        best_solution = max(solutions, key=lambda s: s.get("score", float('-inf')) if s.get("score") is not None else float('-inf'))
        
        return best_solution
    
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
            logger.warning(f"Session {session_id} not found")
            return False
        
        # Save evolution log
        log_path = os.path.join(session_dir, "evolution_log.json")
        try:
            write_json(log_path, evolution_log)
            logger.info(f"Saved evolution log for session {session_id}")
            
            # Update session metadata
            self.update_session(session_id, {"has_evolution_log": True})
            
            return True
        except Exception as e:
            logger.error(f"Error saving evolution log for session {session_id}: {str(e)}")
            return False
    
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
            logger.warning(f"Session {session_id} not found")
            return None
        
        # Load evolution log
        log_path = os.path.join(session_dir, "evolution_log.json")
        if not os.path.exists(log_path):
            logger.warning(f"Evolution log for session {session_id} not found")
            return None
        
        try:
            evolution_log = read_json(log_path)
            return evolution_log
        except Exception as e:
            logger.error(f"Error loading evolution log for session {session_id}: {str(e)}")
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
            logger.info(f"Saved experiment data for experiment {experiment_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving experiment data for experiment {experiment_id}: {str(e)}")
            return False
    
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
            experiment_data = read_json(data_path)
            return experiment_data
        except Exception as e:
            logger.error(f"Error loading experiment data for experiment {experiment_id}: {str(e)}")
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
        session_dirs = [d for d in os.listdir(self.sessions_dir) 
                       if os.path.isdir(os.path.join(self.sessions_dir, d))]
        
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
        experiment_dirs = [d for d in os.listdir(self.experiments_dir) 
                          if os.path.isdir(os.path.join(self.experiments_dir, d))]
        
        # Load data for each experiment
        experiments = []
        for experiment_id in experiment_dirs:
            experiment_data = self.get_experiment(experiment_id)
            if experiment_data:
                experiments.append(experiment_data)
        
        return experiments
