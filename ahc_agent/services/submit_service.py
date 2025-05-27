import logging
import os
from typing import Optional

from ahc_agent.config import Config
from ahc_agent.core.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


class SubmitService:
    def __init__(self, config: Config, knowledge_base: KnowledgeBase):
        self.config = config
        # Assuming knowledge_base is already configured with the correct workspace_dir
        self.knowledge_base = knowledge_base

    def submit_solution(self, session_id: str, output_path: Optional[str] = None) -> dict:
        """
        Retrieves the best solution for a given session and writes it to a file or logs it.
        Returns a dictionary with submission details or raises an error.
        """
        logger.info(f"Attempting to submit solution for session {session_id}")

        # The KnowledgeBase is initialized using workspace_dir from the config in the CLI.
        # Here, self.knowledge_base is assumed to be correctly initialized.
        # No need to re-initialize or pass config.get("workspace.base_dir") again.

        session = self.knowledge_base.get_session(session_id)
        if not session:
            msg = f"Session {session_id} not found"
            logger.error(msg)
            raise ValueError(msg)

        best_solution = self.knowledge_base.get_best_solution(session_id)
        if not best_solution:
            msg = f"No solution found for session {session_id}"
            logger.error(msg)
            raise ValueError(msg)  # Or perhaps a specific custom error like NoSolutionError

        solution_code = best_solution.get("code")
        if not solution_code:
            msg = f"No code found in best solution for session {session_id}"
            logger.error(msg)
            raise ValueError(msg)  # Or NoCodeInSolutionError

        score = best_solution.get("score", "Unknown")

        if output_path:
            try:
                # Ensure the directory for output_path exists if it's nested
                output_dir = os.path.dirname(output_path)
                if output_dir:  # If output_dir is not empty (i.e., not current dir)
                    os.makedirs(output_dir, exist_ok=True)

                with open(output_path, "w") as f:
                    f.write(solution_code)
                logger.info(f"Best solution for session {session_id} written to {output_path}")
            except OSError as e:
                err_msg = f"Error writing solution to {output_path}: {e}"
                logger.error(err_msg)
                raise RuntimeError(err_msg) from e
        else:
            # If no output_path, log the solution code.
            # Using logger.info for consistency, but print() could also be used for direct console output.
            logger.info(f"\n=== Best Solution for Session {session_id} ===")
            # For multi-line code, logging each line or a prefixed block might be cleaner in some log viewers.
            # However, a single log message with newlines is also common.
            for line in solution_code.splitlines():
                logger.info(line)  # Log each line to preserve formatting in logs
            # Alternatively, just: logger.info(f"\n{solution_code}")

        logger.info(f"Score for session {session_id}: {score}")

        return {
            "session_id": session_id,
            "output_path": str(output_path) if output_path else "logged_to_console",  # str() in case output_path is Path object
            "solution_code": solution_code,
            "score": score,
        }
