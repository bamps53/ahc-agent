import datetime  # for _format_timestamp
import logging
import time
from typing import Optional

from ahc_agent.config import Config
from ahc_agent.core.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


class StatusService:
    def __init__(self, config: Config, knowledge_base: KnowledgeBase):
        self.config = config
        # The workspace_dir for KnowledgeBase is typically set globally or per command.
        # Here, we assume the passed knowledge_base instance is already configured
        # with the correct workspace_dir.
        self.knowledge_base = knowledge_base

    @staticmethod
    def _format_timestamp(timestamp) -> str:
        """
        Format a timestamp.
        """
        if not timestamp:
            return "Unknown"
        # Ensure timestamp is a number before formatting
        try:
            return datetime.datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return "Invalid timestamp"

    @staticmethod
    def _format_duration(seconds) -> str:
        """
        Format a duration in seconds.
        """
        if seconds is None:  # Check for None explicitly
            return "Unknown"
        try:
            s = int(seconds)
        except (TypeError, ValueError):
            return "Invalid duration"

        if s == 0:  # Handle 0 seconds specifically if desired, or let it fall through
            return "0s"

        minutes, s = divmod(s, 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {s}s"
        if minutes > 0:
            return f"{minutes}m {s}s"
        return f"{s}s"

    def _format_session_status(self, session: dict) -> list[str]:
        """
        Formats detailed session status into a list of strings.
        """
        lines = []
        session_id = session.get("session_id")

        lines.append("\n=== Session Status ===")
        lines.append(f"Session ID: {session_id}")
        # problem_id can be directly in session or in metadata, prioritize metadata
        problem_id = session.get("metadata", {}).get("problem_id") or session.get("problem_id", "Unknown")
        lines.append(f"Problem: {problem_id}")
        lines.append(f"Created: {self._format_timestamp(session.get('created_at'))}")
        lines.append(f"Updated: {self._format_timestamp(session.get('updated_at'))}")
        lines.append(f"Status: {session.get('status', 'Unknown')}")

        # Check for problem analysis
        problem_analysis = self.knowledge_base.get_problem_analysis(session_id)
        lines.append(f"Problem Analysis: {'Complete' if problem_analysis else 'Not started'}")

        # Check for solution strategy
        solution_strategy = self.knowledge_base.get_solution_strategy(session_id)
        lines.append(f"Solution Strategy: {'Complete' if solution_strategy else 'Not started'}")

        # Check for evolution log
        evolution_log = self.knowledge_base.get_evolution_log(session_id)
        if evolution_log:
            lines.append(f"Evolution: Complete ({evolution_log.get('generations_completed', 0)} generations)")
            lines.append(f"Best Score: {evolution_log.get('best_score', 'Unknown')}")
            lines.append(f"Duration: {self._format_duration(evolution_log.get('duration', 0))}")
        else:
            lines.append("Evolution: Not started")

        # Check for best solution
        best_solution = self.knowledge_base.get_best_solution(session_id)
        if best_solution:
            lines.append(f"Best Solution: Available (score: {best_solution.get('score', 'Unknown')})")
        else:
            lines.append("Best Solution: Not available")

        return lines

    def get_status(self, session_id: Optional[str] = None, watch: bool = False) -> list[str]:
        """
        Retrieves and formats status information for sessions.
        Logs the information and returns it as a list of strings.
        """
        output_lines = []

        # Note: The original CLI command initialized KnowledgeBase within the `status` command.
        # Here, `self.knowledge_base` is already initialized and passed to the service.
        # The workspace_dir should be correctly set in `self.knowledge_base`.

        if session_id:
            session = self.knowledge_base.get_session(session_id)
            if not session:
                msg = f"Session {session_id} not found"
                logger.error(msg)
                # As per prompt, raise ValueError or return error message. Let's raise.
                raise ValueError(msg)

            status_strings = self._format_session_status(session)
            for line in status_strings:
                logger.info(line)  # Logging as per prompt
            output_lines.extend(status_strings)

            if watch:
                logger.info("\nWatching for updates (Ctrl+C to stop)...")
                try:
                    while True:
                        time.sleep(5)
                        current_session_data = self.knowledge_base.get_session(session_id)
                        if current_session_data:
                            logger.info(f"--- Watch update for session {session_id} at {self._format_timestamp(time.time())} ---")
                            watched_status_strings = self._format_session_status(current_session_data)
                            for line in watched_status_strings:
                                logger.info(line)
                            # output_lines can grow very large in watch mode if we keep appending.
                            # The prompt says "return a list of strings", which implies the *current* status.
                            # For watch mode, the primary output is through logging.
                            # We could update output_lines to the latest status if needed.
                            output_lines = watched_status_strings  # Keep only the latest status
                        else:
                            logger.info(f"Session {session_id} no longer found. Stopping watch.")
                            break
                except KeyboardInterrupt:
                    logger.info("\nStopped watching.")
        else:
            sessions = self.knowledge_base.list_sessions()
            if not sessions:
                msg = "No sessions found"
                logger.info(msg)
                output_lines.append(msg)
                return output_lines  # Return early

            summary_header = f"Found {len(sessions)} sessions:"
            logger.info(summary_header)
            output_lines.append(summary_header)

            for session_data in sessions:
                # Mimicking the brief summary from cli.py for list view
                # Get problem_id with metadata priority
                problem_id = session_data.get("metadata", {}).get("problem_id") or session_data.get("problem_id", "Unknown")

                line = (
                    f"\nSession ID: {session_data.get('session_id')}\n"
                    f"  Problem: {problem_id}\n"
                    f"  Created: {self._format_timestamp(session_data.get('created_at'))}\n"
                    f"  Status: {session_data.get('status', 'Unknown')}"
                )
                logger.info(line)
                # For a list of strings, we can split the multi-line f-string or add individual components
                output_lines.append(f"Session ID: {session_data.get('session_id')}")
                output_lines.append(f"  Problem: {session_data.get('metadata', {}).get('problem_id') or session_data.get('problem_id', 'Unknown')}")
                output_lines.append(f"  Created: {self._format_timestamp(session_data.get('created_at'))}")
                output_lines.append(f"  Status: {session_data.get('status', 'Unknown')}")
                output_lines.append("")  # Add a blank line for separation

        return output_lines
