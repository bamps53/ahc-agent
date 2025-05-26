import pytest
from unittest.mock import MagicMock, patch, call
import time # For potentially mocking sleep in watch mode
import datetime

from ahc_agent.services.status_service import StatusService
from ahc_agent.config import Config
from ahc_agent.core.knowledge import KnowledgeBase

class TestStatusService:

    @pytest.fixture
    def mock_config(self):
        return MagicMock(spec=Config)

    @pytest.fixture
    def mock_kb(self):
        return MagicMock(spec=KnowledgeBase)

    def test_format_timestamp(self):
        assert StatusService._format_timestamp(None) == "Unknown"
        assert StatusService._format_timestamp("invalid_data") == "Invalid timestamp"
        # Test with a valid timestamp. The exact output depends on the test environment's timezone.
        # We can check for parts of the timestamp or mock datetime.datetime.fromtimestamp for full control.
        # For this example, let's assume UTC or check for a consistent part.
        # A specific timestamp: 1621234567 is Mon May 17 2021 06:56:07 GMT+0000
        # If your local timezone is different, the output of strftime("%Y-%m-%d %H:%M:%S") will vary.
        # Let's check for the date part which is less likely to be affected by minor tz differences in common setups.
        formatted_time = StatusService._format_timestamp(1621234567)
        assert "2021-05-17" in formatted_time # Check for date part
        # If specific time is needed and test env is not UTC:
        # with patch('datetime.datetime') as mock_dt:
        #     mock_dt.fromtimestamp.return_value.strftime.return_value = "YYYY-MM-DD HH:MM:SS_mocked"
        #     assert StatusService._format_timestamp(1621234567) == "YYYY-MM-DD HH:MM:SS_mocked"

    def test_format_duration(self):
        assert StatusService._format_duration(None) == "Unknown"
        assert StatusService._format_duration("invalid_data") == "Invalid duration"
        assert StatusService._format_duration(0) == "0s"
        assert StatusService._format_duration(59) == "59s"
        assert StatusService._format_duration(60) == "1m 0s"
        assert StatusService._format_duration(3600) == "1h 0m 0s"
        assert StatusService._format_duration(3661) == "1h 1m 1s"
        assert StatusService._format_duration(86400 + 3600 + 60 + 1) == "25h 1m 1s" # Example over 24h

    def test_format_session_status_all_present(self, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        session_data = {"session_id": "s1", "problem_id": "p1", "created_at": 1621234567, "updated_at": 1621234568, "status": "running"}
        
        mock_kb.get_problem_analysis.return_value = {"title": "Analyzed"}
        mock_kb.get_solution_strategy.return_value = {"approach": "Strategized"}
        mock_kb.get_evolution_log.return_value = {"generations_completed": 10, "best_score": 12345, "duration": 300}
        mock_kb.get_best_solution.return_value = {"score": 12345, "code": "dummy_code"} # Added code for completeness

        result_strings = service._format_session_status(session_data)

        # Check for key pieces of information
        output_str = "\n".join(result_strings) # Join for easier substring search
        assert "Session ID: s1" in output_str
        assert "Problem: p1" in output_str
        assert "Problem Analysis: Complete" in output_str
        assert "Solution Strategy: Complete" in output_str
        assert "Evolution: Complete (10 generations)" in output_str
        assert "Best Score: 12345" in output_str # From evolution_log
        assert "Duration: 5m 0s" in output_str # 300 seconds
        assert "Best Solution: Available (score: 12345)" in output_str

    def test_format_session_status_some_missing(self, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        session_data = {"session_id": "s2", "problem_id": "p2", "created_at": None, "updated_at": None, "status": "pending"}
        
        mock_kb.get_problem_analysis.return_value = None
        mock_kb.get_solution_strategy.return_value = None
        mock_kb.get_evolution_log.return_value = None
        mock_kb.get_best_solution.return_value = None
        
        result_strings = service._format_session_status(session_data)
        output_str = "\n".join(result_strings)

        assert "Session ID: s2" in output_str
        assert "Problem: p2" in output_str
        assert "Created: Unknown" in output_str # Handled by _format_timestamp
        assert "Problem Analysis: Not started" in output_str
        assert "Solution Strategy: Not started" in output_str
        assert "Evolution: Not started" in output_str
        assert "Best Solution: Not available" in output_str # Check how "Best Solution: Not available" is represented

    def test_get_status_single_session_found(self, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        session_data = {"session_id": "s1", "problem_id": "p1"}
        mock_kb.get_session.return_value = session_data
        
        # Spy on _format_session_status
        # The return value of _format_session_status is a list of strings.
        expected_formatted_status = ["Formatted status line 1", "Formatted status line 2"]
        with patch.object(service, '_format_session_status', return_value=expected_formatted_status) as spy_format:
            result = service.get_status(session_id="s1")
            spy_format.assert_called_once_with(session_data)
            assert result == expected_formatted_status

    def test_get_status_single_session_not_found(self, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        mock_kb.get_session.return_value = None
        
        with pytest.raises(ValueError, match="Session test_id_not_found not found"):
            service.get_status(session_id="test_id_not_found")

    def test_get_status_list_all_sessions(self, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        sessions_data = [
            {"session_id": "s1", "problem_id": "p1", "created_at": 1621234567, "status": "done", "metadata": {"problem_id": "p1_meta"}},
            {"session_id": "s2", "problem_id": "p2", "created_at": 1621234500, "status": "running"},
        ]
        mock_kb.list_sessions.return_value = sessions_data
        
        result_strings = service.get_status()
        
        assert f"Found {len(sessions_data)} sessions:" == result_strings[0]
        # Check if summary lines for each session are present (simplified check)
        output_str = "\n".join(result_strings)
        assert "Session ID: s1" in output_str
        assert "Problem: p1_meta" in output_str # Checks if metadata problem_id is preferred
        assert "Session ID: s2" in output_str
        assert "Problem: p2" in output_str # Falls back to session.problem_id

    def test_get_status_list_no_sessions(self, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        mock_kb.list_sessions.return_value = []
        
        result_strings = service.get_status()
        
        assert result_strings == ["No sessions found"]

    @patch("ahc_agent.services.status_service.time.sleep") # Mock time.sleep
    def test_get_status_watch_mode_single_iteration(self, mock_sleep, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        session_data_iter1 = {"session_id": "s1", "problem_id": "p1", "status": "running"}
        # Simulate get_session returning data then None to break loop after first actual status display
        mock_kb.get_session.side_effect = [session_data_iter1, None] 
                                        
        expected_formatted_status = ["Formatted status for watch iter1"]
        with patch.object(service, '_format_session_status', return_value=expected_formatted_status) as spy_format:
            # The get_status in watch mode logs updates but returns the first status.
            result = service.get_status(session_id="s1", watch=True)
            
            # _format_session_status should be called for the first iteration
            spy_format.assert_called_once_with(session_data_iter1)
            assert result == expected_formatted_status # Returns status of the first fetch
            
            # Check that time.sleep was called as part of the loop
            mock_sleep.assert_called_once_with(5)
            
            # Check that get_session was called twice by the loop logic
            assert mock_kb.get_session.call_count == 2
            mock_kb.get_session.assert_has_calls([call("s1"), call("s1")])

    @patch("ahc_agent.services.status_service.time.sleep")
    @patch.object(StatusService, '_format_timestamp', side_effect=lambda ts: f"formatted_ts_{ts}")
    def test_get_status_watch_mode_stops_if_session_vanishes(self, mock_format_ts, mock_sleep, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        session_data_initial = {"session_id": "s1", "problem_id": "p1", "status": "processing"}
        
        # First call for initial status, second for first loop check (session gone)
        mock_kb.get_session.side_effect = [session_data_initial, None] 
        
        with patch.object(service, '_format_session_status', return_value=["Initial status"]) as spy_format_status, \
             patch.object(service.logger, 'info') as mock_logger_info: # Spy on logger

            result = service.get_status(session_id="s1", watch=True)

            assert result == ["Initial status"] # Initial status is returned
            spy_format_status.assert_called_once_with(session_data_initial) # Formatted once
            mock_sleep.assert_called_once_with(5) # Slept once

            # Check logger calls for watch start and session vanishing
            # This requires knowing the exact log messages.
            # Example: logger.info("\nWatching for updates (Ctrl+C to stop)...")
            # logger.info(f"Session {session_id} no longer found. Stopping watch.")
            log_calls = [c[0][0] for c in mock_logger_info.call_args_list]
            assert any("\nWatching for updates (Ctrl+C to stop)..." in log_msg for log_msg in log_calls)
            assert any(f"Session s1 no longer found. Stopping watch." in log_msg for log_msg in log_calls)
            # Also check that the initial status lines were logged by get_status
            assert any("Initial status" in log_msg for log_msg in log_calls)
            
    @patch("ahc_agent.services.status_service.time.sleep", side_effect=KeyboardInterrupt("Stop testing watch"))
    def test_get_status_watch_mode_keyboard_interrupt(self, mock_sleep, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        session_data = {"session_id": "s1", "problem_id": "p1"}
        mock_kb.get_session.return_value = session_data # Always returns session data

        with patch.object(service, '_format_session_status', return_value=["Status during watch"]) as spy_format, \
             patch.object(service.logger, 'info') as mock_logger_info:
            
            result = service.get_status(session_id="s1", watch=True)
            
            spy_format.assert_called_once_with(session_data) # Called for the first display
            assert result == ["Status during watch"]
            mock_sleep.assert_called_once_with(5) # time.sleep was called before KeyboardInterrupt

            # Check that "Stopped watching" was logged
            log_calls = [c[0][0] for c in mock_logger_info.call_args_list]
            assert any("\nStopped watching." in log_msg for log_msg in log_calls)

    def test_format_session_status_problem_id_from_metadata(self, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        # Case where session_data has problem_id in metadata (e.g. from batch experiments)
        session_data = {
            "session_id": "s_batch_1", 
            "metadata": {"problem_id": "p_batch_meta"}, # problem_id in metadata
            "created_at": 1621234567, 
            "status": "init"
        }
        # Set all KB calls to None for simplicity, focus on problem_id formatting
        mock_kb.get_problem_analysis.return_value = None
        mock_kb.get_solution_strategy.return_value = None
        mock_kb.get_evolution_log.return_value = None
        mock_kb.get_best_solution.return_value = None

        result_strings = service._format_session_status(session_data)
        output_str = "\n".join(result_strings)
        assert "Problem: p_batch_meta" in output_str # Check if problem_id from metadata is used

    def test_get_status_list_all_sessions_problem_id_handling(self, mock_config, mock_kb):
        service = StatusService(mock_config, mock_kb)
        sessions_data = [
            {"session_id": "s1", "problem_id": "p1_direct", "created_at": 1621234567, "status": "done"}, # Direct problem_id
            {"session_id": "s2", "metadata": {"problem_id": "p2_meta"}, "created_at": 1621234500, "status": "running"}, # Metadata problem_id
            {"session_id": "s3", "created_at": 1621234400, "status": "error"}, # No problem_id specified
        ]
        mock_kb.list_sessions.return_value = sessions_data
        
        result_strings = service.get_status()
        output_str = "\n".join(result_strings)

        assert "Problem: p1_direct" in output_str
        assert "Problem: p2_meta" in output_str
        assert "Problem: Unknown" in output_str # Fallback for s3
        assert "Found 3 sessions:" in result_strings[0]

    def test_format_timestamp_invalid_type(self):
        # Test with a type that cannot be converted to float, e.g., a dictionary
        assert StatusService._format_timestamp({'invalid': 'data'}) == "Invalid timestamp"

    def test_format_duration_invalid_type(self):
        # Test with a type that cannot be converted to int
        assert StatusService._format_duration({'invalid': 'data'}) == "Invalid duration"
