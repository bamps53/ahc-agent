import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from ahc_agent.config import Config
from ahc_agent.core.knowledge import KnowledgeBase
from ahc_agent.services.submit_service import SubmitService


class TestSubmitService:
    @pytest.fixture
    def mock_config(self):
        return MagicMock(spec=Config)

    @pytest.fixture
    def mock_kb(self):
        kb = MagicMock(spec=KnowledgeBase)
        # Default session and solution for most tests
        kb.get_session.return_value = {"session_id": "test_session_id", "problem_id": "test_problem_id"}
        kb.get_best_solution.return_value = {"code": "sample_code();", "score": 12345}
        return kb

    @patch("builtins.open", new_callable=mock_open)
    def test_submit_solution_stdout_success(self, mock_file, mock_config, mock_kb):
        service = SubmitService(config=mock_config, knowledge_base=mock_kb)

        # The service logs to logger.info, so we can patch that to check the output
        with patch("ahc_agent.services.submit_service.logger.info") as mock_logger_info:
            result = service.submit_solution(session_id="test_session_id")

            mock_kb.get_session.assert_called_once_with("test_session_id")
            mock_kb.get_best_solution.assert_called_once_with("test_session_id")
            assert result["solution_code"] == "sample_code();"
            assert result["score"] == 12345
            # Based on SubmitService implementation, output_path is 'logged_to_console'
            assert result["output_path"] == "logged_to_console"

            mock_file.assert_not_called()

            # Check that the code was logged
            # This depends on how the service logs multi-line code.
            # The service logs: logger.info(f"\n=== Best Solution for Session {session_id} ===")
            # then each line: logger.info(line)
            # then score: logger.info(f"Score for session {session_id}: {score}")

            log_calls = [call_args[0][0] for call_args in mock_logger_info.call_args_list]
            assert "\n=== Best Solution for Session test_session_id ===" in log_calls
            assert "sample_code();" in log_calls
            assert "Score for session test_session_id: 12345" in log_calls

    @patch("ahc_agent.services.submit_service.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_submit_solution_file_success(self, mock_file, mock_makedirs, mock_config, mock_kb):
        service = SubmitService(config=mock_config, knowledge_base=mock_kb)
        output_file_path = "output/solution.cpp"

        result = service.submit_solution(session_id="test_session_id", output_path=output_file_path)

        # Check if os.makedirs was called with the correct directory
        expected_dir = os.path.dirname(output_file_path)
        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)

        mock_file.assert_called_once_with(output_file_path, "w")
        mock_file().write.assert_called_once_with("sample_code();")
        assert result["output_path"] == output_file_path
        assert result["solution_code"] == "sample_code();"
        assert result["score"] == 12345

    @patch("ahc_agent.services.submit_service.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_submit_solution_file_success_non_nested_path(self, mock_file, mock_makedirs, mock_config, mock_kb):
        service = SubmitService(config=mock_config, knowledge_base=mock_kb)
        output_file_path = "solution_root.cpp"

        result = service.submit_solution(session_id="test_session_id", output_path=output_file_path)

        # os.makedirs should not be called if os.path.dirname(output_file_path) is empty
        mock_makedirs.assert_not_called()

        mock_file.assert_called_once_with(output_file_path, "w")
        mock_file().write.assert_called_once_with("sample_code();")
        assert result["output_path"] == output_file_path

    def test_submit_solution_session_not_found(self, mock_config, mock_kb):
        mock_kb.get_session.return_value = None
        service = SubmitService(config=mock_config, knowledge_base=mock_kb)

        with pytest.raises(ValueError, match="Session test_session_id not found"):
            service.submit_solution(session_id="test_session_id")

    def test_submit_solution_no_solution_found(self, mock_config, mock_kb):
        mock_kb.get_best_solution.return_value = None
        service = SubmitService(config=mock_config, knowledge_base=mock_kb)

        with pytest.raises(ValueError, match="No solution found for session test_session_id"):
            service.submit_solution(session_id="test_session_id")

    def test_submit_solution_no_code_in_solution(self, mock_config, mock_kb):
        mock_kb.get_best_solution.return_value = {"score": 0}
        service = SubmitService(config=mock_config, knowledge_base=mock_kb)

        with pytest.raises(ValueError, match="No code found in best solution for session test_session_id"):
            service.submit_solution(session_id="test_session_id")

    @patch("ahc_agent.services.submit_service.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_submit_solution_file_write_error(self, mock_file, mock_makedirs, mock_config, mock_kb):
        # mock_makedirs is patched to avoid issues if the path is nested, focus is on open error
        mock_makedirs.return_value = None

        mock_file.side_effect = OSError("Disk full")
        service = SubmitService(config=mock_config, knowledge_base=mock_kb)
        output_file_path = "output/error_solution.cpp"

        with pytest.raises(RuntimeError, match=f"Error writing solution to {output_file_path}: Disk full"):
            service.submit_solution(session_id="test_session_id", output_path=output_file_path)

    @patch("ahc_agent.services.submit_service.os.makedirs", side_effect=OSError("Cannot create dir"))
    @patch("builtins.open", new_callable=mock_open)
    def test_submit_solution_makedirs_error(self, mock_open_file, mock_makedirs, mock_config, mock_kb):
        # This test assumes that if os.makedirs fails, the file open will not be attempted
        # and the error from makedirs (if not caught and re-raised) would propagate.
        # The current SubmitService code has os.makedirs inside the try-except OSError block
        # so an OSError from makedirs should be caught and re-raised as RuntimeError.

        service = SubmitService(config=mock_config, knowledge_base=mock_kb)
        output_file_path = "very/nested/output/solution.cpp"

        expected_message = f"Error writing solution to {output_file_path}: Cannot create dir"
        with pytest.raises(RuntimeError, match=expected_message):
            service.submit_solution(session_id="test_session_id", output_path=output_file_path)

        mock_makedirs.assert_called_once_with(os.path.dirname(output_file_path), exist_ok=True)
        mock_open_file.assert_not_called()
        # builtins.open should not have been called if makedirs failed and error was raised
        # This depends on how builtins.open is patched (globally or locally).
        # If patched globally, we'd need to check its call_count or assert_not_called.
        # For this test, let's assume if RuntimeError from makedirs is raised, open isn't reached.
        # To be explicit, one could add @patch("builtins.open") and assert it's not called.
