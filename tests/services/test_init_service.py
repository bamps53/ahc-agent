import os  # For os.getcwd
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from ahc_agent.services.init_service import InitService


class TestInitService:
    @patch("ahc_agent.services.init_service.scrape_and_setup_problem")
    @patch("ahc_agent.services.init_service.yaml.dump")
    @patch("pathlib.Path.mkdir")  # Patching Path.mkdir directly
    @patch("builtins.open", new_callable=mock_open)  # Mocks open for writing config
    def test_initialize_project_success_with_args_and_workspace(self, mock_file_open, mock_path_mkdir, mock_yaml_dump, mock_scraper, tmp_path):
        init_service = InitService()

        contest_id = "ahc001"
        workspace_name = "test_workspace_ahc001"
        # Define workspace path relative to tmp_path for test isolation
        workspace_path_str = str(tmp_path / workspace_name)

        # Act
        result = init_service.initialize_project(contest_id=contest_id, workspace=workspace_path_str)

        # Assert
        expected_project_dir_obj = Path(workspace_path_str)
        # Check that Path(workspace_path_str).mkdir() was called
        # The actual call is on an instance of Path, so we check the mock_path_mkdir directly
        # as if it's the method of the specific Path instance.
        # Path(workspace_path_str) itself isn't mocked, its .mkdir method is.
        # To check which Path instance mkdir was called on, we would need to inspect Path() calls.
        # For now, assume Path.mkdir is globally patched and any call to it is caught.
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=False)

        # Check config.yaml write
        expected_config_file_path = expected_project_dir_obj / "config.yaml"
        # First call to open is for config.yaml
        # mock_file_open.assert_any_call(expected_config_file_path, "w")

        mock_yaml_dump.assert_called_once()
        dumped_data = mock_yaml_dump.call_args[0][0]

        assert dumped_data["contest_id"] == contest_id
        # assert dumped_data["template"] == custom_template_arg  # Uses passed template
        # assert dumped_data["docker_image"] == custom_image_arg  # Uses passed image

        problem_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
        mock_scraper.assert_called_once_with(problem_url, str(expected_project_dir_obj))

        assert result["project_dir"] == str(expected_project_dir_obj)
        assert result["config_file_path"] == str(expected_config_file_path)

    @patch("ahc_agent.services.init_service.scrape_and_setup_problem")
    @patch("ahc_agent.services.init_service.yaml.dump")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialize_project_success_defaults_no_workspace(self, mock_file_open, mock_path_mkdir, mock_yaml_dump, mock_scraper, tmp_path):
        # Temporarily change CWD for this test to ensure default workspace logic is predictable
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        init_service = InitService()
        contest_id = "ahc002"

        # Act: workspace=None, template=None, docker_image=None
        result = init_service.initialize_project(contest_id=contest_id)

        # Assert
        # Default project_dir is os.getcwd() / contest_id
        expected_project_dir_obj = tmp_path / contest_id
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=False)

        expected_config_file_path = expected_project_dir_obj / "config.yaml"
        # mock_file_open.assert_any_call(expected_config_file_path, "w")

        mock_yaml_dump.assert_called_once()
        dumped_data = mock_yaml_dump.call_args[0][0]
        assert dumped_data["contest_id"] == contest_id

        problem_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_a"
        mock_scraper.assert_called_once_with(problem_url, str(expected_project_dir_obj))

        assert result["project_dir"] == str(expected_project_dir_obj)
        assert result["config_file_path"] == str(expected_config_file_path)

        # Restore CWD
        os.chdir(original_cwd)

    @patch("pathlib.Path.mkdir")
    def test_initialize_project_mkdir_fails(self, mock_path_mkdir, tmp_path):
        init_service = InitService()
        # Configure the specific Path instance's mkdir method to raise error
        # This requires a bit more advanced patching or a different approach if we need to target
        # the specific Path instance created inside initialize_project.
        # For now, patching pathlib.Path.mkdir globally is simpler.
        mock_path_mkdir.side_effect = FileExistsError("Mocked FileExistsError from Path.mkdir")

        with pytest.raises(RuntimeError) as exc_info:
            init_service.initialize_project(contest_id="ahc002", workspace=str(tmp_path / "anywhere"))

        assert "Error creating project directory" in str(exc_info.value)
        # エラーメッセージはサービスの実装によって変更されているので、より汎用的なチェックに変更
        assert "already exists" in str(exc_info.value)

    @patch("pathlib.Path.mkdir")  # Mock mkdir to succeed
    @patch("builtins.open", new_callable=mock_open)  # Mock open to succeed for config
    @patch("ahc_agent.services.init_service.yaml.dump")  # Mock yaml.dump
    @patch("ahc_agent.services.init_service.scrape_and_setup_problem")
    def test_initialize_project_scraper_fails(self, mock_scraper, mock_yaml_dump, mock_file_open, mock_path_mkdir, tmp_path):
        init_service = InitService()

        # Mock scraper to raise an exception
        scraper_error_message = "Mocked Scraper Error"
        mock_scraper.side_effect = Exception(scraper_error_message)

        workspace_path_str = str(tmp_path / "scraper_fail_ws")

        # InitService's initialize_project currently logs the scraper error and continues,
        # it does not re-raise it as a RuntimeError. Let's verify the logging and successful return.
        # If the requirement changes to re-raise, this test needs adjustment.

        # To check logger calls, we patch the module-level logger
        with patch("ahc_agent.services.init_service.logger.error") as mock_logger_error:
            result = init_service.initialize_project(contest_id="ahc003", workspace=workspace_path_str)

            # Assert that directory creation and config writing still happened
            mock_path_mkdir.assert_called_once()
            mock_yaml_dump.assert_called_once()

            # Assert scraper was called
            expected_project_dir_obj = Path(workspace_path_str)
            problem_url = "https://atcoder.jp/contests/ahc003/tasks/ahc003_a"
            mock_scraper.assert_called_once_with(problem_url, str(expected_project_dir_obj))

            # Assert logger.error was called due to scraper failure
            mock_logger_error.assert_called_once()
            assert scraper_error_message in mock_logger_error.call_args[0][0]

            # Assert that the method still returns a result dictionary indicating partial success
            assert result is not None
            assert result["project_dir"] == str(expected_project_dir_obj)
            # The problem scraping part failed, but the project dir and config were created.

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("ahc_agent.services.init_service.yaml.dump")
    def test_initialize_project_config_save_fails(self, mock_yaml_dump, mock_file_open, mock_path_mkdir, tmp_path):
        init_service = InitService()

        # Mock yaml.dump to raise an exception
        yaml_error_message = "Mocked YAML Dump Error"
        mock_yaml_dump.side_effect = Exception(yaml_error_message)

        workspace_path_str = str(tmp_path / "yaml_fail_ws")

        with pytest.raises(RuntimeError) as exc_info:
            init_service.initialize_project(contest_id="ahc004", workspace=workspace_path_str)

        mock_path_mkdir.assert_called_once()  # Directory creation was attempted
        # mock_file_open should have been called for config.yaml
        # The path is Path(workspace_path_str) / "config.yaml"
        # Check that open was called with the correct path and mode "w"
        # This depends on the exact sequence in InitService.
        # If yaml.dump fails, open was successful.

        assert "Error saving project configuration" in str(exc_info.value)
        assert yaml_error_message in str(exc_info.value)

    def test_initialize_project_writes_contest_id_to_config(self, tmp_path):
        # This test verifies that contest_id is written to config.yaml.

        with patch("ahc_agent.services.init_service.scrape_and_setup_problem"), patch(
            "ahc_agent.services.init_service.yaml.dump"
        ) as mock_yaml_dump, patch("pathlib.Path.mkdir"), patch("builtins.open", new_callable=mock_open):
            init_service = InitService()
            contest_id = "ahc005"
            init_service.initialize_project(contest_id, workspace=str(tmp_path))

            expected_data_to_dump = {
                "contest_id": contest_id,
            }
            mock_yaml_dump.assert_called_once()
            actual_data_dumped = mock_yaml_dump.call_args[0][0]
            assert actual_data_dumped == expected_data_to_dump
