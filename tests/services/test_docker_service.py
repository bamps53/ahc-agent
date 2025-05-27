from unittest.mock import MagicMock, patch

import pytest

from ahc_agent.services.docker_service import DockerService


class TestDockerService:
    @pytest.fixture
    def mock_docker_manager(self):
        # DockerManagerクラスのモックを作成
        manager = MagicMock()
        # 必要な属性とメソッドを設定
        manager.image = "test_image:latest"
        # check_docker_availabilityメソッドが実際には存在しないので、モックで追加
        manager.check_docker_availability = MagicMock()
        return manager

    def test_setup_environment_success(self, mock_docker_manager):
        mock_docker_manager.pull_image.return_value = True
        service = DockerService(docker_manager=mock_docker_manager)

        result = service.setup_environment()

        mock_docker_manager.pull_image.assert_called_once()
        assert result is True

    def test_setup_environment_failure_returns_false(self, mock_docker_manager):
        # Test case where pull_image returns False
        mock_docker_manager.pull_image.return_value = False
        # In the service, if pull_image returns False, it logs an error and returns False.
        # It does not raise an exception in this specific path.
        service = DockerService(docker_manager=mock_docker_manager)

        result = service.setup_environment()

        mock_docker_manager.pull_image.assert_called_once()
        assert result is False

    def test_setup_environment_failure_raises_exception(self, mock_docker_manager):
        # Test case where pull_image raises an exception
        mock_docker_manager.pull_image.side_effect = Exception("Pull failed badly")
        service = DockerService(docker_manager=mock_docker_manager)

        with pytest.raises(Exception) as exc_info:
            service.setup_environment()

        mock_docker_manager.pull_image.assert_called_once()
        assert "Pull failed badly" in str(exc_info.value)

    @patch("ahc_agent.services.docker_service.os.getcwd")  # Patch os.getcwd used in get_status
    def test_get_status_all_success(self, mock_os_getcwd, mock_docker_manager):
        mock_os_getcwd.return_value = "/mock/current/dir"  # Mock the return of os.getcwd()

        # check_docker_availability does not raise, meaning it's available
        # check_docker_availabilityメソッドが正常に動作するように設定
        mock_docker_manager.check_docker_availability.return_value = None
        mock_docker_manager.run_command.return_value = {"success": True, "stdout": "Test output", "stderr": ""}
        service = DockerService(docker_manager=mock_docker_manager)

        status = service.get_status()

        mock_docker_manager.check_docker_availability.assert_called_once()
        mock_docker_manager.run_command.assert_called_once_with("echo 'Docker test successful'", "/mock/current/dir")
        assert status["docker_available"] is True
        assert status["test_successful"] is True
        assert "Docker is available." in status["message"]
        assert "Docker test command successful." in status["message"]

    @patch("ahc_agent.services.docker_service.os.getcwd")
    def test_get_status_docker_available_test_fails(self, mock_os_getcwd, mock_docker_manager):
        mock_os_getcwd.return_value = "/mock/current/dir"

        # check_docker_availabilityメソッドが正常に動作するように設定
        mock_docker_manager.check_docker_availability.return_value = None
        mock_docker_manager.run_command.return_value = {"success": False, "stdout": "", "stderr": "Test failed miserably"}
        service = DockerService(docker_manager=mock_docker_manager)

        status = service.get_status()

        mock_docker_manager.check_docker_availability.assert_called_once()
        mock_docker_manager.run_command.assert_called_once_with("echo 'Docker test successful'", "/mock/current/dir")
        assert status["docker_available"] is True
        assert status["test_successful"] is False
        assert "Docker is available." in status["message"]
        assert "Docker test command failed: Test failed miserably" in status["message"]

    def test_get_status_docker_not_available(self, mock_docker_manager):
        # No need to patch os.getcwd here as run_command should not be called
        # check_docker_availabilityメソッドが例外を発生させるように設定
        mock_docker_manager.check_docker_availability.side_effect = RuntimeError("Docker daemon not running")
        service = DockerService(docker_manager=mock_docker_manager)

        status = service.get_status()

        mock_docker_manager.check_docker_availability.assert_called_once()
        mock_docker_manager.run_command.assert_not_called()
        assert status["docker_available"] is False
        assert status["test_successful"] is False  # Ensure this is explicitly False
        assert "Docker is not available: Docker daemon not running" in status["message"]

    @patch("ahc_agent.services.docker_service.os.getcwd")
    def test_get_status_run_command_returns_none(self, mock_os_getcwd, mock_docker_manager):
        mock_os_getcwd.return_value = "/mock/current/dir"
        # check_docker_availabilityメソッドが正常に動作するように設定
        mock_docker_manager.check_docker_availability.return_value = None
        mock_docker_manager.run_command.return_value = None  # Simulate unexpected return from run_command
        service = DockerService(docker_manager=mock_docker_manager)

        status = service.get_status()

        mock_docker_manager.check_docker_availability.assert_called_once()
        mock_docker_manager.run_command.assert_called_once_with("echo 'Docker test successful'", "/mock/current/dir")
        assert status["docker_available"] is True
        assert status["test_successful"] is False
        assert "Docker test command did not return a valid result." in status["message"]

    def test_cleanup_environment_success(self, mock_docker_manager):
        mock_docker_manager.cleanup.return_value = True
        service = DockerService(docker_manager=mock_docker_manager)

        result = service.cleanup_environment()

        mock_docker_manager.cleanup.assert_called_once()
        assert result is True

    def test_cleanup_environment_failure_returns_false(self, mock_docker_manager):
        mock_docker_manager.cleanup.return_value = False
        # Service's cleanup_environment returns False if manager.cleanup() returns False
        service = DockerService(docker_manager=mock_docker_manager)

        result = service.cleanup_environment()

        mock_docker_manager.cleanup.assert_called_once()
        assert result is False

    def test_cleanup_environment_failure_raises_exception(self, mock_docker_manager):
        mock_docker_manager.cleanup.side_effect = Exception("Cleanup failed badly")
        service = DockerService(docker_manager=mock_docker_manager)

        with pytest.raises(Exception) as exc_info:
            service.cleanup_environment()

        mock_docker_manager.cleanup.assert_called_once()
        assert "Cleanup failed badly" in str(exc_info.value)
