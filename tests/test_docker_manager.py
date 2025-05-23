"""
Unit tests for Docker manager.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from ahc_agent_cli.utils.docker_manager import DockerManager

class TestDockerManager:
    """
    Tests for DockerManager.
    """
    
    @pytest.fixture
    def docker_manager(self):
        """
        Create a DockerManager instance for testing.
        """
        config = {
            "enabled": True,
            "image": "mcr.microsoft.com/devcontainers/rust:1-1-bullseye",
            "cpp_compiler": "g++",
            "cpp_flags": "-std=c++17 -O2 -Wall"
        }
        return DockerManager(config)
    
    @patch("docker.from_env")
    def test_init(self, mock_docker_from_env, docker_manager):
        """
        Test initialization.
        """
        # Check attributes
        assert docker_manager.enabled is True
        assert docker_manager.image == "mcr.microsoft.com/devcontainers/rust:1-1-bullseye"
        assert docker_manager.cpp_compiler == "g++"
        assert docker_manager.cpp_flags == "-std=c++17 -O2 -Wall"
    
    @patch("docker.from_env")
    def test_init_disabled(self, mock_docker_from_env):
        """
        Test initialization with Docker disabled.
        """
        config = {
            "enabled": False,
            "image": "mcr.microsoft.com/devcontainers/rust:1-1-bullseye"
        }
        manager = DockerManager(config)
        
        # Check attributes
        assert manager.enabled is False
        assert manager.image == "mcr.microsoft.com/devcontainers/rust:1-1-bullseye"
        
        # Docker client should not be initialized
        mock_docker_from_env.assert_not_called()
    
    @patch("docker.from_env")
    def test_pull_image(self, mock_docker_from_env, docker_manager):
        """
        Test pull_image method.
        """
        # Mock Docker client
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client
        
        # Call pull_image
        result = docker_manager.pull_image()
        
        # Check result
        assert result is True
        
        # Check Docker client calls
        mock_client.images.pull.assert_called_once_with(docker_manager.image)
    
    @patch("docker.from_env")
    def test_pull_image_error(self, mock_docker_from_env, docker_manager):
        """
        Test pull_image method with error.
        """
        # Mock Docker client
        mock_client = MagicMock()
        mock_client.images.pull.side_effect = Exception("Test error")
        mock_docker_from_env.return_value = mock_client
        
        # Call pull_image
        result = docker_manager.pull_image()
        
        # Check result
        assert result is False
    
    @patch("docker.from_env")
    def test_run_command(self, mock_docker_from_env, docker_manager):
        """
        Test run_command method.
        """
        # Mock Docker client and container
        mock_container = MagicMock()
        mock_container.logs.return_value = b"Test output"
        mock_container.wait.return_value = {"StatusCode": 0}
        
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_docker_from_env.return_value = mock_client
        
        # Call run_command
        result = docker_manager.run_command("echo 'test'", "/tmp")
        
        # Check result
        assert result["success"] is True
        assert result["stdout"] == "Test output"
        assert result["stderr"] == ""
        
        # Check Docker client calls
        mock_client.containers.run.assert_called_once()
        call_args = mock_client.containers.run.call_args[1]
        assert call_args["image"] == docker_manager.image
        assert call_args["command"] == "echo 'test'"
        assert call_args["volumes"] == {"/tmp": {"bind": "/workspace", "mode": "rw"}}
        assert call_args["working_dir"] == "/workspace"
        assert call_args["detach"] is True
    
    @patch("docker.from_env")
    def test_run_command_with_input(self, mock_docker_from_env, docker_manager):
        """
        Test run_command method with input.
        """
        # Mock Docker client and container
        mock_container = MagicMock()
        mock_container.logs.return_value = b"Test output"
        mock_container.wait.return_value = {"StatusCode": 0}
        
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_docker_from_env.return_value = mock_client
        
        # Call run_command with input
        result = docker_manager.run_command("cat", "/tmp", input_data="test input")
        
        # Check result
        assert result["success"] is True
        assert result["stdout"] == "Test output"
        assert result["stderr"] == ""
        
        # Check Docker client calls
        mock_client.containers.run.assert_called_once()
        call_args = mock_client.containers.run.call_args[1]
        assert call_args["image"] == docker_manager.image
        assert call_args["command"] == "cat"
        assert call_args["volumes"] == {"/tmp": {"bind": "/workspace", "mode": "rw"}}
        assert call_args["working_dir"] == "/workspace"
        assert call_args["detach"] is True
        
        # Check that input was written to container
        mock_container.exec_run.assert_called_once()
        exec_args = mock_container.exec_run.call_args[1]
        assert exec_args["cmd"] == ["sh", "-c", "echo 'test input' | cat"]
    
    @patch("docker.from_env")
    def test_run_command_error(self, mock_docker_from_env, docker_manager):
        """
        Test run_command method with error.
        """
        # Mock Docker client and container
        mock_container = MagicMock()
        mock_container.logs.side_effect = [b"", b"Test error"]
        mock_container.wait.return_value = {"StatusCode": 1}
        
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_docker_from_env.return_value = mock_client
        
        # Call run_command
        result = docker_manager.run_command("invalid_command", "/tmp")
        
        # Check result
        assert result["success"] is False
        assert result["stdout"] == ""
        assert result["stderr"] == "Test error"
    
    @patch("docker.from_env")
    def test_compile_cpp(self, mock_docker_from_env, docker_manager):
        """
        Test compile_cpp method.
        """
        # Mock Docker client and container
        mock_container = MagicMock()
        mock_container.logs.return_value = b"Compilation successful"
        mock_container.wait.return_value = {"StatusCode": 0}
        
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_docker_from_env.return_value = mock_client
        
        # Call compile_cpp
        result = docker_manager.compile_cpp("test.cpp", "/tmp")
        
        # Check result
        assert result["success"] is True
        assert result["stdout"] == "Compilation successful"
        assert result["stderr"] == ""
        
        # Check Docker client calls
        mock_client.containers.run.assert_called_once()
        call_args = mock_client.containers.run.call_args[1]
        assert call_args["image"] == docker_manager.image
        assert "g++" in call_args["command"]
        assert "-std=c++17" in call_args["command"]
        assert "test.cpp" in call_args["command"]
    
    @patch("docker.from_env")
    def test_run_cpp(self, mock_docker_from_env, docker_manager):
        """
        Test run_cpp method.
        """
        # Mock Docker client and container
        mock_container = MagicMock()
        mock_container.logs.return_value = b"Program output"
        mock_container.wait.return_value = {"StatusCode": 0}
        
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_docker_from_env.return_value = mock_client
        
        # Call run_cpp
        result = docker_manager.run_cpp("test", "/tmp", "test input", 10)
        
        # Check result
        assert result["success"] is True
        assert result["stdout"] == "Program output"
        assert result["stderr"] == ""
        
        # Check Docker client calls
        mock_client.containers.run.assert_called_once()
        call_args = mock_client.containers.run.call_args[1]
        assert call_args["image"] == docker_manager.image
        assert "./test" in call_args["command"]
    
    @patch("docker.from_env")
    def test_cleanup(self, mock_docker_from_env, docker_manager):
        """
        Test cleanup method.
        """
        # Mock Docker client
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client
        
        # Call cleanup
        result = docker_manager.cleanup()
        
        # Check result
        assert result is True
        
        # Check Docker client calls
        mock_client.containers.prune.assert_called_once()
    
    @patch("docker.from_env")
    def test_cleanup_error(self, mock_docker_from_env, docker_manager):
        """
        Test cleanup method with error.
        """
        # Mock Docker client
        mock_client = MagicMock()
        mock_client.containers.prune.side_effect = Exception("Test error")
        mock_docker_from_env.return_value = mock_client
        
        # Call cleanup
        result = docker_manager.cleanup()
        
        # Check result
        assert result is False
