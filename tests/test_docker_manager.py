"""
Unit tests for Docker manager.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ahc_agent.utils.docker_manager import DockerManager


class TestDockerManager:
    """
    Tests for DockerManager.
    """

    @pytest.fixture()
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture()
    def docker_manager(self, temp_workspace):
        """
        Create a DockerManager instance for testing, with _check_docker mocked.
        """
        test_config = {
            "docker.image_name": "test_image_from_config",
            "docker.host_workspace_dir": temp_workspace,
            "docker.container_workspace_dir": "/app/test_container_workspace",
            "docker.enabled": True,
            "output_dir_name": "output",
            "input_dir_name": "input",
            "enabled": True,  # Add enabled: True
        }

        with patch.object(DockerManager, "_check_docker", return_value=True):
            # Pass the test_config directly to DockerManager
            dm = DockerManager(config=test_config)
            dm.workspace_dir = temp_workspace  # Ensure workspace_dir is set for tests
            yield dm

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_run_command_error(self, mock_subprocess_run, docker_manager, temp_workspace):
        mock_subprocess_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")
        result = docker_manager.run_command("error_command", temp_workspace)
        assert not result["success"]
        assert result["stderr"] == "Error"
        mock_subprocess_run.assert_called_once()

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_run_container_with_input_file(self, mock_subprocess_run, docker_manager, temp_workspace):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="test input", stderr="")

        input_filename = "input.txt"
        host_input_file_path = os.path.join(temp_workspace, input_filename)
        input_content = "test input"
        with open(host_input_file_path, "w+") as f:
            f.write(input_content)

        # The command will be executed in the container's working directory,
        # which is docker_manager.mount_path (e.g., /workspace)
        command = f"cat {input_filename}"

        # docker_manager.workspace_dir is set to temp_workspace in the fixture.
        # run_command mounts temp_workspace to docker_manager.mount_path.
        result = docker_manager.run_command(command, work_dir=temp_workspace)
        stdout = result["stdout"]
        stderr = result["stderr"]

        expected_docker_command_args = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.path.abspath(temp_workspace)}:{docker_manager.mount_path}",
            "-w",
            docker_manager.mount_path,
            docker_manager.image,  # Image from DockerManager's config
            "/bin/bash",
            "-c",
            command,
        ]

        mock_subprocess_run.assert_called_once_with(
            expected_docker_command_args, capture_output=True, text=True, check=False, timeout=docker_manager.timeout
        )
        assert stdout.strip() == input_content
        assert stderr == ""

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_init(self, mock_subprocess_run, docker_manager):
        """
        Test initialization.
        """
        # docker.from_env (Python library) is not used by DockerManager's __init__
        mock_subprocess_run.assert_not_called()

        # Check attributes from config
        assert docker_manager.config.get("enabled") is True
        assert docker_manager.image == DockerManager.DEFAULT_CONFIG["image"]  # Use image from DEFAULT_CONFIG
        assert docker_manager.mount_path == "/workspace"
        assert docker_manager.timeout == 300
        assert docker_manager.cpp_compiler == "g++"  # Assert instance attribute
        assert docker_manager.cpp_flags == "-std=c++17 -O2 -Wall"  # Assert instance attribute
        assert docker_manager.rust_compiler == "rustc"  # Assert instance attribute
        assert docker_manager.rust_flags == "-C opt-level=3"  # Assert instance attribute
        assert docker_manager.java_compiler == "javac"  # Assert instance attribute
        assert docker_manager.java_flags == ""  # Assert instance attribute
        assert docker_manager.python_interpreter == "python3"  # Assert instance attribute
        assert docker_manager.python_flags == ""  # Assert instance attribute

    @patch("ahc_agent.utils.docker_manager.DockerManager._check_docker")
    @patch("docker.from_env")
    def test_init_disabled(self, mock_docker_from_env, mock_check_docker_method):
        """
        Test initialization with Docker disabled in config.
        """
        config = {"enabled": False, "image": "mcr.microsoft.com/devcontainers/rust:1-1-bullseye"}
        manager = DockerManager(config)
        mock_check_docker_method.assert_not_called()  # Should not be called when disabled
        mock_docker_from_env.assert_not_called()
        assert manager.enabled is False
        assert manager.image == "mcr.microsoft.com/devcontainers/rust:1-1-bullseye"

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_pull_image(self, mock_subprocess_run, docker_manager):
        """
        Test pull_image method.
        """
        # Mock subprocess.run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        # Call pull_image
        result = docker_manager.pull_image()

        # Check result
        assert result is True

        # Check subprocess.run calls
        mock_subprocess_run.assert_called_once_with(
            ["docker", "pull", docker_manager.image],
            capture_output=True,
            text=True,
            check=False,
            timeout=docker_manager.timeout,  # Use docker_manager.timeout
        )

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_pull_image_error(self, mock_subprocess_run, docker_manager):
        """
        Test pull_image method with error.
        """
        # Mock subprocess.run for error case
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Test error pulling image"
        mock_subprocess_run.return_value = mock_result

        # Call pull_image
        result = docker_manager.pull_image()

        # Check result
        assert result is False

        # Check subprocess.run calls
        mock_subprocess_run.assert_called_once_with(
            ["docker", "pull", docker_manager.image],
            capture_output=True,
            text=True,
            check=False,
            timeout=docker_manager.timeout,  # Use docker_manager.timeout
        )

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_run_command(self, mock_subprocess_run, docker_manager, temp_workspace):
        """
        Test run_command method.
        """
        # Mock subprocess.run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Test output"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        # Call run_command
        result = docker_manager.run_command("echo 'Test output'", temp_workspace)

        # Check result
        assert result["success"] is True
        assert result["stdout"] == "Test output"
        assert result["stderr"] == ""
        assert result["returncode"] == 0

        # Check subprocess.run calls
        expected_docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.path.abspath(temp_workspace)}:{docker_manager.mount_path}",
            "-w",
            docker_manager.mount_path,
            docker_manager.image,
            "/bin/bash",
            "-c",
            "echo 'Test output'",
        ]
        mock_subprocess_run.assert_called_once_with(expected_docker_cmd, capture_output=True, text=True, check=False, timeout=docker_manager.timeout)

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_run_command_with_input(self, mock_subprocess_run, docker_manager, temp_workspace):
        """
        Test run_command method with input (simulated via command string).
        """
        # Mock subprocess.run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Input was: Test input"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        command_with_input = "sh -c 'echo Input was: $(cat input.txt)'"
        # Create a dummy input file for the test to simulate reading from it
        # In a real scenario, the command itself would handle the input file creation if needed
        # or the input would be piped.
        # For simplicity, we assume the command string itself handles input or it's pre-placed.

        # Call run_command
        result = docker_manager.run_command(command_with_input, temp_workspace)

        # Check result
        assert result["success"] is True
        assert result["stdout"] == "Input was: Test input"
        assert result["stderr"] == ""
        assert result["returncode"] == 0

        # Check subprocess.run calls
        expected_docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.path.abspath(temp_workspace)}:{docker_manager.mount_path}",
            "-w",
            docker_manager.mount_path,
            docker_manager.image,
            "/bin/bash",
            "-c",
            command_with_input,
        ]
        mock_subprocess_run.assert_called_once_with(expected_docker_cmd, capture_output=True, text=True, check=False, timeout=docker_manager.timeout)

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_cleanup(self, mock_subprocess_run, docker_manager):
        """
        Test cleanup method.
        """
        # Mock subprocess.run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess_run.return_value = mock_result

        # Call cleanup
        result = docker_manager.cleanup()

        # Check result
        assert result is True

        # Check subprocess.run calls
        mock_subprocess_run.assert_called_once_with(
            ["docker", "container", "prune", "-f"],
            capture_output=True,  # Change to capture_output=True
            check=False,
        )

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_cleanup_error(self, mock_subprocess_run, docker_manager):
        """
        Test cleanup method with error.
        """
        # Mock subprocess.run to raise an exception or return non-zero
        # The cleanup method should catch this and return False
        mock_subprocess_run.return_value = MagicMock(returncode=1, stderr="Cleanup error")
        # Alternatively, to simulate an exception during subprocess.run:
        # mock_subprocess_run.side_effect = Exception("Test error cleaning up")

        # Call cleanup
        result = docker_manager.cleanup()

        # Check result - should be False as an error occurred
        assert result is False

        # Check subprocess.run calls
        mock_subprocess_run.assert_called_once_with(["docker", "container", "prune", "-f"], capture_output=True, check=False)

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_compile_cpp(self, mock_subprocess_run, docker_manager):
        """
        Test compile_cpp method.
        """
        # Mock subprocess.run for the compile command
        mock_compile_result = MagicMock()
        mock_compile_result.returncode = 0
        mock_compile_result.stdout = "Compilation successful"
        mock_compile_result.stderr = ""
        mock_subprocess_run.return_value = mock_compile_result

        # Call compile_cpp
        result = docker_manager.compile_cpp("test.cpp", "/tmp")

        # Check result
        assert result["success"] is True
        assert result["stdout"] == "Compilation successful"
        assert result["stderr"] == ""
        assert result["returncode"] == 0
        assert "executable_path" in result
        assert result["executable_path"] == os.path.join("/tmp", "test")

        # Check subprocess.run calls
        expected_compile_cmd_str_part = "g++ -std=c++17 -O2 -Wall test.cpp -o test"

        # Get the actual command list called
        called_args_list = mock_subprocess_run.call_args[0][0]
        assert called_args_list[0] == "docker"
        assert called_args_list[1] == "run"
        # The actual command is the last element in the list
        actual_compile_command_str = called_args_list[-1]
        assert actual_compile_command_str == expected_compile_cmd_str_part

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_run_cpp(self, mock_subprocess_run, docker_manager):
        """
        Test run_cpp method.
        """
        # Mock subprocess.run for the run command
        mock_run_result = MagicMock()
        mock_run_result.returncode = 0
        # Simulate stderr containing time output and actual program output in stdout
        mock_run_result.stdout = "Program output"
        mock_run_result.stderr = "real    0.01s\nuser    0.00s\nsys     0.00s"
        mock_subprocess_run.return_value = mock_run_result

        # Call run_cpp
        # Note: run_cpp creates a temp file for input_data if provided.
        # The command executed will be something like 'time -p ./test < temp_input_file 2>&1'
        with patch("tempfile.NamedTemporaryFile") as mock_temp_file_constructor:
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/temp_input_file_mock"
            mock_temp_file_constructor.return_value.__enter__.return_value = mock_temp_file

            result = docker_manager.run_cpp("test_exec", "/tmp", input_data="test input", timeout=10)

        # Check result
        assert result["success"] is True
        assert result["stdout"] == "Program output"
        # stderr from subprocess.run includes time output, run_cpp extracts execution_time
        assert "real    0.01s" in result["stderr"]
        assert result["returncode"] == 0
        assert result["execution_time"] == 0.01

        # Check subprocess.run calls
        called_args_list = mock_subprocess_run.call_args[0][0]
        actual_run_command_str = called_args_list[-1]
        # The input file name is dynamic, so we check parts of the command
        assert actual_run_command_str.startswith("time -p ./test_exec < ")
        assert actual_run_command_str.endswith(" 2>&1")

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_copy_to_container(self, mock_subprocess_run, docker_manager, temp_workspace):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        container_id = "test_container_for_copy"

        # Create a dummy source file on the host
        src_filename = "test_source_file.txt"
        host_src_path = os.path.join(temp_workspace, src_filename)
        with open(host_src_path, "w") as f:
            f.write("Test content for copy to container")

        container_dest_path = "/app/destination_file.txt"

        result = docker_manager.copy_to_container(container_id, host_src_path, container_dest_path)

        assert result["success"] is True
        expected_command = ["docker", "cp", host_src_path, f"{container_id}:{container_dest_path}"]
        mock_subprocess_run.assert_called_once_with(expected_command, capture_output=True, text=True, check=False, timeout=docker_manager.timeout)

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_copy_from_container(self, mock_subprocess_run, docker_manager, temp_workspace):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        container_id = "test_container_for_copy_from"
        container_src_path = "/app/source_file_in_container.txt"
        host_dest_path = os.path.join(temp_workspace, "destination_file_on_host.txt")

        # We don't actually create the source file in the container for this mock test,
        # as subprocess.run is mocked.

        result = docker_manager.copy_from_container(container_id, container_src_path, host_dest_path)

        assert result["success"] is True
        expected_command = ["docker", "cp", f"{container_id}:{container_src_path}", host_dest_path]
        mock_subprocess_run.assert_called_once_with(expected_command, capture_output=True, text=True, check=False, timeout=docker_manager.timeout)

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_build_image_existing_dockerfile(self, mock_subprocess_run, docker_manager, temp_workspace):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="Image built successfully", stderr="")

        # Create a dummy Dockerfile in the temporary workspace
        dockerfile_content = "FROM alpine\nCMD echo 'Hello from Docker'"
        dockerfile_path = os.path.join(temp_workspace, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        image_tag = "test-image:latest"
        result = docker_manager.build_image(context_path=temp_workspace, image_tag=image_tag)

        assert result["success"] is True
        assert result["stdout"] == "Image built successfully"
        expected_command = ["docker", "build", "-t", image_tag, temp_workspace]
        mock_subprocess_run.assert_called_once_with(
            expected_command,
            capture_output=True,
            text=True,
            check=False,
            timeout=docker_manager.build_timeout,  # Use build_timeout here
        )

    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_build_image_no_dockerfile_or_context(self, mock_subprocess_run, docker_manager, temp_workspace):
        # Case 1: Context path does not exist
        non_existent_context_path = os.path.join(temp_workspace, "non_existent_dir")
        # subprocess.run should not be called if context path check fails early,
        # but if it were, it would likely be an error from docker CLI.
        # For now, assume DockerManager might not pre-check path validity before calling docker CLI.
        mock_subprocess_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: path not found")

        result_no_context = docker_manager.build_image(context_path=non_existent_context_path)
        assert result_no_context["success"] is False
        # Depending on implementation, stderr might come from our code or docker
        # assert "Error: path not found" in result_no_context["stderr"]
        # subprocess.run should have been called if DockerManager doesn't pre-validate
        mock_subprocess_run.assert_called_with(
            ["docker", "build", "-t", docker_manager.image, non_existent_context_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=docker_manager.build_timeout,
        )
        mock_subprocess_run.reset_mock()  # Reset mock for the next case

        # Case 2: Dockerfile does not exist in a valid context path
        # Create a valid context directory
        valid_context_path = os.path.join(temp_workspace, "valid_context")
        os.makedirs(valid_context_path, exist_ok=True)
        non_existent_dockerfile = "Dockerfile.nonexistent"

        mock_subprocess_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error: Dockerfile not found")
        result_no_dockerfile = docker_manager.build_image(context_path=valid_context_path, dockerfile=non_existent_dockerfile)

        assert result_no_dockerfile["success"] is False
        # assert "Error: Dockerfile not found" in result_no_dockerfile["stderr"]
        expected_command_no_dockerfile = [
            "docker",
            "build",
            "-t",
            docker_manager.image,
            "-f",
            os.path.join(valid_context_path, non_existent_dockerfile),
            valid_context_path,
        ]
        mock_subprocess_run.assert_called_with(
            expected_command_no_dockerfile,
            capture_output=True,
            text=True,
            check=False,
            timeout=docker_manager.build_timeout,
        )

    @pytest.mark.skip(reason="Not yet implemented")
    @patch("ahc_agent.utils.docker_manager.subprocess.run")
    def test_run_container_with_volume_mapping(self, mock_subprocess_run, docker_manager, temp_workspace):
        pass
