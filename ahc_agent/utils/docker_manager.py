import logging
import os
import re
import subprocess
import tempfile
from typing import Any, ClassVar, Dict, Optional

logger = logging.getLogger(__name__)


class DockerManager:
    """
    Manager for Docker containers.
    """

    DEFAULT_CONFIG: ClassVar[Dict[str, Any]] = {
        "image": "ubuntu:latest",
        "mount_path": "/workspace",
        "timeout": 300,  # Default timeout in seconds
        "cpp_compiler": "g++",
        "cpp_flags": "-std=c++17 -O2 -Wall",
        "enabled": True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Docker manager.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.image = self.config.get("image")
        self.mount_path = self.config.get("mount_path")
        self.timeout = self.config.get("timeout")
        self.cpp_compiler = self.config.get("cpp_compiler")
        self.cpp_flags = self.config.get("cpp_flags")
        self.enabled = self.config.get("enabled")
        self.build_timeout = self.config.get("build_timeout", 300)  # Add build_timeout

        if self.enabled:
            self._check_docker()
            self.logger.info(f"Initialized Docker manager with image: {self.image}")
        else:
            self.logger.info("Docker manager is disabled by configuration.")

    def _check_docker(self) -> None:
        """
        Check if Docker is available.

        Raises:
            RuntimeError: If Docker is not available
        """
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logger.error(f"Docker not available: {result.stderr}")
                raise RuntimeError("Docker is not available. Please install Docker and try again.")

            logger.debug(f"Docker version: {result.stdout.strip()}")

        except FileNotFoundError as e:
            logger.error("Docker command not found")
            raise RuntimeError("Docker command not found. Please install Docker and try again.") from e

    def pull_image(self) -> bool:
        """
        Pull the Docker image.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("Docker is disabled, skipping image pull.")
            return True  # Or False, depending on desired behavior for disabled state

        self.logger.info(f"Pulling Docker image: {self.image}")
        try:
            result = subprocess.run(
                ["docker", "pull", self.image],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
            if result.returncode == 0:
                self.logger.info(f"Successfully pulled Docker image: {self.image}")
                return True
            self.logger.error(f"Failed to pull Docker image: {result.stderr.strip()}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout occurred while pulling Docker image: {self.image}")
            return False
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred while pulling Docker image: {self.image}", exc_info=e)
            return False

    def run_command(self, command: str, work_dir: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a command in a Docker container.

        Args:
            command: Command to run
            work_dir: Working directory (will be mounted in container)
            timeout: Timeout in seconds (default: self.timeout)

        Returns:
            Dictionary with command result
                - success: True if successful, False otherwise
                - stdout: Standard output
                - stderr: Standard error
                - returncode: Return code
                - error: Error message if any
        """
        try:
            # Ensure work_dir exists
            os.makedirs(work_dir, exist_ok=True)

            # Get absolute path
            work_dir_abs = os.path.abspath(work_dir)

            # Prepare Docker command
            docker_cmd = [
                "docker",
                "run",
                "--rm",  # Remove container after execution
                "-v",
                f"{work_dir_abs}:{self.mount_path}",  # Mount work_dir
                "-w",
                self.mount_path,  # Set working directory
                self.image,  # Image to use
                "/bin/bash",
                "-c",
                command,  # Command to run
            ]

            logger.debug(f"Running Docker command: {' '.join(docker_cmd)}")

            # Run Docker command
            process = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout or self.timeout, check=False)
            return {
                "success": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "returncode": process.returncode,
                "error": "",
            }

        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout or self.timeout} seconds: {command}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"TimeoutExpired: {e!s}",
                "returncode": -1,  # Or some other indicator of timeout
                "error": f"TimeoutExpired: {e!s}",
            }
        except (FileNotFoundError, OSError, RuntimeError) as e:
            logger.error(f"Error running command '{command}': {e!s}")
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1, "error": str(e)}

    def compile_cpp(self, source_file: str, work_dir: str, output_file: Optional[str] = None, flags: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile a C++ source file in a Docker container.

        Args:
            source_file: Path to source file (relative to work_dir)
            work_dir: Working directory
            output_file: Path to output file (relative to work_dir, default: source_file without extension)
            flags: Compiler flags (default: "-std=c++17 -O2 -Wall")

        Returns:
            Dictionary with compilation result
                - success: True if successful, False otherwise
                - executable_path: Path to executable if successful
                - stdout: Standard output
                - stderr: Standard error
                - error: Error message if any
        """
        try:
            # Get output file name
            if output_file is None:
                output_file = os.path.splitext(source_file)[0]

            # Get compiler flags
            if flags is None:
                flags = "-std=c++17 -O2 -Wall"

            # Prepare compile command
            compile_cmd = f"g++ {flags} {source_file} -o {output_file}"

            # Run command
            result = self.run_command(compile_cmd, work_dir)
            original_stdout = result["stdout"]
            original_stderr = result["stderr"]

            if not result["success"]:
                error_lines = []
                for line in original_stderr.splitlines():
                    if re.search(r".*error:.*|.*warning:.*", line):
                        error_lines.append(line)
                result["stderr"] = "\n".join(error_lines)
                result["stdout"] = "Compilation failed. See stderr for details."
            else:
                result["stdout"] = "Compilation successful."
                result["stderr"] = ""
                # Get absolute path to executable
                executable_path = os.path.join(work_dir, output_file)
                result["executable_path"] = executable_path

            result["original_stdout"] = original_stdout
            result["original_stderr"] = original_stderr

            return result

        except (OSError, RuntimeError) as e:
            logger.error(f"Error compiling C++ file: {e!s}")
            return {"success": False, "stdout": "", "stderr": str(e), "error": str(e), "original_stdout": "", "original_stderr": str(e)}

    def run_cpp(self, executable_file: str, work_dir: str, input_data: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a compiled C++ executable in a Docker container.

        Args:
            executable_file: Path to executable (relative to work_dir)
            work_dir: Working directory
            input_data: Input data to provide to the executable
            timeout: Timeout in seconds (default: self.timeout)

        Returns:
            Dictionary with execution result
                - success: True if successful, False otherwise
                - stdout: Standard output
                - stderr: Standard error
                - returncode: Return code
                - error: Error message if any
                - execution_time: Execution time in seconds (if available)
        """
        try:
            # Create temporary input file if input_data is provided
            input_file = None
            if input_data:
                with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=work_dir) as f:
                    f.write(input_data)
                    input_file = os.path.basename(f.name)

            # Prepare run command
            base_run_cmd = f"./{executable_file} < {input_file}" if input_file else f"./{executable_file}"

            # Add time measurement
            run_cmd = f"time -p {base_run_cmd} 2>&1"

            # Run command
            result = self.run_command(run_cmd, work_dir, timeout)

            # Parse execution time if available
            execution_time = None
            if result["success"]:
                # Try to extract execution time from stderr
                import re

                time_match = re.search(r"real\s+(\d+\.\d+)", result["stderr"])
                if time_match:
                    execution_time = float(time_match.group(1))
                    result["execution_time"] = execution_time

            # Clean up temporary input file
            if input_file:
                try:
                    os.unlink(os.path.join(work_dir, input_file))
                except OSError as e:
                    logger.warning(f"Failed to delete temporary input file: {e!s}")

            return result

        except (FileNotFoundError, OSError, RuntimeError) as e:
            logger.error(f"Error running C++ executable: {e!s}")
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1, "error": str(e)}

    def build_image(self, context_path: str, dockerfile: Optional[str] = None, image_tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Build a Docker image from a Dockerfile.

        Args:
            context_path: Path to the build context.
            dockerfile: Name of the Dockerfile (e.g., 'Dockerfile.custom').
                        If None, defaults to 'Dockerfile' in the context_path.
            image_tag: Tag for the built image (e.g., 'myimage:latest').
                       If None, defaults to self.image.

        Returns:
            Dictionary with operation result:
                - success: True if successful, False otherwise
                - stdout: Standard output
                - stderr: Standard error
                - returncode: Return code
        """
        if not self.enabled:
            self.logger.info("Docker is disabled, skipping image build.")
            return {"success": True, "stdout": "", "stderr": "", "returncode": 0}

        tag_to_use = image_tag if image_tag else self.image
        self.logger.info(f"Building Docker image '{tag_to_use}' from context '{context_path}'")

        cmd = ["docker", "build", "-t", tag_to_use]
        if dockerfile:
            # Ensure dockerfile path is correctly joined with context_path
            dockerfile_path = os.path.join(context_path, dockerfile)
            cmd.extend(["-f", dockerfile_path])
        cmd.append(context_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.build_timeout,  # Consider a longer timeout for builds
            )
            if result.returncode == 0:
                self.logger.info(f"Successfully built image '{tag_to_use}'.")
                return {"success": True, "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
            self.logger.error(f"Failed to build image '{tag_to_use}': {result.stderr.strip()}")
            return {"success": False, "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout occurred while building image '{tag_to_use}'.")
            return {"success": False, "stdout": "", "stderr": "Timeout", "returncode": -1}
        except Exception as e:
            self.logger.error(f"Error building image '{tag_to_use}': {e!s}")
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    def cleanup(self) -> bool:
        """
        Clean up Docker resources (e.g., prune stopped containers).

        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled:
            self.logger.info("Docker is disabled, skipping cleanup.")
            return True

        self.logger.info("Cleaning up Docker resources")
        try:
            # Prune stopped containers
            result = subprocess.run(["docker", "container", "prune", "-f"], capture_output=True, check=False)
            if result.returncode != 0:
                self.logger.error(f"Failed to prune containers: {result.stderr.strip()}")
                return False  # Return False on error
            self.logger.info("Successfully pruned stopped containers.")
            return True
        except Exception as e:
            self.logger.exception("An error occurred during Docker cleanup.", exc_info=e)
            return False
