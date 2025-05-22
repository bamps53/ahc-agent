"""
Docker manager for AHCAgent CLI.

This module provides utilities for managing Docker containers for code execution.
"""

import os
import json
import logging
import tempfile
import subprocess
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class DockerManager:
    """
    Manager for Docker containers.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Docker manager.
        
        Args:
            config: Configuration dictionary
                - image: Docker image (default: from env var AHCAGENT_DOCKER_IMAGE or "mcr.microsoft.com/devcontainers/rust:1-1-bullseye")
                - mount_path: Path to mount in container (default: "/workspace")
                - timeout: Timeout in seconds (default: 300)
        """
        self.config = config or {}
        
        # Get image from config or env var or default
        self.image = self.config.get("image") or os.environ.get("AHCAGENT_DOCKER_IMAGE") or "mcr.microsoft.com/devcontainers/rust:1-1-bullseye"
        
        # Get mount path from config or default
        self.mount_path = self.config.get("mount_path", "/workspace")
        
        # Get timeout from config or default
        self.timeout = self.config.get("timeout", 300)
        
        # Check if Docker is available
        self._check_docker()
        
        logger.info(f"Initialized Docker manager with image: {self.image}")
    
    def _check_docker(self) -> None:
        """
        Check if Docker is available.
        
        Raises:
            RuntimeError: If Docker is not available
        """
        try:
            result = subprocess.run(
                ["docker", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Docker not available: {result.stderr}")
                raise RuntimeError("Docker is not available. Please install Docker and try again.")
            
            logger.debug(f"Docker version: {result.stdout.strip()}")
        
        except FileNotFoundError:
            logger.error("Docker command not found")
            raise RuntimeError("Docker command not found. Please install Docker and try again.")
    
    def pull_image(self) -> bool:
        """
        Pull the Docker image.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling Docker image: {self.image}")
            
            result = subprocess.run(
                ["docker", "pull", self.image],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to pull Docker image: {result.stderr}")
                return False
            
            logger.info(f"Successfully pulled Docker image: {self.image}")
            return True
        
        except Exception as e:
            logger.error(f"Error pulling Docker image: {str(e)}")
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
                "docker", "run",
                "--rm",  # Remove container after execution
                "-v", f"{work_dir_abs}:{self.mount_path}",  # Mount work_dir
                "-w", self.mount_path,  # Set working directory
                self.image,  # Image to use
                command  # Command to run
            ]
            
            logger.debug(f"Running Docker command: {' '.join(docker_cmd)}")
            
            # Run command
            result = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout or self.timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "error": None
            }
        
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout or self.timeout} seconds")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout or self.timeout} seconds",
                "returncode": -1,
                "error": str(e)
            }
        
        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "error": str(e)
            }
    
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
            
            if result["success"]:
                # Get absolute path to executable
                executable_path = os.path.join(work_dir, output_file)
                result["executable_path"] = executable_path
            
            return result
        
        except Exception as e:
            logger.error(f"Error compiling C++ file: {str(e)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "error": str(e)
            }
    
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
            if input_file:
                run_cmd = f"./{executable_file} < {input_file}"
            else:
                run_cmd = f"./{executable_file}"
            
            # Add time measurement
            run_cmd = f"time -p {run_cmd} 2>&1"
            
            # Run command
            result = self.run_command(run_cmd, work_dir, timeout)
            
            # Parse execution time if available
            execution_time = None
            if result["success"]:
                # Try to extract execution time from stderr
                import re
                time_match = re.search(r'real\s+(\d+\.\d+)', result["stderr"])
                if time_match:
                    execution_time = float(time_match.group(1))
                    result["execution_time"] = execution_time
            
            # Clean up temporary input file
            if input_file:
                try:
                    os.unlink(os.path.join(work_dir, input_file))
                except Exception as e:
                    logger.warning(f"Failed to delete temporary input file: {str(e)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error running C++ executable: {str(e)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "error": str(e)
            }
    
    def cleanup(self) -> bool:
        """
        Clean up unused Docker resources.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Cleaning up Docker resources")
            
            # Remove stopped containers
            subprocess.run(
                ["docker", "container", "prune", "-f"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error cleaning up Docker resources: {str(e)}")
            return False
