import logging
import os

from ahc_agent.utils.docker_manager import DockerManager

logger = logging.getLogger(__name__)


class DockerService:
    def __init__(self, docker_manager: DockerManager):
        self.docker_manager = docker_manager

    def setup_environment(self) -> bool:
        """
        Pulls the Docker image specified by the DockerManager.
        Returns True on success, False or raises an exception on failure.
        """
        # DockerManager.image_name was used in cli.py refactor, let's assume it's the correct attribute
        # If not, it should be self.docker_manager.image or similar, based on DockerManager's actual attribute.
        # For now, using self.docker_manager.image as per the example's hint.
        # If DockerManager has `image_name` as the property for the image identifier, that should be used.
        # The example provided `self.docker_manager.image`. I'll stick to that for consistency with the example.
        image_to_pull = getattr(self.docker_manager, "image", getattr(self.docker_manager, "image_name", "unknown_image_attribute"))

        logger.info(f"Attempting to pull Docker image: {image_to_pull}")
        try:
            success = self.docker_manager.pull_image()
            if success:
                logger.info(f"Docker image '{image_to_pull}' pulled successfully (or was already present).")
            else:
                # This path might not be reached if pull_image() raises exceptions on failure.
                logger.error(f"Failed to pull Docker image '{image_to_pull}'. pull_image() returned False.")
            return success
        except Exception as e:
            logger.error(f"Exception occurred while pulling Docker image '{image_to_pull}': {e}")
            # Depending on desired behavior, re-raise or return False
            raise  # Re-raising the exception to indicate failure more strongly.

    def get_status(self) -> dict:
        """
        Checks Docker availability and runs a test command.
        Returns a dictionary with status information.
        """
        status_report = {"docker_available": False, "test_successful": False, "message": ""}
        try:
            self.docker_manager.check_docker_availability()  # This raises RuntimeError if Docker is not available
            status_report["docker_available"] = True
            status_report["message"] = "Docker is available."
            logger.info("Docker is available.")

            # Run test command. os.getcwd() is a sensible default for work_dir if not specified.
            # The specific command "echo 'Docker test successful'" is from the example.
            test_command_result = self.docker_manager.run_command("echo 'Docker test successful'", os.getcwd())

            if test_command_result and test_command_result.get("success"):
                status_report["test_successful"] = True
                status_report["message"] += " Docker test command successful."
                logger.info("Docker test command successful.")
                if test_command_result.get("stdout"):
                    logger.debug(f"Test command stdout: {test_command_result.get('stdout')}")
            elif test_command_result:  # Command executed but failed
                error_message = test_command_result.get("stderr", "No stderr output.")
                status_report["test_successful"] = False
                status_report["message"] += f" Docker test command failed: {error_message}"
                logger.error(f"Docker test command failed: {error_message}")
            else:  # Result was None or not in expected format
                status_report["test_successful"] = False
                status_report["message"] += " Docker test command did not return a valid result."
                logger.error("Docker test command did not return a valid result.")

        except RuntimeError as e:  # Specific exception from check_docker_availability
            logger.error(f"Docker availability check failed: {e!s}")
            status_report["message"] = f"Docker is not available: {e!s}"
            # docker_available and test_successful remain False as initialized
        except Exception as e:  # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during Docker status check: {e!s}")
            status_report["message"] = f"An unexpected error occurred: {e!s}"
            # docker_available and test_successful remain False

        return status_report

    def cleanup_environment(self) -> bool:
        """
        Cleans up Docker resources managed by DockerManager.
        Returns True on success, False or raises an exception on failure.
        """
        logger.info("Attempting to clean up Docker resources.")
        try:
            success = self.docker_manager.cleanup()
            if success:
                logger.info("Docker resources cleaned up successfully.")
            else:
                # This path might not be reached if cleanup() raises exceptions on failure.
                logger.error("Failed to clean up Docker resources. cleanup() returned False.")
            return success
        except Exception as e:
            logger.error(f"Exception occurred during Docker cleanup: {e}")
            # Depending on desired behavior, re-raise or return False
            raise  # Re-raising the exception to indicate failure more strongly.
