"""
File I/O utilities for AHCAgent.

This module provides utilities for file operations.
"""

import json
import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def ensure_directory(directory: str) -> str:
    """
    Ensure a directory exists.

    Args:
        directory: Directory path

    Returns:
        Absolute path to the directory
    """
    abs_path = os.path.abspath(directory)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def read_file(file_path: str) -> str:
    """
    Read a file.

    Args:
        file_path: Path to the file

    Returns:
        File content
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e!s}")
        raise


def write_file(file_path: str, content: str) -> None:
    """
    Write content to a file.

    Args:
        file_path: Path to the file
        content: Content to write
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {e!s}")
        raise


def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file.

    Args:
        file_path: Path to the file

    Returns:
        JSON content as dictionary
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e!s}")
        raise


def write_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """
    Write data to a JSON file.

    Args:
        file_path: Path to the file
        data: Data to write
        indent: Indentation level
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        logger.error(f"Error writing to JSON file {file_path}: {e!s}")
        raise


def read_yaml(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML file.

    Args:
        file_path: Path to the file

    Returns:
        YAML content as dictionary
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {e!s}")
        raise


def write_yaml(file_path: str, data: Dict[str, Any]) -> None:
    """
    Write data to a YAML file.

    Args:
        file_path: Path to the file
        data: Data to write
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error writing to YAML file {file_path}: {e!s}")
        raise
