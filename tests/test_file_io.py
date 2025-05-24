"""
Unit tests for file I/O utilities.
"""

import json
import os
import tempfile

import pytest

from ahc_agent_cli.utils.file_io import ensure_directory, read_file, read_json, write_file, write_json


class TestFileIO:
    """
    Tests for file I/O utilities.
    """

    def test_read_file(self):
        """
        Test read_file function.
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp:
            temp.write("Test content")
            temp_path = temp.name

        try:
            # Read the file
            content = read_file(temp_path)

            # Check content
            assert content == "Test content"
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_read_file_not_found(self):
        """
        Test read_file function with non-existent file.
        """
        # Try to read a non-existent file
        with pytest.raises(FileNotFoundError):
            read_file("/non/existent/file.txt")

    def test_write_file(self):
        """
        Test write_file function.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define file path
            file_path = os.path.join(temp_dir, "test.txt")

            # Write to file
            write_file(file_path, "Test content")

            # Read the file
            with open(file_path) as f:
                content = f.read()

            # Check content
            assert content == "Test content"

    def test_write_file_create_dirs(self):
        """
        Test write_file function with directory creation.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define file path with subdirectories
            file_path = os.path.join(temp_dir, "subdir1", "subdir2", "test.txt")

            # Write to file
            write_file(file_path, "Test content")

            # Check that directories were created
            assert os.path.exists(os.path.dirname(file_path))

            # Read the file
            with open(file_path) as f:
                content = f.read()

            # Check content
            assert content == "Test content"

    def test_read_json(self):
        """
        Test read_json function.
        """
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp:
            json.dump({"key": "value"}, temp)
            temp_path = temp.name

        try:
            # Read the JSON file
            data = read_json(temp_path)

            # Check data
            assert data == {"key": "value"}
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_read_json_invalid(self):
        """
        Test read_json function with invalid JSON.
        """
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp:
            temp.write("Invalid JSON")
            temp_path = temp.name

        try:
            # Try to read the JSON file
            with pytest.raises(json.JSONDecodeError):
                read_json(temp_path)
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_write_json(self):
        """
        Test write_json function.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define file path
            file_path = os.path.join(temp_dir, "test.json")

            # Write JSON to file
            write_json(file_path, {"key": "value"})

            # Read the file
            with open(file_path) as f:
                content = f.read()

            # Parse JSON
            data = json.loads(content)

            # Check data
            assert data == {"key": "value"}

    def test_write_json_indent(self):
        """
        Test write_json function with indent.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define file path
            file_path = os.path.join(temp_dir, "test.json")

            # Write JSON to file with indent
            write_json(file_path, {"key": "value"}, indent=2)

            # Read the file
            with open(file_path) as f:
                content = f.read()

            # Check that content has newlines (indented)
            assert "\n" in content

            # Parse JSON
            data = json.loads(content)

            # Check data
            assert data == {"key": "value"}

    def test_ensure_directory(self):
        """
        Test ensure_directory function.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define directory path
            dir_path = os.path.join(temp_dir, "subdir1", "subdir2")

            # Ensure directory exists
            result = ensure_directory(dir_path)

            # Check result
            assert result == dir_path

            # Check that directory was created
            assert os.path.exists(dir_path)
            assert os.path.isdir(dir_path)

    def test_ensure_directory_existing(self):
        """
        Test ensure_directory function with existing directory.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ensure directory exists
            result = ensure_directory(temp_dir)

            # Check result
            assert result == temp_dir

            # Check that directory exists
            assert os.path.exists(temp_dir)
            assert os.path.isdir(temp_dir)
