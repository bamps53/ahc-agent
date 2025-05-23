"""
Unit tests for CLI module.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from ahc_agent_cli.cli import cli

class TestCLI:
    """
    Tests for CLI module.
    """
    
    @pytest.fixture
    def runner(self):
        """
        Create a CLI runner for testing.
        """
        return CliRunner()
    
    def test_cli_help(self, runner):
        """
        Test CLI help command.
        """
        # Run CLI with --help
        result = runner.invoke(cli, ["--help"])
        
        # Check result
        assert result.exit_code == 0
        assert "AHCAgent CLI" in result.output
        assert "init" in result.output
        assert "solve" in result.output
        assert "status" in result.output
        assert "submit" in result.output
    
    @patch("ahc_agent_cli.cli.Config")
    def test_init_command(self, mock_config, runner):
        """
        Test init command.
        """
        # Mock Config
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        # Run init command
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "--workspace", "./workspace"])
            
            # Check result
            assert result.exit_code == 0
            assert "Initialized AHC project" in result.output
            
            # Check Config calls
            mock_config_instance.set.assert_called_with("workspace.base_dir", os.path.abspath("./workspace"))
            mock_config_instance.save.assert_called_once()
    
    @patch("ahc_agent_cli.cli.asyncio.run")
    def test_solve_command(self, mock_asyncio_run, runner):
        """
        Test solve command.
        """
        # Create a temporary problem file
        with runner.isolated_filesystem():
            # Create problem file
            with open("problem.md", "w") as f:
                f.write("# Test Problem\n\nThis is a test problem.")
            
            # Run solve command
            result = runner.invoke(cli, ["solve", "problem.md"])
            
            # Check result
            assert result.exit_code == 0
            
            # Check asyncio.run call
            mock_asyncio_run.assert_called_once()
    
    @patch("ahc_agent_cli.cli.KnowledgeBase")
    def test_status_command(self, mock_knowledge_base, runner):
        """
        Test status command.
        """
        # Mock KnowledgeBase
        mock_kb_instance = MagicMock()
        mock_kb_instance.get_session.return_value = {
            "session_id": "test-session",
            "problem_id": "Test Problem",
            "created_at": 1621234567,
            "updated_at": 1621234567,
            "status": "completed"
        }
        mock_kb_instance.get_problem_analysis.return_value = {"title": "Test Problem"}
        mock_kb_instance.get_solution_strategy.return_value = {"approach": "Test approach"}
        mock_kb_instance.get_evolution_log.return_value = {
            "generations_completed": 10,
            "best_score": 100,
            "duration": 60
        }
        mock_kb_instance.get_best_solution.return_value = {
            "code": "// Test code",
            "score": 100
        }
        mock_knowledge_base.return_value = mock_kb_instance
        
        # Run status command
        result = runner.invoke(cli, ["status", "test-session"])
        
        # Check result
        assert result.exit_code == 0
        assert "Session ID: test-session" in result.output
        assert "Problem: Test Problem" in result.output
        assert "Status: completed" in result.output
        assert "Problem Analysis: Complete" in result.output
        assert "Solution Strategy: Complete" in result.output
        assert "Evolution: Complete (10 generations)" in result.output
        assert "Best Score: 100" in result.output
    
    @patch("ahc_agent_cli.cli.KnowledgeBase")
    def test_submit_command(self, mock_knowledge_base, runner):
        """
        Test submit command.
        """
        # Mock KnowledgeBase
        mock_kb_instance = MagicMock()
        mock_kb_instance.get_session.return_value = {
            "session_id": "test-session",
            "problem_id": "Test Problem"
        }
        mock_kb_instance.get_best_solution.return_value = {
            "code": "// Test code",
            "score": 100
        }
        mock_knowledge_base.return_value = mock_kb_instance
        
        # Run submit command with output file
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["submit", "test-session", "--output", "solution.cpp"])
            
            # Check result
            assert result.exit_code == 0
            assert "Best solution written to solution.cpp" in result.output
            assert "Score: 100" in result.output
            
            # Check output file
            with open("solution.cpp", "r") as f:
                content = f.read()
                assert content == "// Test code"
    
    @patch("ahc_agent_cli.cli.Config")
    def test_config_get_command(self, mock_config, runner):
        """
        Test config get command.
        """
        # Mock Config
        mock_config_instance = MagicMock()
        mock_config_instance.get.return_value = "gpt-4"
        mock_config.return_value = mock_config_instance
        
        # Run config get command
        result = runner.invoke(cli, ["config", "get", "llm.model"])
        
        # Check result
        assert result.exit_code == 0
        assert "llm.model = gpt-4" in result.output
        
        # Check Config calls
        mock_config_instance.get.assert_called_with("llm.model")
    
    @patch("ahc_agent_cli.cli.Config")
    def test_config_set_command(self, mock_config, runner):
        """
        Test config set command.
        """
        # Mock Config
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        # Run config set command
        result = runner.invoke(cli, ["config", "set", "llm.model", "gpt-3.5-turbo"])
        
        # Check result
        assert result.exit_code == 0
        assert "Set llm.model = gpt-3.5-turbo" in result.output
        
        # Check Config calls
        mock_config_instance.set.assert_called_with("llm.model", "gpt-3.5-turbo")
    
    @patch("ahc_agent_cli.cli.DockerManager")
    def test_docker_status_command(self, mock_docker_manager, runner):
        """
        Test docker status command.
        """
        # Mock DockerManager
        mock_dm_instance = MagicMock()
        mock_dm_instance.run_command.return_value = {
            "success": True,
            "stdout": "Docker test successful",
            "stderr": ""
        }
        mock_docker_manager.return_value = mock_dm_instance
        
        # Run docker status command
        result = runner.invoke(cli, ["docker", "status"])
        
        # Check result
        assert result.exit_code == 0
        assert "Docker is available" in result.output
        assert "Docker test successful" in result.output
