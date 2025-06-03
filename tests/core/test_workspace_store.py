"""
Unit tests for WorkspaceStore.
"""

import os
from pathlib import Path
import tempfile

from ahc_agent.core.workspace_store import WorkspaceStore


class TestWorkspaceStore:
    """
    Tests for WorkspaceStore class.
    """

    def test_init(self):
        """
        Test initialization of WorkspaceStore.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Check that directories were created
            assert os.path.exists(os.path.join(temp_dir, "solutions"))
            assert os.path.exists(os.path.join(temp_dir, "logs"))

            # Check attributes
            assert workspace_store.workspace_path == Path(temp_dir)
            assert workspace_store.problem_id == "test_problem"

    def test_get_workspace_dir(self):
        """
        Test get_workspace_dir method.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Get workspace directory
            workspace_dir = workspace_store.get_workspace_dir()

            # Check result
            assert workspace_dir == Path(temp_dir)

    def test_save_load_problem_text(self):
        """
        Test save_problem_text and load_problem_text methods.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Save problem text
            problem_text = "Test problem text"
            result = workspace_store.save_problem_text(problem_text)

            # Check result
            assert result is True

            # Check that file was created
            assert os.path.exists(os.path.join(temp_dir, "problem_text.md"))

            # Load problem text
            loaded_text = workspace_store.load_problem_text()

            # Check loaded text
            assert loaded_text == problem_text

    def test_save_load_problem_analysis(self):
        """
        Test save_problem_analysis and load_problem_analysis methods.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Save problem analysis
            analysis = {"key": "value"}
            result = workspace_store.save_problem_analysis(analysis)

            # Check result
            assert result is True

            # Check that file was created
            assert os.path.exists(os.path.join(temp_dir, "problem_analysis.json"))

            # Load problem analysis
            loaded_analysis = workspace_store.load_problem_analysis()

            # Check loaded analysis
            assert loaded_analysis == analysis

    def test_save_load_solution_strategy(self):
        """
        Test save_solution_strategy and load_solution_strategy methods.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Save solution strategy
            strategy = {"key": "value"}
            result = workspace_store.save_solution_strategy(strategy)

            # Check result
            assert result is True

            # Check that file was created
            assert os.path.exists(os.path.join(temp_dir, "solution_strategy.json"))

            # Load solution strategy
            loaded_strategy = workspace_store.load_solution_strategy()

            # Check loaded strategy
            assert loaded_strategy == strategy

    def test_save_load_initial_solution(self):
        """
        Test save_solution and load_solution_code methods for initial solution.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Save initial solution
            initial_solution = "Test initial solution"
            workspace_store.save_solution("initial", initial_solution)

            # Check that file was created
            assert os.path.exists(os.path.join(temp_dir, "solutions", "initial.cpp"))

            # Load initial solution
            loaded_solution = workspace_store.load_solution_code("initial")

            # Check loaded solution
            assert loaded_solution == initial_solution

    def test_save_load_best_solution(self):
        """
        Test save_solution and load_solution_code methods for best solution.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Save best solution
            best_solution = "Test best solution"
            workspace_store.save_solution("best", best_solution)

            # Check that file was created
            assert os.path.exists(os.path.join(temp_dir, "solutions", "best.cpp"))

            # Load best solution
            loaded_solution = workspace_store.load_solution_code("best")

            # Check loaded solution
            assert loaded_solution == best_solution

    def test_save_load_evolution_log(self):
        """
        Test save_evolution_log and load_evolution_log methods.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Save evolution logs
            logs = {"key": "value"}
            result = workspace_store.save_evolution_log(logs)

            # Check result
            assert result is True

            # Check that file was created
            assert os.path.exists(os.path.join(temp_dir, "logs", "evolution_log.json"))

            # Load evolution logs
            loaded_logs = workspace_store.load_evolution_log()

            # Check loaded logs
            assert loaded_logs == logs

    def test_save_load_solution(self):
        """
        Test save_solution and load_solution_code methods.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Save solution
            solution_name = "test_solution"
            solution = "Test solution"
            workspace_store.save_solution(solution_name, solution)

            # Check that file was created
            assert os.path.exists(os.path.join(temp_dir, "solutions", f"{solution_name}.cpp"))

            # Load solution
            loaded_solution = workspace_store.load_solution_code(solution_name)

            # Check loaded solution
            assert loaded_solution == solution

    def test_get_best_solution_code_and_meta(self):
        """
        Test get_best_solution_code_and_meta method.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize WorkspaceStore
            workspace_store = WorkspaceStore(temp_dir, "test_problem")

            # Save best solution code
            best_solution_code = "Test best solution"
            workspace_store.save_solution("best", best_solution_code)

            # Save best solution metadata
            best_solution_meta = {"score": 100}
            workspace_store.save_solution("best", best_solution_code, best_solution_meta)

            # Get best solution code and metadata
            loaded_code, loaded_meta = workspace_store.get_best_solution_code_and_meta()

            # Check loaded solution
            assert loaded_code == best_solution_code
            assert loaded_meta == best_solution_meta
