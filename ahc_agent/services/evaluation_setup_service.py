import logging
from pathlib import Path
from typing import Dict, Any

from ahc_agent.config import Config
from ahc_agent.core.problem_logic import ProblemLogic
from ahc_agent.core.workspace_store import WorkspaceStore
from ahc_agent.utils.docker_manager import DockerManager
from ahc_agent.utils.file_io import ensure_directory, write_file

logger = logging.getLogger(__name__)

class EvaluationSetupService:
    def __init__(
        self,
        config: Config,
        workspace_store: WorkspaceStore,
        problem_logic: ProblemLogic,
        docker_manager: DockerManager,
    ):
        self.config = config
        self.workspace_store = workspace_store
        self.problem_logic = problem_logic
        self.docker_manager = docker_manager
        # Attempt to get compiler and flags from debugger_config, then docker_config, then defaults
        debugger_config = self.config.get("debugger", {})
        docker_config = self.config.get("docker", {})

        self.cpp_compiler = debugger_config.get("cpp_compiler", docker_config.get("cpp_compiler", "g++"))
        self.cpp_flags = debugger_config.get("cpp_flags", docker_config.get("cpp_flags", "-std=c++17 -O2 -Wall"))
        self.execution_timeout = debugger_config.get("execution_timeout", docker_config.get("execution_timeout", 10))


    async def create_baseline_cpp(self, problem_analysis_data: Dict[str, Any]) -> Path:
        logger.info("Creating baseline C++ file...")
        setup_dir = self.workspace_store.get_evaluation_setup_dir()
        baseline_path = setup_dir / "baseline.cpp"

        try:
            code = await self.problem_logic.generate_initial_solution(problem_analysis_data)
            if not code or not code.strip(): # Check if code is empty or whitespace
                logger.warning("Generated initial solution was empty. Using basic template.")
                code = self.problem_logic._get_basic_template()
        except Exception as e:
            logger.error(f"Error generating initial solution: {e}. Using basic template.")
            code = self.problem_logic._get_basic_template()

        write_file(baseline_path, code)
        logger.info(f"Baseline C++ file saved to {baseline_path}")
        return baseline_path

    def create_compile_script_info(self) -> Path:
        logger.info("Creating compile script info file...")
        setup_dir = self.workspace_store.get_evaluation_setup_dir()
        script_info_path = setup_dir / "compile_script_info.md"

        # Assuming DockerManager uses these or similar.
        # These are typically set in the debugger config.
        compiler = self.cpp_compiler
        flags = self.cpp_flags

        # Add -DONLINE_JUDGE as it's common in competitive programming
        if "-DONLINE_JUDGE" not in flags:
            flags += " -DONLINE_JUDGE"

        content = f"""# Compilation Script Information

This document describes the compilation process used by the AHC Agent.

## Command

A representative command for compiling a C++ solution (`solution.cpp`) is:

```sh
{compiler} {flags} -o solution solution.cpp
```

## Notes

*   The agent uses Docker to ensure a consistent compilation environment. The actual command inside the Docker container might vary slightly but will achieve the same result.
*   The compiler is typically `{compiler}`.
*   Standard flags include `{flags}`. These optimize the code and enable warnings.
*   The output executable is typically named `solution`.
"""
        write_file(script_info_path, content)
        logger.info(f"Compile script info file saved to {script_info_path}")
        return script_info_path

    def create_run_script_info(self) -> Path:
        logger.info("Creating run script info file...")
        setup_dir = self.workspace_store.get_evaluation_setup_dir()
        script_info_path = setup_dir / "run_script_info.md"

        timeout = self.execution_timeout

        content = f"""# Execution Script Information

This document describes the execution process used by the AHC Agent for evaluating C++ solutions.

## Process

1.  The compiled executable (e.g., `solution`) is run.
2.  Input for a specific test case (e.g., from `input.txt`) is provided to the standard input (stdin) of the executable.
3.  The standard output (stdout) of the executable is captured as the solution's output.
4.  Standard error (stderr) is also captured for debugging.
5.  Execution is subject to a time limit (typically {timeout} seconds).

## Representative Command

A conceptual command to run the solution with an input file and save its output is:

```sh
./solution < input.txt > output.txt
```

(Error redirection and timeout handling are managed by the agent.)

## Notes

*   The agent uses Docker for execution to provide a consistent environment and manage resources.
*   The agent iterates through all test case files found in the `tools/in/` directory (or generated test files). For each input file, it runs the solution and captures the output.
*   The execution time for each test case is measured.
"""
        write_file(script_info_path, content)
        logger.info(f"Run script info file saved to {script_info_path}")
        return script_info_path

    async def create_evaluation_readme(self, problem_analysis_data: Dict[str, Any]) -> Path:
        logger.info("Creating evaluation README file...")
        setup_dir = self.workspace_store.get_evaluation_setup_dir()
        readme_path = setup_dir / "README.md"

        problem_title = problem_analysis_data.get("title", "the current problem")

        content = f"""# Evaluation Environment Setup for {problem_title}

This directory contains files and information describing the evaluation procedure established by the AHC Agent for C++ solutions.

## Key Components:

1.  **Baseline C++ Code (`baseline.cpp`):**
    *   A starting point or template C++ solution generated by the agent.
    *   Path: `baseline.cpp`

2.  **Compilation Information (`compile_script_info.md`):**
    *   Details the command and flags used for compiling C++ code.
    *   Path: `compile_script_info.md`

3.  **Execution Information (`run_script_info.md`):**
    *   Explains how the compiled solution is run against test cases.
    *   Path: `run_script_info.md`

## Evaluation Flow:

The agent follows these general steps to evaluate a C++ solution:

1.  **Compile:** The C++ source code is compiled using the method described in `compile_script_info.md`. If compilation fails, an error is reported.
2.  **Execute per Test Case:**
    *   The agent looks for test case input files in the `../../tools/in/` directory relative to this workspace's root (or uses internally generated test cases if none are found there).
    *   For each test case, the compiled solution is executed as outlined in `run_script_info.md`.
    *   The output from the solution is captured.
3.  **Score Calculation:**
    *   After successful execution, the agent uses a problem-specific scoring function to calculate the score for that test case. This scoring function is derived from the problem statement's scoring rules (typically handled by the agent's `ProblemLogic.create_score_calculator` method).
    *   Scores from individual test cases are aggregated to produce an overall score for the solution.

This setup ensures a consistent and transparent process for evaluating solutions generated during the agent's operation.
"""
        write_file(readme_path, content)
        logger.info(f"Evaluation README file saved to {readme_path}")
        return readme_path

    async def setup_evaluation_environment(self, problem_analysis_data: Dict[str, Any]) -> None:
        logger.info("Setting up evaluation environment...")
        setup_dir = self.workspace_store.get_evaluation_setup_dir()
        ensure_directory(setup_dir)

        await self.create_baseline_cpp(problem_analysis_data)
        self.create_compile_script_info()
        self.create_run_script_info()
        await self.create_evaluation_readme(problem_analysis_data)

        logger.info(f"Evaluation environment setup complete in {setup_dir}")
