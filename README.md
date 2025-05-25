# AHCAgent CLI

A standalone command-line tool for solving AtCoder Heuristic Contest (AHC) problems using an AlphaEvolve-inspired approach.

## ✨ Features

- Problem analysis and solution strategy development utilizing LLMs.
- Solution search and optimization based on evolutionary algorithms.
- Docker integration for C++ code compilation and execution.
- Interactive mode supporting a step-by-step problem-solving process.
- Batch processing mode for efficiently running multiple experiment configurations.
- Knowledge base feature for managing progress and results for each experiment session.
- Flexible configuration management via YAML files, environment variables, and command-line options.

## 🛠️ Installation

```bash
pip install ahc-agent
```

## 📋 Requirements

- Python 3.8 or higher
- Docker (required for C++ code compilation and execution)
- LLM API Access (supports OpenAI, Anthropic, etc., via LiteLLM)

## 🚀 Quick Start

### 1\. Initialize a Project

Creates a workspace directory and a configuration file (`ahc_config.yaml`).

```bash
ahc-agent init --workspace ./workspace --docker-image my-cpp-dev-env:latest
```

- `--workspace PATH` (`-w`): Specify the path for the workspace directory.
- `--docker-image IMAGE` (`-i`): Record the project's default Docker image in the configuration file.

### 2\. Solve a Problem

Solves the problem based on the provided problem description file.

```bash
ahc-agent solve path/to/problem_description.md --time-limit 1800 --generations 50 --population-size 20
```

- `PROBLEM_FILE`: Path to the problem description file.
- `--time-limit SECONDS` (`-t`): Set the time limit (in seconds) for the evolutionary algorithm.
- `--generations NUM` (`-g`): Set the maximum number of generations for the evolutionary algorithm.
- `--population-size NUM` (`-p`): Set the population size for the evolutionary algorithm.

### 3\. Solve a Problem in Interactive Mode

Guides you through the problem-solving process step-by-step.

```bash
ahc-agent solve path/to/problem_description.md --interactive
```

Available commands in interactive mode:

- `analyze`: Execute problem analysis.
- `strategy`: Devise solution strategy.
- `testcases`: Generate test cases and score calculator.
- `initial`: Generate initial solution.
- `evolve`: Run evolutionary search.
- `status`: Display current status.
- `help`: Display command list.
- `exit`: Exit interactive mode.

### 4\. Check Session Status

Display the status of a specific session ID or list all sessions.

```bash
ahc-agent status <session_id>
```

To continuously monitor the status of a specific session:

```bash
ahc-agent status <session_id> --watch
```

### 5\. Submit the Best Solution

Retrieve the best solution from a specified session ID and output it to standard output or a specified file.

```bash
ahc-agent submit <session_id> --output solution.cpp
```

## ⚙️ Configuration

Configuration is managed with the following priority:

1.  Command-line options
2.  Environment variables (e.g., `AHC_LLM_MODEL="o4-mini"`)
3.  Configuration file (`ahc_config.yaml` in the workspace)
4.  Default settings

### Configuration Management Commands

```bash
# Get a specific configuration value
ahc-agent config get llm.model

# Set a configuration value (for the current CLI invocation's context, not persisted to ahc_config.yaml by this command)
ahc-agent config set llm.temperature 0.5

# Export the current effective configuration to a file
ahc-agent config export current_config.yaml

# Import and merge configuration from a file into the current CLI invocation's context
ahc-agent config import custom_config.yaml
```

## 📊 Batch Processing

Run multiple experiments with different parameters, either sequentially or in parallel.

```bash
ahc-agent batch path/to/batch_config.yaml --parallel 4 --output-dir ~/ahc_batch_results
```

- `BATCH_CONFIG`: Path to the batch configuration file (YAML).
- `--parallel NUM` (`-p`): Specify the number of parallel executions.
- `--output-dir PATH` (`-o`): Specify the output directory for batch processing results.

Refer to `architecture.md` for an example of the batch configuration file format.

## 🐳 Docker Management

Set up, check the status of, and clean up the Docker environment used by AHCAgent.

```bash
# Pull the Docker image specified in the configuration file
ahc-agent docker setup

# Display Docker daemon connectivity and run a test command within the configured image
ahc-agent docker status

# Clean up unused Docker resources (e.g., stopped containers created by this tool)
ahc-agent docker cleanup
```

## 📜 License

[MIT](https://www.google.com/search?q=LICENSE)
