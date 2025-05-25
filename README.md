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

Initializes a new AHC project for a specific contest.
Creates a project directory (named after the `CONTEST_ID` by default, or as specified by `--workspace`)
and a configuration file (`ahc_config.yaml`) within it.
This configuration file will store the `contest_id`, the `template` (defaults to "default"),
and the `docker_image` (uses global config or defaults to "ubuntu:latest" if not specified).
The command will also attempt to scrape the problem statement for the given `CONTEST_ID`.

```bash
ahc-agent init ahc001 --workspace ./my_ahc_project --template custom_cpp --docker-image my-cpp-dev-env:latest
```

- `CONTEST_ID`: (Required) The ID of the AtCoder Heuristic Contest (e.g., `ahc001`).
- `--workspace PATH` (`-w`): Specify the directory to create the project in.
  If not set, a directory named after the `CONTEST_ID` is created in the current location.
- `--template NAME` (`-t`): Specify a project template to use. This will be recorded in `ahc_config.yaml`.
  (Default: "default")
- `--docker-image IMAGE` (`-i`): Specify the Docker image for the project. This will be recorded in `ahc_config.yaml`.
  (Default: Value from global config, or "ubuntu:latest")

### 2\. Solve a Problem

Solves the problem based on the problem description and configuration within the specified workspace directory.
**The workspace directory must contain `problem.md` (the problem statement) and `ahc_config.yaml` (the project configuration).**

```bash
ahc-agent solve path/to/your_workspace_directory --time-limit 1800 --generations 50 --population-size 20
```

- `WORKSPACE`: Path to the workspace directory.
- `--time-limit SECONDS` (`-t`): Set the time limit (in seconds) for the evolutionary algorithm.
- `--generations NUM` (`-g`): Set the maximum number of generations for the evolutionary algorithm.
- `--population-size NUM` (`-p`): Set the population size for the evolutionary algorithm.

### 3\. Solve a Problem in Interactive Mode

Guides you through the problem-solving process step-by-step using the problem and configuration in the specified workspace.
**The workspace directory must contain `problem.md` and `ahc_config.yaml`.**

```bash
ahc-agent solve path/to/your_workspace_directory --interactive
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
