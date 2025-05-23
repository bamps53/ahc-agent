# AHCAgent CLI

A standalone command-line tool for solving AtCoder Heuristic Contest problems using AlphaEvolve algorithm.

## Features

- Problem analysis and solution strategy development
- Evolutionary algorithm-based solution optimization
- Docker integration for C++ compilation and execution
- Interactive and batch processing modes
- Experiment tracking and logging
- Configuration management

## Installation

```bash
pip install ahc-agent-cli
```

## Requirements

- Python 3.8 or higher
- Docker (for C++ compilation and execution)
- LLM API access (OpenAI, Anthropic, etc. via LiteLLM)

## Quick Start

### Initialize a project

```bash
ahc-agent init --workspace ~/ahc_workspace
```

### Solve a problem

```bash
ahc-agent solve problem.md
```

### Interactive mode

```bash
ahc-agent solve problem.md --interactive
```

### Check status

```bash
ahc-agent status <session_id>
```

### Submit best solution

```bash
ahc-agent submit <session_id> --output solution.cpp
```

## Configuration

Configuration can be managed via:

- Configuration file (`ahc_config.yaml`)
- Environment variables
- Command-line options

### Export/Import Configuration

```bash
ahc-agent config export config.yaml
ahc-agent config import config.yaml
```

## Batch Processing

Run multiple experiments with different parameters:

```bash
ahc-agent batch batch_config.yaml --parallel 4
```

## Docker Management

```bash
ahc-agent docker setup
ahc-agent docker status
ahc-agent docker cleanup
```

## License

MIT
