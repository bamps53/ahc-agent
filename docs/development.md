# Development Guide

This document outlines the development practices, including setting up the environment, running checks, and the release process for `ahc-agent`.

## Prerequisites

- Python 3.8+
- pip (Python package installer)
- Git version control

## Setting up Development Environment


1.  **Install `uv`** (if you haven't already):
    `uv` は `pip` を使ってインストールできます。
    ```bash
    pip install uv
    ```
    または、公式のインストール手順に従ってください: [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation)

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/bamps53/ahc-agent
    cd ahc-agent
    ```

3.  **Create a virtual environment and install dependencies using `uv`**:
    `uv` は仮想環境の作成と依存関係のインストールを一度に行うことができます。
    ```bash
    uv sync --extra dev
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

4.  **Install pre-commit hooks** (optional but highly recommended):
    This will run linters and formatters automatically before each commit.
    ```bash
    pre-commit install
    ```

## Running Linters and Tests

Before committing or pushing changes, and especially before a release, ensure all checks pass.

-   **Run Linters (Ruff)**:
    If you haven't installed pre-commit hooks, or want to run them manually:
    ```bash
    ruff check .
    ruff format .  # To apply formatting fixes
    ```
    To only check formatting without applying:
    ```bash
    ruff format --check .
    ```

-   **Run Tests (Pytest)**:
    ```bash
    pytest
    ```
    This will discover and run all tests in the `tests/` directory.

## Release Process

This project adheres to [Semantic Versioning (SemVer)](https://semver.org/).

### 1. Prerequisites for Releasing

-   Ensure you have the latest versions of `build` and `twine`:
    ```bash
    pip install --upgrade build twine
    ```
-   Ensure you have an account on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/).
-   You must be a maintainer of the `ahc-agent` package on both platforms.
-   Configure your `~/.pypirc` file with API tokens for `twine` for authentication (recommended over username/password):
    ```ini
    [testpypi]
      username = __token__
      password = <your_testpypi_api_token>

    [pypi]
      username = __token__
      password = <your_pypi_api_token>
    ```
    Replace `<your_testpypi_api_token>` and `<your_pypi_api_token>` with your actual API tokens generated from TestPyPI and PyPI respectively.

### 2. Pre-Release Steps

1.  **Update `main` Branch**:
    Ensure your local `main` branch is synchronized with the remote repository.
    ```bash
    git checkout main
    git pull origin main
    ```

2.  **Create a Release Branch** (optional, but good practice):
    Branch off `main` for the release. Replace `X.Y.Z` with the new version number.
    ```bash
    git checkout -b release/vX.Y.Z
    ```

3.  **Run All Checks**:
    - Ensure linters pass: `ruff check . && ruff format --check .` (or `pre-commit run --all-files`).
    - Ensure all tests pass: `pytest`.
    - Verify that the Continuous Integration (CI) pipeline (e.g., GitHub Actions) is passing for the latest commits on your release branch or `main`.

4.  **Update Version Number**:
    The version number needs to be updated in two places:
    - `pyproject.toml`: Update the `version = "X.Y.Z"` line under the `[project]` section.
    - `ahc_agent_cli/__init__.py`: Update the `__version__ = "X.Y.Z"` string.
    Ensure both versions match exactly.

5.  **Update Changelog**:
    - Maintain a `CHANGELOG.md` file (create it if it doesn't exist).
    - Add a new section for the upcoming release (e.g., `## [vX.Y.Z] - YYYY-MM-DD`).
    - Document all significant changes: new features, bug fixes, breaking changes, deprecations, performance improvements, etc.

6.  **Commit Changes**:
    Commit the version update and changelog.
    ```bash
    git add pyproject.toml ahc_agent_cli/__init__.py CHANGELOG.md
    git commit -m "chore: Prepare release vX.Y.Z"
    ```

7.  **Merge to `main`** (if using a release branch):
    - Create a Pull Request from your release branch (`release/vX.Y.Z`) to `main`.
    - Ensure all checks pass on the PR.
    - Get the PR reviewed and merge it into `main`.
    - After merging, switch back to `main` and pull the latest changes:
      ```bash
      git checkout main
      git pull origin main
      ```

8.  **Tag the Release**:
    Create an annotated Git tag on the `main` branch for the commit that represents the release.
    ```bash
    git tag -a vX.Y.Z -m "Release vX.Y.Z" # Use the same version number
    ```

### 3. Build the Package

1.  **Clean Previous Builds** (recommended for a clean state):
    ```bash
    rm -rf dist/ build/ *.egg-info/
    ```

2.  **Build Source and Wheel Distributions**:
    Use the `build` package:
    ```bash
    python -m build
    ```
    This command will create a source distribution (`.tar.gz`) and a wheel (`.whl`) in the `dist/` directory.

### 4. Publish to TestPyPI (Highly Recommended)

Before publishing to the official PyPI, test the upload and installation process using TestPyPI.

1.  **Upload to TestPyPI**:
    ```bash
    twine upload --repository testpypi dist/*
    ```
    If your `~/.pypirc` is not configured, you will be prompted for your TestPyPI username and API token.

2.  **Verify Installation from TestPyPI**:
    In a new, clean virtual environment, try installing your package from TestPyPI:
    ```bash
    # Create a new temporary directory and virtual environment
    python -m venv test_env
    source test_env/bin/activate
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ahc-agent==X.Y.Z
    ```
    Replace `X.Y.Z` with the version you are releasing.
    Test basic functionality:
    ```bash
    ahc-agent --version
    # Optionally, run a simple command if feasible
    ```
    Deactivate and remove the test environment afterwards.

### 5. Publish to PyPI (Official Release)

Once you are confident that the package works correctly when installed from TestPyPI:

1.  **Upload to PyPI**:
    ```bash
    twine upload dist/*
    ```
    If your `~/.pypirc` is not configured, you will be prompted for your PyPI username and API token.

### 6. Post-Release Steps

1.  **Push Tags to GitHub**:
    Push the newly created Git tag to the remote repository.
    ```bash
    git push origin vX.Y.Z # Push the specific tag
    # Or push all tags (if you have multiple new tags)
    # git push origin --tags
    ```
    Ensure your `main` branch is also pushed if you didn't use a release branch that was merged.
    ```bash
    git push origin main
    ```

2.  **Create a GitHub Release**:
    - Navigate to your repository on GitHub.
    - Go to the "Releases" section.
    - Click on "Draft a new release" or "Create a new release".
    - Select the tag you just pushed (e.g., `vX.Y.Z`).
    - Set the release title (usually the tag name, e.g., `vX.Y.Z`).
    - Copy the relevant release notes from `CHANGELOG.md` into the release description.
    - Optionally, you can upload the generated package files (`dist/*`) as binary attachments to the release.
    - Publish the release.

3.  **(Optional) Bump Version for Development**:
    After the release, you may want to update the version in your `main` branch to the next development version (e.g., `X.Y.(Z+1).dev0` or `X.(Y+1).0.dev0`).
    - Update version in `pyproject.toml` and `ahc_agent_cli/__init__.py`.
    - Commit the change:
      ```bash
      git add pyproject.toml ahc_agent_cli/__init__.py
      git commit -m "chore: Bump version to X.Y.Z.dev0 for development"
      git push origin main
      ```

## Documentation

-   Project architecture is described in `architecture.md` (or potentially `docs/architecture.md`).
-   User-facing documentation is in `README.md`.
-   This development guide is in `docs/development.md`.

When making changes to documentation, commit and push them as you would with code changes.
