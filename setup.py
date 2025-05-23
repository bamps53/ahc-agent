"""
Setup script for AHCAgent CLI.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ahc-agent-cli",
    version="0.1.0",
    author="AHCAgent Team",
    author_email="ahcagent@example.com",
    description="A CLI tool for solving AtCoder Heuristic Contest problems using AlphaEvolve",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ahc-agent-cli",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "pyyaml>=6.0",
        "litellm>=0.1.0",
        "docker>=6.0.0",
        "aiohttp>=3.8.0",
        "tqdm>=4.64.0",
        "colorama>=0.4.4",
    ],
    entry_points={
        "console_scripts": [
            "ahc-agent=ahc_agent_cli.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ahc_agent_cli": ["templates/*.cpp", "templates/*.md", "config/*.yaml"],
    },
)
