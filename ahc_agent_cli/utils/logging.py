"""
Logging utilities for AHCAgent CLI.

This module provides utilities for logging.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up logging.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: None, log to console only)
        log_format: Log format (default: None, use standard format)
        json_format: Whether to use JSON format (default: False)
        
    Returns:
        Logger instance
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logger
    logger = logging.getLogger("ahc_agent_cli")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    if json_format:
        try:
            import json_logging
            json_logging.init_non_web(enable_json=True)
            formatter = None  # json_logging will handle formatting
        except ImportError:
            # Fall back to standard formatter
            formatter = logging.Formatter(log_format)
    else:
        formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    if formatter:
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        if formatter:
            file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger.
    
    Args:
        name: Logger name (default: None, use root logger)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"ahc_agent_cli.{name}")
    else:
        return logging.getLogger("ahc_agent_cli")
