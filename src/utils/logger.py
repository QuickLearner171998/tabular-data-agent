"""Logging configuration for the CPG Data Analysis Agent."""

import logging
import sys
from pathlib import Path
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logger(
    name: str = "cpg_agent",
    level: LogLevel = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file path for logging to file.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level))
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    
    return logger


def get_logger(name: str = "cpg_agent") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name (use dot notation for hierarchy, e.g., 'cpg_agent.tools')
    
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers and not logger.parent.handlers:
        return setup_logger(name)
    
    return logger


# Create default logger
logger = setup_logger()
