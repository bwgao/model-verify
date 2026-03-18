"""
Model Verification System - Logging Configuration

Structured logging with Rich output for beautiful console display.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# Global console instance
console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_output: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for the model verification system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file
        rich_output: Whether to use Rich formatting for console output

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("model_verify")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with Rich formatting
    if rich_output:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            markup=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = (
        "%(message)s" if rich_output else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "model_verify") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Convenience functions for common log patterns
class LogContext:
    """Context manager for timing operations."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[float] = None

    def __enter__(self) -> "LogContext":
        import time

        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time

        elapsed = time.time() - self.start_time if self.start_time else 0
        if exc_type:
            self.logger.error(f"Failed: {self.operation} ({elapsed:.2f}s) - {exc_val}")
        else:
            self.logger.debug(f"Completed: {self.operation} ({elapsed:.2f}s)")


def log_probe_start(logger: logging.Logger, probe_name: str, model: str) -> None:
    """Log the start of a probe execution."""
    logger.info(f"[bold blue]Starting {probe_name}[/] probe for model: {model}")


def log_probe_result(
    logger: logging.Logger,
    probe_name: str,
    score: float,
    verdict: str,
    confidence: float,
) -> None:
    """Log the result of a probe execution."""
    color = "green" if verdict == "PASS" else "yellow" if verdict == "WARN" else "red"
    logger.info(
        f"[bold]{probe_name}[/]: Score={score:.2f}, "
        f"Verdict=[{color}]{verdict}[/], Confidence={confidence:.2f}"
    )


def log_api_call(
    logger: logging.Logger,
    provider: str,
    model: str,
    latency_ms: float,
    tokens: Optional[int] = None,
) -> None:
    """Log an API call."""
    token_info = f", Tokens={tokens}" if tokens else ""
    logger.debug(f"API call: {provider}/{model} - {latency_ms:.0f}ms{token_info}")
