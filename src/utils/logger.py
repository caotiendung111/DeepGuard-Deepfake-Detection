"""
DeepGuard — Logging Setup
Uses loguru for structured, colored logging.
"""
import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/deepguard.log",
    rotation: str = "50 MB",
    retention: str = "30 days",
    colorize: bool = True,
) -> None:
    """
    Configure loguru with console + rotating file handlers.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file (rotated automatically).
        rotation: File rotation size/time (e.g., "50 MB", "1 day").
        retention: How long to keep old log files.
        colorize: Enable colored output in console.
    """
    # Remove default handler
    logger.remove()

    # Console handler with custom format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=colorize,
        enqueue=True,
    )

    # File handler with JSON-like detailed format
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
            "{name}:{function}:{line} | {message}"
        )
        logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            enqueue=True,
        )

    logger.info(f"Logger initialized — level={log_level}, file={log_file}")


def get_logger(name: str):
    """Get a named logger (for module-level use)."""
    return logger.bind(module=name)
