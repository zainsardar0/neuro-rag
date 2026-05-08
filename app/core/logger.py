import sys
from loguru import logger
from app.core.config import get_settings

settings = get_settings()


def setup_logger():
    """Configure application-wide logger."""

    # Remove default logger
    logger.remove()

    # Console handler — colored output
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
    )

    # File handler — saves logs to disk
    logger.add(
        "logs/neurorag.log",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    return logger


app_logger = setup_logger()