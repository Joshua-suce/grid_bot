import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "{message}"
        ),
    )

    logger.add(
        log_path / "grid_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{function}:{line} | {message}",
    )

    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="1 day",
        retention="90 days",
    )
