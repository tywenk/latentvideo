from dataclasses import dataclass
import logging
import sys
from pathlib import Path
from typing import Optional


@dataclass
class Logger:
    """Application-wide logger setup."""

    @staticmethod
    def setup(
        name: str = "latent_video",
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        format: str = "%(asctime)s %(name)s [%(levelname)s] %(message)s",
    ) -> logging.Logger:
        """Set up and return application logger."""
        logger = logging.getLogger(name)

        # Only configure if it hasn't been configured before
        if not logger.handlers:
            logger.setLevel(level)
            formatter = logging.Formatter(format)

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler if log_file is specified
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(str(log_file))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            logger.propagate = False

        return logger
