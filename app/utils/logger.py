import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
def setup_logger():
    logger = logging.getLogger("vector_db")
    logger.setLevel(logging.DEBUG)  
    logger.propagate = False

    if logger.handlers:
        return logger
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    info_handler = RotatingFileHandler(
        log_dir / "info.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    warning_handler = RotatingFileHandler(
        log_dir / "warning.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(formatter)
    logger.addHandler(warning_handler)
    error_handler = RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    return logger
