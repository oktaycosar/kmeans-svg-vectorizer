"""
Utility functions: logging, file I/O, image type helpers.
"""

from __future__ import annotations

import logging
import sys
import math
from pathlib import Path
from typing import Optional, Tuple
import datetime
import json


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger("ImageRestoration")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)-7s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = logging.getLogger("ImageRestoration")


def ensure_output_dir(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_stem(file_path: str) -> str:
    """Return filename without extension, sanitized."""
    return Path(file_path).stem


def get_file_extension(file_path: str) -> str:
    """Return lowercase file extension without dot."""
    return Path(file_path).suffix.lower().lstrip(".")


SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"}


def is_supported_image(file_path: str) -> bool:
    """Check if file extension is a supported image format."""
    return get_file_extension(file_path) in SUPPORTED_FORMATS


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value between bounds."""
    return max(low, min(value, high))


def save_json(data: dict, file_path: str, pretty: bool = True) -> str:
    """Save a dictionary as JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return str(path)


def format_timestamp() -> str:
    """Return current timestamp for filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def image_size_mb(image) -> float:
    """Estimate image size in megabytes."""
    import numpy as np
    return float(image.nbytes) / (1024 * 1024)
