"""
Utility functions: logging setup, file I/O helpers, and numerical helpers.
"""

from __future__ import annotations

import logging
import sys
import math
from pathlib import Path
from typing import Optional, Tuple, List, Any
import datetime


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return the project logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("ImageVectorizer")
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


# Module-level logger
logger = logging.getLogger("ImageVectorizer")


def ensure_output_dir(path: str) -> Path:
    """Ensure an output directory exists, creating it if necessary.

    Args:
        path: Directory path as string.

    Returns:
        Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_stem(file_path: str) -> str:
    """Return the filename without extension.

    Args:
        file_path: Full file path.

    Returns:
        Stem of the filename.
    """
    return Path(file_path).stem


def get_file_extension(file_path: str) -> str:
    """Return the lowercase file extension (without dot).

    Args:
        file_path: Full file path.

    Returns:
        Lowercase extension.
    """
    return Path(file_path).suffix.lower().lstrip(".")


def is_supported_image(file_path: str) -> bool:
    """Check if the file has a supported image extension.

    Args:
        file_path: Full file path.

    Returns:
        True if the extension is supported.
    """
    ext = get_file_extension(file_path)
    return ext in {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"}


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value between low and high bounds.

    Args:
        value: Input value.
        low: Lower bound.
        high: Upper bound.

    Returns:
        Clamped value.
    """
    return max(low, min(value, high))


def normalize_angle_degrees(angle: float) -> float:
    """Normalize an angle to [0, 360) degrees.

    Args:
        angle: Angle in degrees.

    Returns:
        Normalized angle in [0, 360).
    """
    return angle % 360.0


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return math.radians(degrees)


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(radians)


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points.

    Args:
        p1: First point (x, y).
        p2: Second point (x, y).

    Returns:
        Euclidean distance.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def midpoint_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Compute the midpoint of two 2D points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def angle_between_lines(
    start1: Tuple[float, float], end1: Tuple[float, float],
    start2: Tuple[float, float], end2: Tuple[float, float],
) -> float:
    """Compute the acute angle between two line segments in degrees.

    Args:
        start1, end1: Endpoints of first line.
        start2, end2: Endpoints of second line.

    Returns:
        Acute angle in degrees [0, 90].
    """
    dx1, dy1 = end1[0] - start1[0], end1[1] - start1[1]
    dx2, dy2 = end2[0] - start2[0], end2[1] - start2[1]

    len1 = math.hypot(dx1, dy1)
    len2 = math.hypot(dx2, dy2)

    if len1 < 1e-9 or len2 < 1e-9:
        return 0.0

    dot = dx1 * dx2 + dy1 * dy2
    cos_angle = clamp(dot / (len1 * len2), -1.0, 1.0)
    angle_rad = math.acos(abs(cos_angle))
    return math.degrees(angle_rad)


def format_timestamp() -> str:
    """Return current timestamp string for filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide, returning default if denominator is near zero."""
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator


def flatten_list(nested: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested for item in sublist]
