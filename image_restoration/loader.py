"""
Image loader supporting RGB, RGBA, grayscale, and 16-bit images.

Handles alpha channel preservation and format normalization.
"""

from __future__ import annotations

from typing import Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .utils import logger, is_supported_image


def load_image(file_path: str) -> Tuple[np.ndarray, dict]:
    """Load an image with full channel and bit-depth preservation.

    Args:
        file_path: Path to the image file.

    Returns:
        Tuple of (image_array, metadata_dict).
        metadata contains: width, height, channels, has_alpha, bit_depth, mode, file_path.

    Raises:
        FileNotFoundError, ValueError, IOError on failure.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    if not is_supported_image(file_path):
        raise ValueError(f"Unsupported format: '{path.suffix}'")

    # Probe with PIL for metadata
    pil_img = Image.open(file_path)
    pil_mode = pil_img.mode
    has_alpha = pil_mode in ("RGBA", "LA", "PA") or (
        pil_mode == "P" and "transparency" in pil_img.info
    )
    is_16bit = pil_mode in ("I;16", "I;16B", "I;16L", "RGB;16", "RGBA;16")
    pil_img.close()

    # Load with OpenCV preserving depth
    if is_16bit:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    elif has_alpha:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise IOError(f"Failed to load: {file_path}")

    # Normalize channels
    if image.ndim == 2:
        channels = 1
    else:
        channels = image.shape[2]

    # Detect bit depth
    if image.dtype == np.uint16:
        bit_depth = 16
    elif image.dtype == np.uint8:
        bit_depth = 8
    else:
        bit_depth = 8  # Default
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    h, w = image.shape[:2]

    metadata = {
        "width": w,
        "height": h,
        "channels": channels,
        "has_alpha": has_alpha,
        "bit_depth": bit_depth,
        "mode": pil_mode,
        "file_path": str(path.absolute()),
        "file_name": path.name,
    }

    logger.info(
        "Loaded: %s (%dx%d, %dch, %d-bit, alpha=%s)",
        path.name, w, h, channels, bit_depth, has_alpha,
    )

    return image, metadata


def split_alpha(image: np.ndarray, metadata: dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Split alpha channel from image if present.

    Returns (bgr_image, alpha_mask). alpha_mask is None if no alpha.
    """
    if not metadata["has_alpha"] or metadata["channels"] < 4:
        return image, None

    bgr = image[:, :, :3]
    alpha = image[:, :, 3]
    return bgr, alpha


def merge_alpha(bgr: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """Merge alpha channel back into a BGRA image."""
    if alpha is None:
        return bgr
    if bgr.shape[:2] != alpha.shape[:2]:
        return bgr
    return np.dstack([bgr, alpha])


def resize_if_large(
    image: np.ndarray, metadata: dict, max_dimension: int = 4096
) -> Tuple[np.ndarray, dict]:
    """Resize image if its largest dimension exceeds max_dimension."""
    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim <= max_dimension:
        return image, metadata

    scale = max_dimension / max_dim
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    metadata["width"] = new_w
    metadata["height"] = new_h

    logger.info("Resized: %dx%d → %dx%d", w, h, new_w, new_h)
    return resized, metadata


def save_image(image: np.ndarray, file_path: str) -> str:
    """Save an image to disk. Creates parent directories if needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to save: {file_path}")
    logger.debug("Saved: %s", file_path)
    return str(path)


def convert_to_8bit(image: np.ndarray) -> np.ndarray:
    """Convert a 16-bit image to 8-bit using min-max normalization."""
    if image.dtype == np.uint8:
        return image
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
