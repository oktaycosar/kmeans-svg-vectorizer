"""
Image loading module with alpha channel preservation.

Supports PNG, JPG, BMP, TIFF, WebP formats.
Returns image data with metadata for downstream processing.
"""

from __future__ import annotations

from typing import Tuple, Optional, Any
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import ImageInfo
from .utils import logger, is_supported_image


def load_image(file_path: str) -> Tuple[np.ndarray, ImageInfo]:
    """Load an image from disk with all channels preserved.

    The image is loaded in BGR(A) format (OpenCV convention) for processing,
    but metadata tracks the actual format.

    Args:
        file_path: Path to the image file.

    Returns:
        Tuple of (image_array, ImageInfo).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        IOError: If the file cannot be opened as an image.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    if not is_supported_image(file_path):
        raise ValueError(
            f"Unsupported image format: '{path.suffix}'. "
            f"Supported formats: png, jpg, jpeg, bmp, tiff, tif, webp"
        )

    # First try OpenCV for performance
    image: Optional[np.ndarray] = None
    has_alpha: bool = False
    channels: int = 3

    # Check if the image has alpha using PIL first
    try:
        pil_img = Image.open(file_path)
        pil_mode = pil_img.mode
        has_alpha = pil_mode in ("RGBA", "LA", "PA") or (
            pil_mode == "P" and "transparency" in pil_img.info
        )
        pil_img.close()
    except Exception:
        pass  # Fall through to OpenCV loading

    # Load with OpenCV
    if has_alpha:
        # Load with alpha channel (IMREAD_UNCHANGED)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    else:
        # Load as BGR
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise IOError(f"Failed to load image: {file_path}. File may be corrupted.")

    # Determine actual channels
    if image.ndim == 2:
        channels = 1
        # Convert grayscale to 3-channel for consistent processing
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        shape_channels = image.shape[2]
        if shape_channels == 1:
            channels = 1
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif shape_channels == 3:
            channels = 3
            has_alpha = False
        elif shape_channels == 4:
            channels = 4
            has_alpha = True
        else:
            channels = shape_channels

    height, width = image.shape[:2]

    info = ImageInfo(
        width=width,
        height=height,
        channels=channels,
        has_alpha=has_alpha,
        file_path=str(path.absolute()),
    )

    logger.info(
        "Loaded image: %s (%dx%d, %d channels, alpha=%s)",
        path.name, width, height, channels, has_alpha,
    )

    return image, info


def split_alpha_channel(image: np.ndarray, info: ImageInfo) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Split the alpha channel from a BGR image if present.

    Args:
        image: Image array (BGR or BGRA).
        info: Image metadata.

    Returns:
        Tuple of (bgr_image, alpha_channel).
        alpha_channel is None if no alpha is present.
    """
    if not info.has_alpha or image.shape[2] < 4:
        return image, None

    bgr = image[:, :, :3]
    alpha = image[:, :, 3]
    return bgr, alpha


def resize_if_needed(
    image: np.ndarray, info: ImageInfo, max_dimension: int = 1920
) -> Tuple[np.ndarray, ImageInfo]:
    """Resize image if its largest dimension exceeds max_dimension.

    Aspect ratio is preserved. Return (possibly resized) image and updated info.

    Args:
        image: Input image array.
        info: Current image metadata.
        max_dimension: Maximum allowed dimension in pixels.

    Returns:
        Tuple of (image, ImageInfo). May be the same if no resize needed.
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim <= max_dimension:
        return image, info

    scale = max_dimension / max_dim
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    updated_info = ImageInfo(
        width=new_w,
        height=new_h,
        channels=info.channels,
        has_alpha=info.has_alpha,
        file_path=info.file_path,
    )

    logger.info(
        "Resized image: %dx%d → %dx%d (scale=%.2f)",
        w, h, new_w, new_h, scale,
    )

    return resized, updated_info


def extract_alpha_mask(image: np.ndarray, info: ImageInfo) -> Optional[np.ndarray]:
    """Extract a binary alpha mask from the image.

    Args:
        image: Image array (BGR or BGRA).
        info: Image metadata.

    Returns:
        Binary mask (uint8, 0 or 255), or None if no alpha.
    """
    if not info.has_alpha or image.shape[2] < 4:
        return None

    alpha = image[:, :, 3]
    _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    return mask


def save_image(image: np.ndarray, file_path: str) -> None:
    """Save an image to disk.

    Args:
        image: Image array (BGR or BGRA).
        file_path: Output path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to save image: {file_path}")
    logger.debug("Saved image: %s", file_path)
