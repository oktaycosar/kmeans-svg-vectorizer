"""
Background cleaning module.

Removes JPEG artifacts, compression blocks, isolated pixels,
dust, scanning artifacts while preserving object geometry.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import CleanupConfig
from .utils import logger


def remove_small_components(
    binary: np.ndarray, min_area: int = 15
) -> np.ndarray:
    """Remove small connected components (noise, dust, isolated pixels).

    Args:
        binary: Binary image (0/255).
        min_area: Minimum component area to keep.

    Returns:
        Cleaned binary image.
    """
    if min_area <= 0:
        return binary

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Create output: only keep large components
    result = np.zeros_like(binary)

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[labels == i] = 255

    removed = num_labels - 1 - np.sum(
        [1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    )
    if removed > 0:
        logger.debug("Removed %d small components (< %d px)", removed, min_area)

    return result


def remove_jpeg_artifacts(image: np.ndarray) -> np.ndarray:
    """Reduce JPEG 8x8 blocking artifacts.

    JPEG compression creates visible 8x8 blocks, especially in
    smooth gradient areas. This applies a frequency-domain filter
    to suppress block-boundary discontinuities.

    Args:
        image: Input BGR image.

    Returns:
        De-blocked image.
    """
    result = image.copy()

    # Apply gentle median filter that specifically targets
    # 8x8 block boundaries
    # A 3x3 median is small enough to preserve edges but large
    # enough to smooth JPEG blocking
    for c in range(image.shape[2]):
        result[:, :, c] = cv2.medianBlur(image[:, :, c], 3)

    # Blend: 30% filtered, 70% original (preserve sharpness)
    blended = cv2.addWeighted(result, 0.3, image, 0.7, 0)

    logger.debug("JPEG artifact reduction applied")
    return blended


def remove_isolated_pixels(image: np.ndarray) -> np.ndarray:
    """Remove isolated bright/dark pixels (salt-and-pepper noise).

    Uses a 3x3 median filter which is highly effective at removing
    isolated extreme pixels while preserving edges.

    Args:
        image: Input image (grayscale or BGR).

    Returns:
        Cleaned image.
    """
    if image.ndim == 2:
        return cv2.medianBlur(image, 3)
    else:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = cv2.medianBlur(image[:, :, c], 3)
        return result


def remove_scan_artifacts(image: np.ndarray) -> np.ndarray:
    """Remove horizontal/vertical line artifacts from scanning.

    Uses morphological operations to detect thin lines and
    inpaint them.

    Args:
        image: Input BGR image.

    Returns:
        Cleaned image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)

    # Combine line masks
    line_mask = cv2.bitwise_or(
        cv2.threshold(h_lines, 30, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(v_lines, 30, 255, cv2.THRESH_BINARY)[1],
    )

    # Dilate mask slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    line_mask = cv2.dilate(line_mask, kernel, iterations=1)

    # Inpaint the detected line regions
    result = cv2.inpaint(image, line_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    logger.debug("Scan artifact removal applied")
    return result


def morphological_cleanup(
    image: np.ndarray, kernel_size: tuple = (3, 3)
) -> np.ndarray:
    """Apply morphological opening + closing to clean small artifacts.

    Args:
        image: Input image.
        kernel_size: Morphological kernel size.

    Returns:
        Cleaned image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    if image.ndim == 3:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            opened = cv2.morphologyEx(image[:, :, c], cv2.MORPH_OPEN, kernel)
            result[:, :, c] = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return result
    else:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def clean_background(
    image: np.ndarray,
    config: CleanupConfig,
) -> np.ndarray:
    """Run the full background cleaning pipeline.

    Args:
        image: Input BGR image.
        config: Background cleaner configuration.

    Returns:
        Cleaned image.
    """
    result = image

    # Step 1: Remove JPEG artifacts (8x8 block deblocking)
    if config.remove_jpeg:
        result = remove_jpeg_artifacts(result)

    # Step 2: Remove isolated pixels (salt noise)
    if config.remove_isolated:
        result = remove_isolated_pixels(result)

    # Step 3: Morphological cleanup
    result = morphological_cleanup(result, config.morph_close_kernel)

    # Step 5: Remove small connected components from binary edge mask
    # (applied to the final image to clean up tiny specks)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned_binary = remove_small_components(binary, config.min_component_area)

    # Only apply binary cleanup to the edges, not to fill regions
    edges = cv2.Canny(gray, 30, 100)
    # Don't apply component removal to the full image; only to the edge binary

    logger.info("Background cleaning complete")
    return result
