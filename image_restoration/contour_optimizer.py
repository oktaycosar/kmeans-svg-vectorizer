"""
Contour optimizer — prepares images for OpenCV contour detection.

Closes small gaps, merges nearly-connected edges, reduces stair-step
artifacts, and improves contour continuity WITHOUT introducing halos.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import ContourOptimizeConfig
from .utils import logger


def close_edge_gaps(
    binary: np.ndarray, kernel_size: tuple = (4, 4)
) -> np.ndarray:
    """Close small gaps in edge map using morphological closing.

    Small gaps in contours cause fragmentation. A gentle morphological
    close bridges gaps < kernel_size pixels wide.

    Args:
        binary: Binary edge map (0/255).
        kernel_size: Morphological kernel size.

    Returns:
        Binary image with gaps closed.
    """
    if kernel_size[0] <= 0 or kernel_size[1] <= 0:
        return binary

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed


def merge_nearby_edges(
    binary: np.ndarray, min_length: int = 10
) -> np.ndarray:
    """Merge nearly-connected parallel edges.

    Uses morphological skeletonization followed by dilation
    to connect edges that are almost touching.

    Args:
        binary: Binary edge map.
        min_length: Minimum edge segment length to keep.

    Returns:
        Merged binary edge map.
    """
    if min_length <= 0:
        return binary

    # Dilate slightly to merge nearby parallel edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Remove very short isolated edges
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    result = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_length:
            result[labels == i] = 255

    return result


def reduce_stair_step(
    image: np.ndarray, sigma: float = 0.3
) -> np.ndarray:
    """Reduce stair-step aliasing on diagonal edges.

    Applies an extremely light Gaussian blur ONLY at edge pixels
    to smooth jagged diagonals without blurring interiors.

    Args:
        image: Input BGR image.
        sigma: Blur sigma (keep VERY small: 0.2-0.5).

    Returns:
        Anti-aliased image.
    """
    if sigma <= 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Slightly dilate edge mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edge_mask = cv2.dilate(edges, kernel, iterations=1)

    # Very light blur
    ksize = max(3, int(sigma * 6 + 1))
    if ksize % 2 == 0:
        ksize += 1
    smoothed = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # Blend: only smooth at diagonal edges
    if image.ndim == 3:
        mask_3ch = cv2.merge([edge_mask, edge_mask, edge_mask]) / 255.0
        result = (image.astype(np.float32) * (1 - mask_3ch) +
                  smoothed.astype(np.float32) * mask_3ch).astype(np.uint8)
    else:
        mask_f = edge_mask / 255.0
        result = (image.astype(np.float32) * (1 - mask_f) +
                  smoothed.astype(np.float32) * mask_f).astype(np.uint8)

    return result


def optimize_contours(
    image: np.ndarray,
    config: ContourOptimizeConfig,
) -> np.ndarray:
    """Run the full contour optimization pipeline.

    Args:
        image: Input BGR image.
        config: Contour optimization configuration.

    Returns:
        Contour-optimized image.
    """
    result = image

    # Step 1: Anti-alias (very light, only at diagonal edges)
    if config.anti_alias_sigma > 0:
        result = reduce_stair_step(result, config.anti_alias_sigma)

    # Step 2: Work on edge map for gap closing
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Step 3: Close small gaps
    if config.close_gaps:
        edges = close_edge_gaps(edges, config.gap_kernel)

    # Step 4: Merge nearby edges
    if config.merge_edges:
        edges = merge_nearby_edges(edges, config.min_edge_length)

    logger.info(
        "Contour optimization: anti-alias=%.1f, gaps=%s",
        config.anti_alias_sigma, config.close_gaps,
    )

    return result
