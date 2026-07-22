"""
Mandatory color quantization — the KEY to clean vectorization.

Fewer colors = fewer unique contours = better vectorization.
Uses K-Means with automatic optimal-k detection based on image type.

CRITICAL RULE: Output color count MUST be ≤ input color count.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import QuantizationConfig
from .utils import logger


def count_unique_colors(image: np.ndarray, tolerance: int = 4) -> int:
    """Count unique colors with tolerance for near-identical values."""
    if image.ndim != 3:
        return len(np.unique(image // tolerance))
    q = image // tolerance
    combined = (q[:,:,0].astype(np.int32) * 10000 +
                q[:,:,1].astype(np.int32) * 100 +
                q[:,:,2].astype(np.int32))
    return len(np.unique(combined))


def auto_detect_optimal_k(image: np.ndarray, image_type: str = "logo") -> int:
    """Auto-detect optimal number of K-Means clusters.

    Based on image type and current color count.

    Args:
        image: BGR image.
        image_type: From image_analyzer (logo/ui/illustration/photo).

    Returns:
        Optimal k value.
    """
    current_colors = count_unique_colors(image)
    type_ranges = {"logo": (2, 16), "diagram": (2, 16), "ui": (4, 32),
                   "screenshot": (8, 48), "illustration": (8, 64),
                   "photo": (16, 128)}
    min_k, max_k = type_ranges.get(image_type, (4, 32))

    # Never increase colors; clamp to current count
    suggested = min(current_colors, max(min_k, min(current_colors // 2, max_k)))
    suggested = max(2, min(suggested, 256))
    logger.debug("Auto k=%d (current=%d, type=%s)", suggested, current_colors, image_type)
    return suggested


def kmeans_quantize(image: np.ndarray, k: int, iterations: int = 15,
                    epsilon: float = 1.0) -> np.ndarray:
    """K-Means color quantization — MANDATORY step.

    Reduces image to exactly k dominant colors. This is the single
    most important preprocessing step for clean vectorization.

    Args:
        image: BGR image.
        k: Target color count.
        iterations: Max K-Means iterations.
        epsilon: Convergence epsilon.

    Returns:
        Color-quantized image with exactly k colors.
    """
    if k <= 1:
        return image

    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)

    return centers[labels.flatten()].reshape(h, w, -1)


def preserve_white_black(result: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Restore pure white (#FFF) and black (#000) exactly."""
    if original.shape != result.shape:
        return result
    r = result.copy()
    wm = np.all(original >= 252, axis=2) if original.ndim == 3 else (original >= 252)
    bm = np.all(original <= 3, axis=2) if original.ndim == 3 else (original <= 3)
    if original.ndim == 3:
        r[wm] = (255, 255, 255); r[bm] = (0, 0, 0)
    else:
        r[wm] = 255; r[bm] = 0
    return r


def quantize_image(image: np.ndarray, config: QuantizationConfig,
                   original: np.ndarray, image_type: str = "logo") -> np.ndarray:
    """Run mandatory quantization.

    Args:
        image: Input BGR image.
        config: Quantization config.
        original: Original image (for extreme preservation).
        image_type: From image_analyzer.

    Returns:
        Quantized image with fewer or equal colors.
    """
    before_colors = count_unique_colors(image)

    k = auto_detect_optimal_k(image, image_type) if config.auto_k else config.fallback_k
    k = min(k, before_colors)  # NEVER increase colors

    result = kmeans_quantize(image, k, config.kmeans_iterations, config.kmeans_epsilon)

    if config.preserve_extremes:
        result = preserve_white_black(result, original)

    after_colors = count_unique_colors(result)
    logger.info("Quantization: %d → %d colors (k=%d, type=%s)",
                before_colors, after_colors, k, image_type)
    return result
