"""
Edge preservation and anti-alias improvement module.

Preserves sharp object boundaries while reducing jagged edges
(stair-step artifacts) and applying edge-aware smoothing.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import EdgePreserveConfig
from .utils import logger


def guided_filter(
    image: np.ndarray,
    radius: int = 4,
    eps: float = 0.01,
) -> np.ndarray:
    """Apply guided filter for edge-preserving smoothing.

    Similar to bilateral filter but faster and free of gradient
    reversal artifacts. Preserves edges while smoothing flat regions.

    Implementation uses a simplified box-filter approach.

    Args:
        image: Input image (BGR or grayscale).
        radius: Filter radius.
        eps: Regularization parameter.

    Returns:
        Edge-preserved smoothed image.
    """
    if radius <= 0:
        return image

    image_f = image.astype(np.float32) / 255.0

    if image.ndim == 2:
        return _guided_filter_channel(image_f, image_f, radius, eps)

    # Process each channel with the grayscale guide
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    result = np.zeros_like(image_f)
    for c in range(image_f.shape[2]):
        result[:, :, c] = _guided_filter_channel(image_f[:, :, c], gray, radius, eps)

    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def _guided_filter_channel(
    p: np.ndarray, guide: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    """Guided filter for a single channel.

    Args:
        p: Input channel to filter.
        guide: Guidance image (same size).
        radius: Filter radius.
        eps: Regularization.

    Returns:
        Filtered channel (float32, 0-1 range).
    """
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    mean_p = cv2.filter2D(p, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mean_g = cv2.filter2D(guide, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mean_gg = cv2.filter2D(guide * guide, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mean_pg = cv2.filter2D(p * guide, -1, kernel, borderType=cv2.BORDER_REFLECT)

    var_g = mean_gg - mean_g * mean_g
    cov_pg = mean_pg - mean_p * mean_g

    a = cov_pg / (var_g + eps)
    b = mean_p - a * mean_g

    mean_a = cv2.filter2D(a, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.filter2D(b, -1, kernel, borderType=cv2.BORDER_REFLECT)

    return mean_a * guide + mean_b


def reduce_jagged_edges(
    image: np.ndarray, sigma: float = 0.5
) -> np.ndarray:
    """Reduce stair-step (aliasing) artifacts on diagonal edges.

    Applies a very light Gaussian blur ONLY at edge pixels,
    preserving sharp edges elsewhere.

    Args:
        image: Input image.
        sigma: Blur sigma (small).

    Returns:
        Anti-aliased image.
    """
    if sigma <= 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Thin edges to 1-pixel width for precise mask
    # Then slightly dilate to include the jagged neighborhood
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edge_mask = cv2.dilate(edges, kernel, iterations=1)

    # Apply very light Gaussian blur
    ksize = max(3, int(sigma * 4 + 1))
    if ksize % 2 == 0:
        ksize += 1
    smoothed = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # Blend: smooth only at edges
    if image.ndim == 3:
        edge_mask_3ch = cv2.merge([edge_mask, edge_mask, edge_mask]) / 255.0
        result = (image * (1 - edge_mask_3ch) + smoothed * edge_mask_3ch).astype(np.uint8)
    else:
        edge_mask_f = edge_mask / 255.0
        result = (image * (1 - edge_mask_f) + smoothed * edge_mask_f).astype(np.uint8)

    return result


def bilateral_edge_preserve(
    image: np.ndarray, strength: float = 0.5
) -> np.ndarray:
    """Edge-preserving smoothing using bilateral filter with
    strength-adaptive parameters.

    Args:
        image: Input image.
        strength: Smoothing strength.

    Returns:
        Smoothed image with preserved edges.
    """
    if strength <= 0:
        return image

    d = int(5 + strength * 4)
    sigma_color = 40 + strength * 40
    sigma_space = 40 + strength * 40

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def preserve_edges(
    image: np.ndarray,
    config: EdgePreserveConfig,
) -> np.ndarray:
    """Run the full edge preservation + anti-alias pipeline.

    Args:
        image: Input BGR image.
        config: Edge preservation configuration.

    Returns:
        Edge-preserved image.
    """
    result = image

    # Step 1: Edge-aware smoothing (guided filter)
    if config.guided_radius > 0:
        result = guided_filter(result, config.guided_radius, config.guided_eps)

    # Step 2: Bilateral edge preserve
    if config.bilateral_edge_strength > 0:
        result = bilateral_edge_preserve(result, config.bilateral_edge_strength)

    # Step 3: Anti-alias
    if config.anti_alias:
        result = reduce_jagged_edges(result, config.anti_alias_sigma)

    logger.info("Edge preservation + anti-alias complete")
    return result
