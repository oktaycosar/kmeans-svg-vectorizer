"""
Sharpening module with halo-free edge enhancement.

Implements: Unsharp Mask, Laplacian Sharpen, Edge-Aware Sharpening.
All methods are designed to avoid creating halos around edges.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import SharpnessConfig
from .utils import logger, clamp


def unsharp_mask(
    image: np.ndarray,
    amount: float = 1.0,
    radius: float = 1.5,
    threshold: int = 2,
) -> np.ndarray:
    """Apply Unsharp Mask sharpening.

    Classic technique: subtract a blurred version from the original
    to isolate high-frequency details, then add them back amplified.

    Args:
        image: Input image (BGR or grayscale).
        amount: Sharpening strength [0, 5].
        radius: Gaussian blur radius.
        threshold: Minimum brightness difference to sharpen (reduces noise).

    Returns:
        Sharpened image.
    """
    if amount <= 0:
        return image

    # Convert radius to kernel size
    kernel_size = max(3, int(radius * 2) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create blurred version
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), radius)

    # Compute difference (high-frequency detail)
    if image.ndim == 3:
        detail = cv2.subtract(image, blurred)
    else:
        detail = image.astype(np.int16) - blurred.astype(np.int16)

    # Apply threshold: only sharpen where difference is significant
    if threshold > 0:
        mask = np.abs(detail) >= threshold
        if image.ndim == 3:
            mask = np.any(mask, axis=2)
        detail[~mask] = 0

    # Add amplified detail back
    result = image.astype(np.float32) + detail.astype(np.float32) * amount
    return np.clip(result, 0, 255).astype(np.uint8)


def laplacian_sharpen(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Sharpen using Laplacian operator (edge detection kernel).

    The Laplacian highlights regions of rapid intensity change.
    Subtracting it from the original increases edge contrast.

    Args:
        image: Input BGR image.
        strength: Sharpening strength [0, 1].

    Returns:
        Sharpened image.
    """
    if strength <= 0:
        return image

    # Laplacian kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ], dtype=np.float32)

    if image.ndim == 3:
        # Apply to each channel separately
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            laplacian = cv2.filter2D(image[:, :, c].astype(np.float32), -1, kernel)
            result[:, :, c] = image[:, :, c].astype(np.float32) - laplacian * strength
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        laplacian = cv2.filter2D(image.astype(np.float32), -1, kernel)
        result = image.astype(np.float32) - laplacian * strength
        return np.clip(result, 0, 255).astype(np.uint8)


def edge_aware_sharpen(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Sharpen only at detected edge locations.

    Uses Canny edge detection to create an edge mask, then applies
    sharpening only within that mask. This prevents noise amplification
    in flat regions.

    Args:
        image: Input BGR image.
        strength: Sharpening strength [0, 1].

    Returns:
        Edge-sharpened image.
    """
    if strength <= 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Detect edges with Canny
    edges = cv2.Canny(gray, 30, 100)

    # Dilate the edge mask slightly to include edge neighborhood
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_mask = cv2.dilate(edges, kernel_dilate, iterations=1)

    # Apply unsharp mask to the whole image
    sharpened = unsharp_mask(image, amount=strength * 2, radius=1.0, threshold=0)

    # Blend: sharpen only at edges
    if image.ndim == 3:
        edge_mask_3ch = cv2.merge([edge_mask, edge_mask, edge_mask]) / 255.0
        result = (image * (1 - edge_mask_3ch) + sharpened * edge_mask_3ch).astype(np.uint8)
    else:
        edge_mask_f = edge_mask / 255.0
        result = (image * (1 - edge_mask_f) + sharpened * edge_mask_f).astype(np.uint8)

    return result


def enhance_sharpness(
    image: np.ndarray,
    config: SharpnessConfig,
) -> np.ndarray:
    """Run the full sharpening pipeline.

    Applies sharpening methods in sequence: edge-aware first,
    then unsharp mask, then Laplacian for subtle edge enhancement.

    Args:
        image: Input BGR image.
        config: Sharpness configuration.

    Returns:
        Sharpened image.
    """
    result = image

    # Step 1: Edge-aware sharpen (targeted, no halo on flat areas)
    if config.edge_aware_strength > 0:
        result = edge_aware_sharpen(result, config.edge_aware_strength)

    # Step 2: Unsharp mask (global detail enhancement)
    if config.unsharp_amount > 0:
        result = unsharp_mask(
            result,
            amount=config.unsharp_amount,
            radius=config.unsharp_radius,
            threshold=config.unsharp_threshold,
        )

    # Step 3: Light Laplacian sharpen for micro-contrast
    if config.laplacian_strength > 0:
        result = laplacian_sharpen(result, config.laplacian_strength)

    logger.info("Sharpness enhancement complete")
    return result
