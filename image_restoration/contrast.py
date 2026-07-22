"""
Contrast improvement module.

Implements: Histogram Equalization, CLAHE, Gamma Correction,
Adaptive Brightness, Shadow Recovery, Highlight Compression.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import ContrastConfig
from .utils import logger, clamp


def compute_mean_brightness(gray: np.ndarray) -> float:
    """Compute mean brightness of a grayscale image [0, 255].

    Args:
        gray: Grayscale image.

    Returns:
        Mean brightness value.
    """
    return float(np.mean(gray))


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_size: tuple = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    Improves local contrast without amplifying noise.

    Args:
        gray: Grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_size: Grid size for local equalization.

    Returns:
        CLAHE-enhanced image.
    """
    if clip_limit <= 0:
        return gray

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def apply_clahe_color(image: np.ndarray, clip_limit: float = 2.0, tile_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE to a color image via LAB color space.

    Applies CLAHE to the L (lightness) channel to avoid color shifts.

    Args:
        image: BGR image.
        clip_limit: CLAHE clip limit.
        tile_size: Grid size.

    Returns:
        Enhanced BGR image.
    """
    if clip_limit <= 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to the image.

    gamma < 1.0: brighten shadows
    gamma > 1.0: darken highlights

    Args:
        image: Input image (0-255 range).
        gamma: Gamma value.

    Returns:
        Gamma-corrected image.
    """
    if gamma <= 0:
        return image

    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

    if image.ndim == 2:
        return cv2.LUT(image, table)
    else:
        # Apply to each channel
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = cv2.LUT(image[:, :, c], table)
        return result


def auto_gamma_correction(image: np.ndarray) -> np.ndarray:
    """Automatically determine gamma based on mean brightness.

    Dark images get gamma < 1 (brighten), bright images get gamma > 1 (darken).

    Args:
        image: Input image.

    Returns:
        Gamma-corrected image.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mean_brightness = compute_mean_brightness(gray)

    # Target brightness: 128 (mid-gray)
    if mean_brightness < 60:
        gamma = 0.6  # Very dark → brighten a lot
    elif mean_brightness < 100:
        gamma = 0.8  # Dark → brighten
    elif mean_brightness > 180:
        gamma = 1.3  # Very bright → darken
    elif mean_brightness > 150:
        gamma = 1.1  # Bright → slight darken
    else:
        gamma = 1.0  # Normal → no change

    logger.debug("Auto gamma: %.2f (mean brightness: %.0f)", gamma, mean_brightness)

    if abs(gamma - 1.0) < 0.01:
        return image

    return apply_gamma_correction(image, gamma)


def recover_shadows(image: np.ndarray, amount: float = 0.1) -> np.ndarray:
    """Lift shadow details using adaptive thresholding.

    Brightens the darkest regions while preserving midtones and highlights.

    Args:
        image: Input BGR image.
        amount: Shadow recovery amount [0-1].

    Returns:
        Image with lifted shadows.
    """
    if amount <= 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Create shadow mask: pixels below 64 brightness
    _, shadow_mask = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

    # Brighten the shadow regions
    brightened = cv2.addWeighted(image, 1.0 + amount, image, 0, amount * 30)

    # Blend: only apply to shadow regions
    if image.ndim == 3:
        shadow_mask_3ch = cv2.merge([shadow_mask, shadow_mask, shadow_mask]) / 255.0
        result = (image * (1 - shadow_mask_3ch) + brightened * shadow_mask_3ch).astype(np.uint8)
    else:
        shadow_mask_f = shadow_mask / 255.0
        result = (image * (1 - shadow_mask_f) + brightened * shadow_mask_f).astype(np.uint8)

    return result


def compress_highlights(image: np.ndarray, amount: float = 0.1) -> np.ndarray:
    """Compress highlight regions to recover overexposed details.

    Args:
        image: Input BGR image.
        amount: Compression amount [0-1].

    Returns:
        Image with compressed highlights.
    """
    if amount <= 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Highlight mask: pixels above 200 brightness
    _, highlight_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Darken the highlight regions
    darkened = cv2.addWeighted(image, 1.0 - amount * 0.5, image, 0, -amount * 20)

    if image.ndim == 3:
        highlight_mask_3ch = cv2.merge([highlight_mask, highlight_mask, highlight_mask]) / 255.0
        result = (image * (1 - highlight_mask_3ch) + darkened * highlight_mask_3ch).astype(np.uint8)
    else:
        highlight_mask_f = highlight_mask / 255.0
        result = (image * (1 - highlight_mask_f) + darkened * highlight_mask_f).astype(np.uint8)

    return result


def apply_adaptive_brightness(image: np.ndarray) -> np.ndarray:
    """Automatically adjust brightness based on image statistics.

    Uses the LAB color space for perceptually uniform adjustments.

    Args:
        image: Input BGR image.

    Returns:
        Brightness-adjusted image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)

    # Compute L-channel statistics
    mean_l = np.mean(l)
    std_l = np.std(l)

    # Target mean: 128 (mid-gray in LAB L)
    # If too dark or too bright, adjust
    adjustment = (128.0 - mean_l) * 0.3  # 30% correction toward ideal

    l_adjusted = np.clip(l.astype(np.float32) + adjustment, 0, 255).astype(np.uint8)

    lab_adjusted = cv2.merge([l_adjusted, a, b_ch])
    return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)


def enhance_contrast(
    image: np.ndarray,
    config: ContrastConfig,
) -> np.ndarray:
    """Run the full contrast enhancement pipeline.

    Args:
        image: Input BGR image.
        config: Contrast configuration.

    Returns:
        Contrast-enhanced image.
    """
    result = image

    # Step 1: CLAHE
    result = apply_clahe_color(result, config.clahe_clip_limit, config.clahe_tile_size)

    # Step 2: Gamma correction
    if config.auto_gamma:
        result = auto_gamma_correction(result)
    elif config.gamma > 0:
        result = apply_gamma_correction(result, config.gamma)

    # Step 3: Shadow recovery
    result = recover_shadows(result, config.shadow_recovery)

    # Step 4: Highlight compression
    result = compress_highlights(result, config.highlight_compression)

    # Step 5: Adaptive brightness
    if config.adaptive_brightness:
        result = apply_adaptive_brightness(result)

    logger.info("Contrast enhancement complete")
    return result
