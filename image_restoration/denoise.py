"""
Noise removal module with adaptive method selection.

Implements: Median, Gaussian, Bilateral, Non-Local Means,
Morphological operations, and adaptive selection based on image content.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import DenoiseConfig
from .utils import logger


def estimate_noise_level(gray: np.ndarray) -> float:
    """Estimate image noise level using Laplacian variance.

    Higher variance = less noise. Lower = more noise.

    Args:
        gray: Grayscale image.

    Returns:
        Noise score [0-1], where 0 = clean, 1 = very noisy.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    # Normalize: variance < 10 = noisy, > 100 = clean
    noise_score = 1.0 - min(variance / 100.0, 1.0)
    return float(noise_score)


def apply_median_blur(image: np.ndarray, kernel: int) -> np.ndarray:
    """Apply median blur for salt-and-pepper noise removal.

    Args:
        image: Input image.
        kernel: Kernel size (odd). 0 = skip.

    Returns:
        Filtered image or original if kernel <= 0.
    """
    if kernel <= 0:
        return image
    if kernel % 2 == 0:
        kernel += 1
    return cv2.medianBlur(image, kernel)


def apply_gaussian_blur(image: np.ndarray, kernel: int) -> np.ndarray:
    """Apply Gaussian blur for general smoothing.

    Args:
        image: Input image.
        kernel: Kernel size (odd). 0 = skip.

    Returns:
        Filtered image.
    """
    if kernel <= 0:
        return image
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """Apply bilateral filter — edge-preserving noise reduction.

    Smooths while preserving edges by considering both spatial
    and color differences.

    Args:
        image: Input image (grayscale or color).
        d: Diameter of pixel neighborhood.
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.

    Returns:
        Filtered image.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_nlm_denoise(
    image: np.ndarray,
    strength: float = 5.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Apply Fast Non-Local Means denoising.

    Best for Gaussian noise; preserves texture better than blur.

    Args:
        image: Input image.
        strength: Denoising strength.
        template_window: Template window size (odd).
        search_window: Search window size (odd).

    Returns:
        Denoised image.
    """
    if strength <= 0:
        return image

    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(
            image, None, h=strength,
            templateWindowSize=template_window,
            searchWindowSize=search_window,
        )
    else:
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=strength, hColor=strength,
            templateWindowSize=template_window,
            searchWindowSize=search_window,
        )


def apply_morph_open(
    image: np.ndarray, kernel_size: tuple, iterations: int = 1
) -> np.ndarray:
    """Morphological opening: removes small noise points."""
    if kernel_size[0] <= 0 or kernel_size[1] <= 0:
        return image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_morph_close(
    image: np.ndarray, kernel_size: tuple, iterations: int = 1
) -> np.ndarray:
    """Morphological closing: fills small holes."""
    if kernel_size[0] <= 0 or kernel_size[1] <= 0:
        return image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def adaptive_denoise(
    image: np.ndarray,
    config: DenoiseConfig,
) -> np.ndarray:
    """Adaptively select and apply the best denoising method.

    Analyzes noise level and image content to choose the optimal
    combination of denoising filters.

    Args:
        image: Input BGR image.
        config: Denoise configuration.

    Returns:
        Denoised image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    noise_score = estimate_noise_level(gray)
    h, w = image.shape[:2]
    image_area = h * w

    logger.debug("Noise score: %.3f (0=clean, 1=noisy)", noise_score)

    result = image

    if not config.adaptive:
        # Non-adaptive: apply all configured filters
        result = apply_median_blur(result, config.median_kernel)
        result = apply_gaussian_blur(result, config.gaussian_kernel)
        result = apply_bilateral_filter(
            result, config.bilateral_d,
            config.bilateral_sigma_color,
            config.bilateral_sigma_space,
        )
        result = apply_nlm_denoise(
            result, config.nlm_strength,
            config.nlm_template_window,
            config.nlm_search_window,
        )
        return result

    # Adaptive selection based on noise level
    if noise_score < 0.15:
        # Very clean: light bilateral only
        result = apply_bilateral_filter(result, d=5, sigma_color=30, sigma_space=30)
    elif noise_score < 0.35:
        # Slight noise: median + light bilateral
        result = apply_median_blur(result, max(config.median_kernel, 3))
        result = apply_bilateral_filter(result, d=7, sigma_color=40, sigma_space=40)
    elif noise_score < 0.60:
        # Moderate noise: median + bilateral + light NLM
        result = apply_median_blur(result, 3)
        result = apply_bilateral_filter(result, d=9, sigma_color=60, sigma_space=60)
        result = apply_nlm_denoise(result, strength=5.0)
    else:
        # Heavy noise: full pipeline
        result = apply_median_blur(result, 5)
        result = apply_bilateral_filter(result, d=9, sigma_color=75, sigma_space=75)
        result = apply_nlm_denoise(result, strength=10.0)

    logger.info("Adaptive denoise applied (noise_score=%.2f)", noise_score)
    return result
