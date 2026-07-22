"""
Image preprocessing pipeline for the ImageVectorizer.

Provides all preprocessing operations with parameterized configuration:
grayscale, blur, threshold, morphology, denoising, and edge detection.
"""

from __future__ import annotations

from typing import Tuple, Optional

import cv2
import numpy as np

from .config import PreprocessConfig
from .utils import logger


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    If already grayscale (2D), return as-is.

    Args:
        image: Input image (BGR or grayscale).

    Returns:
        Single-channel grayscale image.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur to an image.

    Args:
        image: Input image (grayscale or color).
        kernel_size: Kernel size (odd number). 0 = skip.

    Returns:
        Blurred image, or original if kernel_size is 0.
    """
    if kernel_size <= 0:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply median blur (salt-and-pepper noise removal).

    Args:
        image: Input image.
        kernel_size: Kernel size (odd number). 0 = skip.

    Returns:
        Blurred image, or original if kernel_size is 0.
    """
    if kernel_size <= 0:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def apply_adaptive_threshold(
    gray_image: np.ndarray,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """Apply adaptive (Gaussian) threshold to a grayscale image.

    Args:
        gray_image: Grayscale input.
        block_size: Size of pixel neighborhood (odd).
        c: Constant subtracted from the mean.

    Returns:
        Binary image (0 or 255).
    """
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )


def apply_binary_threshold(
    gray_image: np.ndarray,
    thresh: int = 127,
    max_val: int = 255,
) -> np.ndarray:
    """Apply simple binary threshold.

    Args:
        gray_image: Grayscale input.
        thresh: Threshold value.
        max_val: Maximum value for pixels above threshold.

    Returns:
        Binary image.
    """
    _, binary = cv2.threshold(gray_image, thresh, max_val, cv2.THRESH_BINARY)
    return binary


def apply_otsu_threshold(gray_image: np.ndarray) -> np.ndarray:
    """Apply Otsu's automatic thresholding.

    Args:
        gray_image: Grayscale input.

    Returns:
        Binary image.
    """
    _, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def apply_morphological_opening(
    binary_image: np.ndarray,
    kernel_size: Tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    """Apply morphological opening (erosion followed by dilation).

    Removes small noise/isolated pixels.

    Args:
        binary_image: Binary input.
        kernel_size: Kernel dimensions.
        iterations: Number of iterations.

    Returns:
        Processed binary image.
    """
    if iterations <= 0:
        return binary_image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_morphological_closing(
    binary_image: np.ndarray,
    kernel_size: Tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    """Apply morphological closing (dilation followed by erosion).

    Closes small holes in foreground regions.

    Args:
        binary_image: Binary input.
        kernel_size: Kernel dimensions.
        iterations: Number of iterations.

    Returns:
        Processed binary image.
    """
    if iterations <= 0:
        return binary_image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def apply_denoising(
    image: np.ndarray,
    strength: float = 10.0,
) -> np.ndarray:
    """Apply Non-Local Means denoising.

    Args:
        image: Input image (grayscale or color).
        strength: Denoising strength. 0 = skip.

    Returns:
        Denoised image.
    """
    if strength <= 0:
        return image

    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, h=strength)
    else:
        return cv2.fastNlMeansDenoisingColored(image, None, h=strength, hColor=strength)


def apply_canny_edge(
    gray_image: np.ndarray,
    low: int = 50,
    high: int = 150,
    aperture: int = 3,
    l2_gradient: bool = True,
) -> np.ndarray:
    """Apply Canny edge detection.

    Args:
        gray_image: Grayscale input.
        low: Lower threshold.
        high: Upper threshold.
        aperture: Sobel aperture size.
        l2_gradient: Use L2 gradient norm for better accuracy.

    Returns:
        Binary edge image.
    """
    return cv2.Canny(gray_image, low, high, apertureSize=aperture, L2gradient=l2_gradient)


def preprocess_pipeline(
    image: np.ndarray,
    config: PreprocessConfig,
    output_binary: bool = True,
    output_edges: bool = False,
) -> dict:
    """Run the full preprocessing pipeline.

    Args:
        image: Input BGR image.
        config: Preprocessing configuration.
        output_binary: Include binary threshold result.
        output_edges: Include Canny edge result.

    Returns:
        Dictionary with keys: 'gray', 'binary' (optional), 'edges' (optional).
    """
    results: dict = {}

    # Step 1: Grayscale
    gray = to_grayscale(image)

    # Step 2: Denoising
    gray = apply_denoising(gray, config.denoise_strength)

    # Step 3: Blur (Gaussian)
    gray = apply_gaussian_blur(gray, config.gaussian_blur_kernel)

    # Step 4: Blur (Median) - optional
    gray = apply_median_blur(gray, config.median_blur_kernel)

    results["gray"] = gray

    # Step 5: Threshold
    if output_binary:
        binary = apply_adaptive_threshold(
            gray,
            config.adaptive_threshold_block,
            config.adaptive_threshold_c,
        )

        # Morphological cleanup
        binary = apply_morphological_closing(
            binary,
            config.morph_kernel_size,
            config.morph_close_iterations,
        )
        binary = apply_morphological_opening(
            binary,
            config.morph_kernel_size,
            config.morph_open_iterations,
        )

        results["binary"] = binary

    # Step 6: Canny Edge Detection
    if output_edges:
        edges = apply_canny_edge(
            gray,
            config.canny_low,
            config.canny_high,
            config.canny_aperture,
            config.canny_l2_gradient,
        )
        results["edges"] = edges

    logger.debug("Preprocessing complete: grayscale + binary + edges")
    return results


def preprocess_simple(image: np.ndarray) -> np.ndarray:
    """Quick preprocessing: grayscale, blur, Canny edges.

    Args:
        image: BGR input.

    Returns:
        Canny edge image.
    """
    gray = to_grayscale(image)
    gray = apply_gaussian_blur(gray, 3)
    return apply_canny_edge(gray, 50, 150)
