"""
Debug visualization module for the restoration pipeline.

Generates visual comparisons: edges, noise map, color palette,
before/after side-by-side.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from .utils import logger, ensure_output_dir


def create_side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    labels: tuple = ("Before", "After"),
) -> np.ndarray:
    """Create a side-by-side before/after comparison image.

    Args:
        before: Original image.
        after: Restored image.
        labels: Labels for the two sides.

    Returns:
        Combined BGR image with labels.
    """
    h_before, w_before = before.shape[:2]
    h_after, w_after = after.shape[:2]

    # Match heights
    max_h = max(h_before, h_after)

    # Add label bar
    label_height = 40
    total_w = w_before + w_after + 2  # 2px separator
    total_h = max_h + label_height

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 40  # Dark bg

    # Place images
    canvas[label_height:label_height + h_before, :w_before] = _ensure_bgr(before)
    canvas[label_height:label_height + h_after, w_before + 2:] = _ensure_bgr(after)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, labels[0], (10, 28), font, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, labels[1], (w_before + 12, 28), font, 0.7, (255, 255, 255), 2)

    return canvas


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel BGR."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def visualize_edges(
    image: np.ndarray,
    output_path: str,
    low: int = 50,
    high: int = 150,
) -> str:
    """Generate and save an edge detection visualization.

    Args:
        image: Input BGR image.
        output_path: Output file path.
        low: Canny low threshold.
        high: Canny high threshold.

    Returns:
        Output path.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, low, high)

    # Create colored edge overlay
    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    color_edges[:, :, 2] = edges  # Red edges in BGR

    cv2.imwrite(output_path, color_edges)
    logger.debug("Edge visualization saved: %s", output_path)
    return output_path


def visualize_noise_map(
    image: np.ndarray,
    output_path: str,
) -> str:
    """Generate a noise estimation map.

    Uses local variance to highlight noisy regions.

    Args:
        image: Input BGR image.
        output_path: Output file path.

    Returns:
        Output path.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Local variance via box filter
    kernel_size = 5
    mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
    local_var = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    # Normalize and colorize
    var_norm = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(var_norm, cv2.COLORMAP_HOT)

    cv2.imwrite(output_path, heatmap)
    logger.debug("Noise map saved: %s", output_path)
    return output_path


def visualize_color_palette(
    image: np.ndarray,
    output_path: str,
    max_colors: int = 32,
) -> str:
    """Generate a color palette visualization of dominant colors.

    Args:
        image: Input BGR image.
        output_path: Output file path.
        max_colors: Maximum colors to display.

    Returns:
        Output path.
    """
    if image.ndim != 3:
        # Grayscale: create a gradient strip
        palette = np.linspace(0, 255, 256).reshape(1, -1).astype(np.uint8)
        palette_img = cv2.cvtColor(palette, cv2.COLOR_GRAY2BGR)
        palette_img = cv2.resize(palette_img, (512, 60))
        cv2.imwrite(output_path, palette_img)
        return output_path

    # Reshape and find unique colors
    pixels = image.reshape(-1, 3)
    # Quantize to reduce unique colors
    quantized = pixels // 16 * 16
    unique, counts = np.unique(quantized, axis=0, return_counts=True)

    # Sort by frequency
    sorted_indices = np.argsort(counts)[::-1]
    top_colors = unique[sorted_indices[:max_colors]]
    top_counts = counts[sorted_indices[:max_colors]]

    # Create palette image
    swatch_w = 40
    swatch_h = 30
    cols = min(16, len(top_colors))
    rows = (len(top_colors) + cols - 1) // cols

    palette_img = np.ones((rows * swatch_h + rows * 2, cols * swatch_w, 3), dtype=np.uint8) * 30

    for i, (color, count) in enumerate(zip(top_colors, top_counts)):
        r_idx = i // cols
        c_idx = i % cols
        y1 = r_idx * (swatch_h + 2)
        y2 = y1 + swatch_h
        x1 = c_idx * swatch_w
        x2 = x1 + swatch_w

        bgr_color = tuple(int(c) for c in color)
        palette_img[y1:y2, x1:x2] = bgr_color

    cv2.imwrite(output_path, palette_img)
    logger.debug("Color palette saved: %s", output_path)
    return output_path


def generate_all_debug_visuals(
    before: np.ndarray,
    after: np.ndarray,
    output_dir: str,
    stem: str,
) -> Dict[str, str]:
    """Generate all debug visualization images.

    Args:
        before: Original image.
        after: Restored image.
        output_dir: Output directory.
        stem: Base filename stem.

    Returns:
        Dict of visualization name to file path.
    """
    ensure_output_dir(output_dir)
    paths: Dict[str, str] = {}

    # Side-by-side comparison
    comparison = create_side_by_side(before, after, ("Original", "Restored"))
    comparison_path = f"{output_dir}/{stem}_comparison.png"
    cv2.imwrite(comparison_path, comparison)
    paths["comparison"] = comparison_path

    # Edges visualization
    edges_path = f"{output_dir}/{stem}_debug_edges.png"
    visualize_edges(after, edges_path)
    paths["edges"] = edges_path

    # Noise map (on restored image)
    noise_path = f"{output_dir}/{stem}_debug_noise.png"
    visualize_noise_map(after, noise_path)
    paths["noise"] = noise_path

    # Color palette
    palette_path = f"{output_dir}/{stem}_debug_colors.png"
    visualize_color_palette(after, palette_path)
    paths["colors"] = palette_path

    logger.info("Generated %d debug visualizations in %s", len(paths), output_dir)
    return paths
