"""
Image type analyzer — detects the type of image to select
the optimal preprocessing strategy.

Categories: logo, ui, diagram, illustration, screenshot, photo.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any

from .utils import logger


def analyze_image_type(image: np.ndarray) -> Dict[str, Any]:
    """Analyze the image and classify its type.

    Uses multiple heuristics to determine if the image is a logo,
    UI screenshot, diagram, illustration, or photo.

    Args:
        image: BGR image.

    Returns:
        Dict with: type, confidence, color_count, edge_density,
                   flat_region_ratio, gradient_ratio, text_density.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = image.shape[:2]

    features = {}

    # ── 1. Color analysis ────────────────────────────────────────
    if image.ndim == 3:
        quantized = image // 16 * 16
        pixels = quantized.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))
        features["color_count"] = unique_colors
    else:
        features["color_count"] = 1

    # ── 2. Edge density ──────────────────────────────────────────
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (h * w)
    features["edge_density"] = edge_density

    # ── 3. Flat region ratio ────────────────────────────────────
    # Areas with very low local variance = flat (typical of logos/ui)
    local_std = _local_std(gray, 7)
    flat_ratio = np.count_nonzero(local_std < 5) / local_std.size
    features["flat_region_ratio"] = flat_ratio

    # ── 4. Gradient ratio ────────────────────────────────────────
    # Areas with high gradient = photos/illustrations
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    high_grad = np.count_nonzero(grad_mag > 30) / grad_mag.size
    features["gradient_ratio"] = high_grad

    # ── 5. Text-like structure density ───────────────────────────
    # Many small horizontal edges = text (UI, screenshots)
    grad_x_bin = np.abs(grad_x) > 20
    text_like = _detect_text_regions(grad_x_bin)
    features["text_density"] = text_like

    # ── 6. Classification ─────────────────────────────────────────
    img_type, confidence = _classify(features, h * w)

    result = {
        "type": img_type,
        "confidence": confidence,
        "features": features,
        "recommended_colors": _recommend_colors(img_type),
    }

    logger.info(
        "Image analysis: type=%s (conf=%.2f), colors=%d, edge=%.3f, flat=%.2f",
        img_type, confidence,
        features.get("color_count", 0),
        features.get("edge_density", 0),
        features.get("flat_region_ratio", 0),
    )

    return result


def _local_std(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Compute local standard deviation."""
    mean = cv2.blur(gray.astype(np.float32), (kernel, kernel))
    mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (kernel, kernel))
    var = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(var)


def _detect_text_regions(h_edges: np.ndarray) -> float:
    """Detect text-like horizontal edge clusters."""
    kernel = np.ones((3, 15), np.uint8)
    h_clusters = cv2.morphologyEx(h_edges.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return np.count_nonzero(h_clusters) / h_clusters.size


def _classify(features: dict, image_area: int) -> tuple:
    """Classify image type based on features."""
    colors = features.get("color_count", 100)
    edge_d = features.get("edge_density", 0)
    flat_r = features.get("flat_region_ratio", 0)
    grad_r = features.get("gradient_ratio", 0)
    text_d = features.get("text_density", 0)

    # Logo: few colors, large flat areas, moderate edges
    logo_score = 0.0
    if colors <= 32:
        logo_score += 0.35
    if colors <= 16:
        logo_score += 0.20
    if flat_r > 0.4:
        logo_score += 0.25
    if edge_d < 0.15:
        logo_score += 0.10
    if text_d < 0.05:
        logo_score += 0.10

    # UI: moderate colors, high text density, regular edges
    ui_score = 0.0
    if 8 <= colors <= 64:
        ui_score += 0.25
    if text_d > 0.02:
        ui_score += 0.30
    if 0.05 <= edge_d <= 0.20:
        ui_score += 0.25
    if flat_r > 0.3:
        ui_score += 0.20

    # Diagram: like logo but with specific structure
    diagram_score = 0.0
    if 2 <= colors <= 32:
        diagram_score += 0.30
    if text_d > 0.01:
        diagram_score += 0.15
    if 0.03 <= edge_d <= 0.12:
        diagram_score += 0.25
    # Diagrams have long-ish edges
    if grad_r > 0.05:
        diagram_score += 0.15
    if flat_r > 0.5:
        diagram_score += 0.15

    # Illustration: many colors, moderate gradients
    ill_score = 0.0
    if 32 <= colors <= 512:
        ill_score += 0.30
    if 0.05 <= grad_r <= 0.4:
        ill_score += 0.25
    if edge_d > 0.05:
        ill_score += 0.20
    if flat_r < 0.5:
        ill_score += 0.15
    if text_d < 0.02:
        ill_score += 0.10

    # Screenshot: like UI but more complex
    screenshot_score = ui_score * 0.8

    # Photo: everything else (default low priority)
    photo_score = 0.0
    if colors > 128:
        photo_score += 0.40
    if grad_r > 0.3:
        photo_score += 0.30
    if flat_r < 0.2:
        photo_score += 0.20
    if edge_d > 0.10:
        photo_score += 0.10

    scores = {
        "logo": logo_score,
        "ui": ui_score,
        "diagram": diagram_score,
        "illustration": ill_score,
        "screenshot": screenshot_score,
        "photo": photo_score,
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # If best score is too low, default based on color count
    if best_score < 0.25:
        if colors <= 16:
            best_type = "logo"
        elif colors <= 64:
            best_type = "ui"
        elif colors <= 256:
            best_type = "illustration"
        else:
            best_type = "photo"
        best_score = 0.15

    return best_type, round(best_score, 3)


def _recommend_colors(img_type: str) -> int:
    """Recommend optimal color count for quantization."""
    recommendations = {
        "logo": 8,
        "diagram": 8,
        "ui": 16,
        "screenshot": 24,
        "illustration": 48,
        "photo": 64,
    }
    return recommendations.get(img_type, 32)
