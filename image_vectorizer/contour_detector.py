"""
Contour detection module using OpenCV's findContours.

Detects, filters, and enriches contours with geometric properties
(area, perimeter, bounding box, centroid, rotated box, circularity).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from .config import ContourConfig
from .models import Point, BoundingBox
from .utils import logger


# Mapping of retrieval mode strings to OpenCV constants
_RETRIEVAL_MODES = {
    "external": cv2.RETR_EXTERNAL,
    "list": cv2.RETR_LIST,
    "ccomp": cv2.RETR_CCOMP,
    "tree": cv2.RETR_TREE,
}

_APPROXIMATION_METHODS = {
    "simple": cv2.CHAIN_APPROX_SIMPLE,
    "none": cv2.CHAIN_APPROX_NONE,
    "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
    "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS,
}


def find_contours(
    binary_image: np.ndarray,
    retrieval_mode: str = "tree",
    approximation: str = "simple",
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Find contours in a binary image.

    Args:
        binary_image: Binary (0/255) image.
        retrieval_mode: One of "external", "list", "ccomp", "tree".
        approximation: One of "simple", "none", "tc89_l1", "tc89_kcos".

    Returns:
        Tuple of (contours, hierarchy). hierarchy is None for RETR_EXTERNAL
        or RETR_LIST.
    """
    mode = _RETRIEVAL_MODES.get(retrieval_mode, cv2.RETR_TREE)
    method = _APPROXIMATION_METHODS.get(approximation, cv2.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = cv2.findContours(
        binary_image, mode, method
    )

    # hierarchy shape: (1, N, 4) for each contour: [next, prev, first_child, parent]
    logger.debug("Found %d raw contours (mode=%s)", len(contours), retrieval_mode)
    return list(contours), hierarchy


def compute_contour_area(contour: np.ndarray) -> float:
    """Compute the signed area of a contour using cv2.contourArea.

    Args:
        contour: OpenCV contour array.

    Returns:
        Absolute area in square pixels.
    """
    return abs(cv2.contourArea(contour))


def compute_contour_perimeter(contour: np.ndarray, closed: bool = True) -> float:
    """Compute the perimeter (arc length) of a contour.

    Args:
        contour: OpenCV contour array.
        closed: Whether the contour is closed.

    Returns:
        Perimeter in pixels.
    """
    return cv2.arcLength(contour, closed)


def compute_bounding_box(contour: np.ndarray) -> BoundingBox:
    """Compute the axis-aligned bounding box of a contour.

    Args:
        contour: OpenCV contour array.

    Returns:
        BoundingBox dataclass.
    """
    x, y, w, h = cv2.boundingRect(contour)
    return BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))


def compute_rotated_box(contour: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Compute the minimum-area rotated bounding rectangle.

    Args:
        contour: OpenCV contour array.

    Returns:
        Tuple of (center (x,y), size (w,h), angle_degrees).
    """
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    return center, size, angle


def compute_centroid(contour: np.ndarray) -> Point:
    """Compute the centroid of a contour using image moments.

    Args:
        contour: OpenCV contour array.

    Returns:
        Point representing the centroid.
    """
    moments = cv2.moments(contour)
    if moments["m00"] < 1e-9:
        # Fallback: use bounding box center
        bbox = compute_bounding_box(contour)
        return bbox.center
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return Point(x=float(cx), y=float(cy))


def compute_circularity(contour: np.ndarray) -> float:
    """Compute contour circularity: 4π·area / perimeter².

    1.0 = perfect circle. Values closer to 0 = less circular.

    Args:
        contour: OpenCV contour array.

    Returns:
        Circularity value in [0, 1].
    """
    area = compute_contour_area(contour)
    perimeter = compute_contour_perimeter(contour, True)
    if perimeter < 1e-9:
        return 0.0
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    return float(min(circularity, 1.0))


def compute_rectangularity(contour: np.ndarray) -> float:
    """Compute how rectangular a contour is: contour_area / bounding_box_area.

    1.0 = perfect rectangle (contour fills its bbox).

    Args:
        contour: OpenCV contour array.

    Returns:
        Rectangularity value in [0, 1].
    """
    area = compute_contour_area(contour)
    bbox = compute_bounding_box(contour)
    bbox_area = bbox.area
    if bbox_area < 1e-9:
        return 0.0
    return float(min(area / bbox_area, 1.0))


def compute_convexity(contour: np.ndarray) -> float:
    """Compute convexity: contour_area / convex_hull_area.

    1.0 = fully convex.

    Args:
        contour: OpenCV contour array.

    Returns:
        Convexity value in [0, 1].
    """
    area = compute_contour_area(contour)
    if area < 1e-9:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = compute_contour_area(hull)
    if hull_area < 1e-9:
        return 0.0
    return float(min(area / hull_area, 1.0))


def filter_contours(
    contours: List[np.ndarray],
    config: ContourConfig,
    image_area: float,
) -> List[Dict[str, Any]]:
    """Filter contours by area and perimeter, enrich with properties.

    Args:
        contours: List of OpenCV contour arrays.
        config: Contour detection configuration.
        image_area: Total image area in pixels (width × height).

    Returns:
        List of dictionaries with contour properties, sorted by area descending.
    """
    max_area = config.max_contour_area_ratio * image_area
    enriched: List[Dict[str, Any]] = []

    for i, contour in enumerate(contours):
        area = compute_contour_area(contour)
        perimeter = compute_contour_perimeter(contour, True)

        # Filter by area
        if area < config.min_contour_area:
            continue
        if max_area > 0 and area > max_area:
            continue

        # Filter by perimeter
        if perimeter < config.min_contour_perimeter:
            continue

        bbox = compute_bounding_box(contour)
        centroid = compute_centroid(contour)
        center_rb, size_rb, angle_rb = compute_rotated_box(contour)
        circularity = compute_circularity(contour)
        rectangularity = compute_rectangularity(contour)
        convexity_val = compute_convexity(contour)

        enriched.append({
            "index": i,
            "contour": contour,
            "area": area,
            "perimeter": perimeter,
            "bbox": bbox,
            "centroid": centroid,
            "rotated_center": center_rb,
            "rotated_size": size_rb,
            "rotated_angle": angle_rb,
            "circularity": circularity,
            "rectangularity": rectangularity,
            "convexity": convexity_val,
        })

    # Sort by area descending
    enriched.sort(key=lambda x: x["area"], reverse=True)

    logger.info(
        "Filtered contours: %d kept from %d total (min_area=%.0f)",
        len(enriched), len(contours), config.min_contour_area,
    )

    return enriched


def get_contour_hierarchy_info(
    hierarchy: Optional[np.ndarray],
    contour_index: int,
) -> Dict[str, int]:
    """Extract hierarchy information for a specific contour.

    Args:
        hierarchy: Hierarchy array from findContours (shape: 1,N,4).
        contour_index: Index of the contour in the hierarchy.

    Returns:
        Dict with keys: next, prev, first_child, parent.
    """
    if hierarchy is None or hierarchy.shape[1] <= contour_index:
        return {"next": -1, "prev": -1, "first_child": -1, "parent": -1}

    h = hierarchy[0, contour_index]
    return {
        "next": int(h[0]),
        "prev": int(h[1]),
        "first_child": int(h[2]),
        "parent": int(h[3]),
    }
