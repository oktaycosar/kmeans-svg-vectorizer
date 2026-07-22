"""
Region-based segmentation — the foundation of object-level reconstruction.

Instead of contours (which fragment objects), we segment large
connected color regions. Each region becomes a candidate logical object.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

from image_restoration.utils import logger


def segment_by_color(
    image: np.ndarray,
    tolerance: int = 8,
    min_region_area: int = 50,
) -> List[Dict[str, Any]]:
    """Segment image into connected color regions.

    Groups connected pixels of similar color into regions.
    Each region represents a potential logical object.

    Args:
        image: BGR image (should be pre-quantized).
        tolerance: Color similarity tolerance per channel.
        min_region_area: Minimum region area in pixels.

    Returns:
        List of region dicts: mask, bbox, area, color, centroid, contour.
    """
    h, w = image.shape[:2]
    regions: List[Dict[str, Any]] = []

    # Find unique colors (quantized)
    quantized = image // (tolerance * 2) * (tolerance * 2)
    color_flat = quantized.reshape(-1, 3)
    unique_colors = np.unique(color_flat, axis=0)

    logger.debug("Segmenting %d unique colors into regions...", len(unique_colors))

    for color in unique_colors:
        bgr_color = tuple(int(c) for c in color)

        # Create mask for this color
        diff = np.abs(image.astype(np.int32) - np.array(bgr_color, dtype=np.int32))
        mask = np.all(diff <= tolerance, axis=2).astype(np.uint8) * 255

        if np.count_nonzero(mask) < min_region_area:
            continue

        # Find connected components within this color mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_region_area:
                continue

            # Extract region mask
            region_mask = (labels == label_id).astype(np.uint8) * 255

            # Bounding box
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            bw = stats[label_id, cv2.CC_STAT_WIDTH]
            bh = stats[label_id, cv2.CC_STAT_HEIGHT]

            # Centroid
            cx = centroids[label_id][0]
            cy = centroids[label_id][1]

            # Contour
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            main_contour = max(contours, key=cv2.contourArea) if contours else None

            # Average color from original image (not quantized)
            region_pixels = image[region_mask > 0]
            avg_color = tuple(int(v) for v in np.mean(region_pixels, axis=0)) if len(region_pixels) > 0 else bgr_color

            regions.append({
                "mask": region_mask,
                "bbox": (x, y, bw, bh),
                "area": int(area),
                "color_bgr": avg_color,
                "centroid": (cx, cy),
                "contour": main_contour,
                "pixel_count": int(area),
            })

    # Sort by area descending
    regions.sort(key=lambda r: r["area"], reverse=True)
    logger.info("Segmented %d regions from %d colors", len(regions), len(unique_colors))
    return regions


def compute_region_geometry(region: Dict[str, Any]) -> Dict[str, Any]:
    """Compute detailed geometry for a region.

    Args:
        region: Region dict from segment_by_color.

    Returns:
        Region dict enriched with perimeter, circularity, rectangularity,
        convex_hull, solidity, extent.
    """
    mask = region["mask"]
    contour = region.get("contour")

    if contour is None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        region["contour"] = contour

    if contour is None:
        return region

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    # Rectangularity
    x, y, w, h = region["bbox"]
    bbox_area = w * h
    rectangularity = area / bbox_area if bbox_area > 0 else 0

    # Solidity
    solidity = area / hull_area if hull_area > 0 else 0

    # Extent (ratio of contour area to bounding rect area)
    extent = area / bbox_area if bbox_area > 0 else 0

    # Rotated rectangle
    rect = cv2.minAreaRect(contour)
    center_rb, size_rb, angle_rb = rect

    region.update({
        "perimeter": float(perimeter),
        "circularity": float(min(circularity, 1.0)),
        "rectangularity": float(min(rectangularity, 1.0)),
        "solidity": float(solidity),
        "extent": float(extent),
        "convex_hull_area": float(hull_area),
        "rotated_center": center_rb,
        "rotated_size": size_rb,
        "rotated_angle": float(angle_rb),
    })

    return region


def find_holes_in_region(
    region_mask: np.ndarray,
    min_hole_area: int = 20,
) -> List[Dict[str, Any]]:
    """Find holes (enclosed background areas) within a region.

    Args:
        region_mask: Binary mask of the region.
        min_hole_area: Minimum hole area.

    Returns:
        List of hole dicts with mask, area, contour.
    """
    # Invert mask to find holes
    inverted = cv2.bitwise_not(region_mask)

    # Remove the outer background (flood fill from edges)
    h, w = inverted.shape
    flood_filled = inverted.copy()
    mask_flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood_filled, mask_flood, (0, 0), 0)
    cv2.floodFill(flood_filled, mask_flood, (w - 1, 0), 0)
    cv2.floodFill(flood_filled, mask_flood, (0, h - 1), 0)
    cv2.floodFill(flood_filled, mask_flood, (w - 1, h - 1), 0)

    # Remaining white areas are holes
    holes_mask = flood_filled

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        holes_mask, connectivity=8
    )

    holes = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_hole_area:
            continue

        hole_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        holes.append({
            "mask": hole_mask,
            "area": int(area),
            "contour": contours[0] if contours else None,
        })

    return holes
