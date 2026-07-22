"""
Debug visualization module for the ImageVectorizer.

Generates annotated images showing:
- Detected contours
- Bounding boxes
- Centers/centroids
- Polygon vertices
- Detected circles
- Detected lines
- Labels with confidence scores
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .models import (
    PrimitiveBase, PrimitiveType,
    RectanglePrimitive, SquarePrimitive, CirclePrimitive,
    EllipsePrimitive, TrianglePrimitive, PolygonPrimitive,
    LinePrimitive, PolylinePrimitive,
    Point, Color,
)
from .config import VisualizationConfig
from .utils import logger, ensure_output_dir


def generate_debug_image(
    image_bgr: np.ndarray,
    primitives: List[PrimitiveBase],
    config: VisualizationConfig,
    output_path: str,
) -> str:
    """Generate a debug visualization image with all annotations.

    Args:
        image_bgr: Original BGR image.
        primitives: List of detected primitives.
        config: Visualization configuration.
        output_path: Output file path.

    Returns:
        Output file path.
    """
    if not config.enabled:
        return output_path

    # Create a copy for drawing
    debug_image = image_bgr.copy()

    color_map = config.color_map

    for primitive in primitives:
        type_name = primitive.primitive_type.value
        draw_color = color_map.get(type_name, color_map["default"])

        # Draw bounding box
        if config.draw_bbox and primitive.bounding_box:
            bbox = primitive.bounding_box
            x, y, w, h = bbox.to_opencv_rect()
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), draw_color, config.bbox_thickness)

        # Draw centroid
        if config.draw_centers and primitive.centroid:
            cx, cy = primitive.centroid.to_int_tuple()
            cv2.circle(debug_image, (cx, cy), 4, (0, 0, 255), -1)

        # Draw type-specific
        if isinstance(primitive, CirclePrimitive) and config.draw_circles:
            cx, cy = int(primitive.center_x), int(primitive.center_y)
            r = int(primitive.radius)
            cv2.circle(debug_image, (cx, cy), r, draw_color, config.contour_thickness)
            cv2.circle(debug_image, (cx, cy), 3, (0, 0, 255), -1)

        elif isinstance(primitive, EllipsePrimitive):
            cx, cy = int(primitive.center_x), int(primitive.center_y)
            cv2.ellipse(
                debug_image,
                (cx, cy),
                (int(primitive.semi_major), int(primitive.semi_minor)),
                primitive.rotation, 0, 360,
                draw_color, config.contour_thickness,
            )

        elif isinstance(primitive, (RectanglePrimitive, SquarePrimitive)):
            if config.draw_vertices:
                if isinstance(primitive, RectanglePrimitive):
                    rect = ((primitive.x + primitive.width / 2, primitive.y + primitive.height / 2),
                            (primitive.width, primitive.height), primitive.rotation)
                else:
                    rect = ((primitive.x + primitive.side_length / 2, primitive.y + primitive.side_length / 2),
                            (primitive.side_length, primitive.side_length), primitive.rotation)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(debug_image, [box], 0, draw_color, config.contour_thickness)

                # Draw corner points
                for pt in box:
                    cv2.circle(debug_image, tuple(pt), 3, (0, 255, 255), -1)

        elif isinstance(primitive, TrianglePrimitive) and config.draw_vertices:
            pts = np.array([v.to_int_tuple() for v in primitive.vertices], dtype=np.int32)
            cv2.polylines(debug_image, [pts], True, draw_color, config.contour_thickness)
            for pt in pts:
                cv2.circle(debug_image, tuple(pt), 3, (0, 255, 255), -1)

        elif isinstance(primitive, PolygonPrimitive) and config.draw_vertices:
            pts = np.array([v.to_int_tuple() for v in primitive.vertices], dtype=np.int32)
            cv2.polylines(debug_image, [pts], True, draw_color, config.contour_thickness)
            for pt in pts:
                cv2.circle(debug_image, tuple(pt), 2, (0, 255, 255), -1)

        elif isinstance(primitive, LinePrimitive) and config.draw_lines:
            pt1 = (int(primitive.start_x), int(primitive.start_y))
            pt2 = (int(primitive.end_x), int(primitive.end_y))
            cv2.line(debug_image, pt1, pt2, draw_color, config.contour_thickness)
            cv2.circle(debug_image, pt1, 3, (0, 255, 0), -1)
            cv2.circle(debug_image, pt2, 3, (0, 0, 255), -1)

        elif isinstance(primitive, PolylinePrimitive):
            pts = np.array([p.to_int_tuple() for p in primitive.points], dtype=np.int32)
            cv2.polylines(debug_image, [pts], False, draw_color, config.contour_thickness)

        # Draw label
        if config.draw_labels:
            label_pos = _get_label_position(primitive)
            label_text = f"{primitive.id}:{type_name[:4]}"
            if primitive.confidence > 0:
                label_text += f"({primitive.confidence:.2f})"

            cv2.putText(
                debug_image, label_text,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.font_scale,
                draw_color,
                1,
                cv2.LINE_AA,
            )

    # Add legend
    _draw_legend(debug_image, primitives, config)

    # Save
    ensure_output_dir(str(config.output_dir))
    cv2.imwrite(output_path, debug_image)
    logger.info("Debug image saved: %s", output_path)

    return output_path


def _get_label_position(primitive: PrimitiveBase) -> Tuple[int, int]:
    """Determine a good position for the label near the primitive.

    Args:
        primitive: Any primitive.

    Returns:
        (x, y) pixel position for the label.
    """
    if primitive.centroid:
        x = int(primitive.centroid.x)
        y = int(primitive.centroid.y) - 10
        return (max(0, x), max(10, y))

    if primitive.bounding_box:
        x = int(primitive.bounding_box.x)
        y = int(primitive.bounding_box.y) - 5
        return (max(0, x), max(10, y))

    if isinstance(primitive, CirclePrimitive):
        return (int(primitive.center_x), int(primitive.center_y) - int(primitive.radius) - 5)

    if isinstance(primitive, LinePrimitive):
        x = int(primitive.midpoint.x)
        y = int(primitive.midpoint.y) - 10
        return (max(0, x), max(10, y))

    return (10, 30)


def _draw_legend(
    image: np.ndarray,
    primitives: List[PrimitiveBase],
    config: VisualizationConfig,
) -> None:
    """Draw a color legend on the debug image.

    Args:
        image: Image to draw on (modified in-place).
        primitives: List of primitives for stats.
        config: Visualization configuration.
    """
    h, w = image.shape[:2]
    legend_x = w - 220
    legend_y = 30
    line_height = 18

    # Background
    overlay = image.copy()
    cv2.rectangle(overlay, (legend_x - 10, legend_y - 25),
                  (w - 10, legend_y + 10 * line_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Title
    cv2.putText(image, f"Objects: {len(primitives)}",
                (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Count by type
    type_counts: dict = {}
    for p in primitives:
        name = p.primitive_type.value
        type_counts[name] = type_counts.get(name, 0) + 1

    color_map = config.color_map
    for i, (type_name, count) in enumerate(sorted(type_counts.items())):
        y = legend_y + (i + 1) * line_height
        color = color_map.get(type_name, color_map["default"])
        cv2.putText(image, f"{type_name}: {count}",
                    (legend_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)


def draw_contours_debug(
    image_bgr: np.ndarray,
    contours: List[np.ndarray],
    output_path: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> str:
    """Draw detected contours on an image.

    Args:
        image_bgr: Original image.
        contours: List of OpenCV contours.
        output_path: Output path.
        color: BGR color for contours.
        thickness: Line thickness.

    Returns:
        Output file path.
    """
    debug = image_bgr.copy()
    cv2.drawContours(debug, contours, -1, color, thickness)
    cv2.imwrite(output_path, debug)
    return output_path
