"""
Color analysis module for the ImageVectorizer.

Extracts fill color and stroke (border) color for each detected primitive
by sampling the original image at the primitive's location.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np

from .models import Point, BoundingBox, Color, PrimitiveBase
from .config import ColorConfig
from .utils import logger


def extract_fill_color(
    image: np.ndarray,
    bbox: BoundingBox,
    config: ColorConfig,
    mask: Optional[np.ndarray] = None,
) -> Color:
    """Extract the average fill color from the interior of a bounding box.

    Uses an inset region to avoid border pixels.

    Args:
        image: Original BGR image.
        bbox: Bounding box of the primitive.
        config: Color analysis configuration.
        mask: Optional binary mask to restrict sampling.

    Returns:
        Average Color (RGB).
    """
    h_img, w_img = image.shape[:2]

    # Compute inset region
    inset_frac = config.fill_inset_fraction
    inset_x = max(0, int(bbox.x + bbox.width * inset_frac))
    inset_y = max(0, int(bbox.y + bbox.height * inset_frac))
    inset_w = max(1, int(bbox.width * (1 - 2 * inset_frac)))
    inset_h = max(1, int(bbox.height * (1 - 2 * inset_frac)))

    # Clamp to image bounds
    inset_x = min(inset_x, w_img - 1)
    inset_y = min(inset_y, h_img - 1)
    inset_w = min(inset_w, w_img - inset_x)
    inset_h = min(inset_h, h_img - inset_y)

    if inset_w <= 0 or inset_h <= 0:
        # Fallback: use bbox center pixel
        cx = int(bbox.center.x)
        cy = int(bbox.center.y)
        cx = max(0, min(cx, w_img - 1))
        cy = max(0, min(cy, h_img - 1))
        pixel = image[cy, cx]
        return Color.from_bgr(tuple(pixel[:3].tolist()))

    region = image[inset_y:inset_y + inset_h, inset_x:inset_x + inset_w]

    if region.size == 0:
        return Color(128, 128, 128)

    # Apply mask if provided
    if mask is not None:
        mask_region = mask[inset_y:inset_y + inset_h, inset_x:inset_x + inset_w]
        if mask_region.shape[:2] == region.shape[:2]:
            region = region[mask_region > 0]

    if region.size == 0:
        return Color(128, 128, 128)

    # Handle 3 or 4 channel images
    pixels = region.reshape(-1, region.shape[-1])
    if pixels.shape[1] >= 3:
        bgr_pixels = pixels[:, :3]  # Take BGR, ignore alpha
    else:
        bgr_pixels = pixels  # Grayscale, replicate across channels
        if bgr_pixels.shape[1] == 1:
            bgr_pixels = np.repeat(bgr_pixels, 3, axis=1)

    avg_bgr = np.mean(bgr_pixels, axis=0)
    return Color.from_bgr(tuple(int(v) for v in avg_bgr))


def extract_stroke_color(
    image: np.ndarray,
    contour: np.ndarray,
    config: ColorConfig,
) -> Tuple[Color, float]:
    """Extract the stroke (border) color and estimated width.

    Samples pixels along the contour.

    Args:
        image: Original BGR image.
        contour: OpenCV contour array.
        config: Color analysis configuration.

    Returns:
        Tuple of (stroke_color, estimated_stroke_width).
    """
    h_img, w_img = image.shape[:2]

    # Sample pixels along the contour
    num_samples = min(200, len(contour))
    step = max(1, len(contour) // num_samples)

    colors = []
    stroke_widths = []

    for i in range(0, len(contour), step):
        pt = contour[i][0]
        x, y = int(pt[0]), int(pt[1])

        # Clamp
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))

        # Sample perpendicular direction to estimate stroke width
        pixel = image[y, x]
        if pixel.ndim >= 3:
            colors.append(pixel[:3])

    if not colors:
        return Color(0, 0, 0), 1.0

    avg_bgr = np.mean(colors, axis=0)
    stroke_color = Color.from_bgr(tuple(int(v) for v in avg_bgr))

    # Estimate stroke width from contour area vs perimeter
    # stroke_width ≈ area / perimeter (approximate for thin shapes)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0 and area > 0:
        # For filled shapes, stroke width is typically small relative to area
        # Use a heuristic: look at border gradient
        stroke_width = float(area / (perimeter + 1e-6))
    else:
        stroke_width = 1.0

    return stroke_color, max(stroke_width, 1.0)


def compute_average_hsv(image: np.ndarray, bbox: BoundingBox) -> Tuple[float, float, float]:
    """Compute the average HSV color of a region.

    Args:
        image: BGR image.
        bbox: Bounding box defining the region.

    Returns:
        Tuple of (H, S, V) where H in [0, 360], S and V in [0, 1].
    """
    h_img, w_img = image.shape[:2]

    x = max(0, int(bbox.x))
    y = max(0, int(bbox.y))
    w_box = min(int(bbox.width), w_img - x)
    h_box = min(int(bbox.height), h_img - y)

    if w_box <= 0 or h_box <= 0:
        return (0.0, 0.0, 0.0)

    region = image[y:y + h_box, x:x + w_box]
    if region.size == 0:
        return (0.0, 0.0, 0.0)

    hsv = cv2.cvtColor(region.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)
    avg = np.mean(hsv, axis=0)
    # OpenCV H is [0, 179], S and V are [0, 255]
    return (float(avg[0]) * 2, float(avg[1]) / 255.0, float(avg[2]) / 255.0)


def analyze_primitive_colors(
    image: np.ndarray,
    primitive: PrimitiveBase,
    contour: Optional[np.ndarray] = None,
    config: Optional[ColorConfig] = None,
) -> None:
    """Analyze and assign fill/stroke colors to a primitive in-place.

    Args:
        image: Original BGR image.
        primitive: Primitive to colorize (modified in-place).
        contour: Optional contour (if not available via primitive).
        config: Color configuration.
    """
    if config is None:
        config = ColorConfig()

    bbox = primitive.bounding_box
    if bbox is None:
        return

    # Extract fill color
    primitive.fill_color = extract_fill_color(image, bbox, config)

    # Extract stroke color if contour available
    if contour is not None:
        stroke_color, stroke_width = extract_stroke_color(image, contour, config)
        primitive.stroke_color = stroke_color
        primitive.stroke_width = stroke_width


def quantize_color(color: Color, levels: int = 16) -> Color:
    """Quantize a color to fewer levels for cleaner output.

    Args:
        color: Input Color.
        levels: Number of quantization levels per channel.

    Returns:
        Quantized Color.
    """
    if levels <= 0:
        return color

    def quantize_channel(value: int) -> int:
        step = 256 / levels
        return int(round(round(value / step) * step))

    return Color(
        r=min(quantize_channel(color.r), 255),
        g=min(quantize_channel(color.g), 255),
        b=min(quantize_channel(color.b), 255),
        a=color.a,
    )
