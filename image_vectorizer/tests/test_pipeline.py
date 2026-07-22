"""
Tests for the ImageVectorizer pipeline.

Run with:
    pytest image_vectorizer/tests/ -v
    python -m pytest image_vectorizer/tests/ -v
"""

from __future__ import annotations

import json
import sys
import math
from pathlib import Path

import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from image_vectorizer.models import (
    Point, Size, Color, BoundingBox, PrimitiveType,
    RectanglePrimitive, CirclePrimitive, EllipsePrimitive,
    TrianglePrimitive, PolygonPrimitive, LinePrimitive, PolylinePrimitive,
    VectorizationResult, ImageInfo,
)
from image_vectorizer.config import PipelineConfig, create_fast_config, create_high_quality_config
from image_vectorizer.utils import (
    distance_2d, normalize_angle_degrees, clamp, safe_divide,
    is_supported_image,
)
from image_vectorizer.geometry import (
    simplify_contour, order_vertices_clockwise,
    compute_polygon_angles, merge_nearby_lines,
    fit_line_to_points, fit_ellipse_to_contour,
)
from image_vectorizer.color_analyzer import extract_fill_color, quantize_color
from image_vectorizer.contour_detector import (
    compute_circularity, compute_rectangularity,
    compute_contour_area, compute_contour_perimeter,
)
from image_vectorizer.preprocess import (
    to_grayscale, apply_gaussian_blur, apply_canny_edge,
    apply_adaptive_threshold,
)
from image_vectorizer.primitive_detector import (
    detect_rectangle, detect_circle, detect_triangle,
    detect_polygon, detect_lines_hough,
)
from image_vectorizer.classifier import classify_contours, get_classification_stats
from image_vectorizer.json_exporter import export_to_json_string, export_objects_list
from image_vectorizer.visualization import generate_debug_image


# ════════════════════════════════════════════════════════════════════
# Helper: create synthetic test images
# ════════════════════════════════════════════════════════════════════

def _create_test_image(width: int = 400, height: int = 300) -> np.ndarray:
    """Create a white canvas for drawing synthetic shapes."""
    return np.ones((height, width, 3), dtype=np.uint8) * 255


# ════════════════════════════════════════════════════════════════════
# Model Tests
# ════════════════════════════════════════════════════════════════════

class TestModels:
    """Test data models (Point, Color, BoundingBox, primitives)."""

    def test_point_operations(self):
        p1 = Point(3, 4)
        p2 = Point(6, 8)
        assert p1.distance_to(p2) == 5.0
        assert p1 + p2 == Point(9, 12)
        assert p2 - p1 == Point(3, 4)
        assert p1 * 2 == Point(6, 8)

    def test_color_hex_conversion(self):
        c = Color(255, 128, 0)
        assert c.to_hex() == "#FF8000"
        c2 = Color(0, 0, 0, 128)
        assert c2.to_hex() == "#00000080"

    def test_color_from_bgr(self):
        c = Color.from_bgr((255, 0, 0))  # BGR pure blue → RGB pure red
        assert c.r == 0 and c.g == 0 and c.b == 255

    def test_color_hsv(self):
        c = Color(255, 0, 0)  # Pure red
        h, s, v = c.to_hsv()
        assert abs(h - 0.0) < 1 or abs(h - 360.0) < 1  # Red hue ≈ 0/360

    def test_bounding_box(self):
        bbox = BoundingBox(10, 20, 100, 50)
        assert bbox.area == 5000
        assert bbox.center == Point(60, 45)
        assert bbox.top_left == Point(10, 20)

    def test_rectangle_primitive(self):
        rect = RectanglePrimitive(id=1, x=10, y=20, width=100, height=50, confidence=0.95)
        assert rect.primitive_type == PrimitiveType.RECTANGLE
        assert rect.area == 5000
        assert rect.aspect_ratio == 2.0

    def test_circle_primitive(self):
        circle = CirclePrimitive(id=2, center_x=50, center_y=50, radius=10, confidence=0.99)
        assert circle.primitive_type == PrimitiveType.CIRCLE
        assert abs(circle.area - math.pi * 100) < 1

    def test_triangle_primitive(self):
        tri = TrianglePrimitive(
            id=3,
            vertex_a=Point(0, 0),
            vertex_b=Point(10, 0),
            vertex_c=Point(0, 10),
        )
        assert abs(tri.area - 50.0) < 1
        assert len(tri.vertices) == 3

    def test_line_primitive(self):
        line = LinePrimitive(id=4, start_x=0, start_y=0, end_x=3, end_y=4)
        assert line.length == 5.0

    def test_vectorization_result_serialization(self):
        info = ImageInfo(1920, 1080, 3, False, "test.png")
        rect = RectanglePrimitive(
            id=1, x=210, y=140, width=330, height=90,
            fill_color=Color(255, 170, 34),
            stroke_color=Color(0, 0, 0),
            confidence=0.99,
        )
        result = VectorizationResult(
            image_info=info,
            objects=[rect],
            total_objects=1,
            processing_time_ms=150.0,
        )
        data = result.to_dict()
        assert data["image_width"] == 1920
        assert data["image_height"] == 1080
        assert len(data["objects"]) == 1
        assert data["objects"][0]["fill"] == "#FFAA22"


# ════════════════════════════════════════════════════════════════════
# Utility Tests
# ════════════════════════════════════════════════════════════════════

class TestUtils:
    """Test utility functions."""

    def test_distance_2d(self):
        assert distance_2d((0, 0), (3, 4)) == 5.0

    def test_normalize_angle(self):
        assert normalize_angle_degrees(370) == 10.0
        assert normalize_angle_degrees(-10) == 350.0

    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_safe_divide(self):
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0, default=0) == 0.0

    def test_is_supported_image(self):
        assert is_supported_image("test.png")
        assert is_supported_image("test.jpg")
        assert is_supported_image("test.JPEG")
        assert not is_supported_image("test.gif")
        assert not is_supported_image("test.pdf")


# ════════════════════════════════════════════════════════════════════
# Geometry Tests
# ════════════════════════════════════════════════════════════════════

class TestGeometry:
    """Test geometric utility functions."""

    def test_vertex_ordering(self):
        # Create an unordered set of square vertices
        pts = [Point(10, 0), Point(0, 0), Point(10, 10), Point(0, 10)]
        ordered = order_vertices_clockwise(pts)
        assert len(ordered) == 4
        # Check all points are present
        all_pts = {(p.x, p.y) for p in ordered}
        assert all_pts == {(0, 0), (10, 0), (10, 10), (0, 10)}

    def test_polygon_angles(self):
        # Square should have 90° angles
        square = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        angles = compute_polygon_angles(square)
        assert len(angles) == 4
        for angle in angles:
            assert abs(angle - 90.0) < 1.0

    def test_merge_lines(self):
        # Two collinear line segments
        lines = np.array([
            [[0, 0, 10, 0]],
            [[8, 0, 20, 0]],
        ], dtype=np.int32)
        merged = merge_nearby_lines(lines, angle_threshold_deg=5, distance_threshold=5)
        assert len(merged) == 1

    def test_fit_line_to_points(self):
        pts = [Point(0, 0), Point(1, 2), Point(2, 4), Point(3, 6)]
        start, end, r2 = fit_line_to_points(pts)
        assert r2 > 0.99  # Perfect linear fit

    def test_simplify_contour(self):
        # Create a rectangular contour
        rect = np.array([[[0, 0]], [[100, 0]], [[100, 50]], [[0, 50]]], dtype=np.int32)
        simplified = simplify_contour(rect, epsilon_factor=0.02)
        assert len(simplified) == 4  # Should keep all 4 corners

    def test_ellipse_fit(self):
        # Generate points on an ellipse
        angles = np.linspace(0, 2 * np.pi, 50)
        pts = np.zeros((50, 1, 2), dtype=np.int32)
        for i, a in enumerate(angles):
            pts[i, 0, 0] = int(200 + 80 * np.cos(a))
            pts[i, 0, 1] = int(150 + 40 * np.sin(a))
        result = fit_ellipse_to_contour(pts)
        assert result is not None
        center, axes, angle = result
        assert abs(center[0] - 200) < 10
        assert abs(center[1] - 150) < 10


# ════════════════════════════════════════════════════════════════════
# Contour Detection Tests
# ════════════════════════════════════════════════════════════════════

class TestContourDetection:
    """Test contour detection and property computation."""

    def test_circularity_perfect_circle(self):
        # Perfect synthetic circle contour
        radius = 50
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        contour = np.zeros((100, 1, 2), dtype=np.float32)
        for i, a in enumerate(angles):
            contour[i, 0, 0] = 200 + radius * np.cos(a)
            contour[i, 0, 1] = 150 + radius * np.sin(a)
        contour = contour.astype(np.int32)
        c = compute_circularity(contour)
        assert c > 0.95  # Very close to 1.0

    def test_rectangularity(self):
        # Perfect rectangle contour
        rect = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]], dtype=np.int32)
        r = compute_rectangularity(rect)
        assert r > 0.97  # Integer contour approximation

    def test_contour_area_perimeter(self):
        # Simple square
        square = np.array([[[10, 10]], [[60, 10]], [[60, 60]], [[10, 60]]], dtype=np.int32)
        area = compute_contour_area(square)
        perimeter = compute_contour_perimeter(square, True)
        assert abs(area - 2500) < 100  # 50*50 = 2500
        assert abs(perimeter - 200) < 20  # 50*4 = 200


# ════════════════════════════════════════════════════════════════════
# Preprocessing Tests
# ════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    """Test image preprocessing functions."""

    def test_grayscale(self):
        bgr = _create_test_image(100, 100)
        gray = to_grayscale(bgr)
        assert gray.ndim == 2
        assert gray.shape == (100, 100)
        assert gray.dtype == np.uint8

    def test_gaussian_blur(self):
        img = _create_test_image(100, 100)
        blurred = apply_gaussian_blur(img, 5)
        assert blurred.shape == img.shape

    def test_canny_edges(self):
        # Create an image with a sharp edge
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        edges = apply_canny_edge(img, 50, 150)
        assert edges.shape == (100, 100)
        # Should detect the vertical edge
        edge_pixels = np.count_nonzero(edges)
        assert edge_pixels > 10

    def test_adaptive_threshold(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        img[25:75, 25:75] = 200
        binary = apply_adaptive_threshold(img)
        assert binary.shape == (100, 100)


# ════════════════════════════════════════════════════════════════════
# Color Analysis Tests
# ════════════════════════════════════════════════════════════════════

class TestColorAnalysis:
    """Test color extraction functions."""

    def test_fill_color_extraction(self):
        # Create a simple red rectangle on white background
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[25:75, 25:75] = (0, 0, 255)  # BGR red
        bbox = BoundingBox(25, 25, 50, 50)
        from image_vectorizer.config import ColorConfig
        color = extract_fill_color(img, bbox, ColorConfig())
        # Should be close to red
        assert color.r > 200
        assert color.g < 50
        assert color.b < 50

    def test_color_quantization(self):
        c = Color(241, 128, 67)
        q = quantize_color(c, levels=16)
        # Check values are quantized to multiples of 16
        assert q.r % 16 == 0
        assert q.g % 16 == 0
        assert q.b % 16 == 0


# ════════════════════════════════════════════════════════════════════
# Primitive Detection Tests
# ════════════════════════════════════════════════════════════════════

class TestPrimitiveDetection:
    """Test primitive shape detection on synthetic images."""

    def test_rectangle_detection(self):
        from image_vectorizer.config import RectangleConfig
        from image_vectorizer.contour_detector import compute_bounding_box, compute_centroid

        rect_contour = np.array(
            [[[50, 30]], [[250, 30]], [[250, 130]], [[50, 130]]],
            dtype=np.int32,
        )
        area = cv2.contourArea(rect_contour)
        perimeter = cv2.arcLength(rect_contour, True)
        bbox = compute_bounding_box(rect_contour)
        centroid = compute_centroid(rect_contour)

        contour_data = {
            "index": 0,
            "contour": rect_contour,
            "area": area,
            "perimeter": perimeter,
            "bbox": bbox,
            "centroid": centroid,
            "circularity": 0.7,
            "rectangularity": 0.99,
            "convexity": 1.0,
            "rotated_center": (150.0, 80.0),
            "rotated_size": (200.0, 100.0),
            "rotated_angle": 0.0,
        }

        config = RectangleConfig(min_area=100.0, min_confidence=0.5, epsilon_factor=0.02)
        rect = detect_rectangle(contour_data, config)
        assert rect is not None
        assert rect.primitive_type == PrimitiveType.RECTANGLE
        assert abs(rect.width - 200) < 10
        assert abs(rect.height - 100) < 10

    def test_circle_detection(self):
        from image_vectorizer.config import CircleConfig
        from image_vectorizer.contour_detector import compute_bounding_box, compute_centroid

        # Create circle contour
        r = 50
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        circle_contour = np.zeros((64, 1, 2), dtype=np.int32)
        for i, a in enumerate(angles):
            circle_contour[i, 0, 0] = int(150 + r * np.cos(a))
            circle_contour[i, 0, 1] = int(150 + r * np.sin(a))

        area = cv2.contourArea(circle_contour)
        perimeter = cv2.arcLength(circle_contour, True)
        bbox = compute_bounding_box(circle_contour)
        centroid = compute_centroid(circle_contour)
        circularity = compute_circularity(circle_contour)

        contour_data = {
            "index": 0,
            "contour": circle_contour,
            "area": area,
            "perimeter": perimeter,
            "bbox": bbox,
            "centroid": centroid,
            "circularity": circularity,
            "rectangularity": 0.6,
            "convexity": 1.0,
            "rotated_center": (150.0, 150.0),
            "rotated_size": (100.0, 100.0),
            "rotated_angle": 0.0,
        }

        config = CircleConfig(circularity_threshold=0.8, min_confidence=0.5)
        circle = detect_circle(contour_data, config)
        assert circle is not None
        assert circle.primitive_type == PrimitiveType.CIRCLE
        assert abs(circle.radius - 50) < 5

    def test_triangle_detection(self):
        from image_vectorizer.config import PolygonConfig
        from image_vectorizer.contour_detector import compute_bounding_box, compute_centroid

        tri_contour = np.array(
            [[[50, 150]], [[150, 30]], [[250, 150]]],
            dtype=np.int32,
        )
        area = cv2.contourArea(tri_contour)
        perimeter = cv2.arcLength(tri_contour, True)
        bbox = compute_bounding_box(tri_contour)
        centroid = compute_centroid(tri_contour)

        contour_data = {
            "index": 0,
            "contour": tri_contour,
            "area": area,
            "perimeter": perimeter,
            "bbox": bbox,
            "centroid": centroid,
            "circularity": 0.5,
            "rectangularity": 0.5,
            "convexity": 1.0,
            "rotated_center": (150.0, 90.0),
            "rotated_size": (200.0, 120.0),
            "rotated_angle": 0.0,
        }

        config = PolygonConfig(min_area=100.0, min_confidence=0.5)
        tri = detect_triangle(contour_data, config)
        assert tri is not None
        assert tri.primitive_type == PrimitiveType.TRIANGLE

    def test_polygon_detection(self):
        from image_vectorizer.config import PolygonConfig
        from image_vectorizer.contour_detector import compute_bounding_box, compute_centroid

        # Pentagon
        pentagon = np.array(
            [[[100, 30]], [[200, 30]], [[230, 100]], [[150, 160]], [[70, 100]]],
            dtype=np.int32,
        )
        area = cv2.contourArea(pentagon)
        perimeter = cv2.arcLength(pentagon, True)
        bbox = compute_bounding_box(pentagon)
        centroid = compute_centroid(pentagon)

        contour_data = {
            "index": 0,
            "contour": pentagon,
            "area": area,
            "perimeter": perimeter,
            "bbox": bbox,
            "centroid": centroid,
            "circularity": 0.6,
            "rectangularity": 0.5,
            "convexity": 1.0,
            "rotated_center": (150.0, 80.0),
            "rotated_size": (160.0, 130.0),
            "rotated_angle": 0.0,
        }

        config = PolygonConfig(min_area=100.0, min_confidence=0.5)
        poly = detect_polygon(contour_data, config)
        assert poly is not None
        assert poly.primitive_type == PrimitiveType.POLYGON
        assert poly.num_sides == 5


# ════════════════════════════════════════════════════════════════════
# Configuration Tests
# ════════════════════════════════════════════════════════════════════

class TestConfig:
    """Test configuration presets."""

    def test_default_config(self):
        config = PipelineConfig()
        assert config.preprocess.gaussian_blur_kernel == 3
        assert config.contour.min_contour_area > 0

    def test_fast_config(self):
        config = create_fast_config()
        assert config.max_dimension == 1280
        assert config.contour.min_contour_area > 0

    def test_high_quality_config(self):
        config = create_high_quality_config()
        assert config.max_dimension == 2560
        assert config.visualization.enabled


# ════════════════════════════════════════════════════════════════════
# JSON Export Tests
# ════════════════════════════════════════════════════════════════════

class TestJsonExport:
    """Test JSON serialization."""

    def test_export_to_string(self):
        info = ImageInfo(800, 600, 3, False, "test.png")
        rect = RectanglePrimitive(id=1, x=100, y=100, width=200, height=100)
        result = VectorizationResult(image_info=info, objects=[rect], total_objects=1)
        json_str = export_to_json_string(result, pretty=True)
        data = json.loads(json_str)
        assert data["image_width"] == 800
        assert data["total_objects"] == 1
        assert data["objects"][0]["type"] == "rectangle"


# ════════════════════════════════════════════════════════════════════
# Classification Statistics Tests
# ════════════════════════════════════════════════════════════════════

class TestClassification:
    """Test classification statistics."""

    def test_stats(self):
        primitives = [
            RectanglePrimitive(id=1, confidence=0.9),
            CirclePrimitive(id=2, confidence=0.95),
            RectanglePrimitive(id=3, confidence=0.85),
            TrianglePrimitive(
                id=4,
                vertex_a=Point(0, 0),
                vertex_b=Point(10, 0),
                vertex_c=Point(0, 10),
            ),
        ]
        stats = get_classification_stats(primitives)
        assert stats["rectangle"] == 2
        assert stats["circle"] == 1
        assert stats["triangle"] == 1
