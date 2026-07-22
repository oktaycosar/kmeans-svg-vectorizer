"""
ImageVectorizer — Convert PNG/JPG images to editable vector primitives.

A production-quality computer vision pipeline that detects geometric
primitives (rectangles, circles, ellipses, polygons, lines) from raster
images using deterministic, non-AI methods.

Usage:
    from image_vectorizer.main import run_pipeline
    result = run_pipeline("image.png")
    print(f"Found {result.total_objects} objects")
"""

__version__ = "1.0.0"
__author__ = "ImageVectorizer Team"

from .models import (
    PrimitiveType, Point, Size, Color, BoundingBox,
    RectanglePrimitive, SquarePrimitive, CirclePrimitive,
    EllipsePrimitive, TrianglePrimitive, PolygonPrimitive,
    LinePrimitive, PolylinePrimitive, BezierCandidatePrimitive,
    ImageInfo, VectorizationResult,
)
from .config import PipelineConfig, create_fast_config, create_high_quality_config

__all__ = [
    "__version__",
    # Models
    "PrimitiveType", "Point", "Size", "Color", "BoundingBox",
    "RectanglePrimitive", "SquarePrimitive", "CirclePrimitive",
    "EllipsePrimitive", "TrianglePrimitive", "PolygonPrimitive",
    "LinePrimitive", "PolylinePrimitive", "BezierCandidatePrimitive",
    "ImageInfo", "VectorizationResult",
    # Config
    "PipelineConfig", "create_fast_config", "create_high_quality_config",
]
