"""
Centralized configuration for the ImageVectorizer pipeline.

All tunable parameters are collected here with sensible defaults.
Modify these values to adjust sensitivity, performance, and output quality.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class PreprocessConfig:
    """Image preprocessing parameters."""
    # Gaussian blur kernel size (odd number, 0 = skip)
    gaussian_blur_kernel: int = 3
    # Median blur kernel size (odd number, 0 = skip)
    median_blur_kernel: int = 0
    # Adaptive threshold block size (odd number)
    adaptive_threshold_block: int = 11
    # Adaptive threshold constant
    adaptive_threshold_c: int = 2
    # Binary threshold value (0-255)
    binary_threshold_value: int = 127
    # Binary threshold max value
    binary_threshold_max: int = 255
    # Morphological kernel size for opening/closing
    morph_kernel_size: Tuple[int, int] = (3, 3)
    # Number of morphological opening iterations
    morph_open_iterations: int = 1
    # Number of morphological closing iterations
    morph_close_iterations: int = 1
    # Non-local means denoising strength (0 = skip)
    denoise_strength: float = 0.0
    # Canny edge detection: lower threshold
    canny_low: int = 50
    # Canny edge detection: upper threshold
    canny_high: int = 150
    # Canny aperture size
    canny_aperture: int = 3
    # Use L2 gradient norm for Canny
    canny_l2_gradient: bool = True


@dataclass
class ContourConfig:
    """Contour detection parameters."""
    # Retrieval mode: "external", "list", "ccomp", "tree"
    retrieval_mode: str = "tree"
    # Approximation method: "simple", "none", "tc89_l1", "tc89_kcos"
    approximation_method: str = "simple"
    # Minimum contour area (pixels) to keep
    min_contour_area: float = 50.0
    # Maximum contour area as fraction of image area (0 = no limit)
    max_contour_area_ratio: float = 0.95
    # Minimum contour perimeter to keep
    min_contour_perimeter: float = 10.0


@dataclass
class RectangleConfig:
    """Rectangle/square detection parameters."""
    # Maximum epsilon factor for approxPolyDP (relative to perimeter)
    epsilon_factor: float = 0.02
    # Aspect ratio tolerance for square detection (1.0 = perfect square)
    square_aspect_ratio_tolerance: float = 0.1
    # Minimum rectangle area
    min_area: float = 100.0
    # Minimum confidence to accept a detection
    min_confidence: float = 0.7
    # Maximum angular deviation for rounded corner detection (degrees)
    rounded_corner_max_angle: float = 15.0


@dataclass
class CircleConfig:
    """Circle detection parameters."""
    # HoughCircles: dp (inverse ratio of accumulator resolution)
    hough_dp: float = 1.0
    # HoughCircles: minimum distance between circle centers
    hough_min_dist: float = 20.0
    # HoughCircles: upper threshold for Canny edge detector
    hough_param1: float = 100.0
    # HoughCircles: accumulator threshold
    hough_param2: float = 30.0
    # HoughCircles: minimum radius
    hough_min_radius: int = 5
    # HoughCircles: maximum radius
    hough_max_radius: int = 500
    # Circularity threshold (4π·area/perimeter², 1.0 = perfect circle)
    circularity_threshold: float = 0.85
    # Minimum confidence
    min_confidence: float = 0.7


@dataclass
class EllipseConfig:
    """Ellipse detection parameters."""
    # Minimum contour points to fit ellipse
    min_contour_points: int = 5
    # Minimum confidence
    min_confidence: float = 0.7


@dataclass
class LineConfig:
    """Line detection parameters."""
    # HoughLinesP: rho (distance resolution in pixels)
    hough_rho: float = 1.0
    # HoughLinesP: theta (angular resolution in radians)
    hough_theta: float = 0.0174533  # π/180
    # HoughLinesP: accumulator threshold
    hough_threshold: int = 50
    # HoughLinesP: minimum line length
    hough_min_line_length: int = 30
    # HoughLinesP: maximum gap between line segments
    hough_max_line_gap: int = 10
    # Maximum angular difference to merge lines (degrees)
    merge_angle_threshold: float = 5.0
    # Maximum distance to merge lines (pixels)
    merge_distance_threshold: float = 10.0


@dataclass
class PolygonConfig:
    """Polygon/triangle detection parameters."""
    # Maximum epsilon factor for approxPolyDP
    epsilon_factor: float = 0.02
    # Minimum number of vertices
    min_vertices: int = 3
    # Maximum number of vertices
    max_vertices: int = 12
    # Minimum area
    min_area: float = 100.0
    # Minimum confidence
    min_confidence: float = 0.6


@dataclass
class ColorConfig:
    """Color analysis parameters."""
    # Border sampling width in pixels
    border_width: int = 2
    # Sample region inset for fill color (fraction of bbox, avoids border)
    fill_inset_fraction: float = 0.15
    # Number of color quantization levels (0 = no quantization)
    quantization_levels: int = 16


@dataclass
class VisualizationConfig:
    """Debug visualization parameters."""
    # Enable debug output
    enabled: bool = False
    # Output directory for debug images
    output_dir: str = "debug_output"
    # Draw contours
    draw_contours: bool = True
    # Draw bounding boxes
    draw_bbox: bool = True
    # Draw centers/centroids
    draw_centers: bool = True
    # Draw polygon vertices
    draw_vertices: bool = True
    # Draw detected circles
    draw_circles: bool = True
    # Draw detected lines
    draw_lines: bool = True
    # Draw labels
    draw_labels: bool = True
    # Contour line thickness
    contour_thickness: int = 2
    # Bounding box thickness
    bbox_thickness: int = 2
    # Font scale for labels
    font_scale: float = 0.5
    # Label offset from point
    label_offset: int = 5
    # Colors for different primitive types (BGR)
    color_map: dict = field(default_factory=lambda: {
        "rectangle": (0, 255, 0),       # Green
        "square": (0, 255, 255),         # Cyan
        "circle": (255, 0, 0),           # Blue
        "ellipse": (255, 128, 0),        # Orange
        "triangle": (0, 128, 255),       # Orange-ish
        "polygon": (255, 0, 255),        # Magenta
        "line": (0, 0, 255),             # Red
        "polyline": (128, 0, 255),       # Purple
        "default": (255, 255, 255),      # White
    })


@dataclass
class PipelineConfig:
    """Master configuration for the entire pipeline."""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    contour: ContourConfig = field(default_factory=ContourConfig)
    rectangle: RectangleConfig = field(default_factory=RectangleConfig)
    circle: CircleConfig = field(default_factory=CircleConfig)
    ellipse: EllipseConfig = field(default_factory=EllipseConfig)
    line: LineConfig = field(default_factory=LineConfig)
    polygon: PolygonConfig = field(default_factory=PolygonConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    # Enable the full pipeline or stop after specific stages
    enable_preprocessing: bool = True
    enable_contour_detection: bool = True
    enable_primitive_detection: bool = True
    enable_color_analysis: bool = True
    enable_classification: bool = True
    # Output format
    output_json: bool = True
    pretty_print_json: bool = True
    # Performance: resize large images before processing
    max_dimension: int = 1920
    # Logging level
    log_level: str = "INFO"


# Singleton default config
DEFAULT_CONFIG = PipelineConfig()


def create_fast_config() -> PipelineConfig:
    """Return a performance-optimized configuration for large images."""
    config = PipelineConfig()
    config.preprocess.gaussian_blur_kernel = 3
    config.preprocess.canny_low = 60
    config.preprocess.canny_high = 180
    config.contour.min_contour_area = 100.0
    config.max_dimension = 1280
    return config


def create_high_quality_config() -> PipelineConfig:
    """Return a high-quality configuration for detailed images."""
    config = PipelineConfig()
    config.preprocess.gaussian_blur_kernel = 1
    config.preprocess.canny_low = 30
    config.preprocess.canny_high = 90
    config.preprocess.canny_l2_gradient = True
    config.contour.min_contour_area = 20.0
    config.max_dimension = 2560
    config.visualization.enabled = True
    return config
