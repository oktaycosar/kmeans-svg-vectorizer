"""
Data models for the ImageVectorizer project.

All detected primitives are represented as immutable dataclasses
with type hints for type safety and JSON serialization support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Optional, Tuple, List, Any, Dict


class PrimitiveType(Enum):
    """Enumeration of detectable primitive types."""
    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    SQUARE = "square"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    TRIANGLE = "triangle"
    POLYGON = "polygon"
    LINE = "line"
    POLYLINE = "polyline"
    BEZIER_CANDIDATE = "bezier_candidate"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Point:
    """Immutable 2D point with floating-point coordinates."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def distance_to(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Point:
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Point:
        return Point(self.x / scalar, self.y / scalar)


@dataclass(frozen=True)
class Size:
    """Immutable 2D size."""
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return float("inf")
        return self.width / self.height


@dataclass(frozen=True)
class Color:
    """Immutable RGBA color representation."""
    r: int
    g: int
    b: int
    a: int = 255

    def to_hex(self) -> str:
        """Return '#RRGGBB' or '#RRGGBBAA' hex string."""
        if self.a == 255:
            return f"#{self.r:02X}{self.g:02X}{self.b:02X}"
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}{self.a:02X}"

    @classmethod
    def from_bgr(cls, bgr: Tuple[int, int, int], alpha: int = 255) -> Color:
        """Create Color from OpenCV BGR tuple."""
        return cls(r=int(bgr[2]), g=int(bgr[1]), b=int(bgr[0]), a=alpha)

    @classmethod
    def from_rgb(cls, rgb: Tuple[int, int, int], alpha: int = 255) -> Color:
        """Create Color from RGB tuple."""
        return cls(r=int(rgb[0]), g=int(rgb[1]), b=int(rgb[2]), a=alpha)

    def to_bgr_tuple(self) -> Tuple[int, int, int]:
        return (self.b, self.g, self.r)

    def to_rgb_tuple(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def to_hsv(self) -> Tuple[float, float, float]:
        """Convert to HSV (H: 0-360, S: 0-1, V: 0-1)."""
        import colorsys
        h, s, v = colorsys.rgb_to_hsv(self.r / 255.0, self.g / 255.0, self.b / 255.0)
        return (h * 360.0, s, v)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box."""
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def top_left(self) -> Point:
        return Point(self.x, self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(self.x + self.width, self.y + self.height)

    def to_opencv_rect(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.width), int(self.height))


class PPTShapeType(Enum):
    """PowerPoint shape types for semantic matching."""
    RECTANGLE = "Rectangle"
    ROUNDED_RECTANGLE = "RoundedRectangle"
    OVAL = "Oval"
    CIRCLE = "Circle"
    DIAMOND = "Diamond"
    TRIANGLE = "Triangle"
    RIGHT_TRIANGLE = "RightTriangle"
    PARALLELOGRAM = "Parallelogram"
    TRAPEZOID = "Trapezoid"
    HEXAGON = "Hexagon"
    OCTAGON = "Octagon"
    CHEVRON = "Chevron"
    PENTAGON = "Pentagon"
    STAR = "Star"
    ARROW = "Arrow"
    DOUBLE_ARROW = "DoubleArrow"
    BLOCK_ARROW = "BlockArrow"
    LINE = "Line"
    CONNECTOR = "Connector"
    FREEFORM = "Freeform"
    TEXTBOX = "Textbox"
    TABLE = "Table"
    GROUP = "Group"
    UNKNOWN = "Unknown"


@dataclass
class CandidateShape:
    """A candidate PPT shape match with a score."""
    name: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 4)}


@dataclass
class PrimitiveBase:
    """Base class for all detected primitives."""
    id: int
    primitive_type: PrimitiveType = PrimitiveType.UNKNOWN
    confidence: float = 1.0
    fill_color: Optional[Color] = None
    stroke_color: Optional[Color] = None
    stroke_width: float = 1.0
    bounding_box: Optional[BoundingBox] = None
    contour_area: float = 0.0
    contour_perimeter: float = 0.0
    centroid: Optional[Point] = None

    # ── Post-processing fields ──
    geometry_score: float = 0.0
    ppt_shape: str = ""
    ppt_confidence: float = 0.0
    candidate_shapes: List[CandidateShape] = field(default_factory=list)
    editable: bool = True
    z_order: int = 0
    group_id: int = 0
    layer: str = "foreground"

    # ── Quality metrics ──
    contour_quality: float = 0.0
    color_consistency: float = 0.0
    edge_continuity: float = 0.0
    shape_match_score: float = 0.0


@dataclass
class RectanglePrimitive(PrimitiveBase):
    """Axis-aligned or rotated rectangle."""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    rotation: float = 0.0  # degrees
    is_rounded: bool = False
    corner_radius: float = 0.0

    def __post_init__(self):
        self.primitive_type = (
            PrimitiveType.ROUNDED_RECTANGLE if self.is_rounded else PrimitiveType.RECTANGLE
        )

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return float("inf")
        return self.width / self.height

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class SquarePrimitive(PrimitiveBase):
    """Square (special case of rectangle with aspect ratio ~ 1)."""
    x: float = 0.0
    y: float = 0.0
    side_length: float = 0.0
    rotation: float = 0.0

    def __post_init__(self):
        self.primitive_type = PrimitiveType.SQUARE

    @property
    def area(self) -> float:
        return self.side_length ** 2


@dataclass
class CirclePrimitive(PrimitiveBase):
    """Circle primitive."""
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 0.0

    def __post_init__(self):
        self.primitive_type = PrimitiveType.CIRCLE

    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2

    @property
    def center(self) -> Point:
        return Point(self.center_x, self.center_y)


@dataclass
class EllipsePrimitive(PrimitiveBase):
    """Ellipse primitive."""
    center_x: float = 0.0
    center_y: float = 0.0
    semi_major: float = 0.0  # a
    semi_minor: float = 0.0  # b
    rotation: float = 0.0  # degrees

    def __post_init__(self):
        self.primitive_type = PrimitiveType.ELLIPSE

    @property
    def area(self) -> float:
        return math.pi * self.semi_major * self.semi_minor

    @property
    def center(self) -> Point:
        return Point(self.center_x, self.center_y)


@dataclass
class TrianglePrimitive(PrimitiveBase):
    """Triangle primitive with three ordered vertices."""
    vertex_a: Point = field(default_factory=lambda: Point(0, 0))
    vertex_b: Point = field(default_factory=lambda: Point(0, 0))
    vertex_c: Point = field(default_factory=lambda: Point(0, 0))

    def __post_init__(self):
        self.primitive_type = PrimitiveType.TRIANGLE

    @property
    def vertices(self) -> List[Point]:
        return [self.vertex_a, self.vertex_b, self.vertex_c]

    @property
    def area(self) -> float:
        """Shoelace formula for triangle area."""
        return abs(
            self.vertex_a.x * (self.vertex_b.y - self.vertex_c.y)
            + self.vertex_b.x * (self.vertex_c.y - self.vertex_a.y)
            + self.vertex_c.x * (self.vertex_a.y - self.vertex_b.y)
        ) / 2.0

    @property
    def edge_lengths(self) -> Tuple[float, float, float]:
        ab = self.vertex_a.distance_to(self.vertex_b)
        bc = self.vertex_b.distance_to(self.vertex_c)
        ca = self.vertex_c.distance_to(self.vertex_a)
        return (ab, bc, ca)

    @property
    def angles(self) -> Tuple[float, float, float]:
        """Return angles in degrees: (a_at_A, a_at_B, a_at_C)."""
        a, b, c = self.edge_lengths
        # Angle opposite side a is at vertex C
        angle_c = math.degrees(math.acos(max(-1, min(1, (a**2 + b**2 - c**2) / (2 * a * b))))) if a > 0 and b > 0 else 0
        angle_b = math.degrees(math.acos(max(-1, min(1, (a**2 + c**2 - b**2) / (2 * a * c))))) if a > 0 and c > 0 else 0
        angle_a = 180.0 - angle_b - angle_c
        return (angle_a, angle_b, angle_c)


@dataclass
class PolygonPrimitive(PrimitiveBase):
    """General polygon with ordered vertices."""
    vertices: List[Point] = field(default_factory=list)
    num_sides: int = 0

    def __post_init__(self):
        self.primitive_type = PrimitiveType.POLYGON
        self.num_sides = len(self.vertices)

    @property
    def area(self) -> float:
        """Shoelace formula."""
        verts = self.vertices
        n = len(verts)
        if n < 3:
            return 0.0
        total = 0.0
        for i in range(n):
            j = (i + 1) % n
            total += verts[i].x * verts[j].y
            total -= verts[j].x * verts[i].y
        return abs(total) / 2.0

    @property
    def perimeter(self) -> float:
        n = len(self.vertices)
        if n < 2:
            return 0.0
        total = 0.0
        for i in range(n):
            j = (i + 1) % n
            total += self.vertices[i].distance_to(self.vertices[j])
        return total


@dataclass
class LinePrimitive(PrimitiveBase):
    """Straight line segment."""
    start_x: float = 0.0
    start_y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0

    def __post_init__(self):
        self.primitive_type = PrimitiveType.LINE

    @property
    def start(self) -> Point:
        return Point(self.start_x, self.start_y)

    @property
    def end(self) -> Point:
        return Point(self.end_x, self.end_y)

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def angle_degrees(self) -> float:
        """Angle of the line in degrees (0-360)."""
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        return (math.degrees(math.atan2(dy, dx)) + 360) % 360

    @property
    def midpoint(self) -> Point:
        return Point((self.start_x + self.end_x) / 2, (self.start_y + self.end_y) / 2)


@dataclass
class PolylinePrimitive(PrimitiveBase):
    """Series of connected line segments."""
    points: List[Point] = field(default_factory=list)

    def __post_init__(self):
        self.primitive_type = PrimitiveType.POLYLINE

    @property
    def total_length(self) -> float:
        total = 0.0
        for i in range(len(self.points) - 1):
            total += self.points[i].distance_to(self.points[i + 1])
        return total

    @property
    def segment_count(self) -> int:
        return max(0, len(self.points) - 1)


@dataclass
class BezierCandidatePrimitive(PrimitiveBase):
    """Candidate for Bezier curve fitting."""
    points: List[Point] = field(default_factory=list)
    control_points: List[Point] = field(default_factory=list)

    def __post_init__(self):
        self.primitive_type = PrimitiveType.BEZIER_CANDIDATE


@dataclass
class ImageInfo:
    """Metadata about the loaded image."""
    width: int
    height: int
    channels: int
    has_alpha: bool
    file_path: str = ""


@dataclass
class VectorizationResult:
    """Complete result of the vectorization pipeline."""
    image_info: ImageInfo
    objects: List[Any] = field(default_factory=list)  # List of primitives
    total_objects: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary matching the required JSON output format."""
        objects_list = []
        for i, obj in enumerate(self.objects):
            obj_dict = {
                "id": obj.id,
                "type": obj.primitive_type.value,
                "confidence": round(obj.confidence, 4),
            }

            # Add type-specific fields
            if isinstance(obj, RectanglePrimitive):
                obj_dict.update({
                    "x": obj.x,
                    "y": obj.y,
                    "width": obj.width,
                    "height": obj.height,
                    "rotation": obj.rotation,
                })
                if obj.is_rounded:
                    obj_dict["corner_radius"] = obj.corner_radius
            elif isinstance(obj, SquarePrimitive):
                obj_dict.update({
                    "x": obj.x,
                    "y": obj.y,
                    "side_length": obj.side_length,
                    "rotation": obj.rotation,
                })
            elif isinstance(obj, CirclePrimitive):
                obj_dict.update({
                    "center_x": obj.center_x,
                    "center_y": obj.center_y,
                    "radius": obj.radius,
                })
            elif isinstance(obj, EllipsePrimitive):
                obj_dict.update({
                    "center_x": obj.center_x,
                    "center_y": obj.center_y,
                    "semi_major": obj.semi_major,
                    "semi_minor": obj.semi_minor,
                    "rotation": obj.rotation,
                })
            elif isinstance(obj, TrianglePrimitive):
                obj_dict.update({
                    "vertex_a": [obj.vertex_a.x, obj.vertex_a.y],
                    "vertex_b": [obj.vertex_b.x, obj.vertex_b.y],
                    "vertex_c": [obj.vertex_c.x, obj.vertex_c.y],
                })
            elif isinstance(obj, PolygonPrimitive):
                obj_dict.update({
                    "vertices": [[p.x, p.y] for p in obj.vertices],
                    "num_sides": obj.num_sides,
                })
            elif isinstance(obj, LinePrimitive):
                obj_dict.update({
                    "start_x": obj.start_x,
                    "start_y": obj.start_y,
                    "end_x": obj.end_x,
                    "end_y": obj.end_y,
                })
            elif isinstance(obj, PolylinePrimitive):
                obj_dict.update({
                    "points": [[p.x, p.y] for p in obj.points],
                })

            # Color information
            if obj.fill_color:
                obj_dict["fill"] = obj.fill_color.to_hex()
            if obj.stroke_color:
                obj_dict["stroke"] = obj.stroke_color.to_hex()
                obj_dict["stroke_width"] = obj.stroke_width

            # Bounding box
            if obj.bounding_box:
                obj_dict["bbox"] = {
                    "x": obj.bounding_box.x,
                    "y": obj.bounding_box.y,
                    "width": obj.bounding_box.width,
                    "height": obj.bounding_box.height,
                }

            # ── Post-processing fields ──
            obj_dict["geometry_score"] = round(obj.geometry_score, 4)
            obj_dict["best_match"] = obj.ppt_shape or obj.primitive_type.value
            obj_dict["ppt_shape"] = obj.ppt_shape
            obj_dict["ppt_confidence"] = round(obj.ppt_confidence, 4)
            obj_dict["editable"] = obj.editable
            obj_dict["z_order"] = obj.z_order
            obj_dict["group_id"] = obj.group_id
            obj_dict["layer"] = obj.layer

            # Candidate shapes
            if obj.candidate_shapes:
                obj_dict["candidate_shapes"] = [
                    cs.to_dict() for cs in obj.candidate_shapes
                ]

            # Quality breakdown
            obj_dict["quality"] = {
                "geometry_score": round(obj.geometry_score, 4),
                "contour_quality": round(obj.contour_quality, 4),
                "color_consistency": round(obj.color_consistency, 4),
                "edge_continuity": round(obj.edge_continuity, 4),
                "shape_match_score": round(obj.shape_match_score, 4),
            }

            objects_list.append(obj_dict)

        return {
            "image_width": self.image_info.width,
            "image_height": self.image_info.height,
            "objects": objects_list,
            "total_objects": len(objects_list),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }
