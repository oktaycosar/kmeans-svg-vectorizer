"""
Object graph — hierarchical representation of logo objects.

Each node is a logical object with parent, children, neighbors.
The graph captures the spatial and hierarchical relationships
between all detected regions/objects in the logo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class ObjectNode:
    """A single logical object in the logo."""
    id: int
    mask: np.ndarray = field(repr=False, default=None)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    area: int = 0
    color_bgr: Tuple[int, int, int] = (0, 0, 0)
    color_hex: str = "#000000"

    # Hierarchy
    parent: int = -1
    children: List[int] = field(default_factory=list)
    neighbors: List[int] = field(default_factory=list)
    holes: List[int] = field(default_factory=list)

    # Geometry
    centroid: Tuple[float, float] = (0.0, 0.0)
    circularity: float = 0.0
    rectangularity: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 1.0

    # Classification
    primitive_type: str = "unknown"
    primitive_confidence: float = 0.0
    semantic_type: str = "region"

    # Editable properties
    editable: bool = True
    z_order: int = 0
    layer: str = "foreground"


class ObjectGraph:
    """Graph of all logical objects in a logo."""

    def __init__(self):
        self.nodes: Dict[int, ObjectNode] = {}
        self._next_id = 1

    def add_node(self, region: Dict[str, Any]) -> ObjectNode:
        """Add a region as a new node in the graph.

        Args:
            region: Region dict from segmentation/merging.

        Returns:
            Created ObjectNode.
        """
        node_id = self._next_id
        self._next_id += 1

        x, y, w, h = region.get("bbox", (0, 0, 0, 0))
        bgr = region.get("color_bgr", (0, 0, 0))
        hex_color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

        node = ObjectNode(
            id=node_id,
            mask=region.get("mask"),
            bbox=(int(x), int(y), int(w), int(h)),
            area=int(region.get("area", 0)),
            color_bgr=tuple(int(c) for c in bgr),
            color_hex=hex_color,
            centroid=region.get("centroid", (0.0, 0.0)),
            circularity=float(region.get("circularity", 0)),
            rectangularity=float(region.get("rectangularity", 0)),
            solidity=float(region.get("solidity", 0)),
            aspect_ratio=float(w / h) if h > 0 else 1.0,
            parent=region.get("parent", -1),
            children=region.get("children", []),
            z_order=node_id,
        )

        self.nodes[node_id] = node
        return node

    def build_from_regions(self, regions: List[Dict[str, Any]]) -> ObjectGraph:
        """Populate graph from a list of region dicts.

        Args:
            regions: List of region dicts.

        Returns:
            Self for chaining.
        """
        for region in regions:
            self.add_node(region)
        self._compute_neighbors()
        return self

    def _compute_neighbors(self):
        """Compute spatial neighbors for all nodes."""
        node_ids = list(self.nodes.keys())
        n = len(node_ids)

        for i in range(n):
            ni = self.nodes[node_ids[i]]
            if ni.mask is None:
                continue

            for j in range(i + 1, n):
                nj = self.nodes[node_ids[j]]
                if nj.mask is None:
                    continue

                # Check if masks are adjacent
                dilated = cv2_dilate(ni.mask)
                overlap = np.count_nonzero(cv2_bitwise_and(dilated, nj.mask))
                if overlap > 0:
                    ni.neighbors.append(nj.id)
                    nj.neighbors.append(ni.id)

    def get_root_nodes(self) -> List[ObjectNode]:
        """Return nodes with no parent (top-level objects)."""
        return [n for n in self.nodes.values() if n.parent == -1]

    def get_children(self, node_id: int) -> List[ObjectNode]:
        """Return child nodes of a given node."""
        return [self.nodes[cid] for cid in self.nodes[node_id].children if cid in self.nodes]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        nodes_list = []
        for node in self.nodes.values():
            nodes_list.append({
                "id": node.id,
                "bbox": node.bbox,
                "area": node.area,
                "color": node.color_hex,
                "parent": node.parent,
                "children": node.children,
                "neighbors": node.neighbors[:10],  # Limit
                "circularity": round(node.circularity, 4),
                "rectangularity": round(node.rectangularity, 4),
                "primitive_type": node.primitive_type,
                "primitive_confidence": round(node.primitive_confidence, 4),
                "semantic_type": node.semantic_type,
            })
        return {
            "total_nodes": len(self.nodes),
            "nodes": nodes_list,
        }

    def __len__(self):
        return len(self.nodes)


# Lazy imports to avoid circular deps
def cv2_dilate(mask):
    import cv2
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def cv2_bitwise_and(a, b):
    import cv2
    return cv2.bitwise_and(a, b)
