"""
Text Detector — OCR-based text extraction for SVG.

Detects text in images using Tesseract OCR (subprocess), masks it out 
from the K-Means pipeline, and returns SVG <text> elements with real fonts.

Philosophy: Text shouldn't be vectorized as paths — it should be real <text>.
"""
import cv2
import numpy as np
import subprocess, tempfile, os, re
from typing import List, Tuple, Optional

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def detect_text_regions(
    image: np.ndarray,
    min_confidence: int = 40,
    padding: int = 4,
) -> List[dict]:
    """
    Detect text regions using Tesseract OCR via subprocess.
    """
    h, w = image.shape[:2]
    
    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, image)
    
    try:
        # Run tesseract with TSV output (tab-separated: level, page_num, block_num, 
        # par_num, line_num, word_num, left, top, width, height, conf, text)
        out_base = tmp_path + "_out"
        result = subprocess.run(
            [TESSERACT_EXE, tmp_path, out_base, "-l", "eng", "--psm", "11", "tsv"],
            capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        
        tsv_path = out_base + ".tsv"
        if not os.path.exists(tsv_path):
            return []
        
        # Parse TSV output
        regions = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return []
        
        headers = lines[0].strip().split("\t")
        # Find column indices
        try:
            idx_level = headers.index("level")
            idx_left = headers.index("left")
            idx_top = headers.index("top")
            idx_width = headers.index("width")
            idx_height = headers.index("height")
            idx_conf = headers.index("conf")
            idx_text = headers.index("text")
        except ValueError:
            return []
        
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) < len(headers):
                continue
            
            try:
                level = int(parts[idx_level])
            except ValueError:
                continue
            
            # Only use word-level detections (level 5)
            if level != 5:
                continue
            
            text = parts[idx_text].strip()
            if not text:
                continue
            
            try:
                conf = float(parts[idx_conf])
            except ValueError:
                conf = 0
            
            if conf < min_confidence:
                continue
            
            left = int(float(parts[idx_left]))
            top = int(float(parts[idx_top]))
            tw = int(float(parts[idx_width]))
            th = int(float(parts[idx_height]))
            
            if tw < 4 or th < 4:
                continue
            
            # Add padding
            x = max(0, left - padding)
            y = max(0, top - padding)
            tw = min(w - x, tw + 2 * padding)
            th = min(h - y, th + 2 * padding)
            
            # Sample text color
            sample_region = image[max(0, y):min(h, y+th), max(0, x):min(w, x+tw)]
            if sample_region.size > 0:
                gray = cv2.cvtColor(sample_region, cv2.COLOR_BGR2GRAY)
                dark_mask = gray < np.median(gray)
                if np.any(dark_mask):
                    dark_pixels = sample_region[dark_mask]
                    text_color_bgr = np.median(dark_pixels, axis=0).astype(int)
                else:
                    text_color_bgr = np.array([0, 0, 0])
            else:
                text_color_bgr = np.array([0, 0, 0])
            
            font_size = th * 0.75
            
            regions.append({
                "text": text,
                "x": x, "y": y, "w": tw, "h": th,
                "confidence": conf,
                "color_bgr": text_color_bgr,
                "font_size": font_size,
            })
        
        # Cleanup tsv
        try:
            os.unlink(tsv_path)
        except OSError:
            pass
        
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    
    # Merge overlapping/adjacent regions
    regions = _merge_nearby_regions(regions, image.shape)
    
    return regions


def _merge_nearby_regions(regions: List[dict], image_shape: Tuple) -> List[dict]:
    """Merge text regions that are on the same line and close together."""
    if len(regions) <= 1:
        return regions
    
    h, w = image_shape[:2]
    merged = []
    used = set()
    
    for i, r1 in enumerate(regions):
        if i in used:
            continue
        
        # Find all regions on the same line (similar y, close x)
        same_line = [r1]
        used.add(i)
        
        for j, r2 in enumerate(regions):
            if j in used:
                continue
            
            # Check if on same line: vertical overlap > 50%
            y_overlap = min(r1["y"] + r1["h"], r2["y"] + r2["h"]) - max(r1["y"], r2["y"])
            min_h = min(r1["h"], r2["h"])
            
            if y_overlap > min_h * 0.4:
                # Check horizontal proximity
                gap = max(r2["x"] - (r1["x"] + r1["w"]), r1["x"] - (r2["x"] + r2["w"]))
                if gap < max(r1["h"], r2["h"]) * 3:
                    same_line.append(r2)
                    used.add(j)
        
        if len(same_line) == 1:
            merged.append(r1)
        else:
            # Merge into one region
            texts = [r["text"] for r in same_line]
            xs = [r["x"] for r in same_line]
            ys = [r["y"] for r in same_line]
            
            merged_region = {
                "text": " ".join(texts),
                "x": min(xs),
                "y": min(ys),
                "w": max(r["x"] + r["w"] for r in same_line) - min(xs),
                "h": max(r["h"] for r in same_line),
                "confidence": min(r["confidence"] for r in same_line),
                "color_bgr": same_line[0]["color_bgr"],
                "font_size": max(r["font_size"] for r in same_line),
            }
            merged.append(merged_region)
    
    return merged


def create_text_mask(
    image_shape: Tuple[int, int],
    text_regions: List[dict],
    dilate_kernel: int = 5,
) -> np.ndarray:
    """
    Create a binary mask covering all text regions.
    White (255) = text region (to be masked OUT of K-Means).
    Black (0) = non-text region (to be vectorized normally).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    
    for region in text_regions:
        x1 = region["x"]
        y1 = region["y"]
        x2 = min(w, region["x"] + region["w"])
        y2 = min(h, region["y"] + region["h"])
        
        mask[y1:y2, x1:x2] = 255
    
    # Dilate to ensure text is fully covered
    if dilate_kernel > 1:
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def inpaint_text_regions(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Remove text from image by inpainting. This replaces text pixels
    with surrounding background colors so K-Means doesn't try to
    vectorize individual letters.
    """
    return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)


def regions_to_svg_text(text_regions: List[dict]) -> str:
    """
    Convert detected text regions to SVG <text> elements.
    
    Uses system fonts with appropriate styling.
    """
    lines = []
    
    for region in text_regions:
        b, g, r = region["color_bgr"]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        font_size = max(8, region["font_size"])
        
        # Position text at baseline (bottom of bounding box)
        x = region["x"]
        y = region["y"] + region["h"] * 0.85
        
        text = region["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" '
            f'font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{font_size:.1f}" '
            f'fill="{hex_color}" '
            f'font-weight="bold">{text}</text>'
        )
    
    return "\n".join(lines)
