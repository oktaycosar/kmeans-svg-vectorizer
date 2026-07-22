"""
Vectorizer API — Stable v1
K-Means segmentation + 2x upscale + findContours + SVG paths
No Bezier curves, no AI, no over-engineering.
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, time, cv2, numpy as np, uvicorn
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/api/vectorize")
async def vectorize(
    image: UploadFile = File(...),
    maxColors: int = Form(16),
    detail: int = Form(5),
    maxImageSize: int = Form(1024),
    strokeEnabled: bool = Form(False),
    strokeWidth: float = Form(0.5),
):
    """Vectorize uploaded image: K-Means → per-color mask → contours → SVG paths"""
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        # Load image
        img = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"error": "Failed to load image"}

        h, w = img.shape[:2]
        
        # Extract BGR (handle RGBA)
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
        elif img.ndim == 3:
            bgr = img
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        orig_h, orig_w = h, w

        # Downscale if too large
        if max(h, w) > maxImageSize:
            scale = maxImageSize / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
            h, w = bgr.shape[:2]

        t_start = time.time()

        # --- 2x upscale for sub-pixel edge quality ---
        upscale = 2
        bgr = cv2.resize(bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_LINEAR)
        hu, wu = bgr.shape[:2]

        # --- K-Means color quantization ---
        pixels = bgr.reshape(-1, 3).astype(np.float32)
        k = max(2, min(maxColors, 32))
        _, labels, centers = cv2.kmeans(
            pixels, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            3, cv2.KMEANS_PP_CENTERS
        )
        centers = centers.astype(np.uint8)

        paths = []

        for ci in range(k):
            # Per-color binary mask at 2x resolution
            mask = (labels.flatten() == ci).reshape(hu, wu).astype(np.uint8) * 255
            if np.count_nonzero(mask) < 120:
                continue

            # Smooth the mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            color = centers[ci]
            hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"

            if hierarchy is None:
                continue

            hierarchy = hierarchy[0]  # shape: (1, N, 4)

            # Build parent→children map from hierarchy
            # hierarchy[i] = [next, prev, first_child, parent]
            children_of = {}
            for i, h in enumerate(hierarchy):
                parent = h[3]
                if parent != -1:
                    children_of.setdefault(parent, []).append(i)

            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < 120:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                eps_factor = 0.008 - (detail - 1) * 0.00083
                epsilon = max(0.001, eps_factor) * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = approx.reshape(-1, 2)

                if len(pts) < 3:
                    continue

                pts = pts / upscale

                def pts_to_d(pts_arr):
                    return "M " + " L ".join(f"{p[0]:.2f} {p[1]:.2f}" for p in pts_arr) + " Z"

                d = pts_to_d(pts)

                # If this contour has holes (children), include them with opposite winding
                if i in children_of:
                    hole_paths = []
                    for child_idx in children_of[i]:
                        child_cnt = contours[child_idx]
                        child_area = cv2.contourArea(child_cnt)
                        if child_area < 80:
                            continue
                        child_peri = cv2.arcLength(child_cnt, True)
                        child_eps = max(0.001, eps_factor) * child_peri
                        child_approx = cv2.approxPolyDP(child_cnt, child_eps, True)
                        child_pts = child_approx.reshape(-1, 2)
                        if len(child_pts) < 3:
                            continue
                        child_pts = child_pts / upscale
                        hole_paths.append(pts_to_d(child_pts))

                    if hole_paths:
                        d = d + " " + " ".join(hole_paths)

                sw = strokeWidth if strokeEnabled else 0.3
                fill_rule = ' fill-rule="evenodd"' if (i in children_of and children_of[i]) else ""
                path_str = f'<path d="{d}"{fill_rule} fill="{hex_color}" stroke="{hex_color}" stroke-width="{sw}" stroke-linejoin="round"/>'

                paths.append((area, path_str))

        # --- Z-order: largest shapes first (background), smallest last (foreground) ---
        paths.sort(key=lambda p: p[0], reverse=True)

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {orig_w} {orig_h}" width="{orig_w}" height="{orig_h}">',
            f'<rect width="{orig_h}" height="{orig_h}" fill="white"/>',
        ] + [p[1] for p in paths] + ["</svg>"]

        svg = "\n".join(svg_lines)

        return {
            "svg": svg,
            "width": orig_w,
            "height": orig_h,
            "colorCount": k,
            "pathCount": len(paths),
            "processingTime": int((time.time() - t_start) * 1000),
        }

    finally:
        os.unlink(tmp_path)


@app.get("/")
async def root():
    frontend_path = Path(__file__).parent / "frontend.html"
    if frontend_path.exists():
        return HTMLResponse(frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VektorAI API — Running on :8765</h1>")


if __name__ == "__main__":
    print("🚀 Starting on http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765)
