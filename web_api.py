"""
Vectorizer API — Dual Engine
K-Means (default) + VTracer (spline curves)
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, time, cv2, numpy as np, uvicorn, vtracer
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
    engine: str = Form("kmeans"),
    layered: bool = Form(False),
    flattenGradients: bool = Form(False),
):
    """Vectorize: K-Means (default) or VTracer (spline curves)"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        if flattenGradients:
            tmp_path = _flatten_gradients(tmp_path)

        if engine == "vtracer":
            result = _vectorize_vtracer(tmp_path, maxColors, detail)
        else:
            result = _vectorize_kmeans(tmp_path, maxColors, detail, maxImageSize, strokeEnabled, strokeWidth)
        
        if layered:
            result["svg"] = _add_layers_to_svg(result["svg"])
            result["layered"] = True
        
        return result
    finally:
        os.unlink(tmp_path)


# ── Gradient Flatten ────────────────────────────────────────────

def _flatten_gradients(tmp_path):
    """Gradient bölgeleri düz renge çevir, kenarları koru.
    Mean-shift + bilateral filter ile gradient'leri yok eder."""
    img = cv2.imread(tmp_path)
    if img is None:
        return tmp_path
    
    h, w = img.shape[:2]
    # Büyük resimleri küçült (hız için)
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    # Mean-shift: benzer renkli pikselleri birleştir, gradient'leri düzleştir
    # sp=8 (küçük) → yuvarlak kenarları daha iyi korur
    flat = cv2.pyrMeanShiftFiltering(img, sp=8, sr=25, maxLevel=2)
    
    # Edge-aware filter: yuvarlak hatları mean-shift'ten daha iyi korur
    flat = cv2.edgePreservingFilter(flat, flags=1, sigma_s=40, sigma_r=0.15)
    
    out_path = tmp_path + "_flat.png"
    cv2.imwrite(out_path, flat)
    return out_path


# ── VTracer Engine ──────────────────────────────────────────────

def _vectorize_vtracer(tmp_path, maxColors, detail):
    t_start = time.time()
    out_path = tmp_path + ".svg"
    # adaptive quantization: çok renk + küçük segment = daha yumuşak geçiş
    # maxColors 2→256 → color_precision 4→8 (16→256 renk)
    color_precision = max(4, min(8, int(maxColors ** 0.4) + 2))
    # Yüksek detay → daha düşük eşik = daha çok küçük segment
    corner = max(20, 80 - detail * 8)
    splice = max(15, 65 - detail * 6)
    speckle = max(1, 6 - detail // 2)

    vtracer.convert_image_to_svg_py(
        tmp_path, out_path,
        colormode='color',
        hierarchical='stacked',
        mode='spline',
        filter_speckle=speckle,
        color_precision=color_precision,
        corner_threshold=corner,
        splice_threshold=splice,
    )

    svg = Path(out_path).read_text(encoding="utf-8")
    path_count = svg.count('<path ')

    try: os.unlink(out_path)
    except OSError: pass

    return {
        "svg": svg,
        "colorCount": 2 ** color_precision,
        "pathCount": path_count,
        "processingTime": int((time.time() - t_start) * 1000),
        "engine": "vtracer",
    }


# ── K-Means Engine ──────────────────────────────────────────────

def _vectorize_kmeans(tmp_path, maxColors, detail, maxImageSize, strokeEnabled, strokeWidth):
    t_start = time.time()

    img = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"error": "Failed to load image"}

    h, w = img.shape[:2]
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
    elif img.ndim == 3:
        bgr = img
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    orig_h, orig_w = h, w

    if max(h, w) > maxImageSize:
        scale = maxImageSize / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        h, w = bgr.shape[:2]

    upscale = 2
    bgr = cv2.resize(bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_LINEAR)
    hu, wu = bgr.shape[:2]

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
        mask = (labels.flatten() == ci).reshape(hu, wu).astype(np.uint8) * 255
        if np.count_nonzero(mask) < 120:
            continue

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        color = centers[ci]
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"

        if hierarchy is None:
            continue

        hierarchy = hierarchy[0]
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

            if i in children_of:
                hole_paths = []
                for child_idx in children_of[i]:
                    child_cnt = contours[child_idx]
                    if cv2.contourArea(child_cnt) < 80:
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

    paths.sort(key=lambda p: p[0], reverse=True)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {orig_w} {orig_h}" width="{orig_w}" height="{orig_h}">',
        f'<rect width="{orig_w}" height="{orig_h}" fill="white"/>',
    ] + [p[1] for p in paths] + ["</svg>"]

    svg = "\n".join(svg_lines)

    return {
        "svg": svg,
        "colorCount": k,
        "pathCount": len(paths),
        "processingTime": int((time.time() - t_start) * 1000),
        "engine": "kmeans",
    }


def _add_layers_to_svg(svg_text):
    """Group SVG paths by fill color into named <g> layers for PPT/Illustrator."""
    import re
    paths = re.findall(r'(<path[^>]*?fill="([^"]+)"[^>]*?/>)', svg_text)
    if len(paths) < 2:
        return svg_text
    
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    def color_dist(c1, c2):
        return sum((a-b)**2 for a,b in zip(c1, c2)) ** 0.5
    
    groups = {}
    for path_full, fill_color in paths:
        if fill_color.lower() in ('none', 'white') or fill_color == '#ffffff':
            continue
        try:
            rgb = hex_to_rgb(fill_color)
        except:
            continue
        best_group, best_dist = None, 9999
        for g_color in groups:
            d = color_dist(rgb, g_color)
            if d < best_dist:
                best_dist, best_group = d, g_color
        if best_group and best_dist < 35:
            groups[best_group].append(path_full)
        else:
            groups[rgb] = [path_full]
    
    if len(groups) <= 1:
        return svg_text
    
    # Keep original SVG header + background
    svg_start = svg_text[:svg_text.index('>')+1]
    svg_end = '</svg>'
    bg_match = re.search(r'<rect[^>]*/>', svg_text)
    bg_str = bg_match.group(0) if bg_match else ''
    
    def brightness(rgb):
        return 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    
    sorted_groups = sorted(groups.items(), key=lambda x: brightness(x[0]))
    lines = [svg_start, bg_str]
    for i, (rgb, layer_paths) in enumerate(sorted_groups, 1):
        lines.append(f'  <!-- layer-{i}: {len(layer_paths)} shapes -->')
        lines.append(f'  <g id="layer-{i}">')
        for p in layer_paths:
            lines.append(f'    {p}')
        lines.append(f'  </g>')
    lines.append(svg_end)
    return '\n'.join(lines)


@app.get("/")
async def root():
    frontend_path = Path(__file__).parent / "frontend.html"
    if frontend_path.exists():
        return HTMLResponse(frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VektorAI API - Running on :8765</h1>")


if __name__ == "__main__":
    print("🚀 Starting on http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765)
