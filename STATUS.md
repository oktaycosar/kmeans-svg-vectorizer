# 🎨 K-Means SVG Vectorizer — Proje Durumu

> Son güncelleme: 22 Temmuz 2026

---

## ✅ Dual Engine (v2.0)

| Motor | Tip | Hiz | Kalite |
|---|---|---|---|
| 🎯 **VTracer** (default) | Spline egriler | ~50-80ms | Profesyonel |
| 🔬 **K-Means** (klasik) | Duz cizgiler | ~900ms | Idare eder |

### VTracer avantajlari
- 13x daha hizli (82ms vs 1088ms)
- %40 daha az path (39 vs 68)
- Spline egriler, yumusak gecisler
- O(n) algoritma, akademik atifli
- [VisionCortex VTracer](https://github.com/visioncortex/vtracer) (Rust, MIT)

---

## 📂 Dosya Yapisi

| Dosya | Amac |
|---|---|
| `web_api.py` | FastAPI sunucu + cift motor endpoint'i |
| `frontend.html` | Web arayuzu (motor secimi, drag-drop) |
| `examples/` | Test gorselleri + ornek ciktilar |
| `image_vectorizer/` | K-Means pipeline modulleri |
| `geometry_corrector/` | Deneysel AI duzeltme (devre disi) |

---

## 🚀 Calistirma

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python web_api.py
# → http://localhost:8765
```

## ⭐ Atif

VTracer motoru: [visioncortex/vtracer](https://github.com/visioncortex/vtracer) — MIT License
