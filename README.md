# 🎨 K-Means SVG Vectorizer

**Dual-engine bitmap → SVG vektörizasyon.** PNG/JPG görselleri K-Means renk kuantalama veya VTracer spline motoru ile SVG'ye dönüştürür.

> Powered by [VTracer](https://github.com/visioncortex/vtracer) (Rust, MIT) + custom K-Means pipeline.

## ⚡ Hızlı Başlangıç

```bash
# 1. Repo'yu klonla
git clone https://github.com/oktayc/kmeans-svg-vectorizer.git
cd kmeans-svg-vectorizer

# 2. Sanal ortam kur
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Çalıştır
python web_api.py
```

Tarayıcıda `http://localhost:8765` adresine git, görselini yükle, vectorize et! 🚀

## 🧠 Pipeline

### Motor 1: VTracer (varsayılan)
```
PNG/JPG → VTracer Rust engine → Spline eğriler → SVG
```
| Aşama | Açıklama |
|---|---|
| 🎨 Renk Kuantalama | VisionCortex clustering algorithm |
| 📐 Spline Fitting | O(n) eğri uydurma |
| 🏗️ Hierarchical | Stacked mode, deliksiz shape'ler |

### Motor 2: K-Means (klasik)

## 🖥️ Web Arayüzü

- 🖱️ **Drag & drop** görsel yükleme
- 🎚️ **Slider**: Renk sayısı (2-32), detay seviyesi (1-10)
- � **Motor seçimi**: VTracer (spline) | K-Means (düz çizgi)
- 👁️ **Canlı önizleme**: Orijinal vs SVG yan yana
- 💾 **Tek tıkla SVG indir**

## 📦 Bağımlılıklar

- Python 3.10+
- FastAPI + Uvicorn (web sunucu)
- OpenCV + NumPy (K-Means motoru)
- [VTracer](https://github.com/visioncortex/vtracer) (Rust, MIT — spline motoru)
- visioncortex/vtracer Python binding

## ⭐ Teşekkür

Bu proje [VisionCortex VTracer](https://github.com/visioncortex/vtracer)'ı varsayılan vektörizasyon motoru olarak kullanır. VTracer, MIT lisanslı, Rust ile yazılmış, akademik atıf almış bir raster-to-SVG dönüştürücüdür.

## 📄 Lisans

MIT — istediğin gibi kullan, değiştir, dağıt.

## ⚠️ Bilinen Sınırlamalar

- **Gradient/fotoğraf**: Vektörizasyon düz renkli logo ve grafikler için optimize edilmiştir. Gradyan geçişlerinde posterizasyon (bantlanma) oluşur — bu özellik üzerinde çalışılmaktadır.
- VTracer `color_precision` maksimum 8-bit (256 renk) ile sınırlıdır.

## 🛣️ Yol Haritası

- [ ] ~~Gradient bantlama~~ → v2.1'de kısmi iyileştirme yapıldı, tam çözüm için `<linearGradient>` modülü planlanıyor
- [ ] Anti-aliasing kenar yumuşatma
- [ ] Toplu işleme (batch klasör)
- [ ] Daha hızlı renk azaltma (Median Cut / Octree)
- [ ] SVG optimizasyonu (gereksiz nokta temizleme)
