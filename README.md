# 🎨 K-Means SVG Vectorizer

**Bitmap → SVG vektörizasyon pipeline.** PNG/JPG görselleri K-Means renk kuantalama + kontur tespiti ile SVG'ye dönüştürür.

> Sıfırdan yazıldı, sadece standart OpenCV fonksiyonları kullanır. AI/ML yok.

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

```
PNG/JPG → 2x Upscale → K-Means (12-32 renk) → Per-color Maske
→ Morfolojik Yumuşatma → findContours (RETR_CCOMP, delik tespiti)
→ Douglas-Peucker Basitleştirme → SVG <path> (fill-rule="evenodd")
```

| Aşama | Açıklama |
|---|---|
| 🎨 K-Means | Renkleri N kümeye indirger |
| 🔍 2x Upscale | Sub-pixel kenar kalitesi |
| 🧹 Morfoloji | Maskelerdeki gürültüyü temizler |
| 🕳️ RETR_CCOMP | Delikleri (hole) tespit eder |
| 📐 approxPolyDP | Akıllı düğüm azaltma |
| 📏 Z-order | Büyük şekiller arkada, küçükler önde |

## 🖥️ Web Arayüzü

- 🖱️ **Drag & drop** görsel yükleme
- 🎚️ **Slider**: Renk sayısı (2-32), detay seviyesi (1-10)
- 👁️ **Canlı önizleme**: Orijinal vs SVG yan yana
- 💾 **Tek tıkla SVG indir**

## 📦 Bağımlılıklar

- Python 3.10+
- FastAPI + Uvicorn (web sunucu)
- OpenCV (görüntü işleme)
- NumPy (matris işlemleri)

## 📄 Lisans

MIT — istediğin gibi kullan, değiştir, dağıt.

## 🛣️ Yol Haritası

- [ ] Gradient bantlama (posterizasyon azaltma)
- [ ] Anti-aliasing kenar yumuşatma
- [ ] Toplu işleme (batch klasör)
- [ ] Daha hızlı renk azaltma (Median Cut / Octree)
- [ ] SVG optimizasyonu (gereksiz nokta temizleme)
