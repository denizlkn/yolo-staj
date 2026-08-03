# YOLOv8 ile Gerçek Zamanlı Nesne Tespiti Projesi

Bu proje, Ultralytics YOLOv8 nano modelini kullanarak görseller, klasörler ve canlı kamera yayını üzerinden nesne tespiti yapan, Streamlit web arayüzüne sahip bir bilgisayarlı görü (computer vision) uygulamasıdır.

## 🚀 Özellikler
- **Toplu Görsel İşleme (`detect.py`):** Belirtilen klasördeki görselleri tarar, taranmış sonuçları `outputs/` klasörüne kaydeder ve terminale doğruluk skorlarını bastırır.
- **Canlı Kamera Tespiti (`webcam.py`):** OpenCV altyapısı ile web kamerasından alınan canlı yayında anlık nesne tespiti gerçekleştirir.
- **Web Arayüzü (`app.py`):** Streamlit ile geliştirilmiş, kullanıcının görsel yükleyebildiği ve dinamik olarak Güven Eşiğini (Confidence Threshold) değiştirebildiği arayüz.

## 🛠️ Kurulum
Proje bağımlılıklarını yüklemek için:
```bash
pip install -r requirements.txt