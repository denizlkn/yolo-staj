# YOLOv8 Nesne Tespiti ve Web Arayüzü Projesi

Bu proje, Ultralytics YOLOv8 Derin Öğrenme modelini kullanarak görseller ve canlı kamera akışı üzerinde gerçek zamanlı nesne tespiti yapmak amacıyla geliştirilmiştir.

## 🚀 Özellikler
- **Toplu Görsel İşleme (`detect.py`):** Görselleri tarar, nesne tespiti yapar ve işlenmiş görselleri kaydeder.
- **Canlı Kamera Tespiti (`webcam.py`):** Bilgisayar kamerasından gelen görüntüyü gerçek zamanlı analiz eder.
- **Streamlit Web Arayüzü (`app.py`):** Web arayüzü üzerinden görsel yükleme ve güven skoru ayarlama imkanı sunar.

## 🛠️ Kurulum
```bash
pip install -r requirements.txt
streamlit run app.py
