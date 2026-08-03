import cv2
from ultralytics import YOLO
import os

# --- Ayarlar ---
# İşlenecek video dosyasının yolu. 
# ÖNEMLİ: Proje klasörünüzde bu isimde bir video olmalı veya yolu güncelleyin.
# Örnek: 'images/test_video.mp4'
INPUT_VIDEO_PATH = 'images/test_video.mp4' 

# Çıktı videosunun kaydedileceği klasör ve isim
OUTPUT_DIR = 'outputs'
OUTPUT_VIDEO_NAME = 'processed_video.mp4'
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_NAME)

# Güven eşiği
CONFIRMATION_THRESHOLD = 0.25

# Sadece belirli sınıfları işlemek isterseniz buraya ID'lerini yazın (örn: [0, 2] -> person, car)
# Hepsini işlemek için None bırakın.
CLASSES_TO_DETECT = None 

# --- İşlemler Başlıyor ---
# Çıktı klasörü yoksa oluştur
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"'{OUTPUT_DIR}' klasörü oluşturuldu.")

# YOLOv8 modelini yükle
print("YOLOv8 modeli yükleniyor...")
model = YOLO('yolov8n.pt') 

# Videoyu aç
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    print(f"HATA: Video dosyası açılamadı. Lütfen yolu kontrol edin: {INPUT_VIDEO_PATH}")
    exit()

# Video özelliklerini al
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video yüklendi: {frame_width}x{frame_height}, {fps} FPS, Toplam {frame_count} kare.")

# Çıktı video yazarını (VideoWriter) ayarla
# Windows için genellikle 'mp4v' veya 'XVID' codec bileşenleri kullanılır.
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print(f"İşlem başlıyor... Çıktı buraya kaydedilecek: {OUTPUT_VIDEO_PATH}")
print("İşlem sürerken 'q' tuşuna basarak durdurabilirsiniz (Pencere açıksa).")

current_frame = 0
while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        # Video bitti veya kare okunamadı
        break
    
    current_frame += 1
    
    # --- YOLO Tahmini ---
    # Sınıf filtrelemesi varsa ekle
    if CLASSES_TO_DETECT is not None:
        results = model.predict(source=frame, conf=CONFIRMATION_THRESHOLD, classes=CLASSES_TO_DETECT, verbose=False)
    else:
        results = model.predict(source=frame, conf=CONFIRMATION_THRESHOLD, verbose=False)
    
    # Sonuçları kare üzerine çiz
    annotated_frame = results[0].plot()
    
    # İşlenmiş kareyi çıktı videosuna yaz
    out.write(annotated_frame)
    
    # İlerlemeyi konsola yazdır (Her 30 karede bir)
    if current_frame % 30 == 0:
        print(f"İlerleme: Kare {current_frame}/{frame_count} (%{int(current_frame/frame_count*100)})", end='\r')

    # (Opsiyonel) İşlenen kareyi ekranda göster (Hızlandırır ama işlemciyi yorar)
    # cv2.imshow("YOLOv8 Video İşleme", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     print("\nKullanıcı tarafından durduruldu.")
    #     break

print(f"\n\nİşlem tamamlandı! Toplam {current_frame} kare işlendi.")
print(f"Sonuç videosu kaydedildi: {OUTPUT_VIDEO_PATH}")

# Kaynakları serbest bırak
cap.release()
out.release()
# cv2.destroyAllWindows() # Eğer imshow kullanmadıysanız gerek yok