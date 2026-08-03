import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')

# Bilgisayarın kamerasını başlat
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Model ile kare üzerinde nesne tespiti yap
    results = model(frame, conf=0.25)

    # Tespit edilen kutuları çiz
    annotated_frame = results[0].plot()

    # Ekranda göster
    cv2.imshow("YOLOv8 Canlı Kamera Tespit Testi", annotated_frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
