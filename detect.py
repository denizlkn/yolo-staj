import os
import cv2
from ultralytics import YOLO

# Hazır YOLOv8 nano modelini yükle
model = YOLO("yolov8n.pt")

os.makedirs("outputs", exist_ok=True)

images_dir = "images"
for img_name in os.listdir(images_dir):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        results = model.predict(img, conf=0.25, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            score = float(box.conf[0])

            print(f"[{img_name}] Nesne: {name} | Skor: {score:.2f}")

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{name} {score:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        output_path = os.path.join("outputs", f"out_{img_name}")
        cv2.imwrite(output_path, img)
        print(f"Kaydedildi: {output_path}\n---")

print("Tüm görseller başarıyla işlendi!")