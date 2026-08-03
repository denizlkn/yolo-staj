import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Sayfa başlığı
st.set_page_config(page_title="YOLOv8 Nesne Tespiti - Gelişmiş", layout="wide")
st.title("YOLOv8 Nesne Tespiti ve Web Arayüzü")
st.write("Fotoğraf yükleyin, güven eşiğini ve tespit edilecek sınıfları seçin.")

# Modeli yükle
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

model = load_model()

# Modelin tanıdığı tüm sınıfları al
all_classes = model.names
class_list = list(all_classes.values())

# --- Kenar Çubuğu Ayarları ---
st.sidebar.header("Ayarlar")

# 1. Güven Eşiği Slider
conf_thres = st.sidebar.slider("Güven Eşiği (Confidence Threshold)", 
                               min_value=0.0, max_value=1.0, 
                               value=0.25, step=0.05)

# 2. Sınıf Seçici (Multiselect)
selected_classes = st.sidebar.multiselect(
    "Tespit Edilecek Sınıfları Seçin",
    class_list,
    default=["person", "car"]
)

# Seçilen sınıfların ID'lerini bul
selected_indices = [k for k, v in all_classes.items() if v in selected_classes]

# --- Ana Sayfa İşlemleri ---
uploaded_file = st.file_uploader("Bir fotoğraf seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Orijinal Fotoğraf")
        st.image(image, use_column_width=True)

    if st.button("Tespit Et"):
        with st.spinner('YOLO modeli çalışıyor...'):
            img_array = np.array(image)
            
            # PIL görüntüsü RGB'dir, OpenCV için BGR'ye çevirip modele verelim
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            results = model.predict(source=img_bgr, conf=conf_thres, classes=selected_indices)
            
            res_plotted = results[0].plot()
            
            # Çıktıyı Streamlit için tekrar RGB'ye dönüştür
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Tespit Sonucu")
            st.image(res_plotted_rgb, caption='İşlenmiş Fotoğraf', use_column_width=True)
            
            st.write("### Tespit Detayları:")
            boxes = results[0].boxes
            if len(boxes) == 0:
                st.warning("Seçilen kriterlere uygun nesne bulunamadı.")
            else:
                for box in boxes:
                    c = int(box.cls)
                    name = model.names[c]
                    score = float(box.conf)
                    st.success(f"- **{name}** bulundu! (Güven Skoru: %{score*100:.1f})")

elif uploaded_file is None and len(selected_classes) == 0:
    st.info("Lütfen bir fotoğraf yükleyin.")
elif len(selected_classes) == 0:
    st.warning("Lütfen tespit edilmesini istediğiniz en az bir sınıf seçin (Sol menüden).")