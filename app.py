import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Basit Nesne Tespiti (YOLOv8)")

conf = st.slider("Güven Eşiği (Confidence Threshold)", 0.10, 0.90, 0.25, 0.05)

file = st.file_uploader("Bir görsel yükleyin...", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Orijinal Görsel", use_container_width=True)

    result = model.predict(np.array(image), conf=conf, verbose=False)[0]

    plotted = result.plot()[:, :, ::-1]
    st.image(plotted, caption="Nesne Tespiti Sonucu", use_container_width=True)

    st.subheader("Tespit Edilen Nesneler")
    if len(result.boxes) > 0:
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            score = float(box.conf[0])
            st.write(f"- **{name}**: %{score * 100:.1f} güven skoru ({score:.2f})")
    else:
        st.write("Seçilen güven eşiğinde hiçbir nesne tespit edilemedi.")