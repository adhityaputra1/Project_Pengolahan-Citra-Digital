import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np

model = YOLO("best.pt")  # sesuaikan path

# Inject custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🚀 Deteksi Objek sampah")
st.write("Upload gambar, dan model akan mendeteksi objek!")

# Navigasi menu
menu = st.sidebar.selectbox("Pilih Metode Input", ["Upload Gambar", "Kamera HP"])

if menu == "Upload Gambar":
    uploaded_file = st.file_uploader("📤 Upload Gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())
        st.image(Image.open(img_path), caption="📸 Gambar yang diupload", use_column_width=True)
        st.write("🔍 Mendeteksi...")
        results = model.predict(source=img_path, save=False, conf=0.3, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = annotated[..., ::-1]
        st.image(annotated_rgb, caption="📦 Hasil Deteksi", use_column_width=True)

elif menu == "Kamera HP":
    img_data = st.camera_input("Ambil foto objek")
    if img_data is not None:
        image = Image.open(img_data)
        st.image(image, caption="📸 Gambar dari Kamera", use_column_width=True)
        st.write("🔍 Mendeteksi...")
        image.save("temp_image.jpg")
        results = model.predict(source="temp_image.jpg", save=False, conf=0.3, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = annotated[..., ::-1]
        st.image(annotated_rgb, caption="📦 Hasil Deteksi", use_column_width=True)
