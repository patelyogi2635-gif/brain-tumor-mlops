import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Classification")
st.markdown("Upload an MRI image to detect tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

        result = response.json()

        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Confidence: {result['confidence']}")
