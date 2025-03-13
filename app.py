import streamlit as st
import os
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Streamlit Page Configuration
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# Project Introduction
st.title("🫁 Pneumonia Detection from Chest X-rays")
st.markdown(
    """
    This AI-powered tool helps in detecting **pneumonia** from chest X-ray images using a deep learning model.
    Simply upload an X-ray image, and the model will analyze it for signs of pneumonia. 
    This project is part of my **final-year thesis** on medical AI applications.
    """
)

# Google Drive Model File ID
FILE_ID = "1hWt04b_JaqN8THcCXMT8cHdPsJ4iq3Of"  # Replace with your actual File ID
MODEL_PATH = "pneumonia_model.h5"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("📥 Downloading model from Google Drive... This may take a moment.")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")

# Download and load model
download_model()

@st.cache_resource
def load_pneumonia_model():
    return load_model(MODEL_PATH)

model = load_pneumonia_model()
st.success("✅ Model loaded successfully!")

# Function to Classify Image
def predict_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Ensure correct input shape
    prediction = model.predict(img_array)[0][0]
    return "🫁 **Pneumonia Detected**" if prediction > 0.5 else "✅ **Normal**"

# Upload File
uploaded_file = st.file_uploader(
    "📂 Upload a Chest X-ray Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Display uploaded image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="🖼️ Uploaded X-ray", use_column_width=True)

    # Prediction Button
    if st.button("🔍 Analyze X-ray"):
        result = predict_image(img)
        st.markdown(f"### {result}")

        # Custom Styling for Results
        if "Pneumonia" in result:
            st.error(result)  # Red for Pneumonia
        else:
            st.success(result)  # Green for Normal

# Footer
st.markdown(
    """
    ---
   
    📜 PBL for Big Data Analysis.
    """
)
