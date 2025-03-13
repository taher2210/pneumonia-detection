import streamlit as st
import os
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Google Drive File ID
FILE_ID = "1hWt04b_JaqN8THcCXMT8cHdPsJ4iq3Of"  # Replace with your actual File ID
MODEL_PATH = "pneumonia_model.h5"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
        st.write("Download complete.")

# Download and load model
download_model()
model = load_model(MODEL_PATH)
st.write("✅ Model loaded successfully!")

# Function to classify uploaded image
def predict_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Ensure correct input shape
    prediction = model.predict(img_array)[0][0]
    return "🫁 Pneumonia Detected" if prediction > 0.5 else "✅ Normal"

# Streamlit UI
st.title("Pneumonia Detection from X-ray")
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    result = predict_image(img)
    st.write(f"Prediction: **{result}**")
