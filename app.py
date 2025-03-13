import streamlit as st
import os
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Streamlit Page Configuration
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# Project Introduction
st.title("🫁 Pneumonia Detection from Chest X-rays")
st.markdown(
    """
    This AI-powered tool helps in detecting **pneumonia** from chest X-ray images using a deep learning model.
    Simply upload an X-ray image, and the model will analyze it for signs of pneumonia. 
    """
)

# Google Drive Model File ID
FILE_ID = "1hWt04b_JaqN8THcCXMT8cHdPsJ4iq3Of"  # Replace with your actual File ID
MODEL_PATH = "pneumonia_model.h5"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("📥 Downloading model from Google Drive... This may take a moment.")
        try:
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.error("Please check your internet connection and the file ID.")
            return False
    return True

# Download and load model
if download_model():
    try:
        @st.cache_resource
        def load_pneumonia_model():
            return load_model(MODEL_PATH)
        
        model = load_pneumonia_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
else:
    st.stop()

# Function to Classify Image
def predict_image(img_data):
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(img_data))
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize image
        img = img.resize((150, 150))
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        probability = float(prediction)
        return probability
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

# Upload File
uploaded_file = st.file_uploader(
    "📂 Upload a Chest X-ray Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Read file
        file_bytes = uploaded_file.getvalue()
        
        # Display uploaded image
        st.image(file_bytes, caption="🖼️ Uploaded X-ray", use_column_width=True)
        
        # Prediction Button
        if st.button("🔍 Analyze X-ray"):
            with st.spinner("Analyzing image..."):
                probability = predict_image(file_bytes)
                
                if probability is not None:
                    # Display results
                    if probability > 0.5:
                        result = "🫁 **Pneumonia Detected**"
                        st.error(result)
                        st.write(f"Confidence: {probability:.2%}")
                    else:
                        result = "✅ **Normal**"
                        st.success(result)
                        st.write(f"Confidence: {(1-probability):.2%}")
                        
                    # Add visualization of confidence
                    st.progress(probability if probability > 0.5 else 1-probability)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown(
    """
    ---
    📜 PBL for Big Data Analysis.
    """
)

# Add some helpful information
with st.expander("How to use this app"):
    st.markdown("""
    1. Upload a chest X-ray image (JPG, PNG, or JPEG format)
    2. Click the "Analyze X-ray" button
    3. View the results and confidence score
    
    Note: This app works best with properly oriented, front-view chest X-ray images.
    """)

with st.expander("About the model"):
    st.markdown("""
    This application uses a convolutional neural network (CNN) trained on a dataset of chest X-ray images.
    The model was trained to distinguish between normal chest X-rays and those showing signs of pneumonia.
    
    Please note that this is a educational tool and should not replace professional medical diagnosis.
    Always consult with a healthcare professional for medical advice.
    """)
