import streamlit as st
import os
import sys

# Install required packages if not already installed
st.write("Checking and installing required packages...")
packages_to_install = [
    "gdown",
    "numpy",
    "matplotlib",
    "tensorflow",
    "pillow"
]

def install_packages():
    import subprocess
    for package in packages_to_install:
        try:
            __import__(package)
            st.write(f"✅ {package} is already installed")
        except ImportError:
            st.write(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            st.write(f"✅ {package} installed successfully")

install_packages()

# Now import the required packages
import gdown
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import tensorflow as tf

# Streamlit Page Configuration
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# Project Introduction
st.title("🫁 Pneumonia Detection from Chest X-rays")
st.markdown(
    """
    This AI-powered tool helps in detecting **pneumonia** from chest X-ray images using a deep learning model.
    Simply upload an X-ray image, and the model will analyze it for signs of pneumonia. 
    The heatmap visualization shows which areas of the X-ray influenced the model's decision.
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

# Function to generate GradCAM heatmap
def make_gradcam_heatmap(img_array, model):
    # Create a model that maps the input image to the activations of the last conv layer
    last_conv_layer = None
    
    # Find the last convolutional layer
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer.name
            break
    
    if last_conv_layer is None:
        st.warning("Could not find a convolutional layer for GradCAM visualization")
        return None
    
    last_conv_layer_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output, 
            model.output
        ]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        # Forward pass to get conv layer output and model prediction
        conv_output, predictions = last_conv_layer_model(img_array)
        class_index = 0  # Pneumonia is typically index 0 in binary classification
        loss = predictions[:, class_index]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_output)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight output feature maps with gradients
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    return heatmap

# Function to overlay heatmap on image
def overlay_heatmap(img, heatmap, alpha=0.4):
    # Convert PIL Image to numpy array if it's not already
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img
    
    # Resize heatmap to match input image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((img_array.shape[1], img_array.shape[0]))
    heatmap = np.array(heatmap)
    
    # Apply colormap to heatmap
    heatmap = cm.jet(heatmap)[:, :, :3]  # Drop alpha channel
    heatmap = np.uint8(255 * heatmap)
    
    # Overlay heatmap on original image
    overlay = np.uint8(heatmap * alpha + img_array * (1 - alpha))
    
    return overlay

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

# Function to Classify Image and generate GradCAM
def analyze_image(img_data):
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(img_data))
        
        # Save original image for display
        orig_img = img.copy()
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image for model input
        img_resized = img.resize((150, 150))
        
        # Convert to array and normalize
        img_array = np.array(img_resized) / 255.0
        
        # Add batch dimension
        img_array_batch = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array_batch)[0][0]
        probability = float(prediction)
        
        # Generate GradCAM heatmap
        heatmap = make_gradcam_heatmap(img_array_batch, model)
        
        if heatmap is not None:
            # Create heatmap overlay on original image
            overlay_img = overlay_heatmap(img_resized, heatmap)
        else:
            overlay_img = None
        
        return {
            'original_img': orig_img,
            'probability': probability,
            'heatmap': heatmap,
            'overlay_img': overlay_img
        }
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.error(f"Details: {str(e)}")
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
                results = analyze_image(file_bytes)
                
                if results is not None:
                    # Display results
                    probability = results['probability']
                    
                    if probability > 0.5:
                        result = "🫁 **Pneumonia Detected**"
                        st.error(result)
                        st.write(f"Confidence: {probability:.2%}")
                    else:
                        result = "✅ **Normal**"
                        st.success(result)
                        st.write(f"Confidence: {(1-probability):.2%}")
                    
                    # Display GradCAM visualization if available
                    if results['overlay_img'] is not None:
                        st.subheader("GradCAM Visualization")
                        st.image(results['overlay_img'], caption="Heatmap of areas influencing the model's decision", use_column_width=True)
                        st.info("The highlighted areas (red/yellow) show regions the model focused on when making its prediction.")
                    
                    # Add visualization of confidence
                    st.progress(probability if probability > 0.5 else 1-probability)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error(f"Details: {str(e)}")

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
    3. View the results, confidence score, and heatmap visualization
    
    **About the Heatmap**: The heatmap highlights areas that most influenced the model's decision. Warmer colors (red/yellow) indicate regions with higher importance for the diagnosis.
    
    Note: This app works best with properly oriented, front-view chest X-ray images.
    """)

with st.expander("About the model"):
    st.markdown("""
    This application uses a convolutional neural network (CNN) trained on a dataset of chest X-ray images.
    The model was trained to distinguish between normal chest X-rays and those showing signs of pneumonia.
    
    The GradCAM visualization helps make the model's decision process more transparent by showing which regions of the image contributed most to the classification.
    
    Please note that this is an educational tool and should not replace professional medical diagnosis.
    Always consult with a healthcare professional for medical advice.
    """)
