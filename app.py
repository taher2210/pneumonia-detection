import streamlit as st

# Streamlit Page Configuration must be the first Streamlit command
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# Rest of imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import tensorflow as tf
import gdown

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

# Alternative simple attention visualization
def create_simple_attention_map(img_array, model):
    """
    Create a simple saliency map by taking the gradient of the output with respect to the input image.
    This is a simpler alternative to GradCAM that works with any model architecture.
    """
    # Convert input to tensor if it's not already
    if not isinstance(img_array, tf.Tensor):
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    else:
        img_tensor = img_array
    
    # Make sure it's the right shape (batch, height, width, channels)
    if len(img_tensor.shape) < 4:
        img_tensor = tf.expand_dims(img_tensor, 0)
    
    # Create a GradientTape to track operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Watch the input tensor
        tape.watch(img_tensor)
        # Get the prediction (assuming binary classification)
        pred = model(img_tensor)
        # Get the output for the positive class (pneumonia)
        class_output = pred[:, 0]
    
    # Get the gradient of the output with respect to the input
    grads = tape.gradient(class_output, img_tensor)
    
    # Take the absolute value of the gradients (we care about magnitude, not direction)
    grads = tf.abs(grads)
    
    # Reduce across the color channels to get a single map
    grads = tf.reduce_mean(grads, axis=-1)
    
    # Normalize to [0, 1]
    grads = grads / tf.reduce_max(grads)
    
    # Get as numpy for easier handling
    attention_map = grads[0].numpy()
    
    return attention_map

# Function to overlay heatmap on image - UPDATED for red coloring
def overlay_heatmap(img, heatmap, alpha=0.4):
    # Convert PIL Image to numpy array if it's not already
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img
    
    # Resize heatmap to match input image size
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img = heatmap_img.resize((img_array.shape[1], img_array.shape[0]))
    heatmap = np.array(heatmap_img)
    
    # Apply RED colormap to heatmap (instead of jet which is blue->red)
    # Using a custom colormap that goes from black to red
    colormap = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    # R channel - increases with intensity
    colormap[:, :, 0] = heatmap  
    # G & B channels - lower values for more reddish appearance
    colormap[:, :, 1] = np.zeros_like(heatmap)
    colormap[:, :, 2] = np.zeros_like(heatmap)
    
    # Overlay heatmap on original image
    overlay = np.uint8(colormap * alpha + img_array * (1 - alpha))
    
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

# Function to Classify Image and generate visualization
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
        
        # Try to generate visualization
        try:
            # Use simpler alternative visualization
            heatmap = create_simple_attention_map(img_array_batch, model)
            
            if heatmap is not None:
                # Create heatmap overlay on original image
                overlay_img = overlay_heatmap(img_resized, heatmap)
            else:
                overlay_img = None
        except Exception as viz_error:
            st.warning(f"Visualization could not be generated: {viz_error}")
            heatmap = None
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
                    
                    # Display visualization if available
                    if results['overlay_img'] is not None:
                        st.subheader("Attention Visualization")
                        st.image(results['overlay_img'], caption="Heatmap of areas influencing the model's decision", use_column_width=True)
                        st.info("The highlighted areas (red) show regions the model focused on when making its prediction.")
                    
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
    3. View the results, confidence score, and visualization
    
    **About the Visualization**: The heatmap highlights areas that most influenced the model's decision. Red colors indicate regions with higher importance for the diagnosis.
    
    Note: This app works best with properly oriented, front-view chest X-ray images.
    """)

with st.expander("About the model"):
    st.markdown("""
    This application uses a convolutional neural network (CNN) trained on a dataset of chest X-ray images.
    The model was trained to distinguish between normal chest X-rays and those showing signs of pneumonia.
    
    The visualization helps make the model's decision process more transparent by showing which regions of the image contributed most to the classification.
    
    Please note that this is an educational tool and should not replace professional medical diagnosis.
    Always consult with a healthcare professional for medical advice.
    """)
