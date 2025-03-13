import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
import gdown

# ✅ Set page config
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# ✅ Google Drive Model File ID
FILE_ID = "1hWt04b_JaqN8THcCXMT8cHdPsJ4iq3Of"
MODEL_PATH = "pneumonia_model.h5"

# ✅ Function to download model if not available
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")

# ✅ Ensure model is available
download_model()

# ✅ Load the trained model
try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ✅ Streamlit UI
st.title("🩺 Pneumonia Detection from X-ray Images")
st.write("Upload a chest X-ray image to check for pneumonia.")

# ✅ File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing image..."):
        try:
            # ✅ Read the uploaded file properly
            file_bytes = uploaded_file.getvalue()  # <-- Fixed line
            
            if file_bytes:
                # Convert to NumPy array and decode
                file_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img is None:
                    st.error("❌ Error: Could not decode image. Please upload a valid image.")
                else:
                    st.success("✅ Image uploaded successfully!")
                    st.image(img, caption="Uploaded X-ray", use_container_width=True)
            else:
                st.error("❌ Error: No image data found. Try re-uploading.")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

# ✅ Function to preprocess image
def preprocess_image(img, img_size=(150, 150)):
    """Loads and preprocesses an image for the model."""
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    img_array = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img_array

# ✅ Function to create a class activation map using occlusion sensitivity
def create_occlusion_map(img_array, model, window_size=10, stride=5):
    """
    Creates a heatmap showing which areas of the image most affect the prediction.
    Uses an occlusion sensitivity approach by masking parts of the image.
    """
    # Get original prediction and class
    original_pred = model.predict(img_array)[0][0]
    is_pneumonia = original_pred > 0.5
    
    # Get image dimensions
    height, width, _ = img_array[0].shape
    
    # Create an empty heatmap
    heatmap = np.zeros((height, width))
    
    # Create a copy of the image for occlusion
    for y in range(0, height - window_size, stride):
        for x in range(0, width - window_size, stride):
            # Create a copy with a masked region
            masked_img = img_array.copy()
            
            # Apply mask (gray box)
            masked_img[0, y:y+window_size, x:x+window_size, :] = 0.5
            
            # Get new prediction
            masked_pred = model.predict(masked_img)[0][0]
            
            # Calculate the effect of masking this region
            if is_pneumonia:
                # For pneumonia: if masking reduces confidence, it's important
                importance = original_pred - masked_pred
            else:
                # For normal: if masking increases confidence of pneumonia, it's important
                importance = masked_pred - original_pred
            
            # Update the heatmap - larger values mean more important regions
            heatmap[y:y+window_size, x:x+window_size] += importance
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap

# ✅ Streamlit UI
st.title("🩺 Pneumonia Detection from X-ray Images")
st.write("Upload a chest X-ray image to check for pneumonia.")

# ✅ File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing image..."):
        # ✅ Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ✅ Preprocess image
        img_array = preprocess_image(img_rgb)

        # ✅ Get prediction
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])  # Adjust based on your model output
        
        # Determine label and create appropriate color for visualization
        if confidence > 0.5:
            label = "Pneumonia Detected"
            severity = min(confidence * 1.5, 1.0)  # Scale up for visualization
            color = (255, 0, 0)  # Red for pneumonia
        else:
            label = "Normal"
            severity = 0.3  # Lower baseline for normal cases
            color = (0, 255, 0)  # Green for normal

        # Show results
        st.subheader("Prediction Result")
        st.write(f"**{label}**")

        try:
            # Create occlusion sensitivity map
            st.write("Generating visualization - please wait...")
            heatmap = create_occlusion_map(img_array, model)
            
            # Resize images for display
            img_resized = cv2.resize(img_rgb, (150, 150))
            heatmap_resized = cv2.resize(heatmap, (150, 150))
            
            # Convert heatmap to color
            heatmap_resized = np.uint8(255 * heatmap_resized)
            
            # Use color based on prediction (red for pneumonia, green for normal)
            colored_heatmap = np.zeros((heatmap_resized.shape[0], heatmap_resized.shape[1], 3), dtype=np.uint8)
            
            # Set the appropriate color channel based on prediction
            if confidence > 0.5:
                # Red channel for pneumonia
                colored_heatmap[:,:,0] = heatmap_resized
            else:
                # Green channel for normal
                colored_heatmap[:,:,1] = heatmap_resized
            
            # Create overlay
            superimposed_img = cv2.addWeighted(img_resized, 0.6, colored_heatmap, 0.4, 0)
            
            # Display images
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption="Uploaded X-ray", use_container_width=True)
            with col2:
                if confidence > 0.5:
                    caption = "Highlighted areas indicate pneumonia"
                else:
                    caption = "Healthy lung regions"
                st.image(superimposed_img, caption=caption, use_container_width=True)
                
            # Download option
            _, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
            st.download_button(
                label="Download Analysis",
                data=buffer.tobytes(),
                file_name="pneumonia_analysis.png",
                mime="image/png"
            )
            
            # Add explanation
            st.subheader("Analysis Explanation")
            if confidence > 0.5:
                st.write("""
                **Pneumonia detected:** The highlighted areas (in red) show regions that 
                strongly influenced the model's decision. These are typically areas with 
                opacities or consolidations consistent with pneumonia.
                """)
            else:
                st.write("""
                **Normal lungs detected:** The highlighted areas (in green) show regions 
                that influenced the model's decision. For normal X-rays, these areas tend 
                to show clear lung fields without significant opacities.
                """)
            
        except Exception as e:
            st.error(f"❌ Error generating visualization: {e}")
            # Just display the image without visualization
            st.image(img_rgb, caption="Uploaded X-ray", use_container_width=True)
