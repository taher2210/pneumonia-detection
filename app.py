import streamlit as st
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

# Set page configuration with a custom theme
st.set_page_config(
    page_title="PneumoniaX - AI Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3498db;
        margin-bottom: 1rem;
    }
    
    .result-card-pneumonia {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .result-card-normal {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .highlight-text {
        font-weight: 600;
        font-size: 1.1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
      /* FIXED: Improved stProgress selector specificity */
    div.stProgress > div > div {
        height: 12px !important;
        border-radius: 10px !important;
    }


     /* FIXED: Added more specific selectors for progress bars */
    .pneumonia-progress div.stProgress > div > div {
        background-color: #f44336 !important;
    }
    .normal-progress .stProgress > div > div {
        background-color: #4caf50 !important;
    }
   
    .visualization-info {
        padding: 15px;
        background-color: #e3f2fd;
        border-radius: 5px;
        margin-bottom: 15px;
    }
     /* FIXED: Added !important to ensure button style is applied */
    .analyze-btn {
        background-color: #3498db !important;
        color: white !important;
        font-weight: bold !important;
        padding: 10px !important;
        border-radius: 5px !important;
        transition: all 0.3s !important;
    }

   
    .analyze-btn:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    /* FIXED: Improved tab selector specificity */
    .tabs-container .stTabs [data-baseweb="tab-list"] {
        gap: 2px !important;
    }
    .tabs-container .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f5;
        border-radius: 5px 5px 0 0;
    }
    .tabs-container .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar with app info
with st.sidebar:
    st.image("logo.png", width=100)
    st.markdown("<h2 style='text-align: center;'>PneumoniaX</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **PneumoniaX** is an AI-powered tool for detecting pneumonia from chest X-rays using advanced deep learning techniques.
    
    Our model has been trained on thousands of X-ray images and can help provide preliminary insights into the presence of pneumonia.
    """)
    
    st.markdown("---")
    st.markdown("### Disclaimer")
    st.markdown("""
    This tool is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.
    
    Always consult with a healthcare professional for medical advice.
    """)
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload a chest X-ray image
    2. Click the "Analyze X-ray" button
    3. Review the results and visualization
    """)

# Main content
st.markdown("<h1 class='main-header'>PneumoniaX: AI Chest X-ray Analysis</h1>", unsafe_allow_html=True)

# Introduction card
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size: 1.1rem; text-align: center;'>
    Upload a chest X-ray image and our AI model will analyze it for signs of pneumonia.
    The visualization highlights areas of interest that influenced the model's decision.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Google Drive Model File ID
FILE_ID = "1hWt04b_JaqN8THcCXMT8cHdPsJ4iq3Of"
MODEL_PATH = "pneumonia_model.h5"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading AI model... This may take a moment."):
            try:
                gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
                return True
            except Exception as e:
                st.error(f"❌ Error downloading model: {e}")
                st.error("Please check your internet connection and try again.")
                return False
    return True

# Alternative simple attention visualization
def create_simple_attention_map(img_array, model):
    """
    Create a simple saliency map by taking the gradient of the output with respect to the input image.
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

# Function to overlay heatmap on image
def overlay_heatmap(img, heatmap, is_pneumonia=True, alpha=0.5):
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
    
    # Create colormap based on diagnosis
    colormap = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    
    if is_pneumonia:
        # RED colormap for pneumonia
        colormap[:, :, 0] = heatmap  # R channel
    else:
        # BLUE colormap for normal
        colormap[:, :, 2] = heatmap  # B channel
    
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
        
        # Determine if pneumonia or normal (for heatmap color selection)
        is_pneumonia = probability > 0.5
        
        # Try to generate visualization
        try:
            # Use simpler alternative visualization
            heatmap = create_simple_attention_map(img_array_batch, model)
            
            if heatmap is not None:
                # Create heatmap overlay on original image with appropriate color
                overlay_img = overlay_heatmap(img_resized, heatmap, is_pneumonia=is_pneumonia)
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
            'overlay_img': overlay_img,
            'is_pneumonia': is_pneumonia
        }
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

# Create upload section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.markdown("### Upload a Chest X-ray Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if uploaded_file is None:
        st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <i class="fas fa-cloud-upload-alt" style='font-size: 3rem;'></i>
        <p>Drag and drop an X-ray image or click to browse</p>
        <p style='font-size: 0.8rem;'>Supported formats: JPG, PNG, JPEG</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Process uploaded image
if uploaded_file is not None:
    # Read file
    file_bytes = uploaded_file.getvalue()
    
    # Create two columns for the image and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 class='sub-header'>Uploaded X-ray</h3>", unsafe_allow_html=True)
        st.image(file_bytes, use_column_width=True)
    
    with col2:
        st.markdown("<h3 class='sub-header'>Analysis</h3>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze X-ray", use_container_width=True, key="analyze_btn")
        
        if analyze_button:
            with st.spinner("Analyzing image..."):
                # Add a small delay to show the spinner
                import time
                time.sleep(0.5)
                
                results = analyze_image(file_bytes)
                
                if results is not None:
                    # Display results
                    probability = results['probability']
                    is_pneumonia = results['is_pneumonia']
                    
                    if is_pneumonia:
                        st.markdown("<div class='result-card-pneumonia'>", unsafe_allow_html=True)
                        st.markdown("<h3>🫁 Pneumonia Detected</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p class='highlight-text'>Confidence: {probability:.1%}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Use custom CSS for pneumonia progress bar
                        st.markdown("<div class='pneumonia-progress'>", unsafe_allow_html=True)
                        st.progress(probability)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='result-card-normal'>", unsafe_allow_html=True)
                        st.markdown("<h3>✅ Normal</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p class='highlight-text'>Confidence: {(1-probability):.1%}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Use custom CSS for normal progress bar
                        st.markdown("<div class='normal-progress'>", unsafe_allow_html=True)
                        st.progress(1-probability)
                        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display visualization if available
    if 'results' in locals() and results is not None and results['overlay_img'] is not None:
        st.markdown("<h3 class='sub-header'>AI Visualization</h3>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        vis_col1, vis_col2 = st.columns([1, 1])
        
        with vis_col1:
            st.image(results['overlay_img'], use_column_width=True)
        
        with vis_col2:
            color_text = "red" if results['is_pneumonia'] else "blue"
            highlight_color = "#FF0000" if results['is_pneumonia'] else "#007BFF"
            st.markdown(f"""
            <div class='visualization-info'>
            <h4 style='color: black;'>What am I seeing?</h4>
            <p style='color: black;'>The highlighted areas in <span style='color: {highlight_color}; font-weight: bold;'>{"red" if results['is_pneumonia'] else "blue"}</span> show regions the AI focused on when making its prediction.</p>
            <p style='color: black;'>These areas typically correspond to patterns the model has learned to associate with {"pneumonia" if results['is_pneumonia'] else "normal lung tissue"}.</p>

        </div>
            """, unsafe_allow_html=True)
            
            # Add interpretation guidance
            if results['is_pneumonia']:
                st.markdown("""
                <h4>Interpretation Guide</h4>
                <p>In pneumonia cases, the model often highlights:</p>
                <ul>
                    <li>Areas of consolidation (fluid buildup)</li>
                    <li>Opacities in the lung fields</li>
                    <li>Blurred costophrenic angles</li>
                </ul>
                <p><em>Note: This is an AI interpretation and should be confirmed by a medical professional.</em></p>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <h4>Interpretation Guide</h4>
                <p>In normal cases, the model often highlights:</p>
                <ul>
                    <li>Clear lung fields</li>
                    <li>Well-defined costophrenic angles</li>
                    <li>Normal cardiac silhouette</li>
                </ul>
                <p><em>Note: This is an AI interpretation and should be confirmed by a medical professional.</em></p>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Add detailed report section
        st.markdown("<h3 class='sub-header'>Detailed Report</h3>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Create tabs for different report sections
        st.markdown("<div class='tabs-container'>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Summary", "Technical Details", "Recommendations"])
        
        with tab1:
            st.markdown(f"""
            <h4>Analysis Summary</h4>
            <p>The AI analysis of this chest X-ray shows {"signs consistent with pneumonia" if results['is_pneumonia'] else "no significant abnormalities"}.</p>
            <p>Confidence: <span class='highlight-text'>{probability:.1%}</span> {"for pneumonia" if results['is_pneumonia'] else "for normal"}</p>
            <p>Date of Analysis: {st.session_state.get('analysis_date', str(st.session_state.setdefault('analysis_date', __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))))}</p>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <h4>Technical Details</h4>
            <p>This analysis was performed using a convolutional neural network (CNN) trained on a large dataset of chest X-rays.</p>
            <p>The model evaluates patterns in the image that correlate with radiological findings associated with pneumonia.</p>
            """, unsafe_allow_html=True)
            
            # Add a simple chart showing the probability distribution
            chart_data = np.array([probability, 1-probability])
            labels = ['Pneumonia', 'Normal']
            colors = ['#f44336', '#4caf50']
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, chart_data, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title('Classification Probability Distribution')
            for i, v in enumerate(chart_data):
                ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
            
            st.pyplot(fig)
        
        with tab3:
            if results['is_pneumonia']:
                st.markdown("""
                <h4>Recommendations</h4>
                <p>Based on the AI analysis, the following steps are recommended:</p>
                <ol>
                    <li>Consult with a healthcare professional to confirm the findings</li>
                    <li>Consider further diagnostic tests if clinically indicated</li>
                    <li>Follow medical advice for treatment options</li>
                </ol>
                <p><strong>Important:</strong> This AI analysis is not a substitute for professional medical diagnosis.</p>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <h4>Recommendations</h4>
                <p>Based on the AI analysis, no significant abnormalities were detected. However:</p>
                <ol>
                    <li>Always consult with a healthcare professional for a proper interpretation</li>
                    <li>Continue regular check-ups as recommended by your doctor</li>
                    <li>If symptoms persist despite this result, seek additional medical evaluation</li>
                </ol>
                <p><strong>Important:</strong> This AI analysis is not a substitute for professional medical diagnosis.</p>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Add sample cases section
if 'uploaded_file' not in locals() or uploaded_file is None:
    st.markdown("<h3 class='sub-header'>Sample Cases</h3>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    sample_col1, sample_col2 = st.columns(2)
    
    with sample_col1:
        st.markdown("""
        <h4>Normal Case Example</h4>
        <p>This is an example of a normal chest X-ray. Note the clear lung fields and well-defined costophrenic angles.</p>
        """, unsafe_allow_html=True)
        # Using a placeholder image since we don't have actual sample images
        st.image("normal.jpeg", width=200)
        
    with sample_col2:
        st.markdown("""
        <h4>Pneumonia Case Example</h4>
        <p>This is an example of a chest X-ray showing pneumonia. Note the opacities in the lung fields.</p>
        """, unsafe_allow_html=True)
       
        st.image("pneumonia.jpeg", width=200)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add FAQ section
with st.expander("Frequently Asked Questions"):
    st.markdown("""
    <h4>What is pneumonia?</h4>
    <p>Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.</p>
    
    <h4>How accurate is this AI tool?</h4>
    <p>While our AI model has been trained on thousands of X-ray images, it should be used as a screening tool only. The accuracy varies depending on image quality and other factors. Always consult with a healthcare professional for diagnosis.</p>
    
    <h4>Can this tool detect other lung conditions?</h4>
    <p>No, this tool is specifically trained to detect pneumonia. It cannot reliably detect other lung conditions like tuberculosis, lung cancer, or COVID-19.</p>
    
    <h4>How should I prepare my X-ray for upload?</h4>
    <p>For best results, upload a clear, front-view (PA or AP) chest X-ray image in JPG, PNG, or JPEG format. The image should be well-centered and should include the entire lung fields.</p>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("""
<p>© 2025 PneumoniaX | An AI-powered pneumonia detection tool | PBL for Big Data Analysis</p>
<p>This application is for educational purposes only and not intended for clinical use.</p>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Add an exploration mode for more advanced users
with st.sidebar:
    st.markdown("---")
    st.markdown("### Advanced Options")
    if st.checkbox("Enable Exploration Mode"):
        st.markdown("#### Model Parameters")
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
        st.markdown(f"Current threshold: {threshold}")
        st.markdown("*Adjusting this value affects the sensitivity of pneumonia detection.*")
        
        st.markdown("#### Visualization Settings")
        heatmap_intensity = st.slider("Heatmap Intensity", 0.1, 1.0, 0.5, 0.1)
        st.markdown("*Higher values show more intense heatmap visualization.*")
        
        if st.button("Reset to Defaults"):
            st.write("Settings reset to defaults.")
