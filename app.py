"""
Modern Web UI for U-Net Image Segmentation
==========================================

This Streamlit app provides an interactive interface for:
- Image upload and segmentation
- Model inference and visualization
- Real-time results display
- Model comparison
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import base64
from typing import Tuple, Optional
import time

from unet_model import ModernUNet, calculate_iou, calculate_dice_score
from dataset import SyntheticSegmentationDataset, MedicalImageDataset


# Page configuration
st.set_page_config(
    page_title="U-Net Image Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem;
        font-size: 1rem;
    }
    .stButton > button:hover {
        background-color: #0d5a8a;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str = None) -> ModernUNet:
    """Load the U-Net model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with default parameters
    model = ModernUNet(
        in_channels=3,
        out_channels=1,
        base_channels=64,
        depth=4,
        dropout_rate=0.1
    ).to(device)
    
    # Load weights if available
    if model_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success(f"✅ Model loaded from {model_path}")
        except Exception as e:
            st.warning(f"⚠️ Could not load model from {model_path}: {e}")
            st.info("Using randomly initialized model")
    else:
        st.info("Using randomly initialized model for demo")
    
    model.eval()
    return model


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Preprocess uploaded image for model inference"""
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to tensor and normalize
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def postprocess_mask(mask: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Postprocess model output mask"""
    # Remove batch dimension and convert to numpy
    mask_np = mask.squeeze().cpu().numpy()
    
    # Apply threshold
    mask_binary = (mask_np > threshold).astype(np.uint8)
    
    return mask_binary


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create overlay of mask on original image"""
    # Ensure image is in the right format
    if len(image.shape) == 3 and image.shape[2] == 3:
        overlay = image.copy()
    else:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create colored mask
    colored_mask = np.zeros_like(overlay)
    colored_mask[:, :, 0] = mask * 255  # Red channel
    
    # Blend images
    result = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
    
    return result


def generate_synthetic_image(dataset_type: str = "synthetic") -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic image for demo"""
    if dataset_type == "synthetic":
        dataset = SyntheticSegmentationDataset(num_samples=1, image_size=(256, 256))
    else:
        dataset = MedicalImageDataset(num_samples=1, image_size=(256, 256))
    
    image, mask = dataset[0]
    
    # Convert tensors to numpy
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()
    
    return image_np, mask_np


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 U-Net Image Segmentation</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    use_pretrained = st.sidebar.checkbox("Use Pretrained Model", value=False)
    threshold = st.sidebar.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.01)
    overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.5, 0.01)
    
    # Dataset settings
    st.sidebar.subheader("Demo Dataset")
    dataset_type = st.sidebar.selectbox("Dataset Type", ["synthetic", "medical"])
    
    # Load model
    model_path = "outputs/checkpoint_best.pth" if use_pretrained else None
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Input Image")
        
        # Image upload or generation
        upload_option = st.radio(
            "Choose input method:",
            ["Upload Image", "Generate Synthetic Image"]
        )
        
        if upload_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image for segmentation"
            )
            
            if uploaded_file is not None:
                # Load uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Preprocess for model
                input_tensor = preprocess_image(image)
                
            else:
                st.info("Please upload an image or generate a synthetic one")
                input_tensor = None
                
        else:
            # Generate synthetic image
            if st.button("🎲 Generate New Image"):
                with st.spinner("Generating synthetic image..."):
                    image_np, mask_np = generate_synthetic_image(dataset_type)
                    
                    # Convert to PIL for display
                    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                    st.image(image_pil, caption="Generated Image", use_column_width=True)
                    
                    # Preprocess for model
                    input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
                    input_tensor = input_tensor.unsqueeze(0)
    
    with col2:
        st.subheader("🎯 Segmentation Results")
        
        if input_tensor is not None:
            # Run inference
            if st.button("🚀 Run Segmentation", type="primary"):
                with st.spinner("Running segmentation..."):
                    start_time = time.time()
                    
                    # Move to device
                    input_tensor = input_tensor.to(device)
                    
                    # Run inference
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    inference_time = time.time() - start_time
                    
                    # Postprocess results
                    mask_binary = postprocess_mask(output, threshold)
                    
                    # Convert input back to numpy for visualization
                    input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                    
                    # Create overlay
                    overlay = create_overlay(input_np, mask_binary, overlay_alpha)
                    
                    # Display results
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.image(mask_binary, caption="Segmentation Mask", use_column_width=True)
                    
                    with col2_2:
                        st.image(overlay, caption="Overlay", use_column_width=True)
                    
                    # Calculate metrics (if we have ground truth)
                    if upload_option == "Generate Synthetic Image":
                        # Calculate metrics against ground truth
                        iou = calculate_iou(output, torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0))
                        dice = calculate_dice_score(output, torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0))
                        
                        # Display metrics
                        st.subheader("📊 Performance Metrics")
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("IoU Score", f"{iou:.3f}")
                        
                        with metric_col2:
                            st.metric("Dice Score", f"{dice:.3f}")
                        
                        with metric_col3:
                            st.metric("Inference Time", f"{inference_time:.3f}s")
                    
                    else:
                        # Display inference time only
                        st.subheader("📊 Performance")
                        st.metric("Inference Time", f"{inference_time:.3f}s")
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    
                    # Create downloadable images
                    mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8))
                    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        # Download mask
                        mask_buffer = io.BytesIO()
                        mask_pil.save(mask_buffer, format='PNG')
                        mask_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Mask",
                            data=mask_buffer.getvalue(),
                            file_name="segmentation_mask.png",
                            mime="image/png"
                        )
                    
                    with col_dl2:
                        # Download overlay
                        overlay_buffer = io.BytesIO()
                        overlay_pil.save(overlay_buffer, format='PNG')
                        overlay_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Overlay",
                            data=overlay_buffer.getvalue(),
                            file_name="segmentation_overlay.png",
                            mime="image/png"
                        )
        
        else:
            st.info("Please provide an input image to run segmentation")
    
    # Model information
    st.sidebar.subheader("ℹ️ Model Information")
    st.sidebar.info(f"""
    **Model:** Modern U-Net
    **Parameters:** {sum(p.numel() for p in model.parameters()):,}
    **Device:** {device}
    **Input Size:** 256×256×3
    **Output:** Binary segmentation mask
    """)
    
    # Instructions
    st.sidebar.subheader("📖 Instructions")
    st.sidebar.markdown("""
    1. **Upload an image** or **generate a synthetic one**
    2. **Adjust settings** in the sidebar
    3. **Click 'Run Segmentation'** to process
    4. **View results** and download if needed
    5. **Compare metrics** for synthetic images
    """)


if __name__ == "__main__":
    main()
