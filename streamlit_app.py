"""
Jewelry Vision: Streamlit Web Application
Author: Ragchaa Battsengel (@whysari)
License: MIT
"""

import os
import sys
import pickle
import streamlit as st
from PIL import Image
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from modelFunctions import CNNModel, RNNModel
import dataFunctions

st.set_page_config(
    page_title="Jewelry Vision - AI Caption Generator",
    page_icon="💎",
    layout="wide"
)

st.title("💎 Jewelry Vision: Fine-Grained Recognition & Caption Generation")
st.markdown(
    "Multi-Modal Deep Learning System (VGG-16 / InceptionV3 + GRU Decoder) · "
    "**91.18% CCR** · Developed by [Ragchaa Battsengel (@whysari)](https://github.com/whysari/jewelry-vision)"
)

# Sidebar configurations
st.sidebar.header("⚙️ Model Configuration")
level_choice = st.sidebar.radio(
    "Description Granularity Level",
    (
        "Level 3: Complete Fine-Grained (91.18% CCR)",
        "Level 2: Material & Category (Gold vs. Silver)",
        "Level 1: General Category (Fast Baseline)"
    )
)

MODEL_CONFIGS = {
    "Level 3: Complete Fine-Grained (91.18% CCR)": {
        "level_dir": "Complete",
        "name": "vgg16_gru_False_50_256_16",
        "cnn": "vgg16",
        "details": "VGG-16 (4,096-dim) Visual Encoder + 256-unit GRU Decoder. Fine-grained details with 91.18% Correct Classification Rate."
    },
    "Level 2: Material & Category (Gold vs. Silver)": {
        "level_dir": "Medium",
        "name": "inception_gru_True_50_512_8",
        "cnn": "inception",
        "details": "InceptionV3 (2,048-dim) Visual Encoder + 512-unit GRU Decoder with fastText embeddings. Detects metal and category."
    },
    "Level 1: General Category (Fast Baseline)": {
        "level_dir": "General_Accessories",
        "name": "vgg16_gru_True_50_256_16",
        "cnn": "vgg16",
        "details": "VGG-16 Visual Encoder + 256-unit GRU Decoder. Coarse categorization across necklaces, earrings, rings, and bracelets."
    }
}

@st.cache_resource
def load_pipeline(level_key: str):
    cfg = MODEL_CONFIGS[level_key]
    level_dir = cfg["level_dir"]
    model_name = cfg["name"]
    cnn_type = cfg["cnn"]

    encoder = CNNModel(cnn_type)

    captions_file = os.path.join(BASE_DIR, "input", "captions.txt")
    if os.path.exists(captions_file):
        _, max_length = dataFunctions.getData(captions_file)
    else:
        max_length = 40

    from tensorflow import keras
    model_path = os.path.join(BASE_DIR, "models", level_dir, f"model_{model_name}.keras")
    rnn = RNNModel(max_length=max_length)
    if os.path.exists(model_path):
        rnn.set_model(keras.models.load_model(model_path, compile=False))

    idxtoword_path = os.path.join(BASE_DIR, "models", level_dir, f"idxtoword_{model_name}.pk1")
    wordtoidx_path = os.path.join(BASE_DIR, "models", level_dir, f"wordtoidx_{model_name}.pk1")

    with open(idxtoword_path, "rb") as f:
        idxtoword = pickle.load(f)
    with open(wordtoidx_path, "rb") as f:
        wordtoidx = pickle.load(f)

    return {
        "encoder": encoder,
        "rnn": rnn,
        "wordtoidx": wordtoidx,
        "idxtoword": idxtoword,
        "config": cfg
    }

def translate_text(text: str, dest="en") -> str:
    try:
        from googletrans import Translator
        return Translator().translate(text, src="auto", dest=dest).text
    except Exception:
        return text

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Input Jewelry Image")
    uploaded_file = st.file_uploader("Upload JPG or PNG", type=["jpg", "jpeg", "png"])
    
    st.markdown("**Or test with sample images:**")
    sample_cols = st.columns(4)
    samples = [
        ("Necklace", "samples/gold_necklace.jpg"),
        ("Earrings", "samples/diamond_earrings.jpg"),
        ("Bracelet", "samples/bracelet.jpg"),
        ("Climber", "samples/silver_climber_earrings.jpg")
    ]
    
    selected_sample = None
    for idx, (label, path) in enumerate(samples):
        if sample_cols[idx].button(label):
            selected_sample = path

    image_to_process = None
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        st.image(image_to_process, caption="Uploaded Image", use_column_width=True)
    elif selected_sample and os.path.exists(selected_sample):
        image_to_process = Image.open(selected_sample).convert("RGB")
        st.image(image_to_process, caption=f"Sample: {os.path.basename(selected_sample)}", use_column_width=True)

with col2:
    st.subheader("🔬 AI Identification & Descriptions")
    if image_to_process is not None:
        with st.spinner("Analyzing visual features & generating captions..."):
            pipeline = load_pipeline(level_choice)
            encoded_vec = pipeline["encoder"].encode_image(image_to_process)
            raw_caption = pipeline["rnn"].generate_caption(
                encoded_vec,
                pipeline["wordtoidx"],
                pipeline["idxtoword"]
            )
            clean_caption = raw_caption.replace("startcap", "").replace("endcap", "").strip()
            en_caption = translate_text(clean_caption, "en")
            mn_caption = translate_text(clean_caption, "mn")

            st.success("Analysis Complete!")
            st.markdown(f"### 🇬🇧 English Description\n> **{en_caption or clean_caption}**")
            st.markdown(f"### 🇲🇳 Mongolian Caption (Монгол тайлбар)\n> **{mn_caption or clean_caption}**")
            
            with st.expander("Model Specifications & Metrics"):
                st.write(pipeline["config"]["details"])
    else:
        st.info("Upload an image or click a sample button above to run inference.")
