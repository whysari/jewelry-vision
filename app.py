"""
Jewelry Vision: Fine-Grained Jewelry Recognition & Multi-Lingual Caption Generation
Interactive Gradio Web Application for Hugging Face Spaces (100% Free Hosting)

Author: Ragchaa Battsengel (@whysari)
License: MIT
"""

import os
import sys
import pickle
import logging
from PIL import Image
import numpy as np

# Suppress TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from modelFunctions import CNNModel, RNNModel
import dataFunctions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("JewelryVision")

# Model configuration mappings
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

# Cache loaded models
loaded_pipelines = {}

def get_pipeline(level_key: str):
    if level_key in loaded_pipelines:
        return loaded_pipelines[level_key]

    cfg = MODEL_CONFIGS[level_key]
    level_dir = cfg["level_dir"]
    model_name = cfg["name"]
    cnn_type = cfg["cnn"]

    logger.info(f"Loading weights for {level_key}...")

    # 1. CNN Encoder
    encoder = CNNModel(cnn_type)

    # 2. Vocabulary max length
    captions_file = os.path.join(BASE_DIR, "input", "captions.txt")
    if os.path.exists(captions_file):
        _, max_length = dataFunctions.getData(captions_file)
    else:
        max_length = 40

    # 3. RNN Decoder
    import tensorflow as tf
    from tensorflow import keras

    model_path = os.path.join(BASE_DIR, "models", level_dir, f"model_{model_name}.keras")
    rnn = RNNModel(max_length=max_length)
    if os.path.exists(model_path):
        rnn.set_model(keras.models.load_model(model_path, compile=False))

    # 4. Tokenizers
    idxtoword_path = os.path.join(BASE_DIR, "models", level_dir, f"idxtoword_{model_name}.pk1")
    wordtoidx_path = os.path.join(BASE_DIR, "models", level_dir, f"wordtoidx_{model_name}.pk1")

    with open(idxtoword_path, "rb") as f:
        idxtoword = pickle.load(f)
    with open(wordtoidx_path, "rb") as f:
        wordtoidx = pickle.load(f)

    pipeline = {
        "encoder": encoder,
        "rnn": rnn,
        "wordtoidx": wordtoidx,
        "idxtoword": idxtoword,
        "config": cfg
    }
    loaded_pipelines[level_key] = pipeline
    return pipeline


def translate_text(text: str, dest="en") -> str:
    """Safe translation with offline fallback."""
    try:
        from googletrans import Translator
        translator = Translator()
        return translator.translate(text, src="auto", dest=dest).text
    except Exception:
        return text


def predict_jewelry(image, level_choice):
    if image is None:
        return "Please upload an image.", "", ""

    try:
        pipeline = get_pipeline(level_choice)
        encoder = pipeline["encoder"]
        rnn = pipeline["rnn"]
        w2i = pipeline["wordtoidx"]
        i2w = pipeline["idxtoword"]

        # Ensure RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Encode image
        encoded_vec = encoder.encode_image(image)

        # Generate caption
        raw_caption = rnn.generate_caption(encoded_vec, w2i, i2w)
        clean_caption = raw_caption.replace("startcap", "").replace("endcap", "").strip()

        # Multi-lingual translations
        english_caption = translate_text(clean_caption, dest="en")
        mongolian_caption = translate_text(clean_caption, dest="mn")

        return (
            english_caption if english_caption else clean_caption,
            mongolian_caption if mongolian_caption else clean_caption,
            pipeline["config"]["details"]
        )

    except Exception as e:
        logger.exception("Prediction failed")
        return f"Error: {str(e)}", "", ""


# Build Gradio UI
import gradio as gr

custom_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="indigo",
    neutral_hue="slate"
)

title_markdown = """
# 💎 Jewelry Vision: Fine-Grained Recognition & Caption Generation
### Multi-Modal Computer Vision (VGG-16 / InceptionV3) + Autoregressive GRU Decoder
*Developed by [Ragchaa Battsengel (@whysari)](https://github.com/whysari/jewelry-vision) · 91.18% Correct Classification Rate*
"""

description_markdown = """
Upload any photo of fine jewelry (necklaces, earrings, rings, bracelets) to extract deep visual representations and automatically generate fine-grained linguistic descriptions in both **English** and **Mongolian**.
"""

# Sample example images
sample_examples = []
samples_dir = os.path.join(BASE_DIR, "samples")
if os.path.exists(samples_dir):
    for f in ["gold_necklace.jpg", "diamond_earrings.jpg", "bracelet.jpg", "silver_climber_earrings.jpg"]:
        p = os.path.join(samples_dir, f)
        if os.path.exists(p):
            sample_examples.append([p, "Level 3: Complete Fine-Grained (91.18% CCR)"])

with gr.Blocks(theme=custom_theme, title="Jewelry Vision - AI Caption Generator") as demo:
    gr.Markdown(title_markdown)
    gr.Markdown(description_markdown)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📸 Upload Jewelry Image")
            level_selector = gr.Radio(
                choices=list(MODEL_CONFIGS.keys()),
                value="Level 3: Complete Fine-Grained (91.18% CCR)",
                label="⚙️ Description Granularity Level"
            )
            submit_btn = gr.Button("✨ Generate Description", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_en = gr.Textbox(label="🇬🇧 English Caption", lines=2, interactive=False)
            output_mn = gr.Textbox(label="🇲🇳 Mongolian Caption (Монгол тайлбар)", lines=2, interactive=False)
            output_model = gr.Textbox(label="🔬 Active Model Architecture & Spec", lines=3, interactive=False)

    if sample_examples:
        gr.Examples(
            examples=sample_examples,
            inputs=[input_image, level_selector],
            outputs=[output_en, output_mn, output_model],
            fn=predict_jewelry,
            cache_examples=False,
            label="🔍 Click Any Sample Image to Test Instantly"
        )

    submit_btn.click(
        fn=predict_jewelry,
        inputs=[input_image, level_selector],
        outputs=[output_en, output_mn, output_model]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
