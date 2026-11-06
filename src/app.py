"""
Jewelry Vision: Fine-Grained Jewelry Recognition & Caption Generation Web Server
Author: Ragchaa Battsengel (@whysari)
License: MIT
"""

import os
import sys
import pickle
import pathlib
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("JewelryVision")

# Suppress TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Resolve base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from modelFunctions import CNNModel, RNNModel
import dataFunctions

app = Flask(
    __name__,
    template_folder=os.path.join(SRC_DIR, "templates"),
    static_folder=os.path.join(SRC_DIR, "static")
)
app.config["UPLOAD_FOLDER"] = os.path.join(SRC_DIR, "static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Model configuration mappings
MODEL_CONFIGS = {
    "1": {
        "name": "vgg16_gru_True_50_256_16",
        "level_dir": "General_Accessories",
        "cnn": "vgg16",
        "description": "General Accessories (Fast Baseline)"
    },
    "2": {
        "name": "inception_gru_True_50_512_8",
        "level_dir": "Medium",
        "cnn": "inception",
        "description": "Medium Detail (Material + Category)"
    },
    "3": {
        "name": "vgg16_gru_False_50_256_16",
        "level_dir": "Complete",
        "cnn": "vgg16",
        "description": "Complete Fine-Grained (High Precision - 91.18% CCR)"
    }
}

# In-memory model cache
loaded_models = {}

def get_loaded_pipeline(level: str):
    """Lazily load and cache models and tokenizers for the requested hierarchy level."""
    if level in loaded_models:
        return loaded_models[level]

    if level not in MODEL_CONFIGS:
        raise ValueError(f"Unknown level: {level}")

    cfg = MODEL_CONFIGS[level]
    level_dir = cfg["level_dir"]
    model_name = cfg["name"]
    cnn_type = cfg["cnn"]

    logger.info(f"Loading Pipeline Level {level} ({cfg['description']})...")

    # 1. Dataset captions for vocabulary max length
    captions_file = os.path.join(BASE_DIR, "input", "captions.txt")
    if not os.path.exists(captions_file):
        captions_file = os.path.join(BASE_DIR, "input", level_dir, "captions.txt")
    
    if os.path.exists(captions_file):
        _, max_length = dataFunctions.getData(captions_file)
    else:
        max_length = 40  # Safe default max length

    # 2. CNN Encoder
    encoder = CNNModel(cnn_type)

    # 3. RNN Decoder
    model_path = os.path.join(BASE_DIR, "models", level_dir, f"model_{model_name}.keras")
    import tensorflow as tf
    from tensorflow import keras
    
    rnn = RNNModel(max_length=max_length)
    if os.path.exists(model_path):
        rnn.set_model(keras.models.load_model(model_path, compile=False))
        logger.info(f"Loaded weights from {model_path}")
    else:
        logger.warning(f"Model file not found at {model_path}")

    # 4. Tokenizer mappings
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
    loaded_models[level] = pipeline
    return pipeline


def translate_text(text: str, src="auto", dest="en") -> str:
    """Translate text with graceful fallback if offline."""
    try:
        from googletrans import Translator
        translator = Translator()
        return translator.translate(text, src=src, dest=dest).text
    except Exception as e:
        logger.debug(f"Translation skipped (offline/error): {e}")
        return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    level = request.form.get("level", "3")
    if level not in MODEL_CONFIGS:
        return jsonify({"error": f"Invalid level: {level}. Choose 1, 2, or 3."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        pipeline = get_loaded_pipeline(level)
        encoder = pipeline["encoder"]
        rnn = pipeline["rnn"]
        w2i = pipeline["wordtoidx"]
        i2w = pipeline["idxtoword"]

        # Load & encode image via CNN
        image = Image.open(save_path).convert("RGB")
        image_encoded = encoder.encode_image(image)

        # Generate caption via GRU
        raw_caption = rnn.generate_caption(image_encoded, w2i, i2w)

        # Clean caption tokens
        clean_caption = raw_caption.replace("startcap", "").replace("endcap", "").strip()

        # Provide English and Mongolian versions
        english_caption = translate_text(clean_caption, src="auto", dest="en")

        return jsonify({
            "image_path": filename,
            "caption": english_caption if english_caption else clean_caption,
            "raw_caption": clean_caption,
            "level": level,
            "model": pipeline["config"]["description"]
        })

    except Exception as exc:
        logger.exception("Inference failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Jewelry Vision server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
