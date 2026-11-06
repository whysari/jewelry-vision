"""
Jewelry Vision: Flask Web Application Server (Glassmorphic Custom UI)
Author: Ragchaa Battsengel (@whysari)
License: MIT
"""

import os
import sys
import pickle
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Suppress TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from modelFunctions import CNNModel, RNNModel
import dataFunctions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("JewelryVisionFlask")

app = Flask(
    __name__,
    template_folder=os.path.join(SRC_DIR, "templates"),
    static_folder=os.path.join(SRC_DIR, "static")
)
app.config["UPLOAD_FOLDER"] = os.path.join(SRC_DIR, "static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

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

loaded_models = {}

def get_loaded_pipeline(level: str):
    if level in loaded_models:
        return loaded_models[level]

    cfg = MODEL_CONFIGS[level]
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

    pipeline = {
        "encoder": encoder,
        "rnn": rnn,
        "wordtoidx": wordtoidx,
        "idxtoword": idxtoword,
        "config": cfg
    }
    loaded_models[level] = pipeline
    return pipeline


def translate_text(text: str, dest="en") -> str:
    try:
        from googletrans import Translator
        return Translator().translate(text, src="auto", dest=dest).text
    except Exception:
        return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    level = request.form.get("level", "3")
    if level not in MODEL_CONFIGS:
        return jsonify({"error": f"Invalid level: {level}"}), 400

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
        image = Image.open(save_path).convert("RGB")
        image_encoded = pipeline["encoder"].encode_image(image)
        raw_caption = pipeline["rnn"].generate_caption(image_encoded, pipeline["wordtoidx"], pipeline["idxtoword"])
        clean_caption = raw_caption.replace("startcap", "").replace("endcap", "").strip()
        english_caption = translate_text(clean_caption, dest="en")

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
    logger.info(f"Starting Flask server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
