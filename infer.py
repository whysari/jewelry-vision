"""
Jewelry Vision: Standalone CLI Inference Tool
Author: Ragchaa Battsengel (@whysari)
License: MIT

Usage:
  python infer.py --image samples/gold_necklace.jpg --level 3
"""

import os
import sys
import argparse
import pickle
from PIL import Image
import numpy as np

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from modelFunctions import CNNModel, RNNModel
import dataFunctions

MODEL_CONFIGS = {
    "1": {
        "name": "vgg16_gru_True_50_256_16",
        "level_dir": "General_Accessories",
        "cnn": "vgg16",
        "title": "Level 1: General Accessories (Fast Baseline)"
    },
    "2": {
        "name": "inception_gru_True_50_512_8",
        "level_dir": "Medium",
        "cnn": "inception",
        "title": "Level 2: Medium Detail (Material & Category)"
    },
    "3": {
        "name": "vgg16_gru_False_50_256_16",
        "level_dir": "Complete",
        "cnn": "vgg16",
        "title": "Level 3: Complete Fine-Grained (High Precision - 91.18% CCR)"
    }
}


def run_inference(image_path: str, level: str = "3"):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    cfg = MODEL_CONFIGS.get(level, MODEL_CONFIGS["3"])
    print(f"\n=======================================================")
    print(f" Jewelry Vision Inference Engine")
    print(f" Model: {cfg['title']}")
    print(f" Input: {image_path}")
    print(f"=======================================================")

    # 1. CNN Feature Extractor
    print(f"[*] Initializing {cfg['cnn'].upper()} visual encoder...")
    encoder = CNNModel(cfg["cnn"])

    # 2. Load Image
    image = Image.open(image_path).convert("RGB")
    print(f"[*] Extracting visual feature vector ({encoder.get_output_dim()} dimensions)...")
    encoded_feature = encoder.encode_image(image)

    # 3. Load Model & Tokenizer
    model_path = os.path.join(BASE_DIR, "models", cfg["level_dir"], f"model_{cfg['name']}.keras")
    idxtoword_path = os.path.join(BASE_DIR, "models", cfg["level_dir"], f"idxtoword_{cfg['name']}.pk1")
    wordtoidx_path = os.path.join(BASE_DIR, "models", cfg["level_dir"], f"wordtoidx_{cfg['name']}.pk1")

    import tensorflow as tf
    from tensorflow import keras

    print(f"[*] Loading GRU language decoder weights...")
    captions_file = os.path.join(BASE_DIR, "input", "captions.txt")
    if os.path.exists(captions_file):
        _, max_length = dataFunctions.getData(captions_file)
    else:
        max_length = 40

    rnn = RNNModel(max_length=max_length)
    rnn.set_model(keras.models.load_model(model_path, compile=False))

    with open(idxtoword_path, "rb") as f:
        idxtoword = pickle.load(f)
    with open(wordtoidx_path, "rb") as f:
        wordtoidx = pickle.load(f)

    # 4. Generate Caption
    print(f"[*] Generating fine-grained caption via greedy autoregressive decoding...")
    raw_caption = rnn.generate_caption(encoded_feature, wordtoidx, idxtoword)
    caption = raw_caption.replace("startcap", "").replace("endcap", "").strip()

    print(f"\n----------------- PREDICTION RESULTS -----------------")
    print(f"  Generated Caption:  \"{caption}\"")
    
    # Try translation
    try:
        from googletrans import Translator
        t = Translator()
        en_cap = t.translate(caption, src="auto", dest="en").text
        print(f"  English Translation: \"{en_cap}\"")
    except Exception:
        pass
    print(f"------------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jewelry Vision CLI Image Caption Generator")
    parser.add_argument("--image", required=True, help="Path to input jewelry image (JPG/PNG)")
    parser.add_argument("--level", default="3", choices=["1", "2", "3"], help="Hierarchy level (1, 2, or 3)")
    args = parser.parse_args()

    run_inference(args.image, args.level)
