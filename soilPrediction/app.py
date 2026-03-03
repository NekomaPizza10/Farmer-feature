# ============================================================
# SmartAgro — app.py  (Flask Backend Server)
#
# HOW TO RUN:
#   python app.py
#   → Server starts at  http://localhost:5000
#
# API ENDPOINT:
#   POST /predict
#   Body: multipart/form-data  with key "image" = image file
#   Returns: JSON with soil analysis results
# ============================================================

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

# Allow requests from the frontend HTML file
# (CORS = Cross-Origin Resource Sharing — needed when
#  frontend and backend run on different ports)
CORS(app)


# ─────────────────────────────────────────
# CONFIG — change model path if needed
# ─────────────────────────────────────────
MODEL_PATH = "soil_model.h5"
IMG_SIZE   = (224, 224)

# Class names — must match the folder names used during training (sorted A-Z)
CLASS_NAMES = ["Black", "Clay", "Loam", "Sandy"]


# ─────────────────────────────────────────
# SOIL KNOWLEDGE BASE
# Maps each soil class → detailed properties
# Edit this dictionary to add more crops or adjust values.
# ─────────────────────────────────────────
SOIL_INFO = {
    "Black": {
        "also_known_as": "Regur / Cotton Soil",
        "fertility":     "High",
        "ph_range":      "7.5 – 8.5  (Slightly Alkaline)",
        "moisture":      "High (retains water well)",
        "texture":       "Fine, sticky when wet, cracks when dry",
        "organic_matter":"High",
        "best_crops":    ["Cotton", "Wheat", "Sorghum", "Sunflower", "Chickpea"],
        "improvement":   "Add gypsum to reduce alkalinity. Ensure good drainage to prevent waterlogging.",
    },
    "Clay": {
        "also_known_as": "Heavy Clay Soil",
        "fertility":     "High",
        "ph_range":      "6.0 – 7.0  (Neutral)",
        "moisture":      "High (drains slowly)",
        "texture":       "Smooth, sticky, very dense",
        "organic_matter":"Medium–High",
        "best_crops":    ["Rice", "Wheat", "Sugarcane", "Cabbage", "Broccoli"],
        "improvement":   "Mix in compost or sand to improve drainage. Avoid tilling when wet.",
    },
    "Loam": {
        "also_known_as": "Garden / Ideal Soil",
        "fertility":     "High",
        "ph_range":      "6.0 – 7.0  (Neutral, Ideal)",
        "moisture":      "Medium (balanced drainage)",
        "texture":       "Crumbly, soft, easy to work with",
        "organic_matter":"High",
        "best_crops":    ["Corn", "Tomato", "Pepper", "Soybean", "Strawberry", "Most vegetables"],
        "improvement":   "Maintain with regular compost. Ideal soil — minimal intervention needed.",
    },
    "Sandy": {
        "also_known_as": "Light / Coarse Soil",
        "fertility":     "Low",
        "ph_range":      "5.5 – 6.5  (Slightly Acidic)",
        "moisture":      "Low (drains very fast)",
        "texture":       "Gritty, loose, flows freely",
        "organic_matter":"Low",
        "best_crops":    ["Carrot", "Potato", "Peanut", "Watermelon", "Cassava"],
        "improvement":   "Add heavy compost or clay to improve water retention. Water more frequently.",
    },
}


# ─────────────────────────────────────────
# LOAD MODEL ONCE when server starts
# (loading inside the endpoint would be slow)
# ─────────────────────────────────────────
print("⏳ Loading soil model...")

if not os.path.exists(MODEL_PATH):
    print(f"⚠️  WARNING: {MODEL_PATH} not found.")
    print("   Run  python train_model.py  first to generate the model.")
    model = None
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")


# ─────────────────────────────────────────
# HELPER — preprocess an uploaded image
# ─────────────────────────────────────────
def preprocess_image(file_bytes):
    """
    Reads image bytes → resizes to 224x224 → normalizes → returns numpy array.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0        # normalize 0-1
    arr = np.expand_dims(arr, axis=0)  # add batch dimension: (1, 224, 224, 3)
    return arr


# ─────────────────────────────────────────
# ROUTE: Health check
# Visit http://localhost:5000/  to confirm server is running
# ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "SmartAgro Soil API is alive 🌱",
        "model_loaded": model is not None
    })


# ─────────────────────────────────────────
# ROUTE: Soil Prediction
# POST http://localhost:5000/predict
# ─────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():

    # 1. Check that an image was actually sent
    if "image" not in request.files:
        return jsonify({"error": "No image file sent. Use key 'image' in form-data."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename — please select an image."}), 400

    # 2. Check model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    try:
        # 3. Read and preprocess the image
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)

        # 4. Run prediction
        predictions = model.predict(img_array)[0]  # shape: (num_classes,)

        # 5. Get top prediction
        top_index      = int(np.argmax(predictions))
        top_class      = CLASS_NAMES[top_index]
        confidence     = float(predictions[top_index]) * 100  # e.g. 87.3

        # 6. Get all class confidence scores (for display in frontend)
        all_scores = {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 1)
            for i in range(len(CLASS_NAMES))
        }

        # 7. Look up soil properties
        info = SOIL_INFO[top_class]

        # 8. Build and return JSON response
        return jsonify({
            "predicted_class": top_class,
            "confidence":      round(confidence, 1),
            "all_scores":      all_scores,
            "soil_name":       info["also_known_as"],
            "fertility":       info["fertility"],
            "ph_range":        info["ph_range"],
            "moisture":        info["moisture"],
            "texture":         info["texture"],
            "organic_matter":  info["organic_matter"],
            "best_crops":      info["best_crops"],
            "improvement":     info["improvement"],
        })

    except Exception as e:
        # Return error message if anything goes wrong
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# START SERVER
# debug=True → auto-restarts when you save app.py (great for development)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌾 SmartAgro Soil API starting...")
    print("   Open http://localhost:5000 to check status\n")
    app.run(debug=True, port=5000)