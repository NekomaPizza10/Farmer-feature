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
import io

# ============================================================
# LOAD ENVIRONMENT — Ensure HF_API_KEY is available FIRST
# ============================================================
try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env from {env_file}")
except Exception as e:
    print(f"⚠️  Could not load .env: {e}")

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    print("⚠️  WARNING: HF_API_KEY not set yet")
    print("   To set it permanently on Windows:")
    print("   [System.Environment]::SetEnvironmentVariable('HF_API_KEY', 'your_key', 'User')")
    print("   Then restart PowerShell/CMD\n")
else:
    print(f"✅ HF_API_KEY loaded: {HF_API_KEY[:20]}...")

from leaf_service import predict_leaf

app = Flask(__name__)

# Allow requests from the frontend HTML file
# (CORS = Cross-Origin Resource Sharing — needed when
#  frontend and backend run on different ports)
CORS(app)


# ─────────────────────────────────────────
# CONFIG — change model path if needed
# ─────────────────────────────────────────
HF_SOIL_MODEL_ID = "vinitha003/soil_classification_prediction"
HF_SOIL_API_URL ="https://vinitha003-soil-classification-prediction-67f427e.hf.space/run/predict"


# Class names — mapped from the Hugging Face model
# Note: Ben041/soil-type-classifier has 11 classes, but we only map the ones we care about
# and default to the closest match if a different one is returned.
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
# HELPER: Map HF model output to our classes
# ─────────────────────────────────────────
def map_soil_class(hf_label):
    label_lower = hf_label.lower()
    if "black" in label_lower:
        return "Black"
    elif "clay" in label_lower:
        return "Clay"
    elif "loam" in label_lower or "alluvial" in label_lower or "red" in label_lower: # map alluvial/red to loam for now
        return "Loam"
    elif "sand" in label_lower or "desert" in label_lower or "laterite" in label_lower:
        return "Sandy"
    else:
        return "Loam" # Default fallback


# ─────────────────────────────────────────
# ROUTE: Health check
# Visit http://localhost:5000/  to confirm server is running
# ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "SmartAgro Soil API is alive 🌱",
        "model_loaded": True # Always true now since we use an external API
    })


# ─────────────────────────────────────────
# ROUTE: Soil Prediction
# POST http://localhost:5000/predict
# ─────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    import requests # Import here or top of file

    # 1. Check that an image was actually sent
    if "image" not in request.files:
        return jsonify({"error": "No image file sent. Use key 'image' in form-data."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename — please select an image."}), 400

    current_api_key = os.getenv("HF_API_KEY")
    if not current_api_key:
         return jsonify({"error": "HF_API_KEY is not set. Cannot use Hugging Face API."}), 503

    try:
        import base64
        # 2. Read the image and encode it as Base64 (required by Gradio Spaces)
        img_bytes = file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        mime_type = file.mimetype or "image/jpeg"
        img_data_uri = f"data:{mime_type};base64,{img_b64}"

        # 3. Call Hugging Face Gradio Space API
        # Gradio /run/predict requires a JSON body with a "data" array
        headers = {
            "Authorization": f"Bearer {current_api_key}",
            "Content-Type": "application/json"
        }
        payload = {"data": [img_data_uri]}
        print(f"🌐 Querying Soil Space API (timeout: 30s)...")
        response = requests.post(
            HF_SOIL_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        print("HF Status:", response.status_code)
        print("HF Response:", response.text[:300])

        if response.status_code != 200:
             return jsonify({"error": f"Hugging Face API error: {response.text[:200]}"}), 502
             
        results = response.json()
        print("Parsed JSON:", results)
        
        if "data" not in results or not isinstance(results["data"], list) or len(results["data"]) == 0:
             return jsonify({"error": "Hugging Face API returned invalid format."}), 502

        # 4. Get top prediction — Gradio returns the label in data[0]
        top_prediction = results["data"][0]
        # top_prediction may be a string like "Sandy" or a dict — handle both
        if isinstance(top_prediction, dict):
            hf_label = top_prediction.get("label", str(top_prediction))
            confidence = float(top_prediction.get("score", 0.9)) * 100
        else:
            hf_label = str(top_prediction)
            confidence = 90.0
        
        # 5. Map to our internal classes
        top_class = map_soil_class(hf_label)

        # 6. Look up soil properties
        info = SOIL_INFO[top_class]

        # 7. Build and return JSON response
        return jsonify({
            "predicted_class": top_class,
            "original_hf_label": hf_label, # Include original label for debugging
            "confidence":      round(confidence, 1),
            "soil_name":       info["also_known_as"],
            "fertility":       info["fertility"],
            "ph_range":        info["ph_range"],
            "moisture":        info["moisture"],
            "texture":         info["texture"],
            "organic_matter":  info["organic_matter"],
            "best_crops":      info["best_crops"],
            "improvement":     info["improvement"],
        })

    except requests.exceptions.Timeout:
         return jsonify({"error": "Hugging Face API timed out."}), 504
    except Exception as e:
        # Return error message if anything goes wrong
        return jsonify({"error": str(e)}), 500

@app.route("/leaf/predict", methods=["POST"])
def leaf_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file sent. Use key 'image' in form-data."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename — please select an image."}), 400

    if not (file.mimetype or "").startswith("image/"):
        return jsonify({"error": f"File must be an image. Got {file.mimetype}"}), 400

    result = predict_leaf(file)

    if "error" in result:
        return jsonify(result), 502

    return jsonify(result)

# ─────────────────────────────────────────
# START SERVER
# debug=True → auto-restarts when you save app.py (great for development)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌾 SmartAgro Soil API starting...")
    print("   Open http://localhost:5000 to check status\n")
    app.run(debug=True, port=5000)