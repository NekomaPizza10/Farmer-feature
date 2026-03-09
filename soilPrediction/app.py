# ============================================================
# SmartAgro — Unified Soil Prediction App (Hybrid Mode)
#
# HOW TO RUN:
#   python app.py
#   → App starts at  http://localhost:7860
#
# WHAT THIS DOES:
#   Dynamically switches between Local TensorFlow model and
#   Hugging Face API based on USE_LOCAL_MODEL config flag.
#
# CONFIG:
#   Set USE_LOCAL_MODEL = True  → Uses local soil_model.h5
#   Set USE_LOCAL_MODEL = False → Uses Hugging Face API
# ============================================================

import os
import io
import base64
import requests
import numpy as np
import gradio as gr
from PIL import Image
from pathlib import Path

# ============================================================
# CONFIGURATION — Change this flag to switch modes
# ============================================================
USE_LOCAL_MODEL = True  # ← Set to False for Hugging Face API

# Hugging Face API Config (only used if USE_LOCAL_MODEL = False)
HF_SOIL_MODEL_ID = "vinitha003/soil_classification_prediction"
HF_SOIL_API_URL = "https://vinitha003-soil-classification-prediction-67f427e.hf.space/run/predict"

# Local Model Config (only used if USE_LOCAL_MODEL = True)
MODEL_PATH = "soil_model.h5"
IMG_SIZE = (224, 224)

# ============================================================
# LOAD ENVIRONMENT (for HF API mode)
# ============================================================
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded .env from {env_path}")
except Exception as e:
    print(f"⚠️  Could not load .env: {e}")

HF_API_KEY = os.getenv("HF_API_KEY")

# ============================================================
# SOIL KNOWLEDGE BASE (Shared for both modes)
# ============================================================
CLASS_NAMES = ["Black", "Clay", "Loam", "Sandy"]

SOIL_INFO = {
    "Black": {
        "also_known_as": "Regur / Cotton Soil",
        "fertility": "High",
        "ph_range": "7.5 – 8.5  (Slightly Alkaline)",
        "moisture": "High (retains water well)",
        "texture": "Fine, sticky when wet, cracks when dry",
        "organic_matter": "High",
        "best_crops": ["Cotton", "Wheat", "Sorghum", "Sunflower", "Chickpea"],
        "improvement": "Add gypsum to reduce alkalinity. Ensure good drainage to prevent waterlogging.",
    },
    "Clay": {
        "also_known_as": "Heavy Clay Soil",
        "fertility": "High",
        "ph_range": "6.0 – 7.0  (Neutral)",
        "moisture": "High (drains slowly)",
        "texture": "Smooth, sticky, very dense",
        "organic_matter": "Medium–High",
        "best_crops": ["Rice", "Wheat", "Sugarcane", "Cabbage", "Broccoli"],
        "improvement": "Mix in compost or sand to improve drainage. Avoid tilling when wet.",
    },
    "Loam": {
        "also_known_as": "Garden / Ideal Soil",
        "fertility": "High",
        "ph_range": "6.0 – 7.0  (Neutral, Ideal)",
        "moisture": "Medium (balanced drainage)",
        "texture": "Crumbly, soft, easy to work with",
        "organic_matter": "High",
        "best_crops": ["Corn", "Tomato", "Pepper", "Soybean", "Strawberry", "Most vegetables"],
        "improvement": "Maintain with regular compost. Ideal soil — minimal intervention needed.",
    },
    "Sandy": {
        "also_known_as": "Light / Coarse Soil",
        "fertility": "Low",
        "ph_range": "5.5 – 6.5  (Slightly Acidic)",
        "moisture": "Low (drains very fast)",
        "texture": "Gritty, loose, flows freely",
        "organic_matter": "Low",
        "best_crops": ["Carrot", "Potato", "Peanut", "Watermelon", "Cassava"],
        "improvement": "Add heavy compost or clay to improve water retention. Water more frequently.",
    },
}

# ============================================================
# LOAD LOCAL MODEL (if using local mode)
# ============================================================
model = None
if USE_LOCAL_MODEL:
    try:
        import tensorflow as tf
        print(f"🔄 Loading local model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Local model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
        print(f"⚠️  Falling back to Hugging Face API mode")
        USE_LOCAL_MODEL = False
else:
    if not HF_API_KEY:
        print("⚠️  WARNING: HF_API_KEY not set. Hugging Face API will fail.")
        print("   Set it with: $env:HF_API_KEY = 'your_key_here'")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def map_soil_class(hf_label):
    """Map HF model output to our 4 soil classes."""
    label_lower = hf_label.lower()
    if "black" in label_lower:
        return "Black"
    elif "clay" in label_lower:
        return "Clay"
    elif "loam" in label_lower or "alluvial" in label_lower or "red" in label_lower:
        return "Loam"
    elif "sand" in label_lower or "desert" in label_lower or "laterite" in label_lower:
        return "Sandy"
    else:
        return "Loam"  # Default fallback


def predict_local(image):
    """
    Predict using local TensorFlow model.
    Returns: (predicted_class, confidence, all_probabilities)
    """
    if model is None:
        raise RuntimeError("Local model not loaded")
    
    # Preprocess
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")
    
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    predicted_class = CLASS_NAMES[predicted_class_idx]
    all_probs = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence, all_probs


def predict_hf_api(image):
    """
    Predict using Hugging Face Gradio Space API.
    Returns: (predicted_class, confidence, all_probabilities=None)
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set")
    
    # Convert image to bytes
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")
    
    img_io = io.BytesIO()
    img.save(img_io, format="JPEG")
    img_bytes = img_io.getvalue()
    
    # Encode to base64
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    img_data_uri = f"data:image/jpeg;base64,{img_b64}"
    
    # Call API
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"data": [img_data_uri]}
    
    response = requests.post(
        HF_SOIL_API_URL,
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"HF API error: {response.text[:200]}")
    
    results = response.json()
    
    if "data" not in results or not isinstance(results["data"], list) or len(results["data"]) == 0:
        raise RuntimeError("Invalid API response format")
    
    # Parse result
    top_prediction = results["data"][0]
    if isinstance(top_prediction, dict):
        hf_label = top_prediction.get("label", str(top_prediction))
        confidence = float(top_prediction.get("score", 0.9))
    else:
        hf_label = str(top_prediction)
        confidence = 0.9
    
    predicted_class = map_soil_class(hf_label)
    
    return predicted_class, confidence, None  # No probabilities from HF API


# ============================================================
# MAIN PREDICTION FUNCTION (Gradio interface)
# ============================================================
def predict_soil(image):
    """
    Unified prediction function.
    Automatically switches between Local Model and HF API based on config.
    """
    if image is None:
        return "Please upload an image."
    
    try:
        # Determine which mode to use
        current_mode = "Local Model" if USE_LOCAL_MODEL and model is not None else "Hugging Face API"
        
        if USE_LOCAL_MODEL and model is not None:
            predicted_class, confidence, all_probs = predict_local(image)
        else:
            if not HF_API_KEY:
                return "❌ Error: HF_API_KEY not set. Please set your Hugging Face API key."
            predicted_class, confidence, all_probs = predict_hf_api(image)
        
        # Get soil info
        info = SOIL_INFO[predicted_class]
        
        # Build output
        output = f"""
## 🌱 Soil Analysis Results

**Mode:** {current_mode}  
**Predicted Soil Type:** {predicted_class}  
**Confidence:** {confidence*100:.1f}%

---

### 📊 Soil Properties

| Property | Value |
|----------|-------|
| **Also Known As** | {info['also_known_as']} |
| **Fertility** | {info['fertility']} |
| **pH Range** | {info['ph_range']} |
| **Moisture** | {info['moisture']} |
| **Texture** | {info['texture']} |
| **Organic Matter** | {info['organic_matter']} |

---

### 🌾 Best Crops
{', '.join(info['best_crops'])}

---

### 💡 Improvement Tips
{info['improvement']}
"""
        
        # Add probabilities if available (local model only)
        if all_probs:
            output += "\n---\n\n### 📈 All Class Probabilities\n"
            for class_name in CLASS_NAMES:
                prob = all_probs[class_name]
                bar = "█" * int(prob * 20)
                output += f"\n- **{class_name}:** {prob*100:.1f}% {bar}"
        
        return output
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"


# ============================================================
# GRADIO INTERFACE
# ============================================================
mode_text = "Local TensorFlow Model" if USE_LOCAL_MODEL else "Hugging Face API"

# Determine description based on mode
if USE_LOCAL_MODEL:
    description = """
    Upload a photo of soil and our AI will analyze it to determine the soil type.
    
    **Current Mode:** 🖥️ Local TensorFlow Model (`soil_model.h5`)
    
    **Supported soil types:** Black, Clay, Loam, Sandy
    
    The model provides detailed information about fertility, pH, moisture, best crops to grow, and improvement recommendations.
    
    *To switch to Hugging Face API mode, edit `app.py` and set `USE_LOCAL_MODEL = False`*
    """
else:
    description = """
    Upload a photo of soil and our AI will analyze it to determine the soil type.
    
    **Current Mode:** 🤗 Hugging Face API (`vinitha003/soil_classification_prediction`)
    
    **Supported soil types:** Black, Clay, Loam, Sandy
    
    The model provides detailed information about fertility, pH, moisture, best crops to grow, and improvement recommendations.
    
    *To switch to Local Model mode, edit `app.py` and set `USE_LOCAL_MODEL = True`*
    """

demo = gr.Interface(
    fn=predict_soil,
    inputs=gr.Image(type="pil", label="Upload Soil Image"),
    outputs=gr.Markdown(label="Soil Analysis"),
    title=f"🌾 SmartAgro Soil Classification ({mode_text})",
    description=description,
    examples=[],
    theme=gr.themes.Soft()
)

# ============================================================
# START APP
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌾 SmartAgro Soil Classification App")
    print("="*60)
    print(f"   Mode: {mode_text}")
    print(f"   Local Model Loaded: {model is not None}")
    print(f"   HF_API_KEY Set: {HF_API_KEY is not None}")
    print("-"*60)
    
    if USE_LOCAL_MODEL and model is None:
        print("⚠️  WARNING: Local model failed to load.")
        print("   The app will try to use Hugging Face API if key is set.")
    
    if not USE_LOCAL_MODEL and not HF_API_KEY:
        print("⚠️  WARNING: HF_API_KEY not set!")
        print("   Set it with: $env:HF_API_KEY = 'hf_your_key_here'")
    
    print("\n   Open http://localhost:7860 to use the app\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
