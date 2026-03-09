# ============================================================
# SmartAgro — app.py  (Gradio Soil Prediction App)
#
# HOW TO RUN:
#   python app.py
#   → App starts at  http://localhost:7860
#
# WHAT IT DOES:
#   Loads the local soil_model.h5 and provides a Gradio
#   interface for soil classification.
# ============================================================

import os
import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH = "soil_model.h5"
CLASS_NAMES = ["Black", "Clay", "Loam", "Sandy"]
IMG_SIZE = (224, 224)  # Model expects 224x224 images

# Soil knowledge base (from flask_app_backup.py)
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

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
print(f"🔄 Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def predict_soil(image):
    """
    Predict soil type from an uploaded image.
    Returns a formatted string with all soil information.
    """
    if model is None:
        return "Error: Model not loaded. Please check if soil_model.h5 exists."

    if image is None:
        return "Please upload an image."

    # Preprocess the image
    # Convert to RGB if needed
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")

    # Resize to 224x224 (model's expected input size)
    img = img.resize(IMG_SIZE)

    # Convert to numpy array and normalize (same as training: rescale=1.0/255)
    img_array = np.array(img) / 255.0

    # Add batch dimension (model expects batch of images)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])

    # Get class name
    predicted_class = CLASS_NAMES[predicted_class_idx]

    # Get soil info
    info = SOIL_INFO[predicted_class]

    # Build output
    output = f"""
## 🌱 Soil Analysis Results

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

---

### 📈 All Class Probabilities
"""

    # Add all probabilities
    for i, class_name in enumerate(CLASS_NAMES):
        prob = float(predictions[0][i])
        bar = "█" * int(prob * 20)
        output += f"\n- **{class_name}:** {prob*100:.1f}% {bar}"

    return output


# ─────────────────────────────────────────
# CREATE GRADIO INTERFACE
# ─────────────────────────────────────────
demo = gr.Interface(
    fn=predict_soil,
    inputs=gr.Image(type="pil", label="Upload Soil Image"),
    outputs=gr.Markdown(label="Soil Analysis"),
    title="🌾 SmartAgro Soil Classification",
    description="""
    Upload a photo of soil and our AI will analyze it to determine the soil type.

    **Supported soil types:** Black, Clay, Loam, Sandy

    The model expects clear images of soil and provides detailed information about
    fertility, pH, moisture, best crops to grow, and improvement recommendations.
    """,
    examples=[],
    theme=gr.themes.Soft()
)

# ─────────────────────────────────────────
# START APP
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌾 SmartAgro Soil Classification App starting...")
    print(f"   Model loaded: {model is not None}")
    print(f"   Classes: {CLASS_NAMES}")
    print("   Open http://localhost:7860 to use the app\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
