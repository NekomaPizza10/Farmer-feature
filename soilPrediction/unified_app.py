# ============================================================
# SmartAgro — Unified App (Gradio + FastAPI + Static HTML)
#
# HOW TO RUN:
#   python unified_app.py
#   → App starts at  http://localhost:7860
#
# WHAT THIS DOES:
#   - Serves Gradio interface at / (or /gradio)
#   - Serves HTML UI at /ui or as static files
#   - Provides REST API endpoints for predictions
#   - Supports Cloud (HF API) and Local (TensorFlow) modes
# ============================================================

import os
import io
import base64
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn

# ============================================================
# CONFIGURATION — Change these flags to switch modes
# ============================================================
USE_LOCAL_SOIL_MODEL = True  # Set False to use HF API for soil
USE_LOCAL_LEAF_MODEL = False  # Set True to use local leaf model (if available)

# Hugging Face API Config
HF_API_KEY = os.getenv("HF_API_KEY")
HF_SOIL_MODEL_ID = "vinitha003/soil_classification_prediction"
HF_SOIL_API_URL = "https://vinitha003-soil-classification-prediction-67f427e.hf.space/run/predict"
HF_LEAF_MODEL_ID = "wambugu71/crop_leaf_diseases_vit"
HF_LEAF_API_URL = f"https://api-inference.huggingface.co/models/{HF_LEAF_MODEL_ID}"

# Local Model Config
SOIL_MODEL_PATH = "soil_model.h5"
LEAF_MODEL_PATH = None  # Set if you have a local leaf model
IMG_SIZE = (224, 224)

# ============================================================
# LOAD ENVIRONMENT
# ============================================================
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from {env_path}")
        HF_API_KEY = os.getenv("HF_API_KEY")
except Exception as e:
    print(f"Could not load .env: {e}")

# ============================================================
# SOIL KNOWLEDGE BASE
# ============================================================
SOIL_CLASS_NAMES = ["Black", "Clay", "Loam", "Sandy"]

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
# LOAD LOCAL MODELS
# ============================================================
soil_model = None
leaf_model = None

if USE_LOCAL_SOIL_MODEL:
    try:
        import tensorflow as tf
        print(f"Loading local soil model from {SOIL_MODEL_PATH}...")
        if Path(SOIL_MODEL_PATH).exists():
            soil_model = tf.keras.models.load_model(SOIL_MODEL_PATH)
            print("Local soil model loaded successfully!")
        else:
            print(f"Soil model file not found: {SOIL_MODEL_PATH}")
            USE_LOCAL_SOIL_MODEL = False
    except Exception as e:
        print(f"Error loading local soil model: {e}")
        USE_LOCAL_SOIL_MODEL = False

if USE_LOCAL_LEAF_MODEL and LEAF_MODEL_PATH:
    try:
        import tensorflow as tf
        print(f"Loading local leaf model from {LEAF_MODEL_PATH}...")
        if Path(LEAF_MODEL_PATH).exists():
            leaf_model = tf.keras.models.load_model(LEAF_MODEL_PATH)
            print("Local leaf model loaded successfully!")
        else:
            print(f"Leaf model file not found: {LEAF_MODEL_PATH}")
            USE_LOCAL_LEAF_MODEL = False
    except Exception as e:
        print(f"Error loading local leaf model: {e}")
        USE_LOCAL_LEAF_MODEL = False

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
        return "Loam"


def predict_soil_local(image):
    """Predict using local TensorFlow model."""
    if soil_model is None:
        raise RuntimeError("Local soil model not loaded")
    
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")
    
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = soil_model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    predicted_class = SOIL_CLASS_NAMES[predicted_class_idx]
    all_probs = {SOIL_CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(SOIL_CLASS_NAMES))}
    
    return predicted_class, confidence, all_probs


def predict_soil_hf_api(image):
    """Predict using Hugging Face API."""
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set")
    
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")
    
    img_io = io.BytesIO()
    img.save(img_io, format="JPEG")
    img_bytes = img_io.getvalue()
    
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    img_data_uri = f"data:image/jpeg;base64,{img_b64}"
    
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
    
    top_prediction = results["data"][0]
    if isinstance(top_prediction, dict):
        hf_label = top_prediction.get("label", str(top_prediction))
        confidence = float(top_prediction.get("score", 0.9))
    else:
        hf_label = str(top_prediction)
        confidence = 0.9
    
    predicted_class = map_soil_class(hf_label)
    
    return predicted_class, confidence, None


def analyze_leaf_color_fast(image_bytes):
    """FAST local fallback analysis for leaf disease."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        
        h, w = arr.shape[:2]
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        
        blight_mottled = np.sum(((r > 0.25) & (r < 0.55)) & ((g > 0.2) & (g < 0.48)) & ((b > 0.2) & (b < 0.42))) / (h * w)
        dark_spots = np.sum((r < 0.3) & (g < 0.3) & (b < 0.3)) / (h * w)
        rust_color = np.sum((r > 0.35) & (r > g) & (g < 0.35) & (b < 0.3)) / (h * w)
        yellow_spots = np.sum((r > 0.45) & (g > 0.4) & (b < 0.35)) / (h * w)
        gray_spots = np.sum((np.abs(r-g) < 0.12) & (np.abs(g-b) < 0.12) & (r > 0.15) & (r < 0.55)) / (h * w)
        
        disease_indicator = blight_mottled + dark_spots + rust_color + yellow_spots + gray_spots
        
        if blight_mottled > 0.08 or (gray_spots > 0.10 and dark_spots > 0.05):
            disease = "Early Blight"
            confidence = 0.60
        elif dark_spots > 0.10:
            disease = "Leaf Blast"
            confidence = 0.62
        elif rust_color > 0.08:
            disease = "Common Rust"
            confidence = 0.60
        elif yellow_spots > 0.08:
            disease = "Brown Spot"
            confidence = 0.58
        elif blight_mottled > 0.04 or gray_spots > 0.08:
            disease = "Late Blight"
            confidence = 0.55
        elif disease_indicator > 0.15:
            disease = "Early Blight"
            confidence = 0.52
        else:
            disease = "Healthy"
            confidence = 0.75
        
        crop_type = "Unknown"
        if "Early Blight" in disease or "Late Blight" in disease:
            crop_type = "Potato"
        elif "Leaf Blast" in disease:
            crop_type = "Rice"
        elif "Common Rust" in disease:
            crop_type = "Wheat"
        elif "Brown Spot" in disease:
            crop_type = "Rice"
        
        return {"disease": disease, "score": confidence, "crop_type": crop_type}
    except Exception as e:
        print(f"Local analysis error: {e}")
        return {"disease": "Unknown", "score": 0.5, "crop_type": "Unknown"}


def predict_leaf_hf_api(image_bytes, timeout=20):
    """Call Hugging Face API for leaf disease."""
    if not HF_API_KEY:
        return analyze_leaf_color_fast(image_bytes)
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    try:
        print(f"Calling HF Leaf API (timeout: {timeout}s)...")
        response = requests.post(
            HF_LEAF_API_URL,
            headers=headers,
            data=image_bytes,
            timeout=timeout
        )
        
        if response.status_code == 200:
            results = response.json()
            if isinstance(results, list) and len(results) > 0:
                top = results[0]
                disease_name = top.get("label", "Unknown")
                score = float(top.get("score", 0.5))
                
                # Infer crop from disease name
                crop_type = "Unknown"
                label_lower = disease_name.lower()
                if "corn" in label_lower:
                    crop_type = "Corn"
                elif "potato" in label_lower:
                    crop_type = "Potato"
                elif "rice" in label_lower:
                    crop_type = "Rice"
                elif "wheat" in label_lower:
                    crop_type = "Wheat"
                elif "early blight" in label_lower or "late blight" in label_lower:
                    crop_type = "Potato"
                elif "leaf blast" in label_lower:
                    crop_type = "Rice"
                elif "common rust" in label_lower or "stripe rust" in label_lower:
                    crop_type = "Wheat"
                elif "brown spot" in label_lower:
                    crop_type = "Rice"
                
                return {"disease": disease_name, "score": score, "crop_type": crop_type, "source": "api"}
        
        print(f"API returned {response.status_code}, using local analysis")
        return analyze_leaf_color_fast(image_bytes)
        
    except requests.exceptions.Timeout:
        print("API timeout, using local analysis")
        return analyze_leaf_color_fast(image_bytes)
    except Exception as e:
        print(f"API error: {str(e)[:50]}, using local analysis")
        return analyze_leaf_color_fast(image_bytes)


def get_crop_type_from_label(hf_label):
    """Extract crop type from model output."""
    label_lower = hf_label.lower()
    
    if "corn" in label_lower:
        return "Corn"
    elif "potato" in label_lower:
        return "Potato"
    elif "rice" in label_lower:
        return "Rice"
    elif "wheat" in label_lower:
        return "Wheat"
    elif "early blight" in label_lower or "late blight" in label_lower:
        return "Potato"
    elif "leaf blast" in label_lower:
        return "Rice"
    elif "common rust" in label_lower or "stripe rust" in label_lower:
        return "Wheat"
    elif "brown spot" in label_lower:
        return "Rice"
    else:
        return "Unknown"


# ============================================================
# GRADIO INTERFACE FUNCTION
# ============================================================
def predict_soil_gradio(image):
    """Main prediction function for Gradio interface."""
    if image is None:
        return "Please upload an image."
    
    try:
        current_mode = "Local Model" if USE_LOCAL_SOIL_MODEL and soil_model is not None else "Hugging Face API"
        
        if USE_LOCAL_SOIL_MODEL and soil_model is not None:
            predicted_class, confidence, all_probs = predict_soil_local(image)
        else:
            if not HF_API_KEY:
                return "Error: HF_API_KEY not set. Please set your Hugging Face API key."
            predicted_class, confidence, all_probs = predict_soil_hf_api(image)
        
        info = SOIL_INFO[predicted_class]
        
        output = f"""
## Soil Analysis Results

**Mode:** {current_mode}  
**Predicted Soil Type:** {predicted_class}  
**Confidence:** {confidence*100:.1f}%

---

### Soil Properties

| Property | Value |
|----------|-------|
| **Also Known As** | {info['also_known_as']} |
| **Fertility** | {info['fertility']} |
| **pH Range** | {info['ph_range']} |
| **Moisture** | {info['moisture']} |
| **Texture** | {info['texture']} |
| **Organic Matter** | {info['organic_matter']} |

---

### Best Crops
{', '.join(info['best_crops'])}

---

### Improvement Tips
{info['improvement']}
"""
        
        if all_probs:
            output += "\n---\n\n### All Class Probabilities\n"
            for class_name in SOIL_CLASS_NAMES:
                prob = all_probs[class_name]
                bar = "█" * int(prob * 20)
                output += f"\n- **{class_name}:** {prob*100:.1f}% {bar}"
        
        return output
        
    except Exception as e:
        return f"Error during prediction: {str(e)}"


# ============================================================
# FASTAPI APP SETUP
# ============================================================
app = FastAPI(title="SmartAgro API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine project layout automatically
SCRIPT_DIR = Path(__file__).parent  # Directory containing this .py file

# Try multiple possible locations for the UI folder
POSSIBLE_UI_DIRS = [
    SCRIPT_DIR / "UI",                # Same folder:  backend/UI/
    SCRIPT_DIR / "ui",                # Lowercase:    backend/ui/
    SCRIPT_DIR / "static",            # Alt name:     backend/static/
    SCRIPT_DIR.parent / "UI",         # One level up: project/UI/
    SCRIPT_DIR.parent / "ui",         # One level up: project/ui/
    SCRIPT_DIR.parent / "frontend",   # One level up: project/frontend/
]

STATIC_DIR = None
for d in POSSIBLE_UI_DIRS:
    if d.exists() and d.is_dir():
        STATIC_DIR = d
        break

if STATIC_DIR:
    print(f"✅ Found UI directory: {STATIC_DIR}")
    print(f"   Files: {[f.name for f in STATIC_DIR.iterdir()]}")
else:
    print(f"⚠️  No UI directory found. Searched:")
    for d in POSSIBLE_UI_DIRS:
        print(f"   - {d}  (exists: {d.exists()})")


# --- Serve individual static files at ROOT level ---
# This fixes the 404 for /styles.css and /app.js

@app.get("/styles.css")
async def serve_css():
    """Serve CSS at root path."""
    if STATIC_DIR:
        css_file = STATIC_DIR / "styles.css"
        if css_file.exists():
            return FileResponse(str(css_file), media_type="text/css")
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/app.js")
async def serve_js():
    """Serve JS at root path."""
    if STATIC_DIR:
        js_file = STATIC_DIR / "app.js"
        if js_file.exists():
            return FileResponse(str(js_file), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


@app.get("/")
async def serve_ui():
    """Serve the main HTML UI."""
    if STATIC_DIR:
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path), media_type="text/html")
    # Fallback to Gradio
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/gradio")


# Also mount the full static directory for any other assets (images, fonts, etc.)
if STATIC_DIR and STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    print(f"✅ Mounted /static → {STATIC_DIR}")

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "soil_mode": "local" if USE_LOCAL_SOIL_MODEL and soil_model else "cloud",
        "leaf_mode": "local" if USE_LOCAL_LEAF_MODEL and leaf_model else "cloud",
        "soil_model_loaded": soil_model is not None,
        "leaf_model_loaded": leaf_model is not None,
        "hf_api_key_set": HF_API_KEY is not None
    }


@app.post("/api/soil/predict")
async def api_predict_soil(
    image: UploadFile = File(...),
    force_cloud: bool = Query(False, description="Force cloud/HF API mode")
):
    """
    Predict soil type from image.
    Supports both local and cloud modes.
    """
    try:
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Determine mode
        use_local = USE_LOCAL_SOIL_MODEL and not force_cloud and soil_model is not None
        
        if use_local:
            predicted_class, confidence, all_probs = predict_soil_local(img)
            mode_used = "local"
        else:
            if not HF_API_KEY:
                raise HTTPException(status_code=503, detail="HF_API_KEY not set for cloud mode")
            predicted_class, confidence, all_probs = predict_soil_hf_api(img)
            mode_used = "cloud"
        
        # Get soil info
        info = SOIL_INFO[predicted_class]
        
        response_data = {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 1),
            "mode_used": mode_used,
            "soil_name": info["also_known_as"],
            "fertility": info["fertility"],
            "ph_range": info["ph_range"],
            "moisture": info["moisture"],
            "texture": info["texture"],
            "organic_matter": info["organic_matter"],
            "best_crops": info["best_crops"],
            "improvement": info["improvement"],
        }
        
        if all_probs:
            response_data["all_probabilities"] = {k: round(v * 100, 1) for k, v in all_probs.items()}
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/leaf/predict")
async def api_predict_leaf(
    image: UploadFile = File(...),
    force_local: bool = Query(False, description="Force local analysis mode")
):
    """
    Predict leaf disease from image.
    Supports both local (fast) and cloud (HF API) modes.
    """
    try:
        # Read image
        contents = await image.read()
        
        # Determine mode
        if force_local or not HF_API_KEY:
            result = analyze_leaf_color_fast(contents)
            mode_used = "local"
        else:
            result = predict_leaf_hf_api(contents)
            mode_used = result.get("source", "cloud")
        
        disease_name = result.get("disease", "Unknown")
        confidence = result.get("score", 0.5)
        crop_type = result.get("crop_type", "Unknown")
        
        predictions_list = [
            {
                "label": f"{crop_type} - {disease_name}",
                "score": round(confidence, 2),
                "confidence": f"{confidence*100:.1f}%"
            }
        ]
        
        return JSONResponse(content={
            "success": True,
            "crop_type": crop_type,
            "disease": disease_name,
            "disease_confidence": f"{confidence*100:.1f}%",
            "mode_used": mode_used,
            "top_prediction": predictions_list[0],
            "predictions": predictions_list
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return {
        "use_local_soil_model": USE_LOCAL_SOIL_MODEL and soil_model is not None,
        "use_local_leaf_model": USE_LOCAL_LEAF_MODEL and leaf_model is not None,
        "hf_api_key_set": HF_API_KEY is not None,
        "soil_model_path": str(SOIL_MODEL_PATH) if soil_model else None,
        "leaf_model_path": str(LEAF_MODEL_PATH) if leaf_model else None,
    }


# ============================================================
# GRADIO REDIRECT
# ============================================================
@app.get("/gradio")
async def gradio_redirect():
    """Redirect to Gradio interface."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/gradio/")


# ============================================================
# GRADIO INTERFACE SETUP
# ============================================================
mode_text = "Local TensorFlow Model" if USE_LOCAL_SOIL_MODEL and soil_model else "Hugging Face API"

if USE_LOCAL_SOIL_MODEL and soil_model:
    description = """
    Upload a photo of soil and our AI will analyze it to determine the soil type.
    
    **Current Mode:** Local TensorFlow Model (`soil_model.h5`)
    
    **Supported soil types:** Black, Clay, Loam, Sandy
    
    The model provides detailed information about fertility, pH, moisture, best crops to grow, and improvement recommendations.
    """
else:
    description = """
    Upload a photo of soil and our AI will analyze it to determine the soil type.
    
    **Current Mode:** Hugging Face API (`vinitha003/soil_classification_prediction`)
    
    **Supported soil types:** Black, Clay, Loam, Sandy
    
    The model provides detailed information about fertility, pH, moisture, best crops to grow, and improvement recommendations.
    """

gradio_interface = gr.Interface(
    fn=predict_soil_gradio,
    inputs=gr.Image(type="pil", label="Upload Soil Image"),
    outputs=gr.Markdown(label="Soil Analysis"),
    title=f"SmartAgro Soil Classification ({mode_text})",
    description=description,
    examples=[],
    theme=gr.themes.Soft()
)

# Mount Gradio app
app = gr.mount_gradio_app(app, gradio_interface, path="/gradio")


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SmartAgro Unified App (Gradio + FastAPI + HTML UI)")
    print("="*60)
    print(f"   Soil Mode: {'Local' if USE_LOCAL_SOIL_MODEL and soil_model else 'Cloud (HF API)'}")
    print(f"   Leaf Mode: {'Local' if USE_LOCAL_LEAF_MODEL and leaf_model else 'Cloud (HF API)'}")
    print(f"   Soil Model Loaded: {soil_model is not None}")
    print(f"   Leaf Model Loaded: {leaf_model is not None}")
    print(f"   HF_API_KEY Set: {HF_API_KEY is not None}")
    print("-"*60)
    print("\n   Endpoints:")
    print("   - HTML UI:        http://localhost:7860/")
    print("   - Gradio UI:      http://localhost:7860/gradio")
    print("   - API Health:     http://localhost:7860/api/health")
    print("   - API Soil:       POST http://localhost:7860/api/soil/predict")
    print("   - API Leaf:       POST http://localhost:7860/api/leaf/predict")
    print("   - API Config:     http://localhost:7860/api/config")
    print("-"*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=7860)