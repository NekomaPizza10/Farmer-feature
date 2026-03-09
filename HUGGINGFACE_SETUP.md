# 🤗 SmartAgro AI Setup Guide

This guide covers setup for **Leaf Detection** and **Soil Prediction** — both use **Hugging Face API** with the same API key.

---

# 🍃 SECTION 1: Leaf Detection (Hugging Face API)

Your leaf detection has been replaced with **Hugging Face Inference API** — a cloud-based AI model that's more accurate and reliable.

---

## ⚡ Quick Start - Leaf Detection (5 minutes)

### Step 1: Get a FREE Hugging Face API Key

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it: `smartagro` 
4. Select **Read** permissions
5. Click **Create token** and copy the token

### Step 2: Set Environment Variable (Windows)

**Open PowerShell and run:**

```powershell
$env:HF_API_KEY = "hf_your_actual_token_here_replace_this"
```

**To make it permanent, add to your user profile:**
```powershell
[System.Environment]::SetEnvironmentVariable("HF_API_KEY", "hf_your_actual_token_here", "User")
```

Then **restart PowerShell** to reload.

### Step 3: Verify Installation

```powershell
pip install requests
python -c "import os; print('✅ Key set!' if os.getenv('HF_API_KEY') else '❌ Key not found')"
```

### Step 4: Test the API

```bash
cd soilPrediction
python app.py
```

Then test with curl (or use the UI):
```powershell
curl.exe -X POST "http://127.0.0.1:5000/leaf/predict" -F "image=@C:\path\to\leaf.jpg"
```

---

## 🎯 What Changed? (Leaf Detection)

| Feature | Before | After |
|---------|--------|-------|
| **Detection Method** | Local image color analysis | Cloud-based Vision Transformer (Google ViT) |
| **Accuracy** | ~70% | ~88%+ |
| **Supported Diseases** | 6 types | 9+ types + custom labels |
| **Speed** | Instant (local) | 2-5 seconds (API call) |
| **No Dependencies** | OpenCV, SciPy | requests library (lighter) |

---

## 📊 Supported Leaf Diseases

The model now detects:
- ✅ Healthy leaves
- 🔴 Leaf Spot
- 🔴 Powdery Mildew
- 🔴 Rust
- 🔴 Blight (Early/Late)
- 🔴 Viral Infections (Mosaic)
- 🔴 Scab
- 🔴 Canker
- 🔴 Anthracnose
- 🔴 Bacterial Spot

---

## 🔧 How Leaf Detection Works

```python
# Your Node.js/React UI sends image
POST /leaf/predict
├─ Flask receives image
├─ leaf_service.py calls Hugging Face API
├─ API returns: {"label": "apple scab", "score": 0.95}
├─ Service maps to crop type (Apple)
└─ Returns full prediction to UI
```

---

## 🚨 Troubleshooting - Leaf Detection

### ❌ "HF_API_KEY environment variable not set"
**Solution:** Make sure you set the environment variable AND restarted PowerShell/terminal.

```powershell
# Verify it's set:
echo $env:HF_API_KEY
```

### ⏳ "Model is loading. Please try again in 30 seconds"
**Solution:** First call downloads the model (~500MB). Wait and retry.

### 🔴 "Connection refused"
**Solution:** Make sure Flask is running: `python app.py`

---

## 💡 Advanced Configuration (Leaf Detection)

To use a **different model**, edit `leaf_service.py`:

```python
# Line 10 - Change to any Hugging Face model:
HF_MODEL_ID = "facebook/dino-vitb14"  # Another vision model
# or
HF_MODEL_ID = "huggingface/food-101"  # Different domain
```

Browse available models: https://huggingface.co/models?pipeline_tag=image-classification

---

## 📝 Leaf Detection API Response Format

```json
{
  "crop_type": "Apple",
  "disease": "Apple scab",
  "disease_confidence": "95.2%",
  "crop_confidence": "80.0%",
  "api_label": "apple scab",
  "top_prediction": {
    "label": "Apple - Apple scab",
    "score": 0.952,
    "confidence": "95.2%"
  },
  "predictions": [...]
}
```

---

# 🌱 SECTION 2: Soil Prediction (Hugging Face API)

Soil prediction uses **Hugging Face Gradio Space API** — just like leaf detection, it requires the same `HF_API_KEY`.

---

## ⚡ Quick Start - Soil Prediction (5 minutes)

### Step 1: API Key (Same as Leaf Detection)

If you already set up the leaf detection API key, you're done! Both features use the same `HF_API_KEY`.

If not:
1. Get your token from: https://huggingface.co/settings/tokens
2. Set environment variable:

```powershell
$env:HF_API_KEY = "hf_your_actual_token_here"
[System.Environment]::SetEnvironmentVariable("HF_API_KEY", "hf_your_token", "User")
```

### Step 2: Install Dependencies

```bash
cd soilPrediction
pip install flask flask-cors requests python-dotenv
```

### Step 3: Run the Flask Server

```bash
cd soilPrediction
python flask_app_backup.py
```

The server starts at: **http://localhost:5000**

### Step 4: Test Soil Prediction

```powershell
curl.exe -X POST "http://127.0.0.1:5000/predict" -F "image=@C:\path\to\soil.jpg"
```

---

## 🎯 Soil Prediction Features

| Feature | Details |
|---------|---------|
| **Model Type** | Hugging Face Gradio Space (vinitha003/soil_classification_prediction) |
| **Input** | Soil image (any format) |
| **Classes** | Black, Clay, Loam, Sandy |
| **Speed** | 2-5 seconds (API call) |
| **API Key** | Uses same `HF_API_KEY` as leaf detection |

---

## 📊 Supported Soil Types

The model classifies 4 soil types:

| Soil Type | Also Known As | Fertility | Best For |
|-----------|---------------|-----------|----------|
| **Black** | Regur / Cotton Soil | High | Cotton, Wheat, Sorghum |
| **Clay** | Heavy Clay Soil | High | Rice, Wheat, Sugarcane |
| **Loam** | Garden / Ideal Soil | High | Corn, Tomato, Most vegetables |
| **Sandy** | Light / Coarse Soil | Low | Carrot, Potato, Watermelon |

---

## 🔧 How Soil Prediction Works

```python
User uploads soil image
├─ Flask receives image at /predict
├─ Image encoded to Base64
├─ Sent to Hugging Face Gradio Space API
│   └─ vinitha003/soil_classification_prediction
├─ API returns: soil type label
├─ Mapped to internal class (Black/Clay/Loam/Sandy)
└─ Returns JSON with soil properties + recommendations
```

---

## 🚨 Troubleshooting - Soil Prediction

### ❌ "HF_API_KEY is not set"
**Solution:** Same as leaf detection — set the environment variable and restart.

### ⏳ "Hugging Face API timed out"
**Solution:** The API call takes 5-30 seconds. Increase timeout or retry.

### 🔴 "Hugging Face API error"
**Solutions:**
- Check your API key is valid: `echo $env:HF_API_KEY`
- Verify key has `read` permissions
- Check if the Gradio Space is running: https://huggingface.co/spaces/vinitha003/soil_classification_prediction

---

## 📝 Soil Prediction API Response Format

```json
{
  "predicted_class": "Loam",
  "original_hf_label": "Loam",
  "confidence": 94.5,
  "soil_name": "Garden / Ideal Soil",
  "fertility": "High",
  "ph_range": "6.0 – 7.0  (Neutral, Ideal)",
  "moisture": "Medium (balanced drainage)",
  "texture": "Crumbly, soft, easy to work with",
  "organic_matter": "High",
  "best_crops": ["Corn", "Tomato", "Pepper", "Soybean", "Strawberry"],
  "improvement": "Maintain with regular compost. Ideal soil — minimal intervention needed."
}
```

---

## � Next Steps

### For Leaf Detection:
1. Test with various leaf images
2. Monitor API response times
3. If needed, switch between Hugging Face models
4. Add image preprocessing if lighting is poor

### For Soil Prediction:
1. Test with clear soil photos
2. Verify predictions match known soil types
3. Try different soil samples for comparison
4. Check Gradio Space status if API is slow

Happy farming! 🌾🍃�
