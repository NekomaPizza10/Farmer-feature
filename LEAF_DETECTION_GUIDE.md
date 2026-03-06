# 🌿 Leaf Detection - Hybrid Online/Offline Setup

Your leaf detection system has been upgraded with a **hybrid approach**:
- **Primary**: Hugging Face Inference API (cloud-based, more accurate)
- **Fallback**: Local transformers model (works offline, slower)

---

## ✅ What's Been Done

1. ✅ Created `.env` file with your API key
2. ✅ Updated `leaf_service.py` with:
   - Auto-loading from `.env`  
   - Local classifier fallback
   - Better error handling
3. ✅ Installed dependencies: `python-dotenv`, `transformers`, `torch`

---

## 🚀 Quick Test (Copy & Paste These Commands)

### Step 1: Kill any running Flask
```powershell
taskkill /F /IM python.exe 2>$null
Start-Sleep -Seconds 2
```

### Step 2: Restart Flask
```powershell
cd "c:\year2\year2sem2\SEGP\Extra_features\soilPrediction"
python app.py
```

### Step 3: Test with a leaf image (in new PowerShell window)
```powershell
curl.exe -X POST "http://127.0.0.1:5000/leaf/predict" `
  -F "image=@C:\Users\janaa\Downloads\leaf.jpg"
```

---

## 🔧 How It Works

```
Image Upload
    ↓
Call Hugging Face API
    ├─ Success? → Return results ✅
    ├─ Timeout? → Try local classifier
    ├─ 500/503? → Try local classifier  
    └─ API Error? → Try local classifier
         ↓
    Local Classification (transformers)
         ├─ Success? → Return results ✅
         └─ Fail? → Return error message ❌
```

---

## 📊 Expected Response

```json
{
  "crop_type": "Tomato",
  "disease": "Early Blight",
  "disease_confidence": "87.3%",
  "crop_confidence": "80.0%",
  "predictions": [...]
}
```

---

## 🐛 Troubleshooting

### "API Error: HTML response"
- Model still loading on Hugging Face servers
- Local fallback will activate automatically
- First call may take 30+ seconds

### "Local classification failed"
- `transformers` library not loaded properly
- Try: `pip install --upgrade transformers torch`

### "Connection refused"
- Flask not running
- Run: `python app.py` from soilPrediction folder

---

## 📁 Files Modified

- `leaf_service.py` - New hybrid system with fallback
- `.env` - Stores your API key securely
- `requirements.txt` - Added `python-dotenv`
- `app.py` - Better error logging

---

## ⚡ First Use Notes

**First API call**: May be slow (30-60 seconds) as the Hugging Face model initializes  
**First local call**: Downloads the transformer model (~500MB), then fast  
**Subsequent calls**: Should be 2-5 seconds with online API

---

Try it now! 🌾
