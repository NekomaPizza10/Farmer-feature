# 🌱 Soil Prediction Feature Guide
### SmartAgro Soil Classification using AI — classify soil types from images and get crop recommendations.

## 1. Overview
The Soil Prediction module is an AI-powered image classification system that analyzes soil images and classifies them into one of four categories:

Black
Clay
Loam
Sandy
For each prediction, the system provides:

Soil properties (fertility, pH range, moisture, texture)
Recommended crops
Soil improvement suggestions
Confidence score
This feature supports two inference modes:

Mode || File || Description
Cloud (Hugging Face API) || flask_app_backup.py || Uses external Hugging Face Gradio Space for inference

Local (Offline) || app.py || Uses locally trained TensorFlow model (soil_model.h5)



## 2. System Architecture

### 2.1 Cloud Mode (Hugging Face API)
- **Endpoint**: `/soil/predict`
- **Inference**: Calls Hugging Face Gradio Space (`https://huggingface.co/spaces/your-username/soil-classifier`)
- **Model**: Pre-trained on Hugging Face
- **Speed**: Fast (0.5-2 seconds)
- **Reliability**: High (no local setup needed)
- **Dependencies**: None (just internet connection)
- **Graph**: User → Flask Backend → Hugging Face Space API → Prediction → Flask → JSON Response
- **Port**: http://localhost:5000

### 2.2 Local Mode (Offline)
- **Endpoint**: `/soil/predict/local`
- **Inference**: Uses local TensorFlow model (`soil_model.h5`)
- **Model**: Trained locally with 95% accuracy
- **Speed**: Slower (2-5 seconds)
- **Reliability**: Medium (requires model file)
- **Dependencies**: TensorFlow, NumPy (No API key required)
- **Graph**: User → Gradio UI → TensorFlow Model (.h5) → Prediction
- **Port**: http://localhost:7860

## 3. Datasets
### Source
- **Dataset**: [Soil Dataset](https://www.kaggle.com/datasets/ashishjangra27/soil-dataset)
- **Description**: Contains images of different soil types with labels
- **Classes**: Black, Clay, Loam, Sandy
- **Size**: ~10,000 images
- **Format**: JPEG images

### Structure
- **Training Data**: 80% of dataset
- **Validation Data**: 10% of dataset
- **Test Data**: 10% of dataset
- **Image Size**: 224x224 pixels
- **Color Mode**: RGB

### Task Type
- **Classification**: Multi-class image classification
- **Number of Classes**: 4 (Black, Clay, Loam, Sandy)
- **Input**: Single soil image (JPEG)
- **Output**: Class label + properties + recommendations


## 4. Model Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned on soil dataset
- **Layers**: 
  - Base ResNet50 (frozen)
  - Global Average Pooling
  - Dense layers with dropout
  - Output layer (4 neurons for 4 classes)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Training**: 50 epochs, batch size 32
- **Validation Accuracy**: 95%

## 5. Training Process
- **Data Preprocessing**: 
  - Resize images to 224x224 pixels
  - Normalize pixel values (0-255 → 0-1)
  - Data augmentation (rotation, zoom, flip)
- **Transfer Learning**: 
  - Load pre-trained ResNet50 without top layers
  - Freeze base model layers
  - Add custom classification head
- **Fine-tuning**: 
  - Unfreeze top layers after initial training
  - Train with lower learning rate
- **Evaluation**: 
  - Test accuracy: 95%
  - Confusion matrix analysis
  - Class-specific performance metrics

## 6. API Design (Cloud Mode)
- **Endpoint**: `/soil/predict`
- **Method**: POST
- **Request**
  - ```bash
    curl -X POST "http://127.0.0.1:5000/predict" \
      -F "image=@soil.jpg"
    ```
- **Response**
  - JSON object with prediction results
- **External Model**
  - Hugging Face Space: vinitha003/soil_classification_prediction
  - Requires HF_API_KEY

## 7. Soil Knowledge Base
Each predicted class maps to structured agronomic information:

Soil	Fertility	pH	        Crops
Black	High	  7.5–8.5	Cotton, Wheat
Clay	High	  6.0–7.0	Rice, Sugarcane
Loam	High	  6.0–7.0	Vegetables
Sandy	Low	      5.5–6.5	Carrot, Potato

This rule-based enrichment layer transforms raw AI predictions into actionable agricultural insights.

## 8. Running the System
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Option A - Cloud Mode (External Hugging Face Model)
```powershell
$env:HF_API_KEY="your_key"
python flask_app_backup.py
```

### Option B - Local Mode (Self-Hosted Model)
```bash
python app.py 
```
Open :http://localhost:7860

## 9. Limitations
- Model accuracy depends on dataset quality and size
- Weather conditions may affect soil appearance
- Limited to 4 soil types (Black, Clay, Loam, Sandy)
- Requires good image quality for accurate predictions

## 10. Future Enhancements
- Add more soil types and improve dataset diversity
- Implement real-time camera integration
- Add multi-language support
- Include soil moisture and nutrient prediction
- Develop mobile app for field use
