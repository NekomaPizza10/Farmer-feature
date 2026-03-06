import requests
import os
import json

image_path = r'c:\year2\year2sem2\SEGP\Extra_features\soilPrediction\dataset\test\Alluvial soil\Sample9.90.jpg'

if os.path.exists(image_path):
    print(f'📤 Uploading test image: {os.path.basename(image_path)}\n')
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post('http://127.0.0.1:5000/leaf/predict', files=files, timeout=30)
    
    print(f'Status Code: {response.status_code}\n')
    result = response.json()
    
    print('✅ Full API Response:')
    print(json.dumps(result, indent=2))
else:
    print(f'❌ Image not found: {image_path}')
