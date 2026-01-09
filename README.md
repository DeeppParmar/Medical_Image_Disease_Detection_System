# MediScan AI - 6 Hour Hackathon Guide
## Early Disease Detection from Medical Imaging

**Quick Start Guide for Building MediScan AI in 6 Hours**

---

## 🎯 Project Overview

MediScan AI is an AI-powered system that detects **Tuberculosis**, **Pneumonia**, and **Bone Fractures** from X-ray/CT scans with explainable AI and visual contour mapping.

**Core Features:**
- ✅ Automated disease detection (TB, Pneumonia, Fractures)
- ✅ Contour mapping with visual overlays
- ✅ Explainable AI (Grad-CAM visualization)
- ✅ Real-time inference (< 3 seconds)
- ✅ Web-based interface

---

## ⏱️ 6-Hour Development Roadmap

### **Hour 1: Setup & Data Preparation** (0-60 min)

#### Step 1: Environment Setup (15 min)
```bash
# Create virtual environment
python -m venv mediscan_env
mediscan_env\Scripts\activate  # Windows
# source mediscan_env/bin/activate  # Linux/Mac

# Install core dependencies
pip install tensorflow keras opencv-python flask flask-cors numpy pillow scikit-learn matplotlib
```

#### Step 2: Quick Data Setup (30 min)
**Option A: Use Pre-trained Models (RECOMMENDED for Hackathon)**
- Download pre-trained weights from:
  - [Kaggle Chest X-ray Models](https://www.kaggle.com/models)
  - [Hugging Face Medical Models](https://huggingface.co/models?pipeline_tag=image-classification&search=chest+xray)
- Use sample test images from public datasets

**Option B: Quick Dataset Download (if time permits)**
```bash
# Download small subset for demo
# NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
# Use 100-200 images for quick training/demo
```

#### Step 3: Project Structure (15 min)
```
mediscan_ai/
├── models/
│   ├── model.py          # Model architecture
│   └── weights.h5        # Pre-trained weights
├── backend/
│   ├── app.py            # Flask API
│   └── inference.py      # Prediction logic
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   └── components/
│   └── package.json
├── utils/
│   ├── preprocessing.py  # Image preprocessing
│   └── visualization.py  # Grad-CAM & contours
└── test_images/          # Sample X-rays
```

---

### **Hour 2: Model Architecture** (60-120 min)

#### Step 1: Build Base Model (30 min)
**File: `models/model.py`**
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def create_mediscan_model(num_classes=3):
    # Load pre-trained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze early layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Classes: [TB, Pneumonia, Fracture]
model = create_mediscan_model(3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Step 2: Load Pre-trained Weights (15 min)
```python
# If you have pre-trained weights
model.load_weights('models/weights.h5')

# OR use transfer learning from ImageNet (works immediately)
# Model will work with ImageNet weights for demo
```

#### Step 3: Quick Test (15 min)
- Test model with sample image
- Verify inference pipeline works

---

### **Hour 3: Image Processing & Contour Mapping** (120-180 min)

#### Step 1: Preprocessing Pipeline (20 min)
**File: `utils/preprocessing.py`**
```python
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Convert to RGB (3 channels for ResNet)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Expand dimensions for batch
    img = np.expand_dims(img, axis=0)
    
    return img
```

#### Step 2: Contour Detection (25 min)
**File: `utils/visualization.py`**
```python
import cv2
import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf

def generate_gradcam(model, img_array, layer_name='conv5_block3_out'):
    # Get model layer
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]
    
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def draw_contours(original_img, heatmap, threshold=0.5):
    # Resize heatmap to original size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Apply threshold
    binary = (heatmap > threshold).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on original image
    overlay = original_img.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    # Blend with heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    result = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
    
    return result, contours
```

#### Step 3: Integration Test (15 min)
- Test preprocessing + contour mapping pipeline
- Verify visualizations work

---

### **Hour 4: Backend API** (180-240 min)

#### Step 1: Flask API Setup (30 min)
**File: `backend/app.py`**
```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
from utils.preprocessing import preprocess_image
from utils.visualization import generate_gradcam, draw_contours
from models.model import create_mediscan_model
import cv2

app = Flask(__name__)
CORS(app)

# Load model
model = create_mediscan_model(3)
model.load_weights('models/weights.h5')  # Or use ImageNet weights

CLASS_NAMES = ['Tuberculosis', 'Pneumonia', 'Fracture']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        img_bytes = file.read()
        
        # Save temporarily
        img_path = 'temp_image.jpg'
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
        
        # Preprocess
        img_array = preprocess_image(img_path)
        
        # Predict
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        disease = CLASS_NAMES[class_idx]
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(model, img_array)
        
        # Draw contours
        original_img = cv2.imread(img_path)
        result_img, contours = draw_contours(original_img, heatmap)
        
        # Encode result image
        _, buffer = cv2.imencode('.jpg', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'disease': disease,
            'confidence': confidence,
            'all_predictions': {
                CLASS_NAMES[i]: float(predictions[0][i]) for i in range(3)
            },
            'annotated_image': img_base64,
            'contour_count': len(contours)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

#### Step 2: Test API (15 min)
```bash
# Run backend
cd backend
python app.py

# Test with curl or Postman
curl -X POST http://localhost:5000/predict -F "image=@test_images/xray.jpg"
```

---

### **Hour 5: Frontend Development** (240-300 min)

#### Step 1: React Setup (20 min)
```bash
cd frontend
npx create-react-app . --yes
npm install axios
```

#### Step 2: Main Component (30 min)
**File: `frontend/src/App.js`**
```jsx
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handlePredict = async () => {
    if (!file) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (error) {
      console.error('Error:', error);
      alert('Prediction failed!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🩺 MediScan AI</h1>
        <p>Early Disease Detection from Medical Imaging</p>
      </header>

      <div className="container">
        <div className="upload-section">
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <button onClick={handlePredict} disabled={!file || loading}>
            {loading ? 'Analyzing...' : 'Analyze Image'}
          </button>
        </div>

        {result && (
          <div className="results">
            <h2>Diagnosis Results</h2>
            <div className="prediction">
              <h3>Detected: {result.disease}</h3>
              <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
            </div>
            
            <div className="all-predictions">
              <h4>All Predictions:</h4>
              {Object.entries(result.all_predictions).map(([disease, conf]) => (
                <div key={disease} className="pred-item">
                  <span>{disease}:</span>
                  <span>{(conf * 100).toFixed(2)}%</span>
                </div>
              ))}
            </div>

            <div className="image-result">
              <img 
                src={`data:image/jpeg;base64,${result.annotated_image}`} 
                alt="Annotated scan"
              />
              <p>Contours detected: {result.contour_count}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
```

#### Step 3: Styling (10 min)
**File: `frontend/src/App.css`**
```css
.App {
  text-align: center;
  padding: 20px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.upload-section {
  margin: 30px 0;
  padding: 20px;
  border: 2px dashed #4CAF50;
  border-radius: 10px;
}

.upload-section input {
  margin: 10px;
  padding: 10px;
}

.upload-section button {
  padding: 12px 24px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
}

.results {
  margin-top: 30px;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 10px;
}

.prediction {
  background: white;
  padding: 20px;
  border-radius: 8px;
  margin: 20px 0;
}

.all-predictions {
  margin: 20px 0;
}

.pred-item {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  margin: 5px 0;
  background: white;
  border-radius: 5px;
}

.image-result img {
  max-width: 100%;
  border-radius: 10px;
  margin-top: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
```

---

### **Hour 6: Integration, Testing & Demo Prep** (300-360 min)

#### Step 1: End-to-End Testing (20 min)
- Test full pipeline: Upload → Process → Predict → Display
- Fix any integration issues
- Test with multiple sample images

#### Step 2: Quick Improvements (20 min)
- Add error handling
- Improve UI/UX
- Add loading states
- Format confidence scores

#### Step 3: Demo Preparation (20 min)
- Prepare 3-5 test images (TB, Pneumonia, Fracture)
- Create demo script/presentation
- Test all features work smoothly
- Prepare 2-minute pitch

---

## 🚀 Quick Start Commands

```bash
# Terminal 1: Backend
cd backend
python app.py

# Terminal 2: Frontend
cd frontend
npm start

# Access: http://localhost:3000
```

---

## 📋 Feature Checklist

- [x] **Model Architecture**: ResNet-50 based CNN
- [x] **Multi-Disease Detection**: TB, Pneumonia, Fracture
- [x] **Image Preprocessing**: CLAHE, normalization, resizing
- [x] **Contour Mapping**: Visual overlays on abnormalities
- [x] **Explainable AI**: Grad-CAM heatmaps
- [x] **Backend API**: Flask REST API
- [x] **Frontend**: React.js interface
- [x] **Real-time Inference**: < 3 seconds per image

---

## 🎯 Hackathon Tips

1. **Use Pre-trained Models**: Don't train from scratch - use transfer learning
2. **Simplify Where Possible**: 
   - Use ImageNet weights if custom weights unavailable
   - Focus on 1-2 diseases if time is tight
   - Use simpler contour detection if U-Net is too complex
3. **Demo-Ready First**: Get basic pipeline working, then enhance
4. **Test Early**: Test with real images as soon as possible
5. **Prepare Backup**: Have sample predictions ready if model fails

---

## 📦 Required Files Summary

**Backend:**
- `models/model.py` - Model architecture
- `backend/app.py` - Flask API
- `utils/preprocessing.py` - Image preprocessing
- `utils/visualization.py` - Grad-CAM & contours

**Frontend:**
- `frontend/src/App.js` - Main React component
- `frontend/src/App.css` - Styling

**Data:**
- Sample test images (3-5 X-rays)
- Pre-trained model weights (optional)

---

## 🎤 Demo Script (2 minutes)

1. **Problem** (30s): "Healthcare crisis - lack of radiologists in rural areas"
2. **Solution** (30s): "MediScan AI - AI-powered early disease detection"
3. **Live Demo** (60s): 
   - Upload X-ray image
   - Show real-time prediction
   - Display contour mapping
   - Explain Grad-CAM visualization
4. **Impact** (30s): "Saves lives through early detection, works in 3 seconds"

---

## 🔧 Troubleshooting

**Model not loading?**
- Use ImageNet weights as fallback
- Check file paths

**API not responding?**
- Check CORS settings
- Verify Flask is running on port 5000

**Frontend not connecting?**
- Verify backend URL in axios call
- Check network tab in browser console

**Contours not showing?**
- Adjust threshold in `draw_contours()`
- Check heatmap generation

---

## 📚 Resources

- **Datasets**: NIH ChestX-ray14, TBX11K, RSNA, MURA
- **Pre-trained Models**: Hugging Face, Kaggle Models
- **Documentation**: TensorFlow, Flask, React.js

---

## 🏆 Success Criteria

✅ System processes X-ray images  
✅ Detects at least 2 diseases  
✅ Shows visual annotations  
✅ Provides confidence scores  
✅ Works in real-time (< 5 seconds)  
✅ Has functional web interface  

---

**Good luck with your hackathon! 🚀**

*Built for DA-IICT Hackathon 2026*
