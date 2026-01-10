"""
MURA (Musculoskeletal Radiographs) Inference Module
Detects abnormalities in musculoskeletal X-rays
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Add MURA dataset path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'DenseNet-MURA'))

try:
    from densenet import densenet169
except ImportError:
    # Fallback if densenet module not available
    densenet169 = None

class MURAPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'DenseNet-MURA', 'models', 'model.pth'
        )
        self.load_model()
    
    def load_model(self):
        """Load the MURA DenseNet model"""
        try:
            if densenet169 is None:
                # Use torchvision DenseNet as fallback
                import torchvision.models as models
                self.model = models.densenet169(pretrained=True)
                self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
            else:
                self.model = densenet169(pretrained=True)
                self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
            
            if os.path.isfile(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"Loaded MURA model from {self.model_path}")
            else:
                print(f"Warning: Model file not found at {self.model_path}. Using pretrained weights only.")
            
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading MURA model: {e}")
            # Create model with pretrained weights as fallback
            import torchvision.models as models
            self.model = models.densenet169(pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def preprocess_image(self, image_path):
        """Preprocess image for MURA"""
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)
    
    def predict(self, image_path):
        """Predict abnormality in musculoskeletal X-ray"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(image_tensor)
                # Apply sigmoid for binary classification
                probability = torch.sigmoid(output).cpu().numpy()[0][0]
            
            is_abnormal = probability > 0.5
            
            return {
                'model': 'MURA',
                'description': 'Musculoskeletal radiographs abnormality detection',
                'is_abnormal': bool(is_abnormal),
                'abnormality_probability': float(probability),
                'normal_probability': float(1 - probability),
                'prediction': 'Abnormal' if is_abnormal else 'Normal'
            }
        
        except Exception as e:
            raise Exception(f"Error in MURA prediction: {str(e)}")

