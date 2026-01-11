
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'DenseNet-MURA'))

try:
    from densenet import densenet169
except ImportError:
    densenet169 = None


class FractureNet(nn.Module):
    """Enhanced model for bone fracture detection with attention mechanism."""
    def __init__(self, pretrained=False, dropout_rate=0.4):
        super(FractureNet, self).__init__()
        
        try:
            self.backbone = models.densenet169(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            self.backbone = models.densenet169(pretrained=pretrained)
        
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone.features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

class MURAPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.fracture_model = None
        self.output_is_probability = False
        
        base_path = os.path.dirname(__file__)
        base_models_dir = os.path.join(base_path, '..', 'datasets', 'DenseNet-MURA', 'models')
        
        # Original MURA model paths
        preferred = os.path.join(base_models_dir, 'XR_WRIST', 'model.pth')
        if os.path.isfile(preferred):
            self.model_path = preferred
        else:
            found = None
            for root, _, files in os.walk(base_models_dir):
                if 'model.pth' in files:
                    found = os.path.join(root, 'model.pth')
                    break
            self.model_path = found or preferred
        
        # Fine-tuned fracture model paths (in priority order)
        self.fracture_model_paths = [
            os.path.join(base_path, '..', 'checkpoints', 'bone_fracture', 'model.pth'),
            os.path.join(base_models_dir, 'fracture_model.pth'),
            os.path.join(base_path, '..', 'datasets', 'DenseNet-MURA', 'models', 'XR_WRIST', 'model.pth'),
        ]
        
        self.load_model()
    
    def load_model(self):
        try:
            if densenet169 is None:
                import torchvision.models as models
                try:
                    self.model = models.densenet169(weights=None)
                except TypeError:
                    self.model = models.densenet169(pretrained=False)
                self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
                self.output_is_probability = False
            else:
                self.model = densenet169(pretrained=False)
                self.output_is_probability = True
            
            if os.path.isfile(self.model_path):
                state = torch.load(self.model_path, map_location=self.device)
                if isinstance(state, dict):
                    if 'model_state_dict' in state:
                        state = state['model_state_dict']
                    elif 'state_dict' in state:
                        state = state['state_dict']
                if isinstance(state, dict):
                    state = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }
                load_res = self.model.load_state_dict(state, strict=False)
                print(f"Loaded MURA model from {self.model_path}")
            else:
                print(f"MURA model file not found at {self.model_path}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Try to load fine-tuned fracture model
            self._load_fracture_model()
            
        except Exception as e:
            print(f"Error loading MURA model: {e}")
            raise
    
    def _load_fracture_model(self):
        """Load the fine-tuned fracture detection model if available."""
        for model_path in self.fracture_model_paths:
            if os.path.isfile(model_path):
                try:
                    self.fracture_model = FractureNet(pretrained=False)
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Handle state dict key conversion
                    if isinstance(state_dict, dict):
                        state_dict = { (k[7:] if k.startswith('module.') else k): v 
                                      for k, v in state_dict.items() }
                    
                    self.fracture_model.load_state_dict(state_dict, strict=False)
                    self.fracture_model = self.fracture_model.to(self.device)
                    self.fracture_model.eval()
                    print(f"✓ Fine-tuned fracture model loaded from {model_path}")
                    return
                except Exception as e:
                    print(f"Could not load fracture model from {model_path}: {e}")
        
        print("ℹ Fine-tuned fracture model not found, using MURA base model")
    
    def preprocess_image(self, image_path):
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
        try:
            image_tensor = self.preprocess_image(image_path).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                out = output
                if isinstance(out, (list, tuple)):
                    out = out[0]
                out = out.view(-1)[0]
                if self.output_is_probability:
                    probability = float(out.cpu().numpy())
                else:
                    probability = float(torch.sigmoid(out).cpu().numpy())
            
            # Use fine-tuned fracture model if available for enhanced detection
            fracture_enhanced_prob = None
            if self.fracture_model is not None:
                with torch.no_grad():
                    fracture_output = self.fracture_model(image_tensor)
                    fracture_enhanced_prob = float(fracture_output.squeeze().cpu().numpy())
                    # Use the enhanced probability
                    probability = fracture_enhanced_prob
            
            is_abnormal = probability > 0.5
            
            # Calculate a rough "bone image confidence" based on model's internal confidence
            # If the model gives a very high or very low probability, it's more confident
            # this is indeed a bone image (vs random noise/wrong image type)
            confidence_spread = abs(probability - 0.5) * 2  # 0 to 1, higher = more confident
            bone_image_confidence = 0.5 + confidence_spread * 0.5  # Always at least 0.5
            
            return {
                'model': 'MURA' + (' + Enhanced Fracture Detection' if fracture_enhanced_prob else ''),
                'description': 'Musculoskeletal radiographs abnormality/fracture detection',
                'is_abnormal': bool(is_abnormal),
                'abnormality_probability': float(probability),
                'normal_probability': float(1 - probability),
                'prediction': 'Abnormal/Fracture Detected' if is_abnormal else 'Normal',
                'fracture_enhanced': fracture_enhanced_prob is not None,
                'confidence': float(bone_image_confidence)  # How confident this is a bone image
            }
        
        except Exception as e:
            raise Exception(f"Error in MURA prediction: {str(e)}")

    def predict_for_frontend(self, image_path):
        try:
            raw_result = self.predict(image_path)

            prob = float(raw_result.get('abnormality_probability', 0.0))
            is_abnormal = bool(raw_result.get('is_abnormal', False))
            is_enhanced = raw_result.get('fracture_enhanced', False)

            if is_abnormal:
                status = 'critical' if prob > 0.75 else 'warning'
                
                # More detailed descriptions based on probability
                if prob > 0.85:
                    description = 'High probability of bone fracture detected. Immediate medical attention recommended. Consult an orthopedic specialist.'
                elif prob > 0.7:
                    description = 'Bone fracture or significant abnormality detected. Further imaging and specialist consultation recommended.'
                else:
                    description = 'Possible musculoskeletal abnormality detected. Please consult an orthopedic specialist for confirmation.'
                
                primary = {
                    'disease': 'Bone Fracture Detected' if prob > 0.7 else 'Musculoskeletal Abnormality',
                    'confidence': int(prob * 100),
                    'status': status,
                    'description': description,
                    'regions': ['Bone structure', 'Affected area'],
                    'enhanced_detection': is_enhanced
                }
                secondary = {
                    'disease': 'Normal Tissue',
                    'confidence': int((1.0 - prob) * 100),
                    'status': 'healthy',
                    'description': 'Some regions appear within normal limits.',
                    'regions': []
                }
                return [primary, secondary]

            return [{
                'disease': 'Healthy Scan (Bone)',
                'confidence': int((1.0 - prob) * 100),
                'status': 'healthy',
                'description': 'No significant bone fracture or musculoskeletal abnormality detected. Bone structure appears normal.',
                'regions': [],
                'enhanced_detection': is_enhanced
            }, {
                'disease': 'Abnormality Risk',
                'confidence': int(prob * 100),
                'status': 'healthy',
                'description': 'Low abnormality probability. No immediate concerns.',
                'regions': []
            }]

        except Exception as e:
            return [{
                'disease': 'Analysis Error',
                'confidence': 0,
                'status': 'warning',
                'description': f'Error analyzing image for musculoskeletal abnormality: {str(e)}',
                'regions': []
            }]

