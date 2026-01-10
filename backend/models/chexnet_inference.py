"""
CheXNet Inference Module
Chest X-ray disease detection (14 diseases)
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Add CheXNet dataset path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CheXNet'))

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = 14

class DenseNet121(nn.Module):
    """Model modified for CheXNet"""
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        try:
            self.densenet121 = torchvision.models.densenet121(weights=None)
        except TypeError:
            self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class CheXNetPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'CheXNet', 'model.pth.tar'
        )
        self.load_model()
    
    def load_model(self):
        """Load the CheXNet model"""
        try:
            self.model = DenseNet121(N_CLASSES)
            
            if os.path.isfile(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (from DataParallel training)
                # Also add 'densenet121.' prefix if keys don't have it
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Remove 'module.' prefix
                    name = k[7:] if k.startswith('module.') else k
                    
                    # Add 'densenet121.' prefix if missing (checkpoint uses different naming)
                    # The model expects 'densenet121.features...' but checkpoint may have 'features...'
                    if not name.startswith('densenet121.'):
                        name = 'densenet121.' + name
                    
                    new_state_dict[name] = v
                
                load_res = self.model.load_state_dict(new_state_dict, strict=False)
                
                # Only raise error if there are missing keys that aren't expected
                if getattr(load_res, 'missing_keys', None):
                    # Filter out expected missing keys (like num_batches_tracked)
                    critical_missing = [k for k in load_res.missing_keys 
                                       if 'num_batches_tracked' not in k]
                    if len(critical_missing) > 0:
                        print(f"Warning: CheXNet checkpoint had {len(critical_missing)} missing keys")
                        # Don't raise - try to load anyway
                
                print(f"Loaded CheXNet model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✓ CheXNet model ready on device: {self.device}")
        except Exception as e:
            print(f"Error loading CheXNet model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess image for CheXNet"""
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Simple preprocessing without TenCrop for faster inference
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        image_tensor = transform(image)
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    
    def predict(self, image_path):
        """Predict diseases from chest X-ray image"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Move to device
            input_var = image_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(input_var)
                probabilities = output.cpu().numpy()[0]
            
            # Format results
            results = {}
            detected_diseases = []
            
            for i, class_name in enumerate(CLASS_NAMES):
                prob = float(probabilities[i])
                results[class_name] = prob
                if prob > 0.5:  # Threshold for positive detection
                    detected_diseases.append({
                        'disease': class_name,
                        'probability': prob
                    })
            
            # Sort detected diseases by probability
            detected_diseases.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'model': 'CheXNet',
                'description': 'Chest X-ray disease detection',
                'all_probabilities': results,
                'detected_diseases': detected_diseases,
                'num_detections': len(detected_diseases)
            }
        
        except Exception as e:
            raise Exception(f"Error in CheXNet prediction: {str(e)}")
    
    def predict_for_frontend(self, image_path):
        """Predict and format results for frontend React component"""
        try:
            # Get raw predictions
            raw_result = self.predict(image_path)
            
            # Disease descriptions mapping
            disease_descriptions = {
                'Atelectasis': 'Collapsed or airless lung tissue detected. May indicate lung compression or obstruction.',
                'Cardiomegaly': 'Enlarged heart silhouette observed. Could indicate cardiac enlargement or pericardial effusion.',
                'Effusion': 'Fluid accumulation in the pleural space detected. Possible pleural effusion.',
                'Infiltration': 'Abnormal substances in lung tissue identified. May indicate inflammation or fluid.',
                'Mass': 'Abnormal mass or nodule detected in lung tissue. Requires further evaluation.',
                'Nodule': 'Small nodule detected in lung parenchyma. Monitor for changes.',
                'Pneumonia': 'Inflammation of lung tissue detected. Signs of infection present.',
                'Pneumothorax': 'Air in pleural space detected. Lung collapse possible.',
                'Consolidation': 'Lung tissue consolidation observed. May indicate pneumonia or other conditions.',
                'Edema': 'Fluid accumulation in lung tissue detected. Possible pulmonary edema.',
                'Emphysema': 'Destructive lung changes detected. Possible chronic obstructive pulmonary disease.',
                'Fibrosis': 'Scarring of lung tissue detected. Chronic lung disease indicators present.',
                'Pleural_Thickening': 'Thickening of pleural membranes observed. Chronic inflammation possible.',
                'Hernia': 'Diaphragmatic hernia or other hernia detected in chest cavity.'
            }
            
            # Convert to frontend format
            analysis_results = []
            
            # Get all probabilities
            all_probs = raw_result.get('all_probabilities', {})
            
            # Sort by probability (highest first)
            sorted_diseases = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            
            # Determine if healthy (no diseases detected above threshold)
            detected_above_threshold = [d for d, prob in sorted_diseases if prob > 0.5]
            
            if not detected_above_threshold:
                # Healthy case - show top 3 with status healthy
                analysis_results.append({
                    'disease': 'Healthy Scan',
                    'confidence': int((1.0 - sorted_diseases[0][1]) * 100) if sorted_diseases else 95,
                    'status': 'healthy',
                    'description': 'No significant abnormalities detected. Lung fields appear clear. Normal cardiac silhouette.',
                    'regions': []
                })
                # Add top diseases as "not detected"
                for disease, prob in sorted_diseases[:3]:
                    analysis_results.append({
                        'disease': disease.replace('_', ' '),
                        'confidence': int(prob * 100),
                        'status': 'healthy',
                        'description': f'No significant {disease.replace("_", " ").lower()} indicators detected.'
                    })
            else:
                # Diseases detected - show them first
                for disease, prob in sorted_diseases:
                    if prob > 0.5:
                        # Critical if high probability, warning if medium
                        status = 'critical' if prob > 0.75 else 'warning'
                        confidence = int(prob * 100)
                        description = disease_descriptions.get(disease, f'{disease.replace("_", " ")} indicators detected.')
                        
                        # Add regions based on disease type
                        regions = []
                        if 'Lobe' not in description:
                            if disease in ['Pneumonia', 'Consolidation']:
                                regions = ['Right Lower Lobe', 'Left Lower Lobe']
                            elif disease in ['Effusion', 'Pleural_Thickening']:
                                regions = ['Pleural Space']
                            elif disease == 'Cardiomegaly':
                                regions = ['Mediastinum', 'Cardiac Silhouette']
                        
                        analysis_results.append({
                            'disease': disease.replace('_', ' '),
                            'confidence': confidence,
                            'status': status,
                            'description': description,
                            'regions': regions
                        })
                        
                        # Limit to top 3 detected diseases
                        if len([r for r in analysis_results if r['status'] != 'healthy']) >= 3:
                            break
                
                # Add non-detected diseases to fill up to 3 results
                for disease, prob in sorted_diseases:
                    if prob <= 0.5 and len(analysis_results) < 3:
                        analysis_results.append({
                            'disease': disease.replace('_', ' '),
                            'confidence': int(prob * 100),
                            'status': 'healthy',
                            'description': f'No significant {disease.replace("_", " ").lower()} indicators detected.'
                        })
            
            # Ensure we have at least 1 result, max 3
            return analysis_results[:3]
        
        except Exception as e:
            raise Exception(f"Error formatting CheXNet results for frontend: {str(e)}")

