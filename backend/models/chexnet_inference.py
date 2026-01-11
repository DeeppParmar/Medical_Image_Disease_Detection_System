
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CheXNet'))

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = 14

# Priority classes for pneumonia detection
PNEUMONIA_CLASSES = ['Pneumonia', 'Consolidation', 'Infiltration']

class DenseNet121(nn.Module):
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


class PneumoniaNet(nn.Module):
    """Enhanced model for pneumonia detection with attention mechanism."""
    def __init__(self, num_classes=2, pretrained=False, dropout_rate=0.5):
        super(PneumoniaNet, self).__init__()
        
        try:
            self.backbone = torchvision.models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            self.backbone = torchvision.models.densenet121(pretrained=pretrained)
        
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.attention = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        attention_weights = self.attention(features)
        features = features * attention_weights
        output = self.classifier(features)
        return output

class CheXNetPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.pneumonia_model = None
        
        # Model paths - prioritize fine-tuned models
        base_path = os.path.dirname(__file__)
        self.model_path = os.path.join(base_path, '..', 'datasets', 'CheXNet', 'model.pth.tar')
        
        # Fine-tuned pneumonia model paths (in priority order)
        self.pneumonia_model_paths = [
            os.path.join(base_path, '..', 'checkpoints', 'pneumonia', 'best_pneumonia_model.pth'),
            os.path.join(base_path, '..', 'datasets', 'CheXNet', 'checkpoints', 'pneumonia', 'best_pneumonia_model.pth'),
            os.path.join(base_path, '..', 'datasets', 'CheXNet', 'pneumonia_model_final.pth.tar'),
        ]
        
        self.load_model()
    
    def load_model(self):
        try:
            # Load main CheXNet model
            self.model = DenseNet121(N_CLASSES)
            
            if os.path.isfile(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix and handle key format conversion
                import re
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        name = k[7:]
                    else:
                        name = k
                    
                    name = re.sub(r'\.norm\.(\d+)', r'.norm\1', name)
                    name = re.sub(r'\.conv\.(\d+)', r'.conv\1', name)
                    
                    new_state_dict[name] = v
                
                load_res = self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded CheXNet model from {self.model_path}")
            else:
                print(f"CheXNet model file not found at {self.model_path}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Try to load fine-tuned pneumonia model
            self._load_pneumonia_model()
            
            print(f"✓ CheXNet model ready on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading CheXNet model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_pneumonia_model(self):
        """Load the fine-tuned pneumonia detection model if available."""
        for model_path in self.pneumonia_model_paths:
            if os.path.isfile(model_path):
                try:
                    self.pneumonia_model = PneumoniaNet(num_classes=2, pretrained=False)
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    if 'model_state_dict' in checkpoint:
                        self.pneumonia_model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.pneumonia_model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.pneumonia_model.load_state_dict(checkpoint)
                    
                    self.pneumonia_model = self.pneumonia_model.to(self.device)
                    self.pneumonia_model.eval()
                    print(f"✓ Fine-tuned pneumonia model loaded from {model_path}")
                    return
                except Exception as e:
                    print(f"Could not load pneumonia model from {model_path}: {e}")
        
        print("ℹ Fine-tuned pneumonia model not found, using CheXNet base model")
    
    def preprocess_image(self, image_path):
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        
        image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    
    def predict(self, image_path):
        try:
            image_tensor = self.preprocess_image(image_path)
            
            input_var = image_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(input_var)
                probabilities = output.cpu().numpy()[0]
            
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
            
            # If fine-tuned pneumonia model is available, use it for enhanced pneumonia detection
            pneumonia_enhanced_prob = None
            if self.pneumonia_model is not None:
                with torch.no_grad():
                    pneumonia_output = self.pneumonia_model(input_var)
                    pneumonia_probs = F.softmax(pneumonia_output, dim=1).cpu().numpy()[0]
                    pneumonia_enhanced_prob = float(pneumonia_probs[1])  # Probability of pneumonia
                    
                    # Update pneumonia probability with enhanced model
                    results['Pneumonia'] = pneumonia_enhanced_prob
                    
                    # Update detected diseases list
                    if pneumonia_enhanced_prob > 0.5:
                        # Remove existing pneumonia entry if any
                        detected_diseases = [d for d in detected_diseases if d['disease'] != 'Pneumonia']
                        detected_diseases.append({
                            'disease': 'Pneumonia',
                            'probability': pneumonia_enhanced_prob,
                            'enhanced': True
                        })
            
            detected_diseases.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'model': 'CheXNet' + (' + Enhanced Pneumonia' if pneumonia_enhanced_prob else ''),
                'description': 'Chest X-ray disease detection',
                'all_probabilities': results,
                'detected_diseases': detected_diseases,
                'num_detections': len(detected_diseases),
                'pneumonia_enhanced': pneumonia_enhanced_prob is not None
            }
        
        except Exception as e:
            raise Exception(f"Error in CheXNet prediction: {str(e)}")
    
    def predict_for_frontend(self, image_path):
        try:
            raw_result = self.predict(image_path)
            
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
            
            analysis_results = []
            
            all_probs = raw_result.get('all_probabilities', {})
            
        # Sort by probability
            sorted_diseases = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            
            detected_above_threshold = [d for d, prob in sorted_diseases if prob > 0.5]
            
            if not detected_above_threshold:
                analysis_results.append({
                    'disease': 'Healthy Scan',
                    'confidence': int((1.0 - sorted_diseases[0][1]) * 100) if sorted_diseases else 95,
                    'status': 'healthy',
                    'description': 'No significant abnormalities detected. Lung fields appear clear. Normal cardiac silhouette.',
                    'regions': []
                })
                for disease, prob in sorted_diseases[:3]:
                    analysis_results.append({
                        'disease': disease.replace('_', ' '),
                        'confidence': int(prob * 100),
                        'status': 'healthy',
                        'description': f'No significant {disease.replace("_", " ").lower()} indicators detected.'
                    })
            else:
                for disease, prob in sorted_diseases:
                    if prob > 0.5:
                        status = 'critical' if prob > 0.75 else 'warning'
                        confidence = int(prob * 100)
                        description = disease_descriptions.get(disease, f'{disease.replace("_", " ")} indicators detected.')
                        
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
                        
                        if len([r for r in analysis_results if r['status'] != 'healthy']) >= 3:
                            break
                
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

