"""
UNet Segmentation Inference Module
Medical image segmentation for disease detection
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Add UNet dataset path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'UNet'))

try:
    from unet.unet_model import UNet
    UNET_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'UNet'))
        from unet.unet_model import UNet
        UNET_AVAILABLE = True
    except ImportError:
        UNET_AVAILABLE = False
        print("Warning: UNet module not available. Using fallback implementation.")

class UNetFallback(nn.Module):
    """Fallback UNet implementation"""
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(UNetFallback, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Simplified UNet architecture
        self.inc = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 128, 3, padding=1)
        self.down2 = nn.Conv2d(128, 256, 3, padding=1)
        self.up1 = nn.Conv2d(256, 128, 3, padding=1)
        self.up2 = nn.Conv2d(128, 64, 3, padding=1)
        self.outc = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        x = F.relu(self.inc(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.down1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.down2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.up1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.up2(x))
        x = self.outc(x)
        return x

class UNetPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'UNet', 'MODEL.pth'
        )
        self.load_model()
    
    def load_model(self):
        """Load the UNet model"""
        try:
            if UNET_AVAILABLE:
                self.model = UNet(n_channels=3, n_classes=2, bilinear=False)
            else:
                self.model = UNetFallback(n_channels=3, n_classes=2, bilinear=False)
            
            if os.path.isfile(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device)
                # Handle different state dict formats
                if 'mask_values' in state_dict:
                    mask_values = state_dict.pop('mask_values')
                self.model.load_state_dict(state_dict)
                print(f"Loaded UNet model from {self.model_path}")
            else:
                print(f"Warning: Model file not found at {self.model_path}. Using untrained model.")
            
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading UNet model: {e}")
            # Create model as fallback
            if UNET_AVAILABLE:
                self.model = UNet(n_channels=3, n_classes=2, bilinear=False)
            else:
                self.model = UNetFallback(n_channels=3, n_classes=2, bilinear=False)
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def preprocess_image(self, image_path, scale_factor=0.5):
        """Preprocess image for UNet"""
        image = Image.open(image_path).convert('RGB')
        
        # Resize based on scale factor
        w, h = image.size
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Convert to tensor
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        
        return image_tensor.unsqueeze(0), image.size
    
    def predict(self, image_path, threshold=0.5):
        """Predict segmentation mask from medical image"""
        try:
            # Preprocess image
            image_tensor, original_size = self.preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(image_tensor)
                # Resize output to original size
                output = F.interpolate(output, size=original_size[::-1], mode='bilinear', align_corners=False)
                
                # Check number of classes
                n_classes = getattr(self.model, 'n_classes', 2)
                if n_classes > 1:
                    mask = output.argmax(dim=1)
                else:
                    mask = (torch.sigmoid(output) > threshold).long()
            
            mask_np = mask[0].cpu().numpy()
            
            # Calculate statistics
            total_pixels = mask_np.size
            if n_classes > 1:
                disease_pixels = np.sum(mask_np > 0)
            else:
                disease_pixels = np.sum(mask_np)
            
            disease_percentage = (disease_pixels / total_pixels) * 100
            
            return {
                'model': 'UNet',
                'description': 'Medical image segmentation',
                'disease_percentage': float(disease_percentage),
                'total_pixels': int(total_pixels),
                'disease_pixels': int(disease_pixels),
                'has_disease': bool(disease_percentage > 5.0),  # Threshold: 5% of image
                'mask_shape': list(mask_np.shape)
            }
        
        except Exception as e:
            raise Exception(f"Error in UNet prediction: {str(e)}")

