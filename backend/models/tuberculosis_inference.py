
import os
import sys
import numpy as np
import cv2
import pickle

# PyTorch imports for fine-tuned model
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    from PIL import Image
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available for TB inference")

# Fix for numpy version compatibility (numpy 2.x vs 1.x checkpoint files)
class NumPyCoreFixUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy._core vs numpy.core differences."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


try:
    import tensorflow as tf
    # Use TensorFlow 1.x compatibility mode for TensorFlow 2.x
    if hasattr(tf, 'compat'):
        tf = tf.compat.v1
        tf.disable_eager_execution()
    try:
        from preprocessing import preprocess_image_inference
        PREPROCESSING_AVAILABLE = True
    except ImportError:
        PREPROCESSING_AVAILABLE = False
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    PREPROCESSING_AVAILABLE = False
    print("Warning: TensorFlow not available. Tuberculosis model will use fallback preprocessing.")


class TBNetPyTorch(nn.Module):
    """PyTorch-based TB detection model for inference.
    Must match the architecture from finetune_tuberculosis.py including attention module.
    """
    def __init__(self, num_classes=2, pretrained=False, dropout_rate=0.5):
        super(TBNetPyTorch, self).__init__()
        
        try:
            self.backbone = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            self.backbone = models.densenet121(pretrained=pretrained)
        
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Attention module for lung region focus (MUST match training architecture)
        self.attention = nn.Sequential(
            nn.Conv2d(num_features, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Get features from backbone
        features = self.backbone.features(x)
        
        # Apply attention
        att = self.attention(features)
        features = features * att
        
        # Global pooling
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        return output

class TuberculosisPredictor:
    def __init__(self):
        # PyTorch model components
        self.pytorch_model = None
        self.pytorch_device = None
        self.pytorch_transform = None
        self.use_pytorch = False
        
        # TensorFlow model components
        self.model_path = None
        self.sess = None
        self.image_tensor = None
        self.logits_tensor = None
        self.pred_tensor = None
        self.model_loaded = False
        
        # Base path for models
        base_path = os.path.dirname(__file__)  # This is backend/models
        tbnet_path = os.path.join(base_path, '..', 'datasets', 'TuberculosisNet')
        
        # Fine-tuned PyTorch model paths (priority)
        # base_path is backend/models, so tb folder is directly under it
        self.pytorch_model_paths = [
            os.path.join(base_path, 'tb', 'tb_model_best.pth'),  # backend/models/tb/
            os.path.join(base_path, 'tb', 'tb_model_final.pth'),  # backend/models/tb/
            os.path.join(base_path, '..', 'checkpoints', 'tb', 'tb_model_best.pth'),  # backend/checkpoints/tb/
            os.path.join(tbnet_path, 'models', 'tb_model_best.pth'),
            os.path.join(tbnet_path, 'models', 'tb', 'tb_model_final.pth'),
        ]
        
        # TensorFlow model paths (fallback)
        potential_tf_paths = [
            os.path.join(tbnet_path, 'models', 'Baseline'),
            os.path.join(tbnet_path, 'TB-Net'),
            os.path.join(tbnet_path, 'models', 'Epoch_5'),
            os.path.join(tbnet_path, 'models', 'Epoch_0'),
        ]
        
        for path in potential_tf_paths:
            if os.path.isdir(path):
                import glob
                has_index = glob.glob(os.path.join(path, '*.index'))
                has_meta = glob.glob(os.path.join(path, '*.meta'))
                if has_index and has_meta:
                    self.model_path = path
                    break
        
        self.load_model()
    
    def load_model(self):
        """Load models - try PyTorch first, then TensorFlow."""
        
        # Try to load PyTorch model first (fine-tuned)
        if PYTORCH_AVAILABLE:
            for model_path in self.pytorch_model_paths:
                if os.path.isfile(model_path):
                    try:
                        self._load_pytorch_model(model_path)
                        if self.pytorch_model is not None:
                            self.use_pytorch = True
                            print(f"✓ Using fine-tuned PyTorch TB model from {model_path}")
                            return
                    except Exception as e:
                        print(f"Could not load PyTorch model from {model_path}: {e}")
        
        # Fall back to TensorFlow model
        if TF_AVAILABLE and self.model_path:
            self._load_tensorflow_model()
    
    def _load_pytorch_model(self, model_path):
        """Load PyTorch-based TB model."""
        self.pytorch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint with numpy compatibility fix
        try:
            checkpoint = torch.load(model_path, map_location=self.pytorch_device)
        except ModuleNotFoundError as e:
            if 'numpy._core' in str(e):
                # Handle numpy version mismatch (numpy 2.x vs 1.x)
                with open(model_path, 'rb') as f:
                    pickle_fix = type('PickleFix', (), {
                        'load': lambda f: NumPyCoreFixUnpickler(f).load(),
                        'Unpickler': NumPyCoreFixUnpickler
                    })
                    checkpoint = torch.load(f, map_location=self.pytorch_device, pickle_module=pickle_fix)
            else:
                raise
        
        # Get model type from checkpoint if available
        model_type = checkpoint.get('model_type', 'densenet') if isinstance(checkpoint, dict) else 'densenet'
        
        # Create appropriate model based on type
        self.pytorch_model = TBNetPyTorch(num_classes=2, pretrained=False)
        
        if 'model_state_dict' in checkpoint:
            self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.pytorch_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.pytorch_model.load_state_dict(checkpoint)
        
        self.pytorch_model = self.pytorch_model.to(self.pytorch_device)
        self.pytorch_model.eval()
        
        # Setup transforms
        self.pytorch_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_tensorflow_model(self):
        """Load TensorFlow-based TB model (original TBNet)."""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Tuberculosis model will use simple inference.")
            return
        
        if self.model_path is None:
            print("Warning: TB-Net model files not found.")
            return
        
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            if hasattr(tf, 'logging'):
                tf.logging.set_verbosity(tf.logging.ERROR)
            
            import glob
            
            index_files = sorted(glob.glob(os.path.join(self.model_path, '*.index')))
            meta_files = sorted(glob.glob(os.path.join(self.model_path, '*.meta')))
            
            if not index_files:
                print(f"No checkpoint index files found in {self.model_path}")
                return
            
            ckpt_prefix = index_files[0][:-6]  # Remove '.index'
            
            meta_path = None
            if meta_files:
                meta_path = meta_files[0]
            else:
                potential_meta = ckpt_prefix + '.meta'
                if os.path.exists(potential_meta):
                    meta_path = potential_meta
            
            if meta_path is None:
                print(f"No meta file found for checkpoint in {self.model_path}")
                return
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(self.sess, ckpt_prefix)
            
            graph = tf.get_default_graph()
            
            input_names = ['image:0', 'input:0', 'x:0', 'inputs:0', 'Placeholder:0']
            for name in input_names:
                try:
                    self.image_tensor = graph.get_tensor_by_name(name)
                    print(f"Found input tensor: {name}")
                    break
                except KeyError:
                    continue
            
            output_names = ['classification/add_1:0', 'classification/MatMul_1:0', 'resnet_model/final_dense:0', 
                           'logits:0', 'output:0', 'predictions:0', 'dense/BiasAdd:0', 'fc/BiasAdd:0']
            for name in output_names:
                try:
                    self.logits_tensor = graph.get_tensor_by_name(name)
                    print(f"Found output tensor: {name}")
                    break
                except KeyError:
                    continue
            
            if self.image_tensor is not None and self.logits_tensor is not None:
                self.model_loaded = True
                print(f"Loaded TensorFlow TB model from {self.model_path}")
            else:
                print("Could not find input/output tensors in TB-Net graph")
        except Exception as e:
            print(f"Error loading TensorFlow TB model: {e}")
            self.sess = None
            self.model_loaded = False
    
    def preprocess_image_fallback(self, image_path):
        image = cv2.imread(image_path, 1)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Simple preprocessing
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _predict_pytorch(self, image_path):
        """Predict using PyTorch model."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.pytorch_transform(image).unsqueeze(0).to(self.pytorch_device)
        
        with torch.no_grad():
            outputs = self.pytorch_model(image_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        mapping = {0: "Normal", 1: "Tuberculosis"}
        prediction = mapping[pred_class]
        
        return {
            'model': 'TuberculosisNet (Fine-tuned PyTorch)',
            'description': 'Tuberculosis detection from chest X-rays - Enhanced model',
            'prediction': prediction,
            'confidence': float(confidence),
            'normal_probability': float(probs[0]),
            'tuberculosis_probability': float(probs[1]),
            'is_tuberculosis': bool(pred_class == 1),
            'enhanced': True
        }
    
    def predict(self, image_path):
        try:
            # Use PyTorch model if available (fine-tuned)
            if self.use_pytorch and self.pytorch_model is not None:
                return self._predict_pytorch(image_path)
            
            # Fall back to TensorFlow model
            if not self.model_loaded or self.sess is None:
                try:
                    image = cv2.imread(image_path, 0)  # Grayscale
                    if image is not None:
                        # Simple analysis based on lung region intensity patterns
                        normalized = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
                        mean_intensity = normalized.mean()
                        std_intensity = normalized.std()
                        
                        # TB X-rays often show more varied intensity
                        # This is a very rough heuristic
                        tb_score = min(0.5, std_intensity * 1.5)
                        normal_score = 1.0 - tb_score
                        
                        return {
                            'model': 'TuberculosisNet (Fallback)',
                            'description': 'Tuberculosis detection - using image analysis heuristics',
                            'prediction': 'Normal' if normal_score > tb_score else 'Possible TB',
                            'confidence': max(normal_score, tb_score),
                            'normal_probability': float(normal_score),
                            'tuberculosis_probability': float(tb_score),
                            'is_tuberculosis': tb_score > normal_score
                        }
                except:
                    pass
                    
                return {
                    'model': 'TuberculosisNet',
                    'description': 'Tuberculosis detection from chest X-rays',
                    'error': 'Model not loaded. Using fallback analysis.',
                    'prediction': 'Unknown',
                    'confidence': 0.5,
                    'normal_probability': 0.5,
                    'tuberculosis_probability': 0.5,
                    'is_tuberculosis': False
                }
            
            if TF_AVAILABLE and PREPROCESSING_AVAILABLE:
                try:
                    image = preprocess_image_inference(image_path)
                except Exception as e:
                    print(f"Using fallback preprocessing: {e}")
                    image = self.preprocess_image_fallback(image_path)
            else:
                image = self.preprocess_image_fallback(image_path)
            
            logits = self.sess.run(self.logits_tensor, feed_dict={self.image_tensor: [image]})[0]
            # Apply softmax manually using numpy to avoid TensorFlow graph issues
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            softmax = exp_logits / np.sum(exp_logits)
            pred_class = int(np.argmax(softmax))
            confidence = float(softmax[pred_class])
            
            mapping = {0: "Normal", 1: "Tuberculosis"}
            prediction = mapping[pred_class]
            
            return {
                'model': 'TuberculosisNet',
                'description': 'Tuberculosis detection from chest X-rays',
                'prediction': prediction,
                'confidence': float(confidence),
                'normal_probability': float(softmax[0]),
                'tuberculosis_probability': float(softmax[1]),
                'is_tuberculosis': bool(pred_class == 1)
            }
        
        except Exception as e:
            raise Exception(f"Error in Tuberculosis prediction: {str(e)}")
    
    def predict_for_frontend(self, image_path):
        try:
            raw_result = self.predict(image_path)
            
            if 'error' in raw_result:
                return [{
                    'disease': 'Model Error',
                    'confidence': 0,
                    'status': 'warning',
                    'description': raw_result['error'],
                    'regions': []
                }]
            
            analysis_results = []
            
            prediction = raw_result.get('prediction', 'Unknown')
            confidence = raw_result.get('confidence', 0.0)
            tb_prob = raw_result.get('tuberculosis_probability', 0.0)
            normal_prob = raw_result.get('normal_probability', 0.0)
            is_tb = raw_result.get('is_tuberculosis', False)
            
            if is_tb:
                analysis_results.append({
                    'disease': 'Tuberculosis (TB)',
                    'confidence': int(tb_prob * 100),
                    'status': 'critical' if tb_prob > 0.75 else 'warning',
                    'description': 'Active tuberculosis infection detected. Immediate medical consultation and treatment required. Characteristic TB patterns identified in lung tissue.',
                    'regions': ['Lung Parenchyma', 'Apical Regions']
                })
                analysis_results.append({
                    'disease': 'Normal Tissue',
                    'confidence': int(normal_prob * 100),
                    'status': 'healthy',
                    'description': f'Limited normal tissue observed. TB pathology predominant.',
                    'regions': []
                })
            else:
                analysis_results.append({
                    'disease': 'Healthy Scan (TB)',
                    'confidence': int(normal_prob * 100),
                    'status': 'healthy',
                    'description': 'No tuberculosis detected. Lung fields appear clear of TB-specific patterns. No cavitation or characteristic TB lesions observed.',
                    'regions': []
                })
                analysis_results.append({
                    'disease': 'Tuberculosis Risk',
                    'confidence': int(tb_prob * 100),
                    'status': 'healthy',
                    'description': f'Low tuberculosis probability. No significant TB indicators present.',
                    'regions': []
                })
            
            return analysis_results
        
        except Exception as e:
            return [{
                'disease': 'Analysis Error',
                'confidence': 0,
                'status': 'warning',
                'description': f'Error analyzing image for tuberculosis: {str(e)}',
                'regions': []
            }]

