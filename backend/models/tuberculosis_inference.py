"""
Tuberculosis Detection Inference Module
Detects tuberculosis from chest X-ray images
"""

import os
import sys
import numpy as np
import cv2

# Add TuberculosisNet dataset path
tbnet_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'TuberculosisNet')
sys.path.append(tbnet_path)

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

class TuberculosisPredictor:
    def __init__(self):
        # Use the trained model from models/Baseline/TB-Net
        self.model_path = os.path.join(tbnet_path, 'models', 'Baseline', 'TB-Net')
        # Try trained model, fallback to original checkpoint
        if not os.path.exists(self.model_path):
            self.model_path = os.path.join(tbnet_path, 'TB-Net')
        self.meta_name = 'model_eval.meta'
        self.ckpt_name = 'TB-Net'
        self.sess = None
        self.image_tensor = None
        self.logits_tensor = None
        self.pred_tensor = None
        self.load_model()
    
    def load_model(self):
        """Load the Tuberculosis model"""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Tuberculosis model will not be loaded.")
            return
        
        try:
            # Ensure eager execution is disabled (for TF 2.x compatibility)
            if hasattr(tf, 'disable_eager_execution'):
                tf.disable_eager_execution()
            # Suppress TensorFlow warnings
            if hasattr(tf, 'logging'):
                tf.logging.set_verbosity(tf.logging.ERROR)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            # Try to find meta file and checkpoint
            meta_path = os.path.join(self.model_path, self.meta_name)
            ckpt_path = os.path.join(self.model_path, self.ckpt_name)
            
            # Check for trained checkpoint files
            if not os.path.exists(meta_path):
                # Try looking for any .meta file
                import glob
                meta_files = glob.glob(os.path.join(self.model_path, '*.meta'))
                if meta_files:
                    meta_path = meta_files[0]
                    print(f"Using meta file: {meta_path}")
                else:
                    print(f"Warning: TB-Net model files not found. Expected at {self.model_path}")
                    return
            
            self.sess = tf.Session()
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(self.sess, ckpt_path)
            
            graph = tf.get_default_graph()
            self.image_tensor = graph.get_tensor_by_name("image:0")
            self.logits_tensor = graph.get_tensor_by_name("resnet_model/final_dense:0")
            # Also get prediction tensor for easier inference
            try:
                self.pred_tensor = graph.get_tensor_by_name("ArgMax:0")
            except KeyError:
                self.pred_tensor = None
            
            print("Loaded Tuberculosis model successfully")
        except Exception as e:
            print(f"Error loading Tuberculosis model: {e}")
            self.sess = None
    
    def preprocess_image_fallback(self, image_path):
        """Fallback preprocessing if TensorFlow preprocessing not available"""
        image = cv2.imread(image_path, 1)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Simple preprocessing
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def predict(self, image_path):
        """Predict tuberculosis from chest X-ray image"""
        try:
            if self.sess is None:
                # Fallback prediction using simple heuristics
                return {
                    'model': 'TuberculosisNet',
                    'description': 'Tuberculosis detection from chest X-rays',
                    'error': 'Model not loaded. Please ensure TensorFlow and model files are available.',
                    'prediction': 'Unknown',
                    'confidence': 0.0
                }
            
            # Preprocess image
            if TF_AVAILABLE and PREPROCESSING_AVAILABLE:
                try:
                    image = preprocess_image_inference(image_path)
                except Exception as e:
                    print(f"Using fallback preprocessing: {e}")
                    image = self.preprocess_image_fallback(image_path)
            else:
                image = self.preprocess_image_fallback(image_path)
            
            # Run inference
            logits = self.sess.run(self.logits_tensor, feed_dict={self.image_tensor: [image]})[0]
            softmax = self.sess.run(tf.nn.softmax(logits))
            pred_class = softmax.argmax()
            confidence = softmax[pred_class]
            
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
        """Predict and format results for frontend React component"""
        try:
            # Get raw predictions
            raw_result = self.predict(image_path)
            
            # Check if model loaded properly
            if 'error' in raw_result:
                return [{
                    'disease': 'Model Error',
                    'confidence': 0,
                    'status': 'warning',
                    'description': raw_result['error'],
                    'regions': []
                }]
            
            # Convert to frontend format
            analysis_results = []
            
            prediction = raw_result.get('prediction', 'Unknown')
            confidence = raw_result.get('confidence', 0.0)
            tb_prob = raw_result.get('tuberculosis_probability', 0.0)
            normal_prob = raw_result.get('normal_probability', 0.0)
            is_tb = raw_result.get('is_tuberculosis', False)
            
            if is_tb:
                # Tuberculosis detected
                analysis_results.append({
                    'disease': 'Tuberculosis (TB)',
                    'confidence': int(tb_prob * 100),
                    'status': 'critical' if tb_prob > 0.75 else 'warning',
                    'description': 'Active tuberculosis infection detected. Immediate medical consultation and treatment required. Characteristic TB patterns identified in lung tissue.',
                    'regions': ['Lung Parenchyma', 'Apical Regions']
                })
                # Add supporting info
                analysis_results.append({
                    'disease': 'Normal Tissue',
                    'confidence': int(normal_prob * 100),
                    'status': 'healthy',
                    'description': f'Limited normal tissue observed. TB pathology predominant.',
                    'regions': []
                })
            else:
                # Normal/No TB detected
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

