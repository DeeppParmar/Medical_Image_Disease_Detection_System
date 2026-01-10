"""
Flask Backend for Early Disease Detection
Handles image uploads and routes them to appropriate models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from werkzeug.utils import secure_filename
import traceback

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.chexnet_inference import CheXNetPredictor
from models.mura_inference import MURAPredictor
from models.tuberculosis_inference import TuberculosisPredictor
from models.rsna_inference import RSNPredictor
from models.unet_inference import UNetPredictor

app = Flask(__name__)
# Configure CORS to allow frontend (running on port 8080)
CORS(app, origins=["http://localhost:8080", "http://127.0.0.1:8080"], supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize model predictors (lazy loading)
predictors = {
    'chexnet': None,
    'mura': None,
    'tuberculosis': None,
    'rsna': None,
    'unet': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_predictor(model_name):
    """Lazy load model predictors"""
    if predictors[model_name] is None:
        try:
            if model_name == 'chexnet':
                predictors[model_name] = CheXNetPredictor()
            elif model_name == 'mura':
                predictors[model_name] = MURAPredictor()
            elif model_name == 'tuberculosis':
                predictors[model_name] = TuberculosisPredictor()
            elif model_name == 'rsna':
                predictors[model_name] = RSNPredictor()
            elif model_name == 'unet':
                predictors[model_name] = UNetPredictor()
        except Exception as e:
            print(f"Error loading {model_name} model: {str(e)}")
            raise
    return predictors[model_name]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Early Disease Detection API is running'
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'models': [
            {
                'name': 'chexnet',
                'description': 'Chest X-ray disease detection (14 diseases)',
                'diseases': ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                           'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
                           'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            },
            {
                'name': 'mura',
                'description': 'Musculoskeletal radiographs abnormality detection'
            },
            {
                'name': 'tuberculosis',
                'description': 'Tuberculosis detection from chest X-rays'
            },
            {
                'name': 'rsna',
                'description': 'RSNA Intracranial Hemorrhage Detection'
            },
            {
                'name': 'unet',
                'description': 'Medical image segmentation'
            }
        ]
    })

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    """Predict endpoint for a specific model"""
    if model_name not in predictors:
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictor and make prediction
        predictor = get_predictor(model_name)
        result = predictor.predict(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'model': model_name,
            'result': result
        })
    
    except Exception as e:
        # Clean up on error
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint for frontend - intelligently routes between TB and CheXNet models"""
    if 'image' not in request.files and 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Handle both 'image' (frontend) and 'file' (alternative) field names
    file = request.files.get('image') or request.files.get('file')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Supported: PNG, JPG, JPEG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Try TB model first (fast inference)
        try:
            tb_predictor = get_predictor('tuberculosis')
            tb_raw = tb_predictor.predict(filepath)
            
            # If TB is detected with high confidence, return TB results
            if 'is_tuberculosis' in tb_raw and tb_raw['is_tuberculosis']:
                tb_prob = tb_raw.get('tuberculosis_probability', 0.0)
                if tb_prob > 0.6:  # High TB confidence threshold
                    result = tb_predictor.predict_for_frontend(filepath)
                    os.remove(filepath)
                    return jsonify(result)  # This is already an array
        except Exception as tb_error:
            app.logger.warning(f"TB model failed, falling back to CheXNet: {tb_error}")
        
        # Use CheXNet for comprehensive analysis (default)
        predictor = get_predictor('chexnet')
        result = predictor.predict_for_frontend(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)  # This is already an array
    
    except Exception as e:
        # Clean up on error
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        if os.path.exists(filepath):
            os.remove(filepath)
        
        app.logger.error(f"Analysis error: {traceback.format_exc()}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/predict/all', methods=['POST'])
def predict_all():
    """Predict using all models"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = {}
        
        # Run predictions for all models
        for model_name in predictors.keys():
            try:
                predictor = get_predictor(model_name)
                results[model_name] = predictor.predict(filepath)
            except Exception as e:
                results[model_name] = {
                    'error': str(e)
                }
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        # Clean up on error
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    print("Starting Early Disease Detection API...")
    print("Available models: chexnet, mura, tuberculosis, rsna, unet")
    print("Frontend endpoint: /api/analyze")
    print("Backend running on: http://0.0.0.0:5000")
    print("CORS enabled for: http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=5000)

