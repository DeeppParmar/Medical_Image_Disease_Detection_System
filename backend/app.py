
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import importlib.util
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import traceback
import threading
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from PIL import Image as PILImage

# ── Structured Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MediScan")

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.chexnet_inference import CheXNetPredictor
from models.mura_inference import MURAPredictor
from models.tuberculosis_inference import TuberculosisPredictor
from models.rsna_inference import RSNPredictor
from models.unet_inference import UNetPredictor
from models.gradcam import generate_gradcam, generate_gradcam_full

app = Flask(__name__)

# CORS — allow all origins for development; restrict in production via FRONTEND_URL
CORS(app, origins="*",
     supports_credentials=False,
     expose_headers=["X-Model-Used"])

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*",
                    async_mode='threading')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Thread pool for parallel model inference
executor = ThreadPoolExecutor(max_workers=4)

# Model predictors - will be preloaded
predictors = {
    'chexnet': None,
    'mura': None,
    'tuberculosis': None,
    'rsna': None,
    'unet': None
}

# Model loading status
model_status = {
    'chexnet': {'loaded': False, 'loading': False, 'error': None},
    'mura': {'loaded': False, 'loading': False, 'error': None},
    'tuberculosis': {'loaded': False, 'loading': False, 'error': None},
    'rsna': {'loaded': False, 'loading': False, 'error': None},
    'unet': {'loaded': False, 'loading': False, 'error': None}
}

# Analysis sessions for real-time tracking
analysis_sessions = {}

import gc
def safe_remove(filepath, retries=5, delay=0.3):
    """Safely remove a file, retrying on Windows file-lock errors."""
    for i in range(retries):
        try:
            gc.collect()  # release any lingering Python references
            if os.path.exists(filepath):
                os.remove(filepath)
            return
        except PermissionError:
            if i < retries - 1:
                time.sleep(delay)
            else:
                logger.warning(f"[CLEANUP] Could not delete {filepath} after {retries} retries")

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

def model_checkpoint_available(model_name: str) -> bool:
    base = os.path.join(os.path.dirname(__file__), 'datasets')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    if model_name == 'chexnet':
        return os.path.isfile(os.path.join(base, 'CheXNet', 'model.pth.tar'))
    
    if model_name == 'mura':
        # MURA model now uses ImageNet pretrained weights, always available
        # Can also check for fine-tuned model
        mura_models = os.path.join(base, 'DenseNet-MURA', 'models')
        has_finetuned = False
        if os.path.isdir(mura_models):
            for root, _, files in os.walk(mura_models):
                if 'model.pth' in files:
                    has_finetuned = True
                    break
        # Always return True since we use ImageNet pretrained backbone
        return True
    
    if model_name == 'tuberculosis':
        # First check for PyTorch model (preferred)
        pytorch_paths = [
            os.path.join(models_dir, 'tb', 'tb_model_best.pth'),
            os.path.join(models_dir, 'tb', 'tb_model_final.pth'),
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'tb', 'tb_model_best.pth'),
        ]
        for path in pytorch_paths:
            if os.path.isfile(path):
                return True
        
        # Fallback: Check for TensorFlow model
        if importlib.util.find_spec('tensorflow') is not None:
            tb_base = os.path.join(base, 'TuberculosisNet')
            candidates = [
                os.path.join(tb_base, 'models', 'Baseline'),
                os.path.join(tb_base, 'TB-Net')
            ]
            for d in candidates:
                if os.path.isdir(d):
                    files = os.listdir(d)
                    has_meta = any(f.endswith('.meta') for f in files)
                    has_index = any(f.endswith('.index') for f in files)
                    has_data = any('.data-' in f for f in files)
                    if has_meta and has_index and has_data:
                        return True
        return False
    
    if model_name == 'unet':
        return os.path.isfile(os.path.join(base, 'UNet', 'MODEL.pth'))
    
    if model_name == 'rsna':
        return os.path.isdir(os.path.join(base, 'rsna18'))
    
    return False

def unavailable_model_result(model_name: str):
    return [{
        'disease': 'Model Unavailable',
        'confidence': 0,
        'status': 'warning',
        'description': f'Required checkpoint files for {model_name} were not found under backend/datasets.',
        'regions': []
    }]

def model_error_result(model_name: str, message: str):
    return [{
        'disease': 'Model Error',
        'confidence': 0,
        'status': 'warning',
        'description': f'{model_name}: {message}',
        'regions': []
    }]

def infer_scan_type_from_image(filepath: str, ext: str) -> str:
    """
    Robust scan type detection using multiple advanced features.
    Accurately distinguishes: chest X-rays, bone X-rays (hand/wrist/limb), and CT scans.
    
    Key insight: Pneumonia/TB can disrupt normal bilateral lung pattern, so we need
    to detect chest X-rays even when pathology is present.
    """
    if ext == 'dcm':
        if importlib.util.find_spec('pydicom') is not None:
            try:
                import pydicom
                ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                modality = str(getattr(ds, 'Modality', '')).upper()
                body_part = str(getattr(ds, 'BodyPartExamined', '')).upper()
                
                if modality == 'CT':
                    return 'ct'
                if modality in {'CR', 'DX'}:
                    # Check body part if available
                    bone_parts = ['HAND', 'WRIST', 'FINGER', 'ELBOW', 'FOREARM', 'HUMERUS', 
                                  'SHOULDER', 'ANKLE', 'FOOT', 'KNEE', 'HIP', 'EXTREMITY']
                    if any(bp in body_part for bp in bone_parts):
                        return 'bone'
                    if 'CHEST' in body_part or 'LUNG' in body_part:
                        return 'chest'
                    return 'chest'  # Default CR/DX to chest
            except Exception:
                pass
        return 'unknown'
    
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 'unknown'
    
    original_h, original_w = img.shape[:2]
    aspect_ratio = original_w / original_h if original_h > 0 else 1.0
    
    # Resize for analysis
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img_f = img_resized.astype(np.float32) / 255.0

    # ============ FEATURE EXTRACTION ============
    
    # 1. Symmetry Analysis (chest X-rays are highly symmetric)
    left = img_f[:, :128]
    right = np.fliplr(img_f[:, 128:])
    try:
        symmetry = float(np.corrcoef(left.flatten(), right.flatten())[0, 1])
        if not np.isfinite(symmetry):
            symmetry = 0.0
    except:
        symmetry = 0.0

    # 2. Edge Analysis
    edges = cv2.Canny(img_resized, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    
    # Edge orientation - chest has more horizontal edges (ribs), bone has varied
    sobelx = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)
    horizontal_edges = float(np.sum(np.abs(sobely) > np.abs(sobelx))) / float(img_resized.size)
    
    # 3. Region Analysis - chest has characteristic lung fields (dark center)
    h, w = img_f.shape
    center = img_f[h//4:3*h//4, w//4:3*w//4]
    border = np.concatenate([img_f[:h//4, :].flatten(), img_f[3*h//4:, :].flatten(),
                             img_f[:, :w//4].flatten(), img_f[:, 3*w//4:].flatten()])
    center_mean = float(np.mean(center))
    border_mean = float(np.mean(border))
    center_darkness = border_mean - center_mean  # Positive = darker center (chest-like)
    
    # 4. Lung field detection - check for bilateral dark regions
    left_lung_region = img_f[h//4:3*h//4, w//8:w//2-w//8]
    right_lung_region = img_f[h//4:3*h//4, w//2+w//8:7*w//8]
    left_mean = float(np.mean(left_lung_region))
    right_mean = float(np.mean(right_lung_region))
    mediastinum = img_f[h//4:3*h//4, w//2-w//8:w//2+w//8]
    mediastinum_mean = float(np.mean(mediastinum))
    
    # Chest X-rays: dark lungs on both sides with brighter mediastinum in middle
    # For pneumonia/TB, one lung may be brighter (infected), so relax the difference threshold
    bilateral_lung_pattern = (mediastinum_mean > left_mean and mediastinum_mean > right_mean and
                              abs(left_mean - right_mean) < 0.15)
    
    # Relaxed lung pattern - at least one dark lung region visible
    # This helps detect chest X-rays with unilateral pathology (pneumonia affecting one lung)
    has_any_lung_region = (min(left_mean, right_mean) < 0.5 and mediastinum_mean > min(left_mean, right_mean))
    
    # 5. Aspect Ratio Analysis
    # Chest X-rays are typically square-ish (0.8-1.2) but can be wider (up to 1.6)
    # Hand/wrist X-rays are elongated (aspect < 0.7 or > 1.8)
    is_square = 0.75 < aspect_ratio < 1.35
    is_chest_like_ratio = 0.7 < aspect_ratio < 1.7  # Wider range for chest
    
    # 6. Histogram Analysis - chest has bimodal (lung vs tissue), bone is different
    hist = cv2.calcHist([img_resized], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    
    # Find peaks in histogram
    from scipy.ndimage import maximum_filter
    try:
        local_max = maximum_filter(hist, size=20)
        peaks = np.where((hist == local_max) & (hist > 0.01))[0]
        num_peaks = len(peaks)
    except:
        num_peaks = 1
    
    # 7. Bone structure detection - high intensity thin structures
    _, bone_thresh = cv2.threshold(img_resized, 200, 255, cv2.THRESH_BINARY)
    bone_ratio = float(np.sum(bone_thresh > 0)) / float(bone_thresh.size)
    
    # Detect elongated bright structures (bones)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bone_edges = cv2.morphologyEx(bone_thresh, cv2.MORPH_GRADIENT, kernel)
    elongated_bright = float(np.sum(bone_edges > 0)) / float(bone_edges.size)
    
    # 8. Connected component analysis - hands have multiple separate bone structures
    _, binary = cv2.threshold(img_resized, 180, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Filter small components
    significant_components = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 100)
    
    # 9. Rib pattern detection - chest X-rays have characteristic horizontal rib lines
    # Apply horizontal line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(img_resized, cv2.MORPH_OPEN, horizontal_kernel)
    rib_pattern_strength = float(np.sum(horizontal_lines > 50)) / float(horizontal_lines.size)
    
    # 10. Spine/mediastinum detection - bright vertical structure in center
    center_strip = img_f[:, w//3:2*w//3]
    center_vertical_brightness = float(np.mean(center_strip))
    has_central_spine = center_vertical_brightness > 0.4
    
    # 11. Overall brightness analysis - bone X-rays on dark background are darker overall
    # Chest X-rays have higher overall brightness due to soft tissue
    _, binary_otsu = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_region_ratio = float(np.sum(binary_otsu > 0)) / float(binary_otsu.size)
    overall_brightness = float(np.mean(img_f))
    
    # Bone X-rays on dark background: low brightness (overall < 0.25 is very strong indicator)
    is_dark_background_xray = overall_brightness < 0.25
    is_very_dark_xray = overall_brightness < 0.20  # Very dark = almost certainly bone
    
    # ============ SCORING ============
    
    # Very strong indicators - immediately decide
    # Extreme aspect ratio strongly indicates bone (hand/wrist/finger)
    # Hand/wrist X-rays are typically tall and narrow (aspect < 0.65)
    if aspect_ratio < 0.65:
        print(f"   🔍 Narrow image (aspect={aspect_ratio:.2f}) - detecting as BONE")
        return 'bone'
    
    # Very wide images could be bone panoramic or poorly cropped
    if aspect_ratio > 2.0:
        print(f"   🔍 Very wide image (aspect={aspect_ratio:.2f}) - detecting as BONE")
        return 'bone'
    
    # VERY dark images are almost always bone X-rays (elbow, finger, etc.)
    if is_very_dark_xray:
        print(f"   🔍 Very dark image (brightness={overall_brightness:.2f}) - detecting as BONE")
        return 'bone'
    
    # Dark background X-ray with narrow aspect = likely bone
    if is_dark_background_xray and aspect_ratio < 0.9:
        print(f"   🔍 Dark background X-ray (bright={bright_region_ratio:.2f}, mean={overall_brightness:.2f}) + narrow aspect - detecting as BONE")
        return 'bone'
    
    # LOW symmetry is a strong indicator of bone X-rays (not symmetric like chest)
    if symmetry < 0.4 and is_dark_background_xray:
        print(f"   🔍 Low symmetry ({symmetry:.2f}) + dark background - detecting as BONE")
        return 'bone'
    
    # HIGH SYMMETRY with high brightness is a strong indicator of chest X-ray
    # Even with pneumonia, chest X-rays maintain symmetry > 0.7 and have higher brightness
    if symmetry > 0.75 and is_chest_like_ratio and bone_ratio < 0.15 and bright_region_ratio > 0.5:
        print(f"   🔍 High symmetry ({symmetry:.2f}) + chest-like ratio + high brightness - detecting as CHEST")
        return 'chest'
    
    # Very high symmetry with bilateral pattern = definite chest
    if symmetry > 0.85 and bilateral_lung_pattern:
        print(f"   🔍 High symmetry + bilateral lungs - detecting as CHEST")
        return 'chest'
    
    # Very low symmetry with narrow aspect = likely bone
    if symmetry < 0.4 and aspect_ratio < 0.8:
        print(f"   🔍 Low symmetry + narrow - detecting as BONE")
        return 'bone'
    
    # Chest X-ray indicators
    chest_indicators = [
        symmetry > 0.7,                    # High symmetry (STRONG for chest)
        bilateral_lung_pattern,            # Dark bilateral lung fields
        has_any_lung_region,               # At least one lung visible (for pathology cases)
        center_darkness > 0.0,             # Darker or equal center
        is_chest_like_ratio,               # Chest-compatible aspect ratio
        horizontal_edges > 0.35,           # Horizontal rib patterns (lowered threshold)
        num_peaks >= 1,                    # At least some histogram structure
        bone_ratio < 0.15,                 # Less bright bone areas (STRONG indicator)
        rib_pattern_strength > 0.05,       # Horizontal rib lines detected
        has_central_spine,                 # Bright central spine/mediastinum
        significant_components < 15,       # Fewer separate regions than hand X-rays
        overall_brightness > 0.35,         # Higher overall brightness (chest has soft tissue) - STRONG
    ]
    
    # Bone X-ray indicators
    bone_indicators = [
        symmetry < 0.6,                    # Low symmetry (STRONG for bone)
        not bilateral_lung_pattern,        # No bilateral lung pattern
        not has_any_lung_region,           # No lung region visible
        center_darkness < -0.1,            # Bright center (not lungs)
        not is_chest_like_ratio,           # Non-chest aspect ratio
        edge_density > 0.10,               # High edge density (bone edges)
        bone_ratio > 0.12,                 # Visible bone structures (STRONG)
        elongated_bright > 0.03,           # Elongated bright structures
        significant_components >= 15,      # Multiple separate bone structures (fingers)
        aspect_ratio < 0.75 or aspect_ratio > 1.6,  # Elongated shape
        is_dark_background_xray,           # Dark background X-ray (STRONG for bone)
        overall_brightness < 0.3,          # Low overall brightness (VERY STRONG for bone)
    ]
    
    # Calculate weighted scores
    # Give higher weight to most reliable indicators
    chest_weights = [2.0, 1.5, 1.0, 0.5, 1.0, 0.5, 0.3, 2.0, 0.5, 0.5, 0.5, 2.5]  # symmetry, bone_ratio, brightness weighted higher
    bone_weights = [2.0, 0.5, 1.0, 0.5, 1.0, 0.5, 2.0, 0.5, 1.0, 1.5, 2.5, 3.0]  # low symmetry, dark_background, low brightness weighted highest
    
    chest_score = sum(w * int(ind) for w, ind in zip(chest_weights, chest_indicators)) / sum(chest_weights)
    bone_score = sum(w * int(ind) for w, ind in zip(bone_weights, bone_indicators)) / sum(bone_weights)
    
    # Additional strong indicators
    if bilateral_lung_pattern and symmetry > 0.75:
        chest_score += 0.15
    if has_any_lung_region and symmetry > 0.7 and bone_ratio < 0.1 and overall_brightness > 0.35:
        chest_score += 0.1  # Boost for pathology cases with good symmetry, low bone, high brightness
    if is_dark_background_xray:
        bone_score += 0.15  # Boost for dark background bone X-rays
    if overall_brightness < 0.2:
        bone_score += 0.2  # Strong boost for very dark images (definitely bone)
    if significant_components >= 15 and bone_ratio > 0.15:
        bone_score += 0.15
    if aspect_ratio < 0.65 or aspect_ratio > 1.8:  # Very elongated = likely bone
        bone_score += 0.15
    
    # Normalize
    total = chest_score + bone_score
    if total > 0:
        chest_score /= total
        bone_score /= total
    
    print(f"   🔍 Image Analysis:")
    print(f"      Symmetry: {symmetry:.2f}, Edge density: {edge_density:.2f}")
    print(f"      Bilateral lungs: {bilateral_lung_pattern}, Has any lung: {has_any_lung_region}")
    print(f"      Center darkness: {center_darkness:.2f}, Rib pattern: {rib_pattern_strength:.2f}")
    print(f"      Aspect ratio: {aspect_ratio:.2f}, Bone ratio: {bone_ratio:.2f}")
    print(f"      Components: {significant_components}, Peaks: {num_peaks}")
    print(f"      Bright region: {bright_region_ratio:.2f}, Overall brightness: {overall_brightness:.2f}")
    print(f"      Dark background xray: {is_dark_background_xray}")
    print(f"   📊 Scores: chest={chest_score:.2f}, bone={bone_score:.2f}")
    
    # Decision - lower bone threshold since we improved the indicators
    if bone_score > 0.50:  # Bone threshold
        print(f"   ✅ Detected as: BONE X-ray")
        return 'bone'
    if chest_score > 0.45:  # Lower threshold for chest (prefer chest for medical safety)
        print(f"   ✅ Detected as: CHEST X-ray")
        return 'chest'
    
    # If uncertain, use additional heuristics
    if bilateral_lung_pattern or (has_any_lung_region and bright_region_ratio > 0.5):
        print(f"   ✅ Detected as: CHEST X-ray (lung region detected)")
        return 'chest'
    if significant_components >= 12 or is_dark_background_xray:
        print(f"   ✅ Detected as: BONE X-ray (bone features detected)")
        return 'bone'
    
    print(f"   ⚠️ Uncertain, defaulting to: CHEST")
    return 'chest'  # Default to chest as it's more common

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Early Disease Detection API is running'
    })

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({
        'models': [
            {
                'name': 'chexnet',
                'available': model_checkpoint_available('chexnet'),
                'description': 'Chest X-ray disease detection (14 diseases)',
                'diseases': ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                           'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
                           'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            },
            {
                'name': 'mura',
                'available': model_checkpoint_available('mura'),
                'description': 'Musculoskeletal radiographs abnormality detection'
            },
            {
                'name': 'tuberculosis',
                'available': model_checkpoint_available('tuberculosis'),
                'description': 'Tuberculosis detection from chest X-rays'
            },
            {
                'name': 'rsna',
                'available': model_checkpoint_available('rsna'),
                'description': 'RSNA Intracranial Hemorrhage Detection'
            },
            {
                'name': 'unet',
                'available': model_checkpoint_available('unet'),
                'description': 'Medical image segmentation'
            }
        ]
    })

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if model_name not in predictors:
        return jsonify({'error': f'Model {model_name} not found'}), 404

    if not model_checkpoint_available(model_name):
        resp = jsonify(unavailable_model_result(model_name))
        resp.headers['X-Model-Used'] = model_name
        return resp, 503
    
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
        
        predictor = get_predictor(model_name)
        result = predictor.predict(filepath)
        
        # Clean up uploaded file
        safe_remove(filepath)
        
        resp = jsonify({
            'success': True,
            'model': model_name,
            'result': result
        })
        resp.headers['X-Model-Used'] = model_name
        return resp
    
    except Exception as e:
        # Clean up on error
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        safe_remove(filepath)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ── Grad-CAM++ target layer mapping per model ─────────────────────────
GRADCAM_LAYERS = {
    'chexnet':      'densenet121.features.denseblock4',
    'tuberculosis': 'backbone.features.denseblock4',
    'mura':         'backbone.features.denseblock4',
    'rsna':         'layer4',
}

def _attach_heatmap(result_list, filepath, model_name, predictor):
    """
    Attempt to generate a Grad-CAM++ heatmap and attach views to result_list[0].
    Attaches: heatmap (overlay), heatmap_threshold, heatmap_bbox, heatmap_regions.
    Gracefully sets heatmap=None when generation is not possible.
    """
    heatmap_data = None
    try:
        layer = GRADCAM_LAYERS.get(model_name)
        torch_model = None
        device = None
        preprocess_fn = None

        if model_name == 'chexnet' and hasattr(predictor, 'model') and predictor.model is not None:
            torch_model = predictor.model
            device = predictor.device
            preprocess_fn = predictor.preprocess_image
        elif model_name == 'tuberculosis' and getattr(predictor, 'use_pytorch', False) and predictor.pytorch_model is not None:
            torch_model = predictor.pytorch_model
            device = predictor.pytorch_device
        elif model_name == 'mura' and hasattr(predictor, 'model') and predictor.model is not None:
            torch_model = predictor.model
            device = predictor.device
        elif model_name == 'rsna' and hasattr(predictor, 'model') and predictor.model is not None:
            torch_model = predictor.model
            device = predictor.device

        if torch_model is not None and layer is not None:
            heatmap_data = generate_gradcam_full(filepath, torch_model, layer,
                                                  device=device, preprocess_fn=preprocess_fn)
    except Exception as hm_err:
        logger.error(f"[GRADCAM++] Heatmap generation failed for {model_name}: {hm_err}")

    if result_list and isinstance(result_list, list) and len(result_list) > 0:
        if heatmap_data:
            result_list[0]['heatmap'] = heatmap_data.get('overlay')
            result_list[0]['heatmap_threshold'] = heatmap_data.get('threshold')
            result_list[0]['heatmap_bbox'] = heatmap_data.get('bbox')
            result_list[0]['heatmap_regions'] = heatmap_data.get('regions', [])
        else:
            result_list[0]['heatmap'] = None
            result_list[0]['heatmap_threshold'] = None
            result_list[0]['heatmap_bbox'] = None
            result_list[0]['heatmap_regions'] = []
    return result_list


@app.route('/api/analyze', methods=['POST'])
def analyze():
    logger.info("[REQUEST] " + "="*50)
    logger.info("[REQUEST] New analysis request received")
    
    if 'image' not in request.files and 'file' not in request.files:
        logger.error("[ERROR] no_file: No image file provided")
        return jsonify({'error': 'unsupported_scan', 'message': 'No image file provided'}), 400
    
    file = request.files.get('image') or request.files.get('file')
    
    if file.filename == '':
        logger.error("[ERROR] no_file: No file selected")
        return jsonify({'error': 'unsupported_scan', 'message': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.error(f"[ERROR] unsupported_scan: File type not allowed - {file.filename}")
        return jsonify({'error': 'unsupported_scan', 'message': 'File type not allowed. Supported: PNG, JPG, JPEG, DICOM (.dcm)'}), 415
    
    filepath = None
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"[REQUEST] New analysis request received | filename={filename}")

        # ── Server-side validation ──────────────────────────────────────
        # 1. File size check (15 MB server limit)
        file_size = os.path.getsize(filepath)
        if file_size > 15 * 1024 * 1024:
            safe_remove(filepath)
            logger.error(f"[ERROR] file_too_large: {file_size} bytes")
            return jsonify({'error': 'file_too_large', 'message': 'File exceeds 15MB server limit'}), 413

        # 2. Image decode check
        ext = os.path.splitext(filename.lower())[1].lstrip('.')
        response_warnings = []
        if ext != 'dcm':
            try:
                pil_img = PILImage.open(filepath)
                pil_img.verify()  # verify integrity
                pil_img = PILImage.open(filepath)  # re-open after verify
                img_w, img_h = pil_img.size
            except Exception:
                safe_remove(filepath)
                logger.error(f"[ERROR] unsupported_scan: Cannot decode image {filename}")
                return jsonify({'error': 'unsupported_scan', 'message': 'Cannot decode image'}), 415

            # 3. Resolution check
            if img_w < 32 or img_h < 32:
                safe_remove(filepath)
                logger.error(f"[ERROR] resolution_too_low: {img_w}x{img_h}")
                return jsonify({'error': 'resolution_too_low', 'message': f'Image resolution too low ({img_w}x{img_h}). Minimum 32x32 required.'}), 422

            # 4. Blur detection (Laplacian variance)
            try:
                gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if lap_var < 50:
                        response_warnings.append('low_image_quality')
                        logger.warning(f"[VALIDATION] Low image quality detected | laplacian_var={lap_var:.1f}")
            except Exception:
                pass  # non-blocking

        scan_type = (request.form.get('scan_type') or request.args.get('scan_type') or 'auto').strip().lower()
        requested_model = (request.form.get('model') or request.args.get('model') or '').strip().lower()
        filename_lower = filename.lower()
        if ext == '':
            ext = os.path.splitext(filename_lower)[1].lstrip('.')
        logger.info(f"[REQUEST] scan_type={scan_type} | model={requested_model or 'auto'} | ext={ext}")

        if scan_type == 'auto':
            inferred_scan_type = infer_scan_type_from_image(filepath, ext)
            scan_type = inferred_scan_type
            logger.info(f"[ROUTING] Inferred scan type: {scan_type}")

        bone_keywords = ['wrist', 'hand', 'elbow', 'shoulder', 'humerus', 'finger', 'forearm', 'ankle', 'foot', 'knee', 'hip', 'bone', 'mura']
        ct_keywords = ['ct', 'brain', 'head', 'intracranial', 'rsna']

        inferred_model = None
        if ext == 'dcm':
            inferred_model = 'rsna'
        elif any(k in filename_lower for k in bone_keywords):
            inferred_model = 'mura'
        elif any(k in filename_lower for k in ct_keywords):
            inferred_model = 'rsna'

        # Prioritize explicitly requested model
        if requested_model and requested_model in predictors:
            selected_model = requested_model
        elif scan_type == 'bone':
            selected_model = 'mura'
        elif scan_type == 'ct':
            selected_model = 'rsna'
        elif scan_type == 'chest':
            selected_model = None  # Will try TB then CheXNet
        else:
            selected_model = inferred_model

        model_name_for_log = selected_model or 'auto (TB→CheXNet)'
        logger.info(f"[ROUTING] Model selected: {model_name_for_log}")

        # ── Helper: build response with optional warnings + heatmap ──
        def _build_response(result, used_model, filepath_to_clean):
            t_end = time.time()
            result = _attach_heatmap(result, filepath_to_clean, used_model, predictors.get(used_model))
            if response_warnings and isinstance(result, list) and len(result) > 0:
                result[0]['warnings'] = response_warnings
            safe_remove(filepath_to_clean)
            if isinstance(result, list) and len(result) > 0:
                logger.info(f"[INFERENCE] Model={used_model} | Time={t_end - t0:.2f}s | TopResult={result[0].get('disease', '?')} @ {result[0].get('confidence', 0):.0f}%")
            logger.info(f"[RESPONSE] Returning {len(result) if isinstance(result, list) else 1} results to frontend")
            resp = jsonify(result)
            resp.headers['X-Model-Used'] = used_model
            return resp

        t0 = time.time()

        if selected_model == 'rsna':
            logger.info("[MODEL] Loading predictor: rsna")
            if not model_checkpoint_available('rsna'):
                safe_remove(filepath)
                logger.error("[ERROR] model_not_loaded: RSNA model not available")
                return jsonify({'error': 'model_not_loaded', 'message': 'RSNA model weights not found'}), 503
            predictor = get_predictor('rsna')
            result = predictor.predict_for_frontend(filepath)
            return _build_response(result, 'rsna', filepath)

        if selected_model == 'mura':
            logger.info("[MODEL] Loading predictor: mura")
            if not model_checkpoint_available('mura'):
                safe_remove(filepath)
                logger.error("[ERROR] model_not_loaded: MURA model not available")
                return jsonify({'error': 'model_not_loaded', 'message': 'MURA model weights not found'}), 503
            try:
                predictor = get_predictor('mura')
                result = predictor.predict_for_frontend(filepath)
                return _build_response(result, 'mura', filepath)
            except Exception as e:
                safe_remove(filepath)
                logger.error(f"[ERROR] processing_failed: MURA Error: {e}")
                return jsonify({'error': 'processing_failed', 'message': str(e)}), 500
        
        # Explicitly requested tuberculosis model
        if selected_model == 'tuberculosis':
            logger.info("[MODEL] Loading predictor: tuberculosis")
            if not model_checkpoint_available('tuberculosis'):
                safe_remove(filepath)
                logger.error("[ERROR] model_not_loaded: Tuberculosis model not available")
                return jsonify({'error': 'model_not_loaded', 'message': 'Tuberculosis model weights not found'}), 503
            try:
                tb_predictor = get_predictor('tuberculosis')
                result = tb_predictor.predict_for_frontend(filepath)
                return _build_response(result, 'tuberculosis', filepath)
            except Exception as e:
                safe_remove(filepath)
                logger.error(f"[ERROR] processing_failed: Tuberculosis Error: {e}")
                return jsonify({'error': 'processing_failed', 'message': str(e)}), 500
        
        # Explicitly requested chexnet model
        if selected_model == 'chexnet':
            logger.info("[MODEL] Loading predictor: chexnet")
            if not model_checkpoint_available('chexnet'):
                safe_remove(filepath)
                logger.error("[ERROR] model_not_loaded: CheXNet model not available")
                return jsonify({'error': 'model_not_loaded', 'message': 'CheXNet model weights not found'}), 503
            try:
                predictor = get_predictor('chexnet')
                result = predictor.predict_for_frontend(filepath)
                return _build_response(result, 'chexnet', filepath)
            except Exception as e:
                safe_remove(filepath)
                logger.error(f"[ERROR] processing_failed: CheXNet Error: {e}")
                return jsonify({'error': 'processing_failed', 'message': str(e)}), 500
        
        # Auto-detect flow: Handle based on detected scan type
        
        # For unknown scan type, try MURA first to check if it's a bone image
        if scan_type == 'unknown':
            try:
                if model_checkpoint_available('mura'):
                    logger.info("[ROUTING] Unknown scan type — checking MURA for bone analysis")
                    mura_predictor = get_predictor('mura')
                    mura_raw = mura_predictor.predict(filepath)
                    mura_prob = float(mura_raw.get('abnormality_probability', 0.0))
                    mura_conf = float(mura_raw.get('confidence', 0.0))
                    logger.info(f"[INFERENCE] MURA probe: abnormality={mura_prob:.2%}, confidence={mura_conf:.2%}")
                    if mura_conf > 0.6 or mura_prob > 0.5:
                        result = mura_predictor.predict_for_frontend(filepath)
                        return _build_response(result, 'mura', filepath)
            except Exception as mura_error:
                logger.warning(f"[ROUTING] MURA check failed: {mura_error}")
        
        # Only try TB model for chest X-rays (not unknown)
        tb_checked = False
        tb_is_positive = False
        if scan_type == 'chest':
            logger.info("[MODEL] Loading predictor: tuberculosis (auto-check)")
            try:
                if not model_checkpoint_available('tuberculosis'):
                    raise Exception('Tuberculosis model checkpoint not found')
                tb_predictor = get_predictor('tuberculosis')
                tb_raw = tb_predictor.predict(filepath)
                tb_prob = tb_raw.get('tuberculosis_probability', 0.0)
                normal_prob = tb_raw.get('normal_probability', 0.0)
                logger.info(f"[INFERENCE] TB probe: Normal={normal_prob:.2%}, TB={tb_prob:.2%}")
                tb_checked = True
                
                if 'is_tuberculosis' in tb_raw and tb_raw['is_tuberculosis'] and tb_prob > 0.55:
                    logger.info(f"[INFERENCE] Tuberculosis DETECTED with {tb_prob:.2%} confidence")
                    tb_is_positive = True
                    result = tb_predictor.predict_for_frontend(filepath)
                    return _build_response(result, 'tuberculosis', filepath)
                
                logger.info(f"[ROUTING] No TB detected, falling through to CheXNet")
            except Exception as tb_error:
                logger.warning(f"[ROUTING] TB model check failed: {tb_error}")

        if scan_type != 'chest':
            try:
                if model_checkpoint_available('mura'):
                    logger.info("[MODEL] Loading predictor: mura (non-chest fallback)")
                    mura_predictor = get_predictor('mura')
                    mura_raw = mura_predictor.predict(filepath)
                    mura_prob = float(mura_raw.get('abnormality_probability', 0.0))
                    logger.info(f"[INFERENCE] MURA probe: abnormality={mura_prob:.2%}")
                    if mura_prob > 0.65:
                        result = mura_predictor.predict_for_frontend(filepath)
                        return _build_response(result, 'mura', filepath)
            except Exception as mura_error:
                logger.warning(f"[ROUTING] MURA check failed: {mura_error}")
        
        # Default to CheXNet for general chest X-ray analysis
        logger.info("[MODEL] Loading predictor: chexnet (default)")
        if not model_checkpoint_available('chexnet'):
            safe_remove(filepath)
            logger.error("[ERROR] model_not_loaded: CheXNet model not available")
            return jsonify({'error': 'model_not_loaded', 'message': 'CheXNet model weights not found'}), 503
        try:
            predictor = get_predictor('chexnet')
            result = predictor.predict_for_frontend(filepath)
            return _build_response(result, 'chexnet', filepath)
        except Exception as e:
            safe_remove(filepath)
            logger.error(f"[ERROR] processing_failed: CheXNet Error: {e}")
            return jsonify({'error': 'processing_failed', 'message': str(e)}), 500
    
    except Exception as e:
        # Clean up on error
        safe_remove(filepath)
        
        logger.error(f"[ERROR] processing_failed: {traceback.format_exc()}")
        return jsonify({
            'error': 'processing_failed',
            'message': str(e)
        }), 500

@app.route('/predict/all', methods=['POST'])
def predict_all():
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
        
        for model_name in predictors.keys():
            try:
                if not model_checkpoint_available(model_name):
                    results[model_name] = {'error': unavailable_model_result(model_name)[0]['description']}
                    continue
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
    port = int(os.environ.get("PORT", 5000))
    print("Starting Early Disease Detection API...")
    print("Available models: chexnet, mura, tuberculosis, rsna, unet")
    print("Frontend endpoint: /api/analyze")
    print(f"Backend running on: http://0.0.0.0:{port}")
    print("CORS enabled for: * (all origins)")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)

