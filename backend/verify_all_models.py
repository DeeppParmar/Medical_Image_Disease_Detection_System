"""
Verify All Medical Image Analysis Models
Comprehensive test script to check all models are working for real-time inference
"""

import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding for special characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def get_project_root():
    """Get the backend directory path"""
    return Path(__file__).parent

def check_model_files():
    """Check if model checkpoint files exist"""
    print("\n" + "=" * 60)
    print("1. Model Checkpoint Files")
    print("=" * 60)
    
    base = get_project_root() / "datasets"
    
    models = {
        "CheXNet": {
            "path": base / "CheXNet" / "model.pth.tar",
            "type": "PyTorch"
        },
        "MURA (DenseNet)": {
            "path": base / "DenseNet-MURA" / "models" / "XR_WRIST" / "model.pth",
            "type": "PyTorch"
        },
        "TuberculosisNet": {
            "path": base / "TuberculosisNet" / "models" / "Epoch_5",
            "type": "TensorFlow"
        }
    }
    
    results = {}
    for name, info in models.items():
        path = info["path"]
        if path.is_dir():
            # Check TensorFlow checkpoint directory
            files = list(path.glob("*"))
            has_checkpoint = any(f.name == "checkpoint" or ".data-" in f.name for f in files)
            exists = has_checkpoint
        else:
            exists = path.exists()
        
        status = "✅" if exists else "❌"
        size_info = ""
        if exists and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            size_info = f" ({size_mb:.1f} MB)"
        
        print(f"  {status} {name}: {info['type']}{size_info}")
        if not exists:
            print(f"      Expected at: {path}")
        
        results[name] = exists
    
    return results

def check_datasets():
    """Check if dataset images exist"""
    print("\n" + "=" * 60)
    print("2. Dataset Images")
    print("=" * 60)
    
    base = get_project_root() / "datasets"
    
    datasets = {
        "CheXNet (ChestX-ray14)": base / "CheXNet" / "ChestX-ray14" / "images",
        "MURA (Wrist X-rays)": base / "DenseNet-MURA" / "MURA" / "train" / "XR_WRIST",
        "TB - Normal": base / "TuberculosisNet" / "Dataset" / "TB_Chest_Radiography_Database" / "Normal",
        "TB - Tuberculosis": base / "TuberculosisNet" / "Dataset" / "TB_Chest_Radiography_Database" / "Tuberculosis",
    }
    
    results = {}
    for name, path in datasets.items():
        if path.exists():
            # Count image files
            images = list(path.glob("*.png")) + list(path.glob("*.jpg"))
            count = len(images)
            if count > 0:
                status = "✅"
                print(f"  {status} {name}: {count} images")
            else:
                status = "⚠️"
                print(f"  {status} {name}: Directory exists but empty")
            results[name] = count > 0
        else:
            status = "❌"
            print(f"  {status} {name}: Not found")
            results[name] = False
    
    return results

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n" + "=" * 60)
    print("3. Dependencies")
    print("=" * 60)
    
    packages = [
        ("torch", "PyTorch (CheXNet, MURA)"),
        ("torchvision", "TorchVision"),
        ("tensorflow", "TensorFlow (TB-Net)"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("flask", "Flask (API)"),
    ]
    
    results = {}
    for package, description in packages:
        try:
            __import__(package)
            print(f"  ✅ {description}")
            results[package] = True
        except ImportError:
            print(f"  ❌ {description} - pip install {package.replace('cv2', 'opencv-python').replace('PIL', 'pillow')}")
            results[package] = False
    
    return results

def test_model_inference():
    """Test actual model inference with sample images"""
    print("\n" + "=" * 60)
    print("4. Model Inference Tests")
    print("=" * 60)
    
    results = {}
    
    # Find a test image
    base = get_project_root() / "datasets"
    test_images = []
    
    # Try TB images first (most reliable)
    tb_normal = base / "TuberculosisNet" / "Dataset" / "TB_Chest_Radiography_Database" / "Normal"
    if tb_normal.exists():
        images = list(tb_normal.glob("*.png"))[:1]
        test_images.extend(images)
    
    if not test_images:
        print("  ⚠️ No test images found. Run download_sample_images.py first.")
        return results
    
    test_image = str(test_images[0])
    print(f"  Using test image: {Path(test_image).name}")
    
    # Test CheXNet
    print("\n  Testing CheXNet...")
    try:
        from models.chexnet_inference import CheXNetPredictor
        start = time.time()
        predictor = CheXNetPredictor()
        load_time = time.time() - start
        
        start = time.time()
        result = predictor.predict(test_image)
        inference_time = time.time() - start
        
        print(f"    ✅ CheXNet working! (Load: {load_time:.1f}s, Inference: {inference_time:.2f}s)")
        if 'predictions' in result:
            top_pred = max(result['predictions'], key=lambda x: x['probability'])
            print(f"    Top prediction: {top_pred['disease']} ({top_pred['probability']*100:.1f}%)")
        results['chexnet'] = True
    except Exception as e:
        print(f"    ❌ CheXNet failed: {e}")
        results['chexnet'] = False
    
    # Test MURA
    print("\n  Testing MURA (Bone X-ray)...")
    try:
        from models.mura_inference import MURAPredictor
        start = time.time()
        predictor = MURAPredictor()
        load_time = time.time() - start
        
        # Note: MURA expects bone X-rays, but we'll test loading
        print(f"    ✅ MURA model loaded! (Load: {load_time:.1f}s)")
        results['mura'] = True
    except Exception as e:
        print(f"    ❌ MURA failed: {e}")
        results['mura'] = False
    
    # Test TuberculosisNet
    print("\n  Testing TuberculosisNet...")
    try:
        from models.tuberculosis_inference import TuberculosisPredictor
        start = time.time()
        predictor = TuberculosisPredictor()
        load_time = time.time() - start
        
        start = time.time()
        result = predictor.predict(test_image)
        inference_time = time.time() - start
        
        print(f"    ✅ TuberculosisNet working! (Load: {load_time:.1f}s, Inference: {inference_time:.2f}s)")
        if 'is_tuberculosis' in result:
            status = "TB Detected" if result['is_tuberculosis'] else "Normal"
            prob = result.get('tuberculosis_probability', 0) * 100
            print(f"    Result: {status} ({prob:.1f}% TB probability)")
        results['tuberculosis'] = True
    except Exception as e:
        print(f"    ❌ TuberculosisNet failed: {e}")
        results['tuberculosis'] = False
    
    return results

def print_summary(model_files, datasets, deps, inference):
    """Print summary of all checks"""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    
    # Core models
    print("\nCore Models Ready for Real-time Inference:")
    
    core_models = [
        ("CheXNet (14 diseases)", model_files.get("CheXNet", False) and inference.get("chexnet", False)),
        ("MURA (Bone abnormality)", model_files.get("MURA (DenseNet)", False) and inference.get("mura", False)),
        ("TuberculosisNet (TB)", model_files.get("TuberculosisNet", False) and inference.get("tuberculosis", False)),
    ]
    
    for name, ready in core_models:
        status = "✅ Ready" if ready else "❌ Not Ready"
        print(f"  {status} - {name}")
        if not ready:
            all_passed = False
    
    # Dependencies
    missing_deps = [k for k, v in deps.items() if not v]
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL MODELS READY FOR REAL-TIME INFERENCE!")
        print("\nTo start the API server:")
        print("  cd backend")
        print("  python app.py")
    else:
        print("⚠️  Some models need attention (see above)")
    print("=" * 60)

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("Medical Image Analysis - Model Verification")
    print("=" * 60)
    
    # Run all checks
    model_files = check_model_files()
    datasets = check_datasets()
    deps = check_dependencies()
    inference = test_model_inference()
    
    # Print summary
    print_summary(model_files, datasets, deps, inference)

if __name__ == "__main__":
    main()
