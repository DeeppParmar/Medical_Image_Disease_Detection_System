"""
Simple test script for the Flask API
Tests all endpoints with sample requests
"""

import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_list_models():
    """Test models listing endpoint"""
    print("\nTesting /models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict(model_name, image_path):
    """Test prediction endpoint"""
    print(f"\nTesting /predict/{model_name} endpoint...")
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/predict/{model_name}", files=files)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_all(image_path):
    """Test predict all endpoint"""
    print(f"\nTesting /predict/all endpoint...")
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/predict/all", files=files)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response keys: {list(result.keys())}")
        # Print summary for each model
        if 'results' in result:
            for model, prediction in result['results'].items():
                print(f"\n{model.upper()}:")
                if 'error' in prediction:
                    print(f"  Error: {prediction['error']}")
                else:
                    print(f"  Success: {prediction.get('model', 'N/A')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Flask API Test Suite")
    print("=" * 50)
    
    # Test basic endpoints
    test_health()
    test_list_models()
    
    # Test predictions (requires a test image)
    # You can provide a path to a test image
    test_image = input("\nEnter path to test image (or press Enter to skip): ").strip()
    
    if test_image and os.path.exists(test_image):
        # Test individual models
        for model in ['chexnet', 'mura', 'tuberculosis', 'rsna', 'unet']:
            test_predict(model, test_image)
        
        # Test all models
        test_predict_all(test_image)
    else:
        print("\nSkipping prediction tests (no image provided)")
    
    print("\n" + "=" * 50)
    print("Test Suite Complete")
    print("=" * 50)

if __name__ == '__main__':
    main()

