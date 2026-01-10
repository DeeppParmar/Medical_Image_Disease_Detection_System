"""
Example usage of the Early Disease Detection API
Demonstrates how to use the Flask backend programmatically
"""

import requests
import json
import os

# API Configuration
BASE_URL = "http://localhost:5000"

def example_health_check():
    """Example: Check API health"""
    print("=" * 50)
    print("Example 1: Health Check")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def example_list_models():
    """Example: List all available models"""
    print("=" * 50)
    print("Example 2: List Available Models")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/models")
    models = response.json()['models']
    
    for model in models:
        print(f"\nModel: {model['name']}")
        print(f"Description: {model['description']}")
        if 'diseases' in model:
            print(f"Diseases: {', '.join(model['diseases'][:5])}...")
    print()

def example_predict_chexnet(image_path):
    """Example: Predict using CheXNet"""
    print("=" * 50)
    print("Example 3: CheXNet Prediction")
    print("=" * 50)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/predict/chexnet", files=files)
    
    result = response.json()
    
    if result['success']:
        print(f"Model: {result['result']['model']}")
        print(f"Detected Diseases: {result['result']['num_detections']}")
        
        if result['result']['detected_diseases']:
            print("\nTop Detections:")
            for disease in result['result']['detected_diseases'][:5]:
                print(f"  - {disease['disease']}: {disease['probability']:.2%}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    print()

def example_predict_all_models(image_path):
    """Example: Predict using all models"""
    print("=" * 50)
    print("Example 4: Predict with All Models")
    print("=" * 50)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/predict/all", files=files)
    
    result = response.json()
    
    if result['success']:
        for model_name, prediction in result['results'].items():
            print(f"\n{model_name.upper()}:")
            if 'error' in prediction:
                print(f"  Error: {prediction['error']}")
            else:
                print(f"  Status: Success")
                if 'prediction' in prediction:
                    print(f"  Prediction: {prediction['prediction']}")
                if 'detected_diseases' in prediction:
                    print(f"  Detections: {prediction['num_detections']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    print()

def example_batch_predict(image_paths, model_name='chexnet'):
    """Example: Batch prediction for multiple images"""
    print("=" * 50)
    print(f"Example 5: Batch Prediction with {model_name}")
    print("=" * 50)
    
    results = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Skipping {image_path} (not found)")
            continue
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/predict/{model_name}", files=files)
        
        result = response.json()
        results.append({
            'image': os.path.basename(image_path),
            'success': result['success'],
            'result': result.get('result', {})
        })
    
    print(f"\nProcessed {len(results)} images:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['image']}")
    print()

def main():
    """Run all examples"""
    print("\n" + "=" * 50)
    print("Early Disease Detection API - Usage Examples")
    print("=" * 50 + "\n")
    
    # Example 1: Health check
    try:
        example_health_check()
    except Exception as e:
        print(f"Error in health check: {e}")
        print("Make sure the Flask server is running on http://localhost:5000")
        return
    
    # Example 2: List models
    try:
        example_list_models()
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Examples 3-5 require an image file
    # Replace with path to your test image
    test_image = input("Enter path to test image (or press Enter to skip prediction examples): ").strip()
    
    if test_image and os.path.exists(test_image):
        # Example 3: Single model prediction
        try:
            example_predict_chexnet(test_image)
        except Exception as e:
            print(f"Error in CheXNet prediction: {e}")
        
        # Example 4: All models prediction
        try:
            example_predict_all_models(test_image)
        except Exception as e:
            print(f"Error in all models prediction: {e}")
        
        # Example 5: Batch prediction
        try:
            example_batch_predict([test_image], 'chexnet')
        except Exception as e:
            print(f"Error in batch prediction: {e}")
    else:
        print("\nSkipping prediction examples (no image provided)")
    
    print("\n" + "=" * 50)
    print("Examples Complete!")
    print("=" * 50)

if __name__ == '__main__':
    main()

