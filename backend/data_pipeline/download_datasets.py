"""
Data Download Pipeline for All Datasets
Downloads datasets in real-time for training
"""

import os
import sys
import subprocess
import zipfile
import tarfile
import requests
from pathlib import Path
import json

class DatasetDownloader:
    def __init__(self, base_dir='datasets'):
        self.base_dir = base_dir
        self.datasets_config = {
            'chexnet': {
                'name': 'ChestX-ray14',
                'url': 'https://nihcc.app.box.com/v/ChestXray-NIHCC',
                'type': 'manual',  # Requires manual download
                'instructions': 'Download from https://nihcc.app.box.com/v/ChestXray-NIHCC and extract to datasets/CheXNet/ChestX-ray14/images/'
            },
            'mura': {
                'name': 'MURA',
                'url': 'https://stanfordmlgroup.github.io/competitions/mura/',
                'type': 'kaggle',
                'kaggle_dataset': 'cjinny/mura-v11',
                'path': 'datasets/DenseNet-MURA/data'
            },
            'tuberculosis': {
                'name': 'Tuberculosis Chest X-ray',
                'url': 'https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset',
                'type': 'kaggle',
                'kaggle_dataset': 'tawsifurrahman/tuberculosis-tb-chest-xray-dataset',
                'path': 'datasets/TuberculosisNet/data'
            },
            'rsna': {
                'name': 'RSNA Intracranial Hemorrhage Detection',
                'url': 'https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection',
                'type': 'kaggle',
                'kaggle_dataset': 'c/rsna-intracranial-hemorrhage-detection',
                'path': 'datasets/rsna18/data'
            },
            'unet': {
                'name': 'Carvana Image Masking',
                'url': 'https://www.kaggle.com/c/carvana-image-masking-challenge',
                'type': 'kaggle',
                'kaggle_dataset': 'c/carvana-image-masking-challenge',
                'path': 'datasets/UNet/data'
            }
        }
    
    def check_kaggle_credentials(self):
        """Check if Kaggle API credentials are set up"""
        # Check environment variable first
        if os.environ.get('KAGGLE_API_TOKEN'):
            print("✓ Kaggle API token found in environment variable")
            return True
        
        # Check kaggle.json file
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if kaggle_json.exists():
            print("✓ Kaggle API credentials found in kaggle.json")
            return True
        
        print("Kaggle API credentials not found!")
        print("Please run: python setup_kaggle_token.py")
        print("Or set environment variable: export KAGGLE_API_TOKEN=your_token")
        return False
    
    def download_kaggle_dataset(self, dataset_name, output_path):
        """Download dataset from Kaggle"""
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            print(f"Downloading {dataset_name} from Kaggle...")
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Initialize Kaggle API
            api = KaggleApi()
            
            # Authenticate - try environment variable first, then kaggle.json
            if os.environ.get('KAGGLE_API_TOKEN'):
                # Set token from environment
                os.environ['KAGGLE_USERNAME'] = os.environ.get('KAGGLE_USERNAME', 'kaggle')
                api.authenticate()
            else:
                # Use kaggle.json
                api.authenticate()
            
            # Download dataset
            print(f"Downloading to {output_path}...")
            print("This may take a while depending on dataset size...")
            api.dataset_download_files(dataset_name, path=output_path, unzip=True)
            print(f"✓ Successfully downloaded {dataset_name} to {output_path}")
            return True
        except ImportError:
            print(f"Error: kaggle package not installed")
            print("Install it with: pip install kaggle")
            return False
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            print("\nTroubleshooting:")
            print("1. Ensure kaggle package is installed: pip install kaggle")
            print("2. Run: python setup_kaggle_token.py")
            print("3. Or set environment variable: export KAGGLE_API_TOKEN=your_token")
            return False
    
    def download_chexnet(self):
        """Download CheXNet dataset (manual download required)"""
        print("\n=== CheXNet Dataset ===")
        print("CheXNet dataset requires manual download.")
        print("Please visit: https://nihcc.app.box.com/v/ChestXray-NIHCC")
        print("Download and extract images to: datasets/CheXNet/ChestX-ray14/images/")
        
        # Check if images directory exists
        images_dir = os.path.join(self.base_dir, 'CheXNet', 'ChestX-ray14', 'images')
        if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
            print(f"✓ Images directory found at {images_dir}")
            return True
        else:
            print(f"✗ Images directory not found at {images_dir}")
            return False
    
    def download_mura(self):
        """Download MURA dataset"""
        print("\n=== MURA Dataset ===")
        if not self.check_kaggle_credentials():
            print("Attempting to download MURA dataset...")
            print("MURA dataset can be downloaded from: https://stanfordmlgroup.github.io/competitions/mura/")
            print("Or from Kaggle: https://www.kaggle.com/cjinny/mura-v11")
            return False
        
        config = self.datasets_config['mura']
        output_path = os.path.join(self.base_dir, 'DenseNet-MURA')
        
        # MURA dataset structure expects MURA-v1.0 in the root
        if os.path.exists(os.path.join(output_path, 'MURA-v1.0')):
            print(f"✓ MURA dataset found at {output_path}")
            return True
        
        try:
            return self.download_kaggle_dataset(config['kaggle_dataset'], output_path)
        except:
            print("Kaggle download failed. Please download manually from:")
            print("https://stanfordmlgroup.github.io/competitions/mura/")
            return False
    
    def download_tuberculosis(self):
        """Download Tuberculosis dataset"""
        print("\n=== Tuberculosis Dataset ===")
        if not self.check_kaggle_credentials():
            return False
        
        config = self.datasets_config['tuberculosis']
        output_path = os.path.join(self.base_dir, 'TuberculosisNet', 'data')
        return self.download_kaggle_dataset(config['kaggle_dataset'], output_path)
    
    def download_rsna(self):
        """Download RSNA dataset"""
        print("\n=== RSNA Dataset ===")
        if not self.check_kaggle_credentials():
            return False
        
        config = self.datasets_config['rsna']
        output_path = os.path.join(self.base_dir, 'rsna18', 'data')
        return self.download_kaggle_dataset(config['kaggle_dataset'], output_path)
    
    def download_unet(self):
        """Download UNet dataset"""
        print("\n=== UNet Dataset ===")
        if not self.check_kaggle_credentials():
            return False
        
        config = self.datasets_config['unet']
        output_path = os.path.join(self.base_dir, 'UNet', 'data')
        return self.download_kaggle_dataset(config['kaggle_dataset'], output_path)
    
    def download_all(self):
        """Download all datasets"""
        print("=" * 50)
        print("Starting dataset download pipeline...")
        print("=" * 50)
        
        results = {
            'chexnet': self.download_chexnet(),
            'mura': self.download_mura(),
            'tuberculosis': self.download_tuberculosis(),
            'rsna': self.download_rsna(),
            'unet': self.download_unet()
        }
        
        print("\n" + "=" * 50)
        print("Download Summary:")
        print("=" * 50)
        for dataset, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"{dataset.upper()}: {status}")
        
        return results

if __name__ == '__main__':
    downloader = DatasetDownloader()
    
    import argparse
    parser = argparse.ArgumentParser(description='Download datasets for early disease detection')
    parser.add_argument('--dataset', type=str, choices=['chexnet', 'mura', 'tuberculosis', 'rsna', 'unet', 'all'],
                       default='all', help='Dataset to download')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        downloader.download_all()
    elif args.dataset == 'chexnet':
        downloader.download_chexnet()
    elif args.dataset == 'mura':
        downloader.download_mura()
    elif args.dataset == 'tuberculosis':
        downloader.download_tuberculosis()
    elif args.dataset == 'rsna':
        downloader.download_rsna()
    elif args.dataset == 'unet':
        downloader.download_unet()

