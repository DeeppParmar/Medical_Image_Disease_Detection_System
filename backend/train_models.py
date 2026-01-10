"""
Training Script for All Models
Trains all 5 models for early disease detection
"""

import os
import sys
import argparse
import subprocess

def train_chexnet():
    """Train CheXNet model"""
    print("\n" + "=" * 50)
    print("Training CheXNet Model")
    print("=" * 50)
    
    chexnet_dir = os.path.join('datasets', 'CheXNet')
    train_script = os.path.join(chexnet_dir, 'train.py')
    
    if not os.path.exists(train_script):
        # Create a basic training script if it doesn't exist
        print("Creating CheXNet training script...")
        # Training would be done using the model.py file
        print("Note: CheXNet training requires the ChestX-ray14 dataset")
        print("Please ensure dataset is downloaded before training")
        return False
    
    try:
        os.chdir(chexnet_dir)
        subprocess.run([sys.executable, 'model.py'], check=True)
        os.chdir('..')
        os.chdir('..')
        return True
    except Exception as e:
        print(f"Error training CheXNet: {e}")
        os.chdir('..')
        os.chdir('..')
        return False

def train_mura():
    """Train MURA model"""
    print("\n" + "=" * 50)
    print("Training MURA Model")
    print("=" * 50)
    
    mura_dir = os.path.join('datasets', 'DenseNet-MURA')
    train_script = os.path.join(mura_dir, 'main.py')
    
    if not os.path.exists(train_script):
        print("MURA training script not found")
        return False
    
    try:
        os.chdir(mura_dir)
        subprocess.run([sys.executable, 'main.py'], check=True)
        os.chdir('..')
        os.chdir('..')
        return True
    except Exception as e:
        print(f"Error training MURA: {e}")
        os.chdir('..')
        os.chdir('..')
        return False

def train_tuberculosis():
    """Train Tuberculosis model"""
    print("\n" + "=" * 50)
    print("Training Tuberculosis Model")
    print("=" * 50)
    
    tb_dir = os.path.join('datasets', 'TuberculosisNet')
    train_script = os.path.join(tb_dir, 'train_tbnet.py')
    
    if not os.path.exists(train_script):
        print("Tuberculosis training script not found")
        return False
    
    try:
        os.chdir(tb_dir)
        subprocess.run([sys.executable, 'train_tbnet.py'], check=True)
        os.chdir('..')
        os.chdir('..')
        return True
    except Exception as e:
        print(f"Error training Tuberculosis model: {e}")
        os.chdir('..')
        os.chdir('..')
        return False

def train_rsna():
    """Train RSNA model"""
    print("\n" + "=" * 50)
    print("Training RSNA Model")
    print("=" * 50)
    
    rsna_dir = os.path.join('datasets', 'rsna18')
    train_script = os.path.join(rsna_dir, 'train.sh')
    
    if not os.path.exists(train_script):
        print("RSNA training script not found")
        return False
    
    try:
        os.chdir(rsna_dir)
        # On Windows, use bash or convert to Python
        if os.name == 'nt':
            print("Note: RSNA training uses shell scripts. Please run train.sh manually on Linux/Mac")
            return False
        else:
            subprocess.run(['bash', 'train.sh'], check=True)
        os.chdir('..')
        os.chdir('..')
        return True
    except Exception as e:
        print(f"Error training RSNA model: {e}")
        os.chdir('..')
        os.chdir('..')
        return False

def train_unet():
    """Train UNet model"""
    print("\n" + "=" * 50)
    print("Training UNet Model")
    print("=" * 50)
    
    unet_dir = os.path.join('datasets', 'UNet')
    train_script = os.path.join(unet_dir, 'train.py')
    
    if not os.path.exists(train_script):
        print("UNet training script not found")
        return False
    
    try:
        os.chdir(unet_dir)
        subprocess.run([sys.executable, 'train.py', '--epochs', '10'], check=True)
        os.chdir('..')
        os.chdir('..')
        return True
    except Exception as e:
        print(f"Error training UNet: {e}")
        os.chdir('..')
        os.chdir('..')
        return False

def train_all():
    """Train all models"""
    print("=" * 50)
    print("Training All Models for Early Disease Detection")
    print("=" * 50)
    
    results = {
        'chexnet': train_chexnet(),
        'mura': train_mura(),
        'tuberculosis': train_tuberculosis(),
        'rsna': train_rsna(),
        'unet': train_unet()
    }
    
    print("\n" + "=" * 50)
    print("Training Summary:")
    print("=" * 50)
    for model, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{model.upper()}: {status}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models for early disease detection')
    parser.add_argument('--model', type=str, 
                       choices=['chexnet', 'mura', 'tuberculosis', 'rsna', 'unet', 'all'],
                       default='all', help='Model to train')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all()
    elif args.model == 'chexnet':
        train_chexnet()
    elif args.model == 'mura':
        train_mura()
    elif args.model == 'tuberculosis':
        train_tuberculosis()
    elif args.model == 'rsna':
        train_rsna()
    elif args.model == 'unet':
        train_unet()

