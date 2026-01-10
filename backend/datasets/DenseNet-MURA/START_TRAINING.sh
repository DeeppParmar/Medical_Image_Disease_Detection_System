#!/bin/bash
# Start MURA training - Run this in WSL

cd /mnt/e/FINAL/backend

# Activate virtual environment
source project/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision scikit-learn --quiet || echo "Packages may already be installed"

# Go to MURA directory
cd datasets/DenseNet-MURA

# Create symlink if needed
[ ! -e "MURA-v1.0" ] && [ -d "MURA-v1.1" ] && ln -sf MURA-v1.1 MURA-v1.0 && echo "Created symlink"

# Create models directory
mkdir -p models

# Start training
echo ""
echo "=========================================="
echo "Starting MURA Training"
echo "=========================================="
echo "This will take 3-6 hours on GPU"
echo "Training log will be saved to training.log"
echo ""

python3 train_mura.py 2>&1 | tee training.log

echo ""
echo "=========================================="
if [ -f "models/model.pth" ]; then
    echo "SUCCESS! Model trained and saved!"
    ls -lh models/model.pth
else
    echo "Training finished. Check training.log for details."
fi
echo "=========================================="

