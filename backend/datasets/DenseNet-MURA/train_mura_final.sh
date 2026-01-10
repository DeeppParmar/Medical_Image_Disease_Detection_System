#!/bin/bash
# Complete MURA training script with environment setup

set -e

cd /mnt/e/FINAL/backend

echo "=========================================="
echo "MURA Model Training - Complete Setup"
echo "=========================================="

# Activate virtual environment
if [ -d "project" ]; then
    source project/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found"
fi

# Install PyTorch if needed
echo ""
echo "Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} OK')" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip install torch torchvision --quiet
}

# Go to MURA directory
cd datasets/DenseNet-MURA

# Create symlink if needed
if [ ! -e "MURA-v1.0" ] && [ -d "MURA-v1.1" ]; then
    echo "Creating symlink: MURA-v1.0 -> MURA-v1.1"
    ln -sf MURA-v1.1 MURA-v1.0
fi

# Create models directory
mkdir -p models

# Check data
if [ ! -d "MURA-v1.0" ] && [ ! -d "MURA-v1.1" ]; then
    echo "Error: MURA dataset not found!"
    exit 1
fi

echo ""
echo "Starting MURA training..."
echo "This will take several hours..."
echo "Training log: training.log"
echo ""

# Start training
python3 train_mura.py 2>&1 | tee training.log

echo ""
echo "=========================================="
if [ -f "models/model.pth" ]; then
    echo "Training Complete! Model saved."
    ls -lh models/model.pth
else
    echo "Training finished. Check training.log for details."
fi
echo "=========================================="

