# MURA Training Guide - DenseNet-169

## 📋 Overview

Train a DenseNet-169 model for detecting abnormalities in musculoskeletal radiographs (MURA dataset). The model classifies X-ray studies as normal or abnormal across 7 different body parts.

## 📁 Dataset Structure

The MURA dataset should be located in `MURA-v1.1/` or `MURA-v1.0/` with this structure:

```
MURA-v1.1/
├── train/
│   ├── XR_ELBOW/
│   │   ├── patient00001/
│   │   │   └── study1_positive/
│   │   │       ├── image1.png
│   │   │       ├── image2.png
│   │   │       └── ...
│   │   └── ...
│   ├── XR_FINGER/
│   ├── XR_FOREARM/
│   ├── XR_HAND/
│   ├── XR_HUMERUS/
│   ├── XR_SHOULDER/
│   └── XR_WRIST/
└── valid/
    ├── XR_ELBOW/
    ├── XR_FINGER/
    └── ... (same structure as train)
```

## 🚀 Quick Start

### 1. Check Dataset
```bash
cd backend/datasets/DenseNet-MURA
python train_mura_single.py --check_data
```

This will verify:
- ✓ Dataset exists in correct location
- ✓ Directory structure is correct
- ✓ Available study types

### 2. Train on Single Study Type

**Train on Wrist X-rays (recommended for quick testing):**
```bash
python train_mura_single.py --study_type XR_WRIST --epochs 10
```

**Train on Shoulder X-rays:**
```bash
python train_mura_single.py --study_type XR_SHOULDER --epochs 15 --lr 0.0001
```

**Available study types:**
- `XR_ELBOW` - Elbow X-rays
- `XR_FINGER` - Finger X-rays
- `XR_FOREARM` - Forearm X-rays
- `XR_HAND` - Hand X-rays
- `XR_HUMERUS` - Humerus X-rays
- `XR_SHOULDER` - Shoulder X-rays
- `XR_WRIST` - Wrist X-rays

### 3. Train on All Study Types
```bash
python train_mura_single.py --study_type all --epochs 20
```

## 🎯 Training Options

```bash
python train_mura_single.py [OPTIONS]

Options:
  --study_type STR      Study type to train (default: XR_WRIST)
                        Options: XR_ELBOW, XR_FINGER, XR_FOREARM, XR_HAND,
                                XR_HUMERUS, XR_SHOULDER, XR_WRIST, all
  
  --epochs INT          Number of training epochs (default: 10)
  
  --lr FLOAT            Learning rate (default: 0.0001)
  
  --batch_size INT      Batch size (default: 1)
                        Note: Study-level training uses batch_size=1
  
  --save_dir DIR        Directory for saved models (default: models)
  
  --check_data          Check dataset and exit (no training)
```

## 📊 Training Examples

### Quick Test (5 minutes)
```bash
# Fast training for testing (fewer epochs)
python train_mura_single.py --study_type XR_WRIST --epochs 3
```

### Recommended Training
```bash
# Good balance of time and performance
python train_mura_single.py --study_type XR_WRIST --epochs 10 --lr 0.0001
```

### Full Training
```bash
# Best performance (longer training time)
python train_mura_single.py --study_type XR_WRIST --epochs 20 --lr 0.0001
```

### Train All Body Parts
```bash
# Train separate models for all 7 study types (takes several hours)
python train_mura_single.py --study_type all --epochs 15
```

## 🔧 System Requirements

### Minimum Requirements
- **RAM**: 8GB+
- **Storage**: 20GB for dataset + models
- **Time**: ~30-60 minutes per study type (10 epochs, CPU)

### Recommended Requirements
- **GPU**: NVIDIA GPU with 6GB+ VRAM (20x faster)
- **RAM**: 16GB+
- **CUDA**: PyTorch with CUDA support
- **Time**: ~5-10 minutes per study type (10 epochs, GPU)

## 📦 Dependencies

Install required packages:
```bash
pip install torch torchvision tqdm pandas pillow matplotlib scikit-learn
```

Or if you have a requirements file:
```bash
pip install -r requirements.txt
```

## 📈 Training Process

The training script will:

1. **Load Dataset**
   - Reads study-level data (multiple images per study)
   - Splits into train/validation sets

2. **Calculate Class Weights**
   - Balances abnormal vs normal cases
   - Prevents bias towards majority class

3. **Initialize Model**
   - DenseNet-169 pretrained on ImageNet
   - Modified final layer for binary classification

4. **Training Loop**
   - Shows progress for each epoch
   - Displays loss and accuracy
   - Prints confusion matrix

5. **Model Checkpointing**
   - Saves best model based on validation accuracy
   - Learning rate decay on plateau

6. **Final Evaluation**
   - Reports final metrics on validation set

## 📊 Expected Output

```
================================================================================
MURA Model Training - DenseNet-169
================================================================================
Device: cuda
Study Type: XR_WRIST
Batch Size: 1
Epochs: 10
Learning Rate: 0.0001
================================================================================
✓ Using MURA dataset: MURA-v1.1
✓ Available study types: XR_WRIST, XR_SHOULDER, ...

================================================================================
Training for: XR_WRIST
================================================================================
Loading study level data...
Training studies: 4251
Validation studies: 537

Class distribution:
Train - Abnormal: 12345, Normal: 10123
Valid - Abnormal: 1567, Normal: 1234
Weight for positive (abnormal): train=0.4510, valid=0.4401
Weight for negative (normal): train=0.5490, valid=0.5599

Initializing model...

================================================================================
Starting Training
================================================================================
Train batches: 4251
Valid batches: 537

Epoch 1/10
----------
  train: 4251/4251
train  Loss: 0.4523 Acc: 0.7834        
Confusion Matrix:
[[0.78 0.22]
 [0.21 0.79]]
  valid: 537/537
valid  Loss: 0.3891 Acc: 0.8156        
Confusion Matrix:
[[0.82 0.18]
 [0.19 0.81]]
Learning rate: 0.000100
Time elapsed: 8m 23s

Epoch 2/10
----------
...

Training complete in 83m 45s
Best valid Acc: 0.8523

Model saved to: models/XR_WRIST/model.pth

================================================================================
Final Metrics
================================================================================
valid Loss: 0.3245 Acc: 0.8523
Confusion Matrix:
[[0.86 0.14]
 [0.15 0.85]]

================================================================================
Training Complete!
================================================================================
Total time: 83m 45s
Models saved in: models/
```

## 🗂️ Output Structure

After training, models are saved in:
```
models/
├── XR_WRIST/
│   └── model.pth
├── XR_SHOULDER/
│   └── model.pth
└── ... (other study types)
```

## 🐛 Troubleshooting

### Issue: Dataset Not Found
```
❌ Error: MURA dataset not found!
```
**Solution**: 
- Download MURA dataset from https://stanfordmlgroup.github.io/competitions/mura/
- Extract to `MURA-v1.1/` directory
- Verify structure with `--check_data`

### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Training already uses batch_size=1 (minimum)
- Try CPU training: Training will use CPU automatically if CUDA unavailable
- Close other GPU-using applications

### Issue: Slow Training (CPU)
**Expected**: 30-60 minutes per epoch on CPU
**Solution**:
- Use GPU for 20x speedup
- Reduce number of epochs
- Train on smaller study type (XR_FINGER has fewer samples)

### Issue: NaN Loss
```
Loss: nan
```
**Solution**: Already fixed! The loss function now includes:
- Epsilon clamping to prevent log(0)
- Input clamping to valid range [0, 1]

### Issue: Low Accuracy
**Possible causes**:
- Too few epochs (try 15-20)
- Learning rate too high/low
- Imbalanced dataset (weights are automatically calculated)

## 🔍 Monitoring Training

### Watch GPU Usage (if available)
```bash
# In another terminal
nvidia-smi -l 1
```

### Check Training Progress
Look for:
- ✓ Decreasing loss over epochs
- ✓ Increasing accuracy over epochs
- ✓ Learning rate decreases when validation plateaus
- ✓ Confusion matrix shows good diagonal values

### Good Training Signs:
- Train loss decreasing steadily
- Valid loss decreasing (with some fluctuation)
- Accuracy > 75% after 10 epochs
- Confusion matrix diagonal > 0.75

### Warning Signs:
- Train loss stuck or increasing
- Valid loss increasing (overfitting)
- Large gap between train/valid accuracy
- Very unbalanced confusion matrix

## 🎨 Integration with Backend

After training, the model can be used with the inference script:

```python
from models.mura_inference import MURAPredictor

predictor = MURAPredictor(study_type='XR_WRIST')
result = predictor.predict('path/to/xray.png')
print(result)  # {'abnormal': True, 'confidence': 0.87, ...}
```

## 📚 Training Tips

1. **Start Small**: Train XR_WRIST first (smaller dataset, faster training)
2. **Monitor**: Watch the first few epochs - if loss doesn't decrease, stop and check
3. **GPU Recommended**: 20x faster than CPU
4. **Epochs**: 10-15 epochs usually sufficient, 20+ for best results
5. **Learning Rate**: 0.0001 is a good default, decrease if loss oscillates
6. **Save Checkpoints**: Models auto-saved to `models/` directory

## 🚀 Next Steps

After training:

1. **Test the model**:
   ```python
   python -c "from models.mura_inference import MURAPredictor; p = MURAPredictor('XR_WRIST'); print(p.predict('test_image.png'))"
   ```

2. **Integrate with backend**:
   - Model already integrated in `backend/models/mura_inference.py`
   - API endpoint: `/predict/mura`

3. **Train more study types**:
   ```bash
   python train_mura_single.py --study_type all --epochs 15
   ```

4. **Improve performance**:
   - Train for more epochs (20-30)
   - Try different learning rates
   - Add data augmentation (already included)

## 📞 Support

Common issues and solutions documented above. For other issues:
1. Check console output for specific error messages
2. Verify dataset structure with `--check_data`
3. Ensure PyTorch and dependencies are installed
4. Try training on a smaller study type first

---

**Status**: Ready for training
**Last Updated**: January 10, 2026
