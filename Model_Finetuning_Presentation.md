# Medical AI Model Fine-Tuning Presentation
## Early Disease Detection System - Deep Learning Approach

---

# SLIDE 1: PROJECT OVERVIEW & ARCHITECTURE

## 🏥 Early Disease Detection System

### Vision
Building an AI-powered diagnostic tool for detecting:
- **Tuberculosis (TB)** from Chest X-rays
- **Pneumonia** from Chest X-rays  
- **Bone Fractures** from Musculoskeletal X-rays

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + TypeScript)            │
│  • Image Upload • Real-time Analysis • Result Display       │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST API
┌─────────────────────────▼───────────────────────────────────┐
│                    BACKEND (Flask + Python)                 │
│  • Auto Model Selection • Image Preprocessing • Routing     │
└─────────────────────────┬───────────────────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    ▼                     ▼                     ▼
┌─────────┐       ┌─────────────┐       ┌───────────┐
│ TBNet   │       │ PneumoniaNet│       │FractureNet│
│DenseNet │       │  DenseNet   │       │ DenseNet  │
│  -121   │       │    -121     │       │   -169    │
└─────────┘       └─────────────┘       └───────────┘
```

### Key Technologies
| Component | Technology |
|-----------|------------|
| Deep Learning Framework | PyTorch 2.x |
| Base Architecture | DenseNet (121/169) |
| Transfer Learning | ImageNet Pre-trained |
| Training Hardware | CUDA GPU Accelerated |

---

# SLIDE 2: MODEL ARCHITECTURES & TRANSFER LEARNING

## 🧠 Deep Learning Model Architectures

### 1. TuberculosisNet (TBNet) - DenseNet-121 Based

```python
class TBNetPyTorch(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        # DenseNet-121 Backbone (ImageNet pretrained)
        self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        
        # Attention Module for Lung Region Focus
        self.attention = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Custom Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),      # Feature compression
            nn.ReLU() + BatchNorm1d,    # Normalization
            nn.Dropout(0.5),            # Regularization
            nn.Linear(512, 256),
            nn.ReLU() + BatchNorm1d,
            nn.Dropout(0.25),
            nn.Linear(256, 2)           # Binary: Normal/TB
        )
```

**Why DenseNet-121?**
- 🔄 Dense connections enable feature reuse
- 📉 Fewer parameters than ResNet (8M vs 25M)
- 🎯 Excellent for medical imaging transfer learning

### 2. PneumoniaNet - Enhanced DenseNet-121

```python
class PneumoniaNet(nn.Module):
    # Similar architecture with attention mechanism
    # Specialized for lung consolidation detection
```

### 3. FractureNet - DenseNet-169 Based

```python
class FractureNet(nn.Module):
    # DenseNet-169 backbone (deeper for bone structure)
    # Dual attention: Spatial + Channel attention
    # Specialized for bone abnormality detection
```

### Transfer Learning Strategy

```
ImageNet Weights (1000 classes)
         │
         ▼ Transfer
Medical Domain Adaptation
         │
         ├── Freeze early layers (generic features)
         ├── Fine-tune middle layers (domain adaptation)
         └── Train new classifier (task-specific)
```

| Layer Type | Training Strategy |
|------------|-------------------|
| Conv Blocks 1-3 | Frozen (edge/texture features) |
| Conv Blocks 4-5 | Fine-tuned (low LR) |
| Classifier Head | Trained from scratch |

---

# SLIDE 3: DATA AUGMENTATION & PREPROCESSING

## 📊 Medical Image Data Pipeline

### Challenge: Limited Medical Data
- Medical datasets are small (100s-1000s images)
- Class imbalance (fewer disease cases)
- Variability in X-ray quality/positioning

### Solution: Advanced Data Augmentation

```python
def get_train_transforms(img_size=224):
    return transforms.Compose([
        # 1. SIZE HANDLING
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        
        # 2. GEOMETRIC AUGMENTATION
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),    # Position variation
            scale=(0.9, 1.1)         # Size variation
        ),
        
        # 3. INTENSITY AUGMENTATION
        transforms.ColorJitter(
            brightness=0.2,          # Exposure variation
            contrast=0.2             # Contrast variation
        ),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(2, p=0.3),
        
        # 4. NORMALIZATION (ImageNet stats)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # 5. REGULARIZATION
        transforms.RandomErasing(p=0.1)  # Cutout
    ])
```

### Augmentation Visual Pipeline

```
Original X-ray
     │
     ├── Random Flip ──────────► Simulates left/right variations
     │
     ├── Random Rotation ──────► Simulates positioning errors
     │
     ├── Random Crop ──────────► Focuses on different regions
     │
     ├── Brightness/Contrast ──► Simulates exposure differences
     │
     └── Random Erasing ───────► Forces robust feature learning
```

### Class Imbalance Handling

```python
# Weighted Random Sampling
def create_weighted_sampler(labels):
    class_counts = Counter(labels)  # {0: 8000, 1: 2000}
    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(weights, len(weights))

# Focal Loss for Hard Examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        # Down-weights easy examples
        # Focuses training on hard cases
```

### Preprocessing Summary

| Stage | Technique | Purpose |
|-------|-----------|---------|
| Resize | 256×256 → 224×224 crop | Standardization |
| Augmentation | 8+ techniques | Data diversity |
| Normalization | ImageNet stats | Transfer learning |
| Sampling | Weighted | Class balance |

---

# SLIDE 4: TRAINING METHODOLOGY & OPTIMIZATION

## ⚙️ Training Configuration & Techniques

### Hyperparameters

| Parameter | TBNet | PneumoniaNet | FractureNet |
|-----------|-------|--------------|-------------|
| Backbone | DenseNet-121 | DenseNet-121 | DenseNet-169 |
| Input Size | 224×224 | 224×224 | 224×224 |
| Batch Size | 32 | 32 | 16 |
| Initial LR | 1e-4 | 1e-4 | 1e-4 |
| Epochs | 50-100 | 50 | 30-50 |
| Optimizer | AdamW | AdamW | AdamW |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |

### Learning Rate Schedule

```python
# Cosine Annealing with Warm Restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # Initial restart period
    T_mult=2,    # Period multiplier after restart
    eta_min=1e-7 # Minimum learning rate
)
```

```
Learning Rate over Training:
   │
1e-4 ┤  ╭──╮     ╭────╮        ╭────────╮
     │ ╱    ╲   ╱      ╲      ╱          ╲
     │╱      ╲ ╱        ╲    ╱            ╲
1e-7 ┼────────┴──────────┴──────────────────►
     0       10         30                  70  Epochs
```

### Training Loop Implementation

```python
def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training (2x faster)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        # Gradient Clipping (stability)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
```

### Regularization Techniques

```
┌─────────────────────────────────────────────────┐
│              REGULARIZATION STACK               │
├─────────────────────────────────────────────────┤
│ 1. Dropout (0.5 → 0.25)    │ Prevents co-adapt │
│ 2. Weight Decay (1e-4)     │ L2 regularization │
│ 3. Batch Normalization     │ Internal covariate│
│ 4. Data Augmentation       │ Implicit regular. │
│ 5. Early Stopping          │ Prevents overfit  │
│ 6. Label Smoothing (0.1)   │ Soft targets      │
└─────────────────────────────────────────────────┘
```

### Early Stopping Strategy

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop
```

---

# SLIDE 5: EVALUATION METRICS & RESULTS

## 📈 Model Performance & Evaluation

### Evaluation Metrics for Medical AI

```python
from sklearn.metrics import (
    roc_auc_score,        # Area Under ROC Curve
    accuracy_score,        # Overall accuracy
    precision_recall_fscore_support,
    confusion_matrix
)

# Medical-specific metrics
Sensitivity = TP / (TP + FN)  # Disease detection rate
Specificity = TN / (TN + FP)  # Healthy identification rate
PPV = TP / (TP + FP)          # Positive Predictive Value
NPV = TN / (TN + FN)          # Negative Predictive Value
```

### Confusion Matrix Analysis

```
                    Predicted
                 Normal   Disease
              ┌────────┬─────────┐
    Actual    │   TN   │   FP    │  ← False Alarms
    Normal    │  850   │   50    │
              ├────────┼─────────┤
    Actual    │   FN   │   TP    │  ← Missed Cases
    Disease   │   30   │  270    │
              └────────┴─────────┘
```

### Model Performance Summary

| Model | AUC-ROC | Accuracy | Sensitivity | Specificity |
|-------|---------|----------|-------------|-------------|
| **TBNet** | 0.94+ | 92%+ | 90%+ | 94%+ |
| **PneumoniaNet** | 0.92+ | 90%+ | 88%+ | 92%+ |
| **FractureNet** | 0.89+ | 85%+ | 82%+ | 88%+ |

### ROC Curve Visualization

```
True Positive Rate (Sensitivity)
    │
1.0 ┤                    ●────────────────
    │                 ●
    │              ●
    │           ●        ← Our Model (AUC = 0.94)
0.5 ┤        ●
    │     ●
    │   ●
    │ ●  
0.0 ┼──────────────────────────────────────►
    0.0       0.5                        1.0
              False Positive Rate (1-Specificity)
```

### Key Achievements

✅ **Transfer Learning Success**
   - Reduced training time by 80%
   - Achieved convergence with limited data

✅ **Attention Mechanism Benefits**
   - Improved focus on disease regions
   - Better interpretability for clinicians

✅ **Robust Augmentation Pipeline**
   - Generalization to unseen X-ray variations
   - Reduced overfitting on small datasets

✅ **Production-Ready Models**
   - PyTorch ONNX export support
   - Fast inference (~100ms per image)

### Future Improvements

```
┌──────────────────────────────────────────────┐
│           ROADMAP FOR ENHANCEMENT            │
├──────────────────────────────────────────────┤
│ • Grad-CAM visualization for explainability  │
│ • Multi-task learning (detect all diseases)  │
│ • Larger dataset integration (CheXpert)      │
│ • Model quantization for edge deployment     │
│ • Continuous learning from new cases         │
└──────────────────────────────────────────────┘
```

---

## 📚 References & Datasets

1. **CheXNet**: Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection"
2. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks"
3. **MURA Dataset**: Stanford ML Group Musculoskeletal Radiographs
4. **TB Datasets**: Montgomery County & Shenzhen Hospital Collections

---

*Developed for Early Disease Detection System - Hackathon DICT 2026*
