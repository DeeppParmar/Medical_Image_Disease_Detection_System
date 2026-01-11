"""
CheXNet Fine-tuning Script for Pneumonia Detection
===================================================
This script fine-tunes a pre-trained DenseNet121 model for accurate
pneumonia detection from chest X-ray images.

Features:
- Transfer learning from ImageNet or existing CheXNet weights
- Advanced data augmentation for medical images
- Class-balanced sampling to handle imbalanced datasets
- Early stopping and learning rate scheduling
- Mixed precision training for faster training
- Comprehensive evaluation metrics (AUC, Sensitivity, Specificity)

Datasets supported:
- NIH ChestX-ray14 (Pneumonia subset)
- RSNA Pneumonia Detection Challenge
- Chest X-Ray Images (Kaggle)

Usage:
    python finetune_pneumonia.py --data_dir ./data --epochs 50 --batch_size 32
    python finetune_pneumonia.py --resume --checkpoint ./checkpoints/best_model.pth
"""

import os
import sys
import argparse
import numpy as np
import random
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

set_seed(42)

# ========================== Model Definition ==========================

class PneumoniaNet(nn.Module):
    """
    DenseNet121-based model for Pneumonia detection.
    Uses transfer learning with attention mechanism for improved accuracy.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        super(PneumoniaNet, self).__init__()
        
        # Load pretrained DenseNet121
        try:
            self.backbone = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            self.backbone = models.densenet121(pretrained=pretrained)
        
        # Get the number of features from backbone
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Identity()
        
        # Attention mechanism for focusing on relevant regions
        self.attention = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        output = self.classifier(features)
        return output


class SimplePneumoniaNet(nn.Module):
    """
    Simpler DenseNet121 model for binary pneumonia classification.
    More stable training with sigmoid output.
    """
    def __init__(self, pretrained=True):
        super(SimplePneumoniaNet, self).__init__()
        
        try:
            self.densenet = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            self.densenet = models.densenet121(pretrained=pretrained)
        
        num_features = self.densenet.classifier.in_features
        
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.densenet(x)


# ========================== Dataset Classes ==========================

class PneumoniaDataset(Dataset):
    """
    Custom dataset for pneumonia detection.
    Supports multiple data formats and augmentation strategies.
    """
    def __init__(self, image_paths, labels, transform=None, is_training=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ChestXrayDataset(Dataset):
    """
    Dataset for NIH ChestX-ray14 format data.
    """
    def __init__(self, data_dir, image_list_file, transform=None, target_class='Pneumonia'):
        self.data_dir = data_dir
        self.transform = transform
        self.target_class = target_class
        
        self.image_paths = []
        self.labels = []
        
        with open(image_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    findings = parts[1] if len(parts) > 1 else ''
                    
                    self.image_paths.append(os.path.join(data_dir, img_name))
                    # Binary label: 1 if target class present, 0 otherwise
                    label = 1 if target_class in findings else 0
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ========================== Data Augmentation ==========================

def get_train_transforms():
    """
    Advanced augmentation pipeline for training.
    Medical image-specific augmentations for better generalization.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2
        ),
        # Histogram equalization simulation
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        # Random erasing for regularization
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])


def get_val_transforms():
    """Validation/Test transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_tta_transforms():
    """Test-time augmentation transforms for improved prediction."""
    return [
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((280, 280)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ]


# ========================== Training Functions ==========================

def create_weighted_sampler(labels):
    """Create weighted sampler for class imbalance."""
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels.float())
                    preds = (outputs > 0.5).long()
                else:
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels.float())
                preds = (outputs > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels.float())
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.argmax(dim=1)
            
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except:
        epoch_auc = 0.0
    
    return epoch_loss, epoch_acc, epoch_auc, all_labels, all_preds


def print_metrics(labels, preds, title="Evaluation Metrics"):
    """Print comprehensive evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nSensitivity (Recall):  {sensitivity:.4f}")
    print(f"Specificity:           {specificity:.4f}")
    print(f"PPV (Precision):       {ppv:.4f}")
    print(f"NPV:                   {npv:.4f}")
    print(f"Accuracy:              {accuracy_score(labels, preds):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Normal', 'Pneumonia']))


# ========================== Dataset Preparation ==========================

def prepare_kaggle_chest_xray(data_dir):
    """
    Prepare data from Kaggle Chest X-Ray Images dataset.
    Expected structure:
    data_dir/
        train/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
    """
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    for split, paths, labels in [('train', train_paths, train_labels), 
                                   ('test', test_paths, test_labels)]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name, class_label in [('NORMAL', 0), ('PNEUMONIA', 1)]:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        paths.append(os.path.join(class_dir, img_name))
                        labels.append(class_label)
    
    # Create validation split from training data
    if train_paths:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=0.15, stratify=train_labels, random_state=42
        )
    else:
        val_paths, val_labels = [], []
    
    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


def prepare_rsna_pneumonia(data_dir):
    """
    Prepare data from RSNA Pneumonia Detection Challenge.
    Expected structure:
    data_dir/
        train_images/
        train_labels.csv (contains patientId, Target columns)
    """
    import csv
    
    images_dir = os.path.join(data_dir, 'train_images')
    labels_file = os.path.join(data_dir, 'train_labels.csv')
    
    if not os.path.exists(labels_file):
        print(f"Labels file not found: {labels_file}")
        return None
    
    paths, labels = [], []
    
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row['patientId']
            target = int(row['Target'])
            
            # Try different image extensions
            for ext in ['.dcm', '.png', '.jpg']:
                img_path = os.path.join(images_dir, f"{patient_id}{ext}")
                if os.path.exists(img_path):
                    paths.append(img_path)
                    labels.append(target)
                    break
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


def auto_detect_dataset(data_dir):
    """Auto-detect and prepare dataset based on directory structure."""
    
    # Check for Kaggle structure
    if os.path.exists(os.path.join(data_dir, 'train', 'NORMAL')):
        print("Detected: Kaggle Chest X-Ray Images dataset")
        return prepare_kaggle_chest_xray(data_dir)
    
    # Check for RSNA structure
    if os.path.exists(os.path.join(data_dir, 'train_labels.csv')):
        print("Detected: RSNA Pneumonia Detection dataset")
        return prepare_rsna_pneumonia(data_dir)
    
    # Check for ChestX-ray14 structure
    if os.path.exists(os.path.join(data_dir, 'images')) and \
       os.path.exists(os.path.join(data_dir, 'labels')):
        print("Detected: NIH ChestX-ray14 dataset")
        # Return None to use ChestXrayDataset class
        return None
    
    print("Warning: Could not auto-detect dataset format")
    return None


# ========================== Main Training Function ==========================

def train_model(args):
    """Main training function."""
    print("\n" + "="*70)
    print("PNEUMONIA DETECTION MODEL - FINE-TUNING")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Prepare data
    print("\n" + "-"*50)
    print("Loading Dataset...")
    print("-"*50)
    
    data = auto_detect_dataset(args.data_dir)
    
    if data is not None:
        train_paths, train_labels = data['train']
        val_paths, val_labels = data['val']
        test_paths, test_labels = data['test']
        
        print(f"Training samples:   {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")
        print(f"Test samples:       {len(test_paths)}")
        
        # Class distribution
        train_dist = Counter(train_labels)
        print(f"\nTraining class distribution:")
        print(f"  Normal: {train_dist[0]} | Pneumonia: {train_dist[1]}")
        
        # Create datasets
        train_dataset = PneumoniaDataset(
            train_paths, train_labels, 
            transform=get_train_transforms(), 
            is_training=True
        )
        val_dataset = PneumoniaDataset(
            val_paths, val_labels, 
            transform=get_val_transforms()
        )
        test_dataset = PneumoniaDataset(
            test_paths, test_labels, 
            transform=get_val_transforms()
        )
        
        # Create weighted sampler for imbalanced data
        sampler = create_weighted_sampler(train_labels) if args.weighted_sampling else None
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=args.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        print("Using ChestX-ray14 format")
        # Placeholder for ChestX-ray14 specific loading
        raise NotImplementedError("Please provide dataset in Kaggle or RSNA format")
    
    # Initialize model
    print("\n" + "-"*50)
    print("Initializing Model...")
    print("-"*50)
    
    if args.simple_model:
        model = SimplePneumoniaNet(pretrained=True)
        criterion = nn.BCELoss()
        print("Using SimplePneumoniaNet (Binary output with Sigmoid)")
    else:
        model = PneumoniaNet(num_classes=2, pretrained=True, dropout_rate=args.dropout)
        # Use weighted CrossEntropy for class imbalance
        weights = torch.tensor([train_dist[1] / train_dist[0], 1.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using PneumoniaNet with attention mechanism")
    
    model = model.to(device)
    
    # Load checkpoint if resuming
    if args.resume and args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
    else:
        start_epoch = 0
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "-"*50)
    print("Starting Training...")
    print("-"*50)
    
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc, val_auc, val_labels_all, val_preds_all = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_auc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            
            checkpoint_path = os.path.join(args.save_dir, 'best_pneumonia_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"✓ Saved best model (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, checkpoint_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation AUC: {best_auc:.4f}")
    
    # Load best model for final evaluation
    best_checkpoint = os.path.join(args.save_dir, 'best_pneumonia_model.pth')
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    test_loss, test_acc, test_auc, test_labels_all, test_preds_all = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    print_metrics(test_labels_all, test_preds_all, "Final Test Metrics")
    
    # Save final model in CheXNet compatible format
    final_model_path = os.path.join(args.save_dir, 'pneumonia_model_final.pth.tar')
    torch.save({
        'state_dict': model.state_dict(),
        'test_auc': test_auc,
        'test_acc': test_acc,
        'epochs': args.epochs,
    }, final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CheXNet for Pneumonia Detection')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/pneumonia',
                        help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/pneumonia',
                        help='Directory to save checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Model arguments
    parser.add_argument('--simple_model', action='store_true',
                        help='Use simpler model architecture')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Training options
    parser.add_argument('--weighted_sampling', action='store_true', default=True,
                        help='Use weighted sampling for class imbalance')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['cosine', 'plateau'],
                        help='Learning rate scheduler')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    train_model(args)


if __name__ == '__main__':
    main()
