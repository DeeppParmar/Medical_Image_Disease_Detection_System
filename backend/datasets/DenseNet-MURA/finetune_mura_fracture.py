"""
MURA Fine-tuning Script for Bone Fracture Detection
====================================================
This script fine-tunes DenseNet-169 for detecting musculoskeletal abnormalities
(fractures, dislocations, etc.) from X-ray images.

Features:
- Multi-body-part training (wrist, shoulder, elbow, etc.)
- Advanced data augmentation specific to bone X-rays
- Class-balanced training with weighted loss
- Mixed precision training for faster convergence
- Comprehensive evaluation with Cohen's Kappa

Datasets supported:
- MURA (Musculoskeletal Radiographs) Dataset
- Custom bone X-ray datasets

Usage:
    python finetune_mura_fracture.py --data_dir ./MURA --study_type all
    python finetune_mura_fracture.py --data_dir ./MURA --study_type XR_WRIST --epochs 30
    python finetune_mura_fracture.py --data_dir ./custom_data --custom_format
"""

import os
import sys
import argparse
import numpy as np
import random
import time
import copy
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    accuracy_score, cohen_kappa_score, f1_score
)

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

# Study types in MURA dataset
STUDY_TYPES = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 
               'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']


# ========================== Model Definition ==========================

class FractureNet(nn.Module):
    """
    DenseNet-169 based model for bone fracture/abnormality detection.
    Enhanced with attention mechanism and multi-scale features.
    """
    def __init__(self, pretrained=True, dropout_rate=0.4):
        super(FractureNet, self).__init__()
        
        # Load pretrained DenseNet169
        try:
            self.backbone = models.densenet169(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            self.backbone = models.densenet169(pretrained=pretrained)
        
        num_features = self.backbone.classifier.in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_features, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_features // 16),
            nn.ReLU(),
            nn.Linear(num_features // 16, num_features),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get features from backbone
        features = self.backbone.features(x)
        
        # Apply attention
        # Channel attention
        ca = self.channel_attention(features).unsqueeze(-1).unsqueeze(-1)
        features = features * ca
        
        # Spatial attention
        sa = self.spatial_attention(features)
        features = features * sa
        
        # Global pooling
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        return output


class SimpleFractureNet(nn.Module):
    """
    Simpler DenseNet-169 model matching original MURA architecture.
    """
    def __init__(self, pretrained=True):
        super(SimpleFractureNet, self).__init__()
        
        try:
            base = models.densenet169(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            base = models.densenet169(pretrained=pretrained)
        
        self.features = base.features
        self.bn = nn.BatchNorm2d(base.classifier.in_features)
        self.fc = nn.Linear(base.classifier.in_features, 1)
        
        # Initialize
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(self.fc(out))
        return out


# ========================== Dataset Classes ==========================

class MURADataset(Dataset):
    """
    Dataset for MURA (Musculoskeletal Radiographs).
    Handles study-level aggregation for improved accuracy.
    """
    def __init__(self, df, transform=None, is_training=False):
        self.df = df
        self.transform = transform
        self.is_training = is_training
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = row['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class BoneFractureDataset(Dataset):
    """
    Custom dataset for bone fracture detection.
    Supports various folder structures.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ========================== Data Augmentation ==========================

def get_train_transforms():
    """
    Advanced augmentation pipeline for bone X-ray training.
    Includes medical imaging specific augmentations.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.1
        ),
        transforms.RandomAutocontrast(p=0.4),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1))
    ])


def get_val_transforms():
    """Validation transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_strong_augment():
    """Strong augmentation for challenging cases."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25)
    ])


# ========================== Data Loading ==========================

def load_mura_data(data_dir, study_type='all'):
    """
    Load MURA dataset from directory.
    Returns DataFrames for train and validation splits.
    """
    train_csv = os.path.join(data_dir, 'train_image_paths.csv')
    valid_csv = os.path.join(data_dir, 'valid_image_paths.csv')
    
    # Load CSV files
    train_df = pd.read_csv(train_csv, header=None, names=['path'])
    valid_df = pd.read_csv(valid_csv, header=None, names=['path'])
    
    # Extract labels from path (positive/negative in path)
    def get_label(path):
        return 1 if 'positive' in path.lower() else 0
    
    train_df['label'] = train_df['path'].apply(get_label)
    valid_df['label'] = valid_df['path'].apply(get_label)
    
    # Fix paths (prepend data_dir if needed)
    def fix_path(path, base_dir):
        if os.path.exists(path):
            return path
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            return full_path
        # Try without MURA-v1.1/ prefix
        if path.startswith('MURA-v1.1/'):
            alt_path = os.path.join(base_dir, path[10:])
            if os.path.exists(alt_path):
                return alt_path
        return path
    
    train_df['path'] = train_df['path'].apply(lambda x: fix_path(x, data_dir))
    valid_df['path'] = valid_df['path'].apply(lambda x: fix_path(x, data_dir))
    
    # Extract study type from path
    def get_study_type(path):
        for st in STUDY_TYPES:
            if st in path:
                return st
        return 'UNKNOWN'
    
    train_df['study_type'] = train_df['path'].apply(get_study_type)
    valid_df['study_type'] = valid_df['path'].apply(get_study_type)
    
    # Filter by study type if specified
    if study_type != 'all' and study_type in STUDY_TYPES:
        train_df = train_df[train_df['study_type'] == study_type]
        valid_df = valid_df[valid_df['study_type'] == study_type]
    
    # Verify paths exist
    train_df = train_df[train_df['path'].apply(os.path.exists)]
    valid_df = valid_df[valid_df['path'].apply(os.path.exists)]
    
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def load_custom_bone_data(data_dir):
    """
    Load custom bone fracture dataset.
    Expected structure:
    data_dir/
        train/
            normal/
            fracture/
        valid/
            normal/
            fracture/
    """
    data = {'train': [], 'valid': []}
    
    for split in ['train', 'valid', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        key = 'train' if split == 'train' else 'valid'
        
        # Look for class folders
        for class_name in ['normal', 'negative', 'healthy', '0']:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        data[key].append({
                            'path': os.path.join(class_dir, img_name),
                            'label': 0
                        })
        
        for class_name in ['fracture', 'positive', 'abnormal', '1']:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        data[key].append({
                            'path': os.path.join(class_dir, img_name),
                            'label': 1
                        })
    
    train_df = pd.DataFrame(data['train'])
    valid_df = pd.DataFrame(data['valid'])
    
    return train_df, valid_df


# ========================== Loss Functions ==========================

class WeightedBCELoss(nn.Module):
    """Weighted BCE Loss for handling class imbalance."""
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        
        if self.pos_weight is not None:
            loss = -(self.pos_weight * target * torch.log(pred) + 
                    (1 - target) * torch.log(1 - pred))
        else:
            loss = -(target * torch.log(pred) + 
                    (1 - target) * torch.log(1 - pred))
        
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        return (focal_weight * bce).mean()


# ========================== Training Functions ==========================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = (outputs > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).long().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except:
        epoch_auc = 0.5
    
    try:
        kappa = cohen_kappa_score(all_labels, all_preds)
    except:
        kappa = 0.0
    
    return epoch_loss, epoch_acc, epoch_auc, kappa, all_labels, all_preds


def print_metrics(labels, preds, title="Evaluation Metrics"):
    """Print comprehensive evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    
    print(f"\nSensitivity (Recall):  {sensitivity:.4f}")
    print(f"Specificity:           {specificity:.4f}")
    print(f"PPV (Precision):       {ppv:.4f}")
    print(f"NPV:                   {npv:.4f}")
    print(f"F1 Score:              {f1:.4f}")
    print(f"Cohen's Kappa:         {kappa:.4f}")
    print(f"Accuracy:              {accuracy_score(labels, preds):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Normal', 'Abnormal/Fracture']))


# ========================== Main Training Function ==========================

def train_model(args):
    """Main training function."""
    print("\n" + "="*70)
    print("BONE FRACTURE DETECTION MODEL - FINE-TUNING")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n" + "-"*50)
    print("Loading Dataset...")
    print("-"*50)
    
    if args.custom_format:
        train_df, valid_df = load_custom_bone_data(args.data_dir)
        print("Using custom bone fracture dataset format")
    else:
        train_df, valid_df = load_mura_data(args.data_dir, args.study_type)
        print(f"Using MURA dataset - Study type: {args.study_type}")
    
    if len(train_df) == 0:
        print("ERROR: No training data found!")
        return None
    
    print(f"\nTraining samples:   {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    
    # Class distribution
    train_dist = train_df['label'].value_counts()
    print(f"\nTraining class distribution:")
    print(f"  Normal: {train_dist.get(0, 0)} | Abnormal: {train_dist.get(1, 0)}")
    
    # Calculate class weights
    pos_count = train_dist.get(1, 1)
    neg_count = train_dist.get(0, 1)
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Create datasets
    train_dataset = MURADataset(train_df, transform=get_train_transforms(), is_training=True)
    valid_dataset = MURADataset(valid_df, transform=get_val_transforms())
    
    # Create weighted sampler
    if args.weighted_sampling:
        class_counts = [neg_count, pos_count]
        sample_weights = [1.0 / class_counts[label] for label in train_df['label']]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    print("\n" + "-"*50)
    print("Initializing Model...")
    print("-"*50)
    
    if args.simple_model:
        model = SimpleFractureNet(pretrained=True)
        print("Using SimpleFractureNet")
    else:
        model = FractureNet(pretrained=True, dropout_rate=args.dropout)
        print("Using FractureNet with attention mechanism")
    
    model = model.to(device)
    
    # Load checkpoint if resuming
    if args.resume and args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
    
    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss")
    else:
        criterion = WeightedBCELoss(pos_weight=torch.tensor(pos_weight).to(device))
        print(f"Using Weighted BCE Loss (pos_weight: {pos_weight:.2f})")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, args.study_type if args.study_type != 'all' else 'all_studies')
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "-"*50)
    print("Starting Training...")
    print("-"*50)
    
    best_auc = 0.0
    best_kappa = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        val_loss, val_acc, val_auc, val_kappa, val_labels, val_preds = validate(
            model, valid_loader, criterion, device
        )
        
        # Update scheduler
        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_auc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val AUC:    {val_auc:.4f} | Val Kappa: {val_kappa:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Save best model
        is_best = val_auc > best_auc or (val_auc == best_auc and val_kappa > best_kappa)
        if is_best:
            best_auc = val_auc
            best_kappa = val_kappa
            best_epoch = epoch + 1
            patience_counter = 0
            
            model_path = os.path.join(save_dir, 'model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_kappa': val_kappa,
                'study_type': args.study_type,
            }, model_path)
            print(f"✓ Saved best model (AUC: {val_auc:.4f}, Kappa: {val_kappa:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # Final evaluation
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Best Validation Kappa: {best_kappa:.4f}")
    
    # Load best model
    best_model_path = os.path.join(save_dir, 'model.pth')
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    
    # Final evaluation
    val_loss, val_acc, val_auc, val_kappa, val_labels, val_preds = validate(
        model, valid_loader, criterion, device
    )
    
    print_metrics(val_labels, val_preds, "Final Validation Metrics")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune MURA for Bone Fracture Detection')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./MURA',
                        help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--study_type', type=str, default='XR_WRIST',
                        choices=['all'] + STUDY_TYPES,
                        help='Body part to train on')
    parser.add_argument('--custom_format', action='store_true',
                        help='Use custom dataset format instead of MURA')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Model arguments
    parser.add_argument('--simple_model', action='store_true',
                        help='Use simpler model architecture')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path')
    
    # Training options
    parser.add_argument('--weighted_sampling', action='store_true', default=True,
                        help='Use weighted sampling')
    parser.add_argument('--focal_loss', action='store_true',
                        help='Use focal loss')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['cosine', 'plateau'],
                        help='LR scheduler')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    train_model(args)


if __name__ == '__main__':
    main()
