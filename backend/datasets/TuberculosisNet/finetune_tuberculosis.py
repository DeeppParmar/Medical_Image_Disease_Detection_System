"""
TuberculosisNet Fine-tuning Script for TB Detection
====================================================
This script fine-tunes models for accurate tuberculosis detection
from chest X-ray images using transfer learning.

Features:
- PyTorch-based training (more stable than TF1.x)
- DenseNet/ResNet backbone with attention
- Advanced data augmentation for chest X-rays
- Class-balanced training
- Comprehensive TB-specific evaluation metrics
- Support for multiple TB datasets

Datasets supported:
- Montgomery County TB Dataset
- Shenzhen Hospital TB Dataset
- TBX11K Dataset
- Custom TB datasets

Usage:
    python finetune_tuberculosis.py --data_dir ./data --epochs 50
    python finetune_tuberculosis.py --data_dir ./data --model resnet50 --batch_size 16
"""

import os
import sys
import argparse
import numpy as np
import random
from datetime import datetime
from collections import Counter
import glob
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import csv

from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_recall_curve, roc_curve
)
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

set_seed(42)


# ========================== Model Definitions ==========================

class TBNetPyTorch(nn.Module):
    """
    PyTorch-based TB detection model using DenseNet-121.
    Includes attention mechanism for focusing on lung regions.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        super(TBNetPyTorch, self).__init__()
        
        # Load DenseNet121 backbone
        try:
            self.backbone = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            self.backbone = models.densenet121(pretrained=pretrained)
        
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Attention module for lung region focus
        self.attention = nn.Sequential(
            nn.Conv2d(num_features, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get features from backbone
        features = self.backbone.features(x)
        
        # Apply attention
        att = self.attention(features)
        features = features * att
        
        # Global pooling
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        return output


class ResNetTB(nn.Module):
    """
    ResNet-50 based TB detection model.
    Alternative architecture for TB detection.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        super(ResNetTB, self).__init__()
        
        try:
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            self.backbone = models.resnet50(pretrained=pretrained)
        
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class EfficientNetTB(nn.Module):
    """
    EfficientNet-B4 based TB detection model.
    Modern efficient architecture for TB detection.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.4):
        super(EfficientNetTB, self).__init__()
        
        try:
            self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1' if pretrained else None)
        except (TypeError, AttributeError):
            try:
                self.backbone = models.efficientnet_b4(pretrained=pretrained)
            except:
                # Fallback to EfficientNet-B0
                try:
                    self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
                except TypeError:
                    self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class SimpleTBNet(nn.Module):
    """
    Simple DenseNet-121 model for binary TB classification.
    Matches the original TBNet output format.
    """
    def __init__(self, pretrained=True):
        super(SimpleTBNet, self).__init__()
        
        try:
            self.densenet = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        except TypeError:
            self.densenet = models.densenet121(pretrained=pretrained)
        
        num_features = self.densenet.classifier.in_features
        
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.densenet(x)


# ========================== Dataset Classes ==========================

class TBDataset(Dataset):
    """
    Dataset for TB detection.
    Supports various data formats and augmentation.
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
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CSVTBDataset(Dataset):
    """
    Dataset that loads from CSV file format.
    Compatible with original TBNet CSV format.
    """
    def __init__(self, csv_path, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    img_path = row[0]
                    label = int(row[1])
                    
                    # Build full path
                    full_path = os.path.join(data_dir, img_path)
                    if not os.path.exists(full_path):
                        full_path = img_path
                    
                    if os.path.exists(full_path):
                        self.image_paths.append(full_path)
                        self.labels.append(label)
    
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

def get_train_transforms(img_size=224):
    """
    Advanced augmentation for TB chest X-rays.
    Includes TB-specific augmentations.
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
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
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08))
    ])


def get_val_transforms(img_size=224):
    """Validation transforms."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ========================== Data Preparation ==========================

def prepare_montgomery_shenzhen(data_dir):
    """
    Prepare Montgomery County and Shenzhen Hospital datasets.
    Expected structure:
    data_dir/
        Montgomery/
            CXR_png/
            clinical_readings/
        Shenzhen/
            CXR_png/
            clinical_readings/
    OR:
    data_dir/
        normal/ or Normal/
        tuberculosis/ or Tuberculosis/
    """
    paths = []
    labels = []
    
    # Check for folder-based structure
    for normal_folder in ['normal', 'Normal', 'NORMAL', 'healthy', 'Healthy']:
        folder = os.path.join(data_dir, normal_folder)
        if os.path.exists(folder):
            for img_name in os.listdir(folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(folder, img_name))
                    labels.append(0)
    
    for tb_folder in ['tuberculosis', 'Tuberculosis', 'TB', 'tb', 'positive', 'Positive']:
        folder = os.path.join(data_dir, tb_folder)
        if os.path.exists(folder):
            for img_name in os.listdir(folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(folder, img_name))
                    labels.append(1)
    
    # Check for Montgomery/Shenzhen structure
    for dataset in ['Montgomery', 'Shenzhen', 'montgomery', 'shenzhen']:
        cxr_dir = os.path.join(data_dir, dataset, 'CXR_png')
        if os.path.exists(cxr_dir):
            for img_name in os.listdir(cxr_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Label from filename: 0 if 'normal', 1 if 'tb' or 'tuberculosis'
                    img_lower = img_name.lower()
                    if 'normal' in img_lower or img_lower.startswith('mcucxr_0'):
                        label = 0
                    else:
                        label = 1
                    paths.append(os.path.join(cxr_dir, img_name))
                    labels.append(label)
    
    if not paths:
        # Try to find images recursively
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            for img_path in glob.glob(os.path.join(data_dir, '**', ext), recursive=True):
                # Try to infer label from path
                path_lower = img_path.lower()
                if 'normal' in path_lower or 'healthy' in path_lower or '/0/' in path_lower:
                    label = 0
                elif 'tb' in path_lower or 'tuberculosis' in path_lower or '/1/' in path_lower:
                    label = 1
                else:
                    continue
                paths.append(img_path)
                labels.append(label)
    
    return paths, labels


def prepare_tbx11k(data_dir):
    """
    Prepare TBX11K dataset.
    Expected structure:
    data_dir/
        imgs/
            health/
            sick/
            tb/
    """
    paths = []
    labels = []
    
    imgs_dir = os.path.join(data_dir, 'imgs')
    if not os.path.exists(imgs_dir):
        imgs_dir = data_dir
    
    # Health (normal)
    for folder in ['health', 'healthy', 'Health', 'Healthy']:
        health_dir = os.path.join(imgs_dir, folder)
        if os.path.exists(health_dir):
            for img_name in os.listdir(health_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(health_dir, img_name))
                    labels.append(0)
    
    # TB (positive)
    for folder in ['tb', 'TB', 'tuberculosis', 'Tuberculosis']:
        tb_dir = os.path.join(imgs_dir, folder)
        if os.path.exists(tb_dir):
            for img_name in os.listdir(tb_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(tb_dir, img_name))
                    labels.append(1)
    
    return paths, labels


def load_csv_data(csv_path, data_dir):
    """Load data from CSV file."""
    paths = []
    labels = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                img_path = os.path.join(data_dir, row[0])
                if not os.path.exists(img_path):
                    img_path = row[0]
                if os.path.exists(img_path):
                    paths.append(img_path)
                    labels.append(int(row[1]))
    
    return paths, labels


def auto_detect_and_prepare(data_dir):
    """Auto-detect dataset format and prepare data."""
    
    print(f"\nSearching for data in: {os.path.abspath(data_dir)}")
    print(f"Directory exists: {os.path.exists(data_dir)}")
    
    if os.path.exists(data_dir):
        print(f"Contents: {os.listdir(data_dir)[:10]}...")  # Show first 10 items
    
    # Check for CSV files (original TBNet format)
    train_csv = os.path.join(data_dir, 'train_split.csv')
    val_csv = os.path.join(data_dir, 'val_split.csv')
    test_csv = os.path.join(data_dir, 'test_split.csv')
    
    if os.path.exists(train_csv):
        print("Detected CSV format (TBNet style)")
        train_paths, train_labels = load_csv_data(train_csv, data_dir)
        
        if os.path.exists(val_csv):
            val_paths, val_labels = load_csv_data(val_csv, data_dir)
        else:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths, train_labels, test_size=0.15, stratify=train_labels
            )
        
        if os.path.exists(test_csv):
            test_paths, test_labels = load_csv_data(test_csv, data_dir)
        else:
            test_paths, test_labels = val_paths, val_labels
        
        return {
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels),
            'test': (test_paths, test_labels)
        }
    
    # Check for folder structure
    print("\nLooking for folder-based structure (Normal/Tuberculosis folders)...")
    all_paths, all_labels = prepare_montgomery_shenzhen(data_dir)
    
    if not all_paths:
        print("Trying TBX11K format...")
        all_paths, all_labels = prepare_tbx11k(data_dir)
    
    if not all_paths:
        print("\nERROR: Could not find any images in the data directory")
        print("\nExpected folder structure:")
        print("  data_dir/")
        print("    Normal/     (or normal/, NORMAL/)")
        print("      *.png, *.jpg")
        print("    Tuberculosis/  (or tuberculosis/, TB/)")
        print("      *.png, *.jpg")
        print(f"\nProvided path: {os.path.abspath(data_dir)}")
        
        # Suggest correct path if we can find it
        parent_dir = os.path.dirname(data_dir)
        if os.path.exists(parent_dir):
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path):
                    normal_exists = any(os.path.exists(os.path.join(item_path, f)) for f in ['Normal', 'normal', 'NORMAL'])
                    tb_exists = any(os.path.exists(os.path.join(item_path, f)) for f in ['Tuberculosis', 'tuberculosis', 'TB', 'tb'])
                    if normal_exists or tb_exists:
                        print(f"\nDid you mean: {item_path} ?")
        
        return None
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.25, stratify=all_labels, random_state=42
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


# ========================== Training Functions ==========================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
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
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except:
        epoch_auc = 0.5
    
    return epoch_loss, epoch_acc, epoch_auc, all_labels, all_preds, all_probs


def print_tb_metrics(labels, preds, probs=None, title="TB Detection Metrics"):
    """Print TB-specific evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              Normal  TB")
    print(f"Actual Normal   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       TB       {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Sensitivity (TB Detection Rate): {sensitivity:.4f}")
    print(f"   Specificity:                     {specificity:.4f}")
    print(f"   Precision (PPV):                 {ppv:.4f}")
    print(f"   NPV:                             {npv:.4f}")
    print(f"   F1 Score:                        {f1:.4f}")
    print(f"   Accuracy:                        {accuracy:.4f}")
    
    if probs is not None:
        try:
            auc = roc_auc_score(labels, probs)
            print(f"   AUC-ROC:                         {auc:.4f}")
        except:
            pass
    
    # TB screening metrics
    print(f"\n🏥 TB Screening Performance:")
    print(f"   Detection Rate (Sensitivity): {sensitivity*100:.1f}%")
    print(f"   False Negative Rate:          {(1-sensitivity)*100:.1f}%")
    print(f"   False Positive Rate:          {(1-specificity)*100:.1f}%")
    
    print("\n" + classification_report(labels, preds, 
                                        target_names=['Normal', 'Tuberculosis']))


# ========================== Main Training Function ==========================

def train_model(args):
    """Main training function."""
    print("\n" + "="*70)
    print("TUBERCULOSIS DETECTION MODEL - FINE-TUNING")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Prepare data
    print("\n" + "-"*50)
    print("Loading Dataset...")
    print("-"*50)
    
    data = auto_detect_and_prepare(args.data_dir)
    
    if data is None:
        print("ERROR: Could not load data!")
        return None
    
    train_paths, train_labels = data['train']
    val_paths, val_labels = data['val']
    test_paths, test_labels = data['test']
    
    print(f"Training samples:   {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples:       {len(test_paths)}")
    
    # Class distribution
    train_dist = Counter(train_labels)
    print(f"\nTraining class distribution:")
    print(f"  Normal: {train_dist[0]} | TB: {train_dist[1]}")
    
    # Create datasets
    train_dataset = TBDataset(train_paths, train_labels, transform=get_train_transforms(args.img_size))
    val_dataset = TBDataset(val_paths, val_labels, transform=get_val_transforms(args.img_size))
    test_dataset = TBDataset(test_paths, test_labels, transform=get_val_transforms(args.img_size))
    
    # Weighted sampler for imbalanced data
    if args.weighted_sampling:
        class_counts = [train_dist[0], train_dist[1]]
        sample_weights = [1.0 / class_counts[label] for label in train_labels]
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
    
    # Initialize model
    print("\n" + "-"*50)
    print("Initializing Model...")
    print("-"*50)
    
    if args.model == 'densenet':
        model = TBNetPyTorch(num_classes=2, pretrained=True, dropout_rate=args.dropout)
        print("Using TBNetPyTorch (DenseNet-121 with attention)")
    elif args.model == 'resnet50':
        model = ResNetTB(num_classes=2, pretrained=True, dropout_rate=args.dropout)
        print("Using ResNetTB (ResNet-50)")
    elif args.model == 'efficientnet':
        model = EfficientNetTB(num_classes=2, pretrained=True, dropout_rate=args.dropout)
        print("Using EfficientNetTB")
    else:
        model = SimpleTBNet(pretrained=True)
        print("Using SimpleTBNet")
    
    model = model.to(device)
    
    # Load checkpoint if resuming
    if args.resume and args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    
    # Loss function with class weights
    class_weights = torch.tensor([train_dist[1] / train_dist[0], 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using weighted CrossEntropyLoss")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader)
        )
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "-"*50)
    print("Starting Training...")
    print("-"*50)
    
    best_auc = 0.0
    best_sensitivity = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        val_loss, val_acc, val_auc, val_labels, val_preds, val_probs = validate(
            model, val_loader, criterion, device
        )
        
        # Calculate sensitivity
        cm = confusion_matrix(val_labels, val_preds)
        if cm.size >= 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            sensitivity = 0
        
        # Update scheduler
        if args.scheduler == 'onecycle':
            pass  # OneCycleLR updates per batch
        elif args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_auc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val AUC:    {val_auc:.4f} | Sensitivity: {sensitivity:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Save best model (prioritize sensitivity for TB detection)
        is_best = (val_auc > best_auc) or (val_auc >= best_auc - 0.01 and sensitivity > best_sensitivity)
        
        if is_best:
            best_auc = val_auc
            best_sensitivity = sensitivity
            best_epoch = epoch + 1
            patience_counter = 0
            
            model_path = os.path.join(args.save_dir, 'tb_model_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'sensitivity': sensitivity,
                'model_type': args.model,
            }, model_path)
            print(f"✓ Saved best model (AUC: {val_auc:.4f}, Sensitivity: {sensitivity:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # Load best model
    best_model_path = os.path.join(args.save_dir, 'tb_model_best.pth')
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    test_loss, test_acc, test_auc, test_labels, test_preds, test_probs = validate(
        model, test_loader, criterion, device
    )
    
    print_tb_metrics(test_labels, test_preds, test_probs, "Test Set Performance")
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'tb_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_auc': test_auc,
        'model_type': args.model,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Best Sensitivity: {best_sensitivity:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune TB Detection Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='./models/tb',
                        help='Directory to save models')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='densenet',
                        choices=['densenet', 'resnet50', 'efficientnet', 'simple'],
                        help='Model architecture')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path')
    
    # Training options
    parser.add_argument('--weighted_sampling', action='store_true', default=True,
                        help='Use weighted sampling')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['cosine', 'plateau', 'onecycle'],
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
