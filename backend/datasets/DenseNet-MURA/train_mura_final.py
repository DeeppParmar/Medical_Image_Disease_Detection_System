"""
FINAL MURA TRAINING SCRIPT
Complete training solution for musculoskeletal abnormality detection

Usage:
    python train_mura_final.py                           # Train XR_WRIST with 10 epochs
    python train_mura_final.py --study_type XR_SHOULDER  # Train shoulder
    python train_mura_final.py --epochs 20 --lr 0.00005  # Custom settings
    python train_mura_final.py --study_type all          # Train all body parts
"""

import os
import sys
import time
import copy
import argparse
import torch
import torch.nn as nn
from densenet import densenet169
from utils import n_p, get_count
from train import train_model, get_metrics
from pipeline import get_study_level_data, get_dataloaders

# All available study types in MURA dataset
STUDY_TYPES = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

class WeightedBCELoss(torch.nn.Module):
    """Weighted Binary Cross Entropy Loss with numerical stability"""
    def __init__(self, Wt1, Wt0):
        super(WeightedBCELoss, self).__init__()
        self.Wt1 = Wt1
        self.Wt0 = Wt0
        self.eps = 1e-7
        
    def forward(self, inputs, targets, phase):
        # Clamp inputs to prevent log(0)
        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)
        loss = -(self.Wt1[phase] * targets * torch.log(inputs) + 
                 self.Wt0[phase] * (1 - targets) * torch.log(1 - inputs))
        return loss.mean()

def find_mura_dataset():
    """Find MURA dataset location"""
    possible_paths = ['MURA', 'MURA-v1.1', 'MURA-v1.0']
    
    for path in possible_paths:
        if os.path.exists(path):
            train_dir = os.path.join(path, 'train')
            valid_dir = os.path.join(path, 'valid')
            if os.path.exists(train_dir) and os.path.exists(valid_dir):
                return path
    return None

def check_dataset():
    """Check if dataset exists and show info"""
    mura_path = find_mura_dataset()
    
    if not mura_path:
        print("\n" + "=" * 80)
        print("❌ MURA DATASET NOT FOUND")
        print("=" * 80)
        print("\nSearched in:")
        print("  - MURA/")
        print("  - MURA-v1.1/")
        print("  - MURA-v1.0/")
        print("\nPlease ensure the dataset is extracted to one of these locations.")
        print("\nExpected structure:")
        print("  MURA/")
        print("    ├── train/")
        print("    │   ├── XR_WRIST/")
        print("    │   ├── XR_SHOULDER/")
        print("    │   └── ... (other study types)")
        print("    └── valid/")
        print("        └── ... (same structure)")
        return False
    
    print("\n" + "=" * 80)
    print("✅ MURA DATASET FOUND")
    print("=" * 80)
    print(f"\nLocation: {os.path.abspath(mura_path)}")
    
    train_dir = os.path.join(mura_path, 'train')
    valid_dir = os.path.join(mura_path, 'valid')
    
    # List available study types
    try:
        study_types = [d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('XR_')]
        study_types.sort()
        
        print(f"\n📊 Available Study Types: {len(study_types)}")
        print("-" * 80)
        
        for study_type in study_types:
            train_path = os.path.join(train_dir, study_type)
            valid_path = os.path.join(valid_dir, study_type)
            
            train_patients = len([d for d in os.listdir(train_path) 
                                 if os.path.isdir(os.path.join(train_path, d))])
            valid_patients = len([d for d in os.listdir(valid_path) 
                                 if os.path.isdir(os.path.join(valid_path, d))]) if os.path.exists(valid_path) else 0
            
            print(f"  {study_type:15s} - Train: {train_patients:4d} patients | Valid: {valid_patients:4d} patients")
        
        print("-" * 80)
        print(f"\n✓ Dataset verified successfully")
        print(f"✓ Ready for training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error checking dataset: {e}")
        return False

def train_single_study(study_type, num_epochs, learning_rate, batch_size, device, save_dir):
    """Train model for a single study type"""
    
    print("\n" + "=" * 80)
    print(f"🏥 TRAINING: {study_type}")
    print("=" * 80)
    
    try:
        # Load study data
        print("\n📂 Loading study data...")
        study_data = get_study_level_data(study_type=study_type)
        
        # Create dataloaders
        data_cat = ['train', 'valid']
        print("🔄 Creating data loaders...")
        dataloaders = get_dataloaders(study_data, batch_size=batch_size)
        dataset_sizes = {x: len(study_data[x]) for x in data_cat}
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Training studies:   {dataset_sizes['train']:,}")
        print(f"   Validation studies: {dataset_sizes['valid']:,}")
        
        # Calculate class weights for balanced training
        tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
        tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
        
        # Avoid division by zero
        Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) if (tni[x] + tai[x]) > 0 else n_p(0.5) for x in data_cat}
        Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) if (tni[x] + tai[x]) > 0 else n_p(0.5) for x in data_cat}
        
        print(f"\n⚖️  Class Distribution:")
        print(f"   Train - Abnormal: {tai['train']:,} | Normal: {tni['train']:,}")
        print(f"   Valid - Abnormal: {tai['valid']:,} | Normal: {tni['valid']:,}")
        print(f"   Weight (Abnormal): train={Wt1['train'].item():.4f} | valid={Wt1['valid'].item():.4f}")
        print(f"   Weight (Normal):   train={Wt0['train'].item():.4f} | valid={Wt0['valid'].item():.4f}")
        
        # Initialize model
        print(f"\n🤖 Initializing DenseNet-169 model...")
        model = densenet169(pretrained=True)
        model = model.to(device)
        print(f"   ✓ Model loaded on: {device}")
        
        # Setup training components
        criterion = WeightedBCELoss(Wt1, Wt0)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=1, factor=0.1
        )
        
        print(f"\n🎯 Training Configuration:")
        print(f"   Optimizer:      Adam")
        print(f"   Learning Rate:  {learning_rate}")
        print(f"   Batch Size:     {batch_size}")
        print(f"   Epochs:         {num_epochs}")
        print(f"   Loss Function:  Weighted BCE")
        
        # Train model
        print("\n" + "=" * 80)
        print("🚀 STARTING TRAINING")
        print("=" * 80 + "\n")
        
        model = train_model(
            model, criterion, optimizer, dataloaders, 
            scheduler, dataset_sizes, num_epochs=num_epochs
        )
        
        # Save model
        study_save_dir = os.path.join(save_dir, study_type)
        os.makedirs(study_save_dir, exist_ok=True)
        model_path = os.path.join(study_save_dir, 'model.pth')
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'study_type': study_type,
            'epochs': num_epochs,
            'learning_rate': learning_rate,
        }, model_path)
        
        print(f"\n💾 Model saved: {model_path}")
        
        # Final evaluation
        print("\n" + "=" * 80)
        print("📈 FINAL EVALUATION")
        print("=" * 80 + "\n")
        get_metrics(model, criterion, dataloaders, dataset_sizes)
        
        print("\n" + "=" * 80)
        print(f"✅ TRAINING COMPLETE: {study_type}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR training {study_type}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='MURA Training - Musculoskeletal Abnormality Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_mura_final.py
  python train_mura_final.py --study_type XR_SHOULDER --epochs 15
  python train_mura_final.py --study_type all --epochs 20
  python train_mura_final.py --check_only
        """
    )
    
    parser.add_argument('--study_type', type=str, default='XR_WRIST',
                        help=f'Study type: {", ".join(STUDY_TYPES)}, or "all"')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    parser.add_argument('--check_only', action='store_true',
                        help='Only check dataset and exit')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("🏥 MURA TRAINING - DENSENET-169")
    print("   Musculoskeletal Radiograph Abnormality Detection")
    print("=" * 80)
    
    # Check dataset
    if not check_dataset():
        print("\n❌ Dataset check failed. Please verify dataset location.")
        return 1
    
    if args.check_only:
        print("\n✓ Dataset check complete. Ready for training!")
        return 0
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("⚙️  CONFIGURATION")
    print("=" * 80)
    print(f"  Device:       {device}")
    if device.type == 'cuda':
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Study Type:   {args.study_type}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Learning Rate:{args.lr}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Save Dir:     {args.save_dir}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Start training
    start_time = time.time()
    success_count = 0
    failed_studies = []
    
    if args.study_type.lower() == 'all':
        print("\n" + "=" * 80)
        print(f"🔄 TRAINING ALL {len(STUDY_TYPES)} STUDY TYPES")
        print("=" * 80)
        
        for i, study_type in enumerate(STUDY_TYPES, 1):
            print(f"\n[{i}/{len(STUDY_TYPES)}] Processing: {study_type}")
            
            if train_single_study(study_type, args.epochs, args.lr, 
                                 args.batch_size, device, args.save_dir):
                success_count += 1
            else:
                failed_studies.append(study_type)
    else:
        # Train single study type
        if train_single_study(args.study_type, args.epochs, args.lr,
                             args.batch_size, device, args.save_dir):
            success_count = 1
        else:
            failed_studies.append(args.study_type)
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING SESSION COMPLETE")
    print("=" * 80)
    print(f"  Total Time:      {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
    print(f"  Successful:      {success_count}")
    
    if failed_studies:
        print(f"  Failed:          {len(failed_studies)}")
        print(f"  Failed Studies:  {', '.join(failed_studies)}")
    
    print(f"  Models Saved:    {args.save_dir}/")
    
    print("\n💡 Next Steps:")
    print("  1. Check training metrics above")
    print("  2. Use models for inference: python -c \"from models.mura_inference import MURAPredictor; ...\"")
    print("  3. Integrate with backend API")
    
    if failed_studies:
        print("\n⚠️  Some studies failed. Check error messages above.")
        return 1
    
    print("\n✅ All training completed successfully!")
    return 0

if __name__ == '__main__':
    exit(main())
