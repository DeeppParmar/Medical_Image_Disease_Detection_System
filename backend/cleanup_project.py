"""
Project Cleanup Script
Removes unnecessary files and keeps only:
- Tuberculosis Detection (TuberculosisNet)
- Pneumonia Detection (CheXNet)
- Bone Fracture Detection (DenseNet-MURA)
- GradCAM for explainability
"""

import os
import shutil
import sys

def remove_path(path, description):
    """Safely remove file or directory"""
    try:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
                print(f"✓ Removed file: {description}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"✓ Removed directory: {description}")
            return True
        else:
            print(f"⊘ Not found: {description}")
            return False
    except Exception as e:
        print(f"✗ Error removing {description}: {e}")
        return False

def main():
    print("=" * 80)
    print("PROJECT CLEANUP - Keeping Only Required Models")
    print("=" * 80)
    print("\nRequired Models:")
    print("  ✓ TuberculosisNet - TB Detection")
    print("  ✓ CheXNet - Pneumonia & Chest Diseases")
    print("  ✓ DenseNet-MURA - Bone Fracture Detection")
    print("  ✓ grad-cam - Explainable AI visualization")
    print("\nRemoving:")
    print("  ✗ rsna18 - Intracranial Hemorrhage (not needed)")
    print("  ✗ UNet - Medical segmentation (not needed)")
    print("  ✗ Unnecessary documentation files")
    print("  ✗ Setup scripts (already set up)")
    print("  ✗ License files")
    print("=" * 80)
    
    response = input("\nProceed with cleanup? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cleanup cancelled.")
        return
    
    print("\n" + "=" * 80)
    print("CLEANUP IN PROGRESS")
    print("=" * 80)
    
    os.chdir('E:/FINAL/backend')
    
    # ====================================
    # 1. Remove Unnecessary Dataset Models
    # ====================================
    print("\n[1/5] Removing unnecessary dataset models...")
    
    remove_path('datasets/rsna18', 'RSNA Intracranial Hemorrhage dataset')
    remove_path('datasets/UNet', 'UNet segmentation model')
    
    # ====================================
    # 2. Remove License Files
    # ====================================
    print("\n[2/5] Removing license files...")
    
    license_files = [
        'datasets/CheXNet/LICENSE',
        'datasets/DenseNet-MURA/LICENSE',
        'datasets/TuberculosisNet/LICENSE.md',
        'datasets/grad-cam/LICENSE',
    ]
    
    for license_file in license_files:
        remove_path(license_file, f'License: {license_file}')
    
    # ====================================
    # 3. Remove Setup/Installation Scripts
    # ====================================
    print("\n[3/5] Removing setup scripts (already configured)...")
    
    setup_files = [
        'activate_kaggle_token.sh',
        'fix_wsl_network.sh',
        'install_python310.sh',
        'install_wsl.md',
        'setup_kaggle_token.py',
        'setup_kaggle_wsl.sh',
        'setup_wsl.sh',
        'set_kaggle_token.bat',
        'set_kaggle_token.sh',
        'setup_and_train.sh',
        'cleanup_datasets.py',
        'train_all_models.py',
        'train_all_wsl.sh',
        'train_models_step_by_step.py',
        'train_with_setup.py',
        'start_tuberculosis_training.py',
        'RUN_TRAINING_NOW.sh',
        '=2.13.0',
    ]
    
    for setup_file in setup_files:
        remove_path(setup_file, f'Setup: {setup_file}')
    
    # ====================================
    # 4. Remove Redundant Documentation
    # ====================================
    print("\n[4/5] Removing redundant documentation...")
    
    redundant_docs = [
        'COMPLETE_SETUP.md',
        'KAGGLE_SETUP.md',
        'QUICKSTART.md',
        'START_TRAINING.md',
        'START_TUBERCULOSIS_TRAINING.md',
        'TRAINING_STATUS.md',
        'TRAIN_NOW.md',
        'WSL_QUICK_FIX.md',
        'MURA_TRAINING_INSTRUCTIONS.md',
        'datasets/CheXNet/README.md',
        'datasets/DenseNet-MURA/README.md',
        'datasets/DenseNet-MURA/QUICKSTART_MURA.md',
        'datasets/DenseNet-MURA/download_mura.py',
        'datasets/DenseNet-MURA/quick_train_mura.bat',
        'datasets/DenseNet-MURA/quick_train_mura.sh',
        'datasets/TuberculosisNet/README.md',
        'datasets/TuberculosisNet/docs',
        'datasets/grad-cam/README.md',
        'datasets/grad-cam/MANIFEST.in',
        'datasets/grad-cam/setup.cfg',
        'datasets/grad-cam/setup.py',
        'datasets/grad-cam/pyproject.toml',
    ]
    
    for doc_file in redundant_docs:
        remove_path(doc_file, f'Doc: {doc_file}')
    
    # ====================================
    # 5. Remove Unused Training Scripts
    # ====================================
    print("\n[5/5] Removing old/unused training scripts...")
    
    unused_scripts = [
        'datasets/DenseNet-MURA/main.py',
        'datasets/DenseNet-MURA/train.py',
        'datasets/DenseNet-MURA/train_mura.py',
        'datasets/DenseNet-MURA/train_mura_complete.sh',
        'datasets/DenseNet-MURA/train_mura_single.py',
        'datasets/DenseNet-MURA/check_training.sh',
        'datasets/DenseNet-MURA/quick_train.sh',
        'datasets/TuberculosisNet/train_tbnet.py',
        'datasets/TuberculosisNet/train_tbnet_wrapper.py',
        'datasets/TuberculosisNet/train_tuberculosis_setup.py',
        'datasets/TuberculosisNet/create_dataset.py',
        'datasets/TuberculosisNet/download_checkpoint.py',
        'datasets/TuberculosisNet/run_tbnet_pipeline.sh',
        'datasets/TuberculosisNet/verify_and_trim_splits.py',
    ]
    
    for script in unused_scripts:
        remove_path(script, f'Script: {script}')
    
    # ====================================
    # Summary
    # ====================================
    print("\n" + "=" * 80)
    print("✅ CLEANUP COMPLETE")
    print("=" * 80)
    
    print("\n📦 Remaining Models:")
    print("  ✓ datasets/TuberculosisNet/")
    print("  ✓ datasets/CheXNet/")
    print("  ✓ datasets/DenseNet-MURA/")
    print("  ✓ datasets/grad-cam/")
    
    print("\n📄 Key Files Kept:")
    print("  ✓ app.py - Backend API")
    print("  ✓ models/ - Inference modules")
    print("  ✓ requirements.txt")
    print("  ✓ README.md")
    print("  ✓ API_CONFIG.md")
    print("  ✓ PROJECT_SUMMARY.md")
    print("  ✓ TRAINING_GUIDE.md")
    
    print("\n🚀 Your Project Now Supports:")
    print("  1. Tuberculosis Detection")
    print("  2. Pneumonia Detection")
    print("  3. Bone Fracture Detection")
    print("  4. Explainable AI (GradCAM)")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Test backend: python app.py")
    print("  2. Check models: ls -la models/")
    print("  3. Start frontend: cd ../frontend && npm run dev")
    print("=" * 80)

if __name__ == '__main__':
    main()
