"""
Script to prepare the Tuberculosis dataset for training.
Moves and renames files from TB_Chest_Radiography_Database to data/
"""

import os
import shutil
import glob

def prepare_data():
    source_root = os.path.join('data', 'TB_Chest_Radiography_Database')
    dest_dir = 'data'
    
    if not os.path.exists(source_root):
        print(f"Error: Source directory {source_root} not found!")
        return

    # Process Normal images
    print("Processing Normal images...")
    normal_src = os.path.join(source_root, 'Normal')
    normal_files = glob.glob(os.path.join(normal_src, '*.png'))
    
    for f in normal_files:
        filename = os.path.basename(f)
        # Normal files strictly keep their name: Normal-X.png
        dest_path = os.path.join(dest_dir, filename)
        if not os.path.exists(dest_path):
            shutil.copy2(f, dest_path)
    print(f"Processed {len(normal_files)} Normal images.")

    # Process Tuberculosis images
    print("Processing Tuberculosis images...")
    tb_src = os.path.join(source_root, 'Tuberculosis')
    tb_files = glob.glob(os.path.join(tb_src, '*.png'))
    
    for f in tb_files:
        filename = os.path.basename(f)
        # Rename Tuberculosis-X.png to TB-X.png
        new_filename = filename.replace('Tuberculosis-', 'TB-')
        dest_path = os.path.join(dest_dir, new_filename)
        if not os.path.exists(dest_path):
            shutil.copy2(f, dest_path)
    print(f"Processed {len(tb_files)} Tuberculosis images.")
    
    print("\nDataset preparation complete!")

if __name__ == '__main__':
    prepare_data()
