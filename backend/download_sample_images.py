"""
Download Sample Chest X-ray Images for CheXNet Testing
Downloads public domain chest X-ray images for testing inference
"""

import os
import sys
import urllib.request
import shutil
from pathlib import Path

# Sample chest X-ray images from public sources
# These are placeholder URLs - using NIH Clinical Center public samples
SAMPLE_IMAGES = {
    # Using images from the public domain COVID-19 dataset (chest X-rays)
    # These work well for CheXNet testing since they're chest X-rays
    "sample_chest_xray_1.png": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1.CXRCTThowormo.png",
    "sample_chest_xray_2.png": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0140673620303706-fx1_lrg.jpg",
}

def get_project_root():
    """Get the backend directory path"""
    return Path(__file__).parent

def create_directories():
    """Create necessary directories for images"""
    base_path = get_project_root() / "datasets" / "CheXNet" / "ChestX-ray14" / "images"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def download_image(url, save_path):
    """Download a single image from URL"""
    try:
        print(f"  Downloading from: {url[:60]}...")
        
        # Create a request with headers to avoid 403 errors
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        with urllib.request.urlopen(request, timeout=30) as response:
            with open(save_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        
        print(f"  ✅ Saved to: {save_path.name}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def copy_tb_images_as_samples():
    """
    Copy some TB dataset images to CheXNet test folder.
    TB chest X-rays work perfectly for CheXNet testing since they're chest X-rays.
    """
    base_path = get_project_root()
    tb_normal_path = base_path / "datasets" / "TuberculosisNet" / "Dataset" / "TB_Chest_Radiography_Database" / "Normal"
    tb_disease_path = base_path / "datasets" / "TuberculosisNet" / "Dataset" / "TB_Chest_Radiography_Database" / "Tuberculosis"
    target_path = base_path / "datasets" / "CheXNet" / "ChestX-ray14" / "images"
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    
    # Copy some normal chest X-rays
    if tb_normal_path.exists():
        normal_images = list(tb_normal_path.glob("*.png"))[:5]
        for img in normal_images:
            dest = target_path / f"chexnet_test_{img.name}"
            if not dest.exists():
                shutil.copy2(img, dest)
                print(f"  ✅ Copied: {dest.name}")
                copied += 1
    
    # Copy some TB chest X-rays
    if tb_disease_path.exists():
        tb_images = list(tb_disease_path.glob("*.png"))[:5]
        for img in tb_images:
            dest = target_path / f"chexnet_test_{img.name}"
            if not dest.exists():
                shutil.copy2(img, dest)
                print(f"  ✅ Copied: {dest.name}")
                copied += 1
    
    return copied

def download_from_urls():
    """Download images from URLs"""
    images_path = create_directories()
    
    successful = 0
    for filename, url in SAMPLE_IMAGES.items():
        save_path = images_path / filename
        if save_path.exists():
            print(f"  ⏭️  Already exists: {filename}")
            successful += 1
            continue
        
        if download_image(url, save_path):
            successful += 1
    
    return successful

def main():
    """Main function to download sample images"""
    print("=" * 60)
    print("CheXNet Sample Images Downloader")
    print("=" * 60)
    
    # First, try to copy from existing TB dataset (most reliable)
    print("\n📁 Copying sample images from TB dataset...")
    copied = copy_tb_images_as_samples()
    
    if copied > 0:
        print(f"\n✅ Successfully copied {copied} images from TB dataset!")
    else:
        print("\n⚠️  No TB images found, trying URL downloads...")
        # Try downloading from URLs as fallback
        print("\n🌐 Downloading from URLs...")
        downloaded = download_from_urls()
        print(f"\n✅ Downloaded {downloaded} images from URLs")
    
    # Verify images exist
    images_path = get_project_root() / "datasets" / "CheXNet" / "ChestX-ray14" / "images"
    image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Images directory: {images_path}")
    print(f"Total images available: {len(image_files)}")
    
    if len(image_files) > 0:
        print("\n✅ CheXNet is ready for inference testing!")
        print("\nSample images:")
        for img in image_files[:5]:
            print(f"  - {img.name}")
        if len(image_files) > 5:
            print(f"  ... and {len(image_files) - 5} more")
    else:
        print("\n⚠️  No sample images available. Manual download may be required.")
        print("Tip: Download any chest X-ray image and place it in:")
        print(f"     {images_path}")

if __name__ == "__main__":
    main()
