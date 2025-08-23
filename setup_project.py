#!/usr/bin/env python3
"""
RT-DETR PCB Project Setup Script
================================

Sets up the project structure and downloads dataset automatically.
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """Create organized project structure"""
    print("Creating project structure...")
    
    folders = [
        'data',           # Dataset configurations and cached data
        'models',         # Trained models and checkpoints
        'results',        # Training results and logs
        'logs',           # Training logs
        'docs',           # Documentation
        'inference',      # Inference outputs
    ]
    
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        print(f"[OK] Created: {folder}/")
    
    print("[OK] Project structure created!")

def download_dataset():
    """Download PCB dataset using kagglehub"""
    print("\nDownloading PCB defect dataset...")
    
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub
    
    try:
        # Download latest version
        dataset_path = kagglehub.dataset_download("norbertelter/pcb-defect-dataset")
        print(f"[OK] Dataset downloaded to: {dataset_path}")
        
        # Create data.yaml
        create_data_config(dataset_path)
        return dataset_path
        
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        print("You can download manually later or the script will auto-download during training.")
        return None

def create_data_config(dataset_path):
    """Create data.yaml configuration"""
    config = f"""# RT-DETR PCB Dataset Configuration
path: {dataset_path}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 6

# Class names
names:
  0: mouse_bite
  1: spur
  2: missing_hole
  3: short
  4: open_circuit
  5: spurious_copper
"""
    
    with open('data/data.yaml', 'w') as f:
        f.write(config)
    
    print("[OK] Data configuration created: data/data.yaml")

def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")
    import subprocess
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Requirements installed successfully!")
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        print("Please run: pip install -r requirements.txt")

def main():
    print("RT-DETR PCB DEFECT DETECTION - PROJECT SETUP")
    print("="*60)
    
    # Create project structure
    create_project_structure()
    
    # Install requirements
    install_requirements()
    
    # Download dataset
    dataset_path = download_dataset()
    
    print("\n" + "="*60)
    print("SETUP COMPLETED!")
    print("="*60)
    
    if dataset_path:
        print("[SUCCESS] Project is ready for training!")
        print("\nNext steps:")
        print("1. Start training: python main.py train --epochs 100 --batch 20")
        print("2. Or quick test: python main.py train --epochs 5 --batch 16")
    else:
        print("[WARNING] Dataset needs to be downloaded manually or will auto-download during training")
        print("\nNext steps:")
        print("1. Try training (will auto-download): python main.py train --epochs 100 --batch 20")
    
    print(f"\nProject structure:")
    print("├── main.py           # Main RT-DETR script")
    print("├── data/             # Dataset configs")
    print("├── models/           # Trained models")
    print("├── results/          # Training results")
    print("├── inference/        # Inference outputs")
    print("└── requirements.txt  # Dependencies")

if __name__ == "__main__":
    main()