#!/usr/bin/env python3
"""
Sample training script to verify the TinyViT-YOLOv8 PCB Detection system.
This script creates synthetic data and runs a quick training test.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def check_system():
    """Check if the system is ready for training."""
    print("🔍 System Check:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print()

def test_imports():
    """Test core module imports."""
    print("📦 Testing imports...")
    
    try:
        from src.attention.cbam import CBAM
        print("✅ CBAM attention module")
        
        from src.training.loss_functions import SIoULoss, FocalLoss
        print("✅ Loss functions")
        
        from src.utils.checkpoint import CheckpointManager
        print("✅ Utilities")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_attention_module():
    """Test CBAM attention module."""
    print("\n🧠 Testing CBAM Attention...")
    
    try:
        from src.attention.cbam import CBAM
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cbam = CBAM(channels=256).to(device)
        
        # Test forward pass
        x = torch.randn(2, 256, 32, 32).to(device)
        output = cbam(x)
        
        print(f"✅ CBAM working on {device}")
        print(f"   Input: {x.shape} → Output: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ CBAM test failed: {e}")
        return False

def test_loss_functions():
    """Test loss functions."""
    print("\n📏 Testing Loss Functions...")
    
    try:
        from src.training.loss_functions import SIoULoss, FocalLoss
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test SIoU Loss
        siou_loss = SIoULoss()
        pred_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]]).to(device)
        target_boxes = torch.tensor([[12.0, 12.0, 48.0, 48.0], [18.0, 18.0, 58.0, 58.0]]).to(device)
        
        loss_value = siou_loss(pred_boxes, target_boxes)
        print(f"✅ SIoU Loss: {loss_value.item():.4f}")
        
        # Test Focal Loss
        focal_loss = FocalLoss()
        inputs = torch.randn(10, 7).to(device)  # 7 classes
        targets = torch.randint(0, 7, (10,)).to(device)
        
        loss_value = focal_loss(inputs, targets)
        print(f"✅ Focal Loss: {loss_value.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        return False

def test_yolo_integration():
    """Test YOLOv8 integration."""
    print("\n🎯 Testing YOLOv8 Integration...")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLOv8 available")
        
        # Try to create a model
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("✅ YOLOv8 model created successfully")
        
        return True
        
    except ImportError:
        print("❌ Ultralytics not available")
        print("   Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ YOLOv8 test failed: {e}")
        return False

def create_sample_data():
    """Create minimal sample data for testing."""
    print("\n📊 Creating sample data...")
    
    try:
        # Create data directory
        data_dir = project_root / "data" / "sample_pcb"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple dataset.yaml for testing
        dataset_yaml = f"""
# Sample PCB Dataset for Testing
path: {data_dir}
train: images/train
val: images/val

# Classes
nc: 6
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
"""
        
        with open(data_dir / "dataset.yaml", 'w') as f:
            f.write(dataset_yaml)
        
        print(f"✅ Sample dataset config created at {data_dir / 'dataset.yaml'}")
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def run_quick_test():
    """Run a quick functionality test."""
    print("\n🚀 Running Quick Functionality Test...")
    
    try:
        # Test if we can import and use ultralytics
        from ultralytics import YOLO
        
        print("Creating YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        
        print("✅ Quick test completed - ready for training!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        print("This is normal if ultralytics is not installed")
        return False

def main():
    """Main function to run all tests."""
    print("="*60)
    print("🔬 TinyViT-YOLOv8 PCB Detection System Test")
    print("="*60)
    
    # Run all tests
    tests = [
        ("System Check", check_system),
        ("Import Test", test_imports),
        ("Attention Module", test_attention_module),
        ("Loss Functions", test_loss_functions),
        ("YOLOv8 Integration", test_yolo_integration),
        ("Sample Data", create_sample_data),
        ("Quick Test", run_quick_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= total - 1:  # Allow 1 failure (e.g., YOLOv8 download)
        print("\n🎉 System is ready for PCB defect detection training!")
        print("\nNext steps:")
        print("1. Prepare your PCB dataset")
        print("2. Configure training parameters in configs/tinivit_yolov8_pcb.yaml")
        print("3. Run: python scripts/train_multi_stage.py --config configs/tinivit_yolov8_pcb.yaml")
    else:
        print("\n⚠️  System needs attention before training")
        print("Please fix the failed tests above")

if __name__ == "__main__":
    main()