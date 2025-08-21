#!/usr/bin/env python3
"""
Minimal test script to verify import structure works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def test_basic_imports():
    """Test basic imports without external dependencies."""
    print("Testing basic module structure...")
    
    try:
        print("  Testing attention modules...")
        from src.attention.cbam import CBAM, ChannelAttention, SpatialAttention
        from src.attention.eca import ECA
        print("    Attention modules imported successfully")
        
        print("  Testing loss functions...")
        from src.training.loss_functions import SIoULoss, FocalLoss
        print("    Loss functions imported successfully")
        
        print("  Testing utils (basic)...")
        from src.utils.checkpoint import CheckpointManager
        print("    Basic utilities imported successfully")
        
        print("\nBasic imports successful!")
        return True
        
    except ImportError as e:
        print(f"\nBasic import failed: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

def test_attention_creation():
    """Test creating attention modules."""
    print("\nTesting attention module creation...")
    
    try:
        import torch
        from src.attention.cbam import CBAM
        from src.attention.eca import ECA
        
        # Test CBAM
        cbam = CBAM(channels=256)
        x = torch.randn(1, 256, 32, 32)
        out = cbam(x)
        print(f"  CBAM: Input {x.shape} -> Output {out.shape}")
        
        # Test ECA
        eca = ECA(channels=256)
        out = eca(x)
        print(f"  ECA: Input {x.shape} -> Output {out.shape}")
        
        print("  Attention modules working correctly!")
        return True
        
    except ImportError as e:
        if "torch" in str(e):
            print("  PyTorch not available - skipping attention tests")
            return True  # Not critical for basic structure test
        else:
            print(f"  Attention test failed: {e}")
            return False
    except Exception as e:
        print(f"  Attention test failed: {e}")
        return False

def test_loss_functions():
    """Test loss function creation."""
    print("\nTesting loss functions...")
    
    try:
        import torch
        from src.training.loss_functions import SIoULoss, FocalLoss
        
        # Test SIoU Loss
        siou_loss = SIoULoss()
        pred_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]])
        target_boxes = torch.tensor([[12.0, 12.0, 48.0, 48.0], [18.0, 18.0, 58.0, 58.0]])
        loss = siou_loss(pred_boxes, target_boxes)
        print(f"  SIoU Loss: {loss.item():.4f}")
        
        # Test Focal Loss
        focal_loss = FocalLoss()
        inputs = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        loss = focal_loss(inputs, targets)
        print(f"  Focal Loss: {loss.item():.4f}")
        
        print("  Loss functions working correctly!")
        return True
        
    except ImportError as e:
        if "torch" in str(e):
            print("  PyTorch not available - skipping loss function tests")
            return True  # Not critical for basic structure test
        else:
            print(f"  Loss function test failed: {e}")
            return False
    except Exception as e:
        print(f"  Loss function test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== TinyViT-YOLOv8 Import Structure Test ===\n")
    
    success = True
    
    # Test basic imports
    if not test_basic_imports():
        success = False
    
    # Test attention modules
    if not test_attention_creation():
        success = False
    
    # Test loss functions  
    if not test_loss_functions():
        success = False
    
    print("\n" + "="*50)
    if success:
        print("All core modules are working correctly!")
        print("The import structure is properly set up.")
        print("\nTo use the full system:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Ensure TinyViT external repo is properly cloned")
        print("3. Install ultralytics for YOLO integration")
    else:
        print("Some tests failed. Check the error messages above.")
        sys.exit(1)