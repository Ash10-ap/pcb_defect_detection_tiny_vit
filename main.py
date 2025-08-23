#!/usr/bin/env python3
"""
RT-DETR PCB Defect Detection System
===================================

Pure RT-DETR implementation for PCB defect detection.
No YOLO, no confusion - just RT-DETR working optimally.

Usage:
    python main.py train --epochs 100 --batch 16
    python main.py infer --model best.pt --img image.jpg
    python main.py validate --model best.pt
"""

import argparse
import os
import sys
import yaml
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class RTDETRDetector:
    """Pure RT-DETR PCB Defect Detection System"""
    
    def __init__(self):
        self.model = None
        self.class_names = ['mouse_bite', 'spur', 'missing_hole', 'short', 'open_circuit', 'spurious_copper']
        self.dataset_name = 'norbertelter/pcb-defect-dataset'
        
    def setup_dataset(self) -> str:
        """Setup dataset - download if needed"""
        print("="*60)
        print("DATASET SETUP")
        print("="*60)
        
        # Check if dataset exists locally
        potential_paths = [
            'pcb-defect-dataset',
            './pcb-defect-dataset',
            # Kagglehub cache locations
            os.path.expanduser('~/.cache/kagglehub/datasets/norbertelter/pcb-defect-dataset/versions/2'),
            os.path.expanduser('~/.cache/kagglehub/datasets/norbertelter/pcb-defect-dataset/versions/1'), 
            # Legacy kaggle API cache
            os.path.expanduser('~/.cache/kagglehub/datasets/norbertelter/pcb-defect-dataset/versions/2/pcb-defect-dataset'),
        ]
        
        dataset_path = None
        for path in potential_paths:
            if Path(path).exists() and self._check_dataset_structure(path):
                dataset_path = str(Path(path).absolute())
                print(f"✓ Found dataset at: {dataset_path}")
                break
        
        if not dataset_path:
            dataset_path = self._download_dataset()
        
        # Create data.yaml for RT-DETR
        data_yaml = self._create_data_yaml(dataset_path)
        print(f"✓ Dataset configuration: {data_yaml}")
        return data_yaml
    
    def _check_dataset_structure(self, path: str) -> bool:
        """Check if dataset has correct structure"""
        try:
            path = Path(path)
            required = ['train/images', 'val/images', 'test/images']
            for req in required:
                if not (path / req).exists():
                    return False
            # Check if there are images
            train_images = list((path / 'train/images').glob('*.jpg'))
            return len(train_images) > 100  # Should have many images
        except:
            return False
    
    def _download_dataset(self) -> str:
        """Download dataset using kagglehub"""
        print("Dataset not found locally. Attempting download...")
        
        try:
            import kagglehub
            print(f"Downloading {self.dataset_name} using kagglehub...")
            
            # Download latest version using kagglehub
            dataset_path = kagglehub.dataset_download(self.dataset_name)
            print(f"✓ Dataset downloaded to: {dataset_path}")
            
            # Verify the downloaded dataset
            if self._check_dataset_structure(dataset_path):
                return dataset_path
            else:
                raise FileNotFoundError("Downloaded dataset structure is invalid")
            
        except ImportError:
            print("❌ kagglehub not available. Installing...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
                import kagglehub
                dataset_path = kagglehub.dataset_download(self.dataset_name)
                print(f"✓ Dataset downloaded to: {dataset_path}")
                return dataset_path
            except Exception as e:
                print(f"❌ Failed to install kagglehub: {e}")
                self._show_manual_download_instructions()
                raise
        except Exception as e:
            print(f"❌ Download failed: {e}")
            self._show_manual_download_instructions()
            raise
    
    def _show_manual_download_instructions(self):
        """Show manual download instructions"""
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*60)
        print("Option 1 - Use kagglehub (Recommended):")
        print("  pip install kagglehub")
        print("  python -c \"import kagglehub; print(kagglehub.dataset_download('norbertelter/pcb-defect-dataset'))\"")
        print("")
        print("Option 2 - Manual download:")
        print("  1. Go to: https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset")
        print("  2. Download the dataset")
        print("  3. Extract to current directory as 'pcb-defect-dataset'")
        print("="*60)
    
    def _create_data_yaml(self, dataset_path: str) -> str:
        """Create data.yaml for RT-DETR training"""
        config = {
            'path': dataset_path,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 6,
            'names': {
                0: 'mouse_bite',
                1: 'spur',
                2: 'missing_hole',
                3: 'short',
                4: 'open_circuit',
                5: 'spurious_copper'
            }
        }
        
        # Ensure data directory exists
        Path('data').mkdir(exist_ok=True)
        data_yaml_path = 'data/data.yaml'
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return data_yaml_path
    
    def verify_gpu(self) -> bool:
        """Verify GPU setup"""
        print("="*60)
        print("GPU VERIFICATION")
        print("="*60)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU: {gpu_name}")
            print(f"✓ VRAM: {gpu_memory:.1f} GB")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("❌ No CUDA GPU detected")
            print("Training will be very slow on CPU")
            return False
    
    def train_rtdetr(self, args) -> bool:
        """Train RT-DETR model"""
        print("\n" + "="*60)
        print("RT-DETR TRAINING")
        print("="*60)
        
        # Setup dataset
        data_yaml = self.setup_dataset()
        
        # Verify GPU
        has_gpu = self.verify_gpu()
        device = args.device if has_gpu else 'cpu'
        batch_size = args.batch if has_gpu else min(args.batch, 4)
        
        print(f"\nTraining Configuration:")
        print(f"Model: RT-DETR-{args.model_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {args.imgsz}")
        print(f"Device: {device}")
        
        try:
            # Import RT-DETR
            from ultralytics import RTDETR
            
            # Load RT-DETR model
            model_name = f'rtdetr-{args.model_size}.pt'
            print(f"\nLoading {model_name}...")
            model = RTDETR(model_name)
            
            # Start training with optimized settings for RT-DETR
            print("\nStarting RT-DETR training...")
            results = model.train(
                # Dataset
                data=data_yaml,
                
                # Training parameters
                epochs=args.epochs,
                batch=batch_size,
                imgsz=args.imgsz,
                device=device,
                
                # RT-DETR optimized settings
                optimizer='AdamW',  # Best for transformers
                lr0=1e-4,          # Lower LR for transformer
                lrf=0.1,           # Final LR factor
                weight_decay=1e-4, # L2 regularization
                
                # Training strategy
                patience=20,       # Early stopping
                save_period=5,     # Save every 5 epochs
                
                # Data augmentation (lighter for PCB precision)
                degrees=3.0,       # Small rotation
                translate=0.05,    # Minimal translation
                scale=0.2,         # Moderate scaling
                shear=1.0,         # Small shear
                perspective=0.0,   # No perspective (PCBs are flat)
                flipud=0.0,        # No vertical flip
                fliplr=0.3,        # Some horizontal flip
                
                # Mixed precision for speed
                amp=True,
                
                # Output settings
                project='results',
                name=f'rtdetr_{args.model_size}_pcb',
                exist_ok=True,
                save=True,
                plots=True,
                
                # Validation
                val=True,
                
                # Workers (set to 0 to avoid multiprocessing issues)
                workers=0,
                
                # RT-DETR specific
                close_mosaic=10,   # Close mosaic augmentation early
                verbose=True
            )
            
            print(f"\n✓ Training completed!")
            print(f"✓ Best model: {results.save_dir}/weights/best.pt")
            print(f"✓ Results: {results.save_dir}")
            
            self.model = model
            return True
            
        except ImportError:
            print("❌ Error: ultralytics not installed or RT-DETR not available")
            print("Install with: pip install ultralytics>=8.0.200")
            return False
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_model(self, model_path: str) -> bool:
        """Load trained RT-DETR model"""
        try:
            from ultralytics import RTDETR
            print(f"Loading RT-DETR model from: {model_path}")
            self.model = RTDETR(model_path)
            print("✓ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def run_inference(self, image_path: str, args) -> Optional[object]:
        """Run inference on image"""
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        try:
            print(f"Running inference on: {image_path}")
            
            results = self.model(
                image_path,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False
            )
            
            return results[0] if results else None
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return None
    
    def save_results(self, result, image_path: str, output_dir: str = 'results'):
        """Save inference results with visualization"""
        try:
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            # Generate output filename
            input_name = Path(image_path).stem
            output_path = Path(output_dir) / f"{input_name}_rtdetr_detections.jpg"
            
            # Get annotated image
            annotated = result.plot(
                conf=True,
                line_width=2,
                font_size=12,
                pil=False
            )
            
            # Convert RGB to BGR for OpenCV
            if len(annotated.shape) == 3:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            
            # Save result
            success = cv2.imwrite(str(output_path), annotated)
            if success:
                print(f"✓ Results saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return None
    
    def print_detections(self, result):
        """Print detection details"""
        print("\n" + "="*50)
        print("DETECTION RESULTS")
        print("="*50)
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            print(f"Total detections: {len(boxes)}")
            print("-" * 50)
            
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                
                print(f"Detection {i+1}:")
                print(f"  Defect: {class_name}")
                print(f"  Confidence: {conf:.3f}")
                print(f"  Location: ({xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f})")
                print()
        else:
            print("No defects detected")
    
    def validate_model(self, model_path: str, data_yaml: str = None) -> bool:
        """Validate RT-DETR model"""
        print("\n" + "="*60)
        print("RT-DETR VALIDATION")
        print("="*60)
        
        if not self.load_model(model_path):
            return False
        
        if data_yaml is None:
            data_yaml = self.setup_dataset()
        
        try:
            print("Running validation on test dataset...")
            results = self.model.val(
                data=data_yaml,
                split='test',
                imgsz=640,
                batch=16,
                device='0' if torch.cuda.is_available() else 'cpu',
                save_json=True,
                plots=True,
                verbose=True
            )
            
            print(f"\n✓ Validation Results:")
            print(f"mAP@0.5: {results.box.map50:.4f}")
            print(f"mAP@0.5:0.95: {results.box.map:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='RT-DETR PCB Defect Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train RT-DETR
    python main.py train --epochs 100 --batch 16
    
    # Run inference
    python main.py infer --model best.pt --img test.jpg
    
    # Validate model
    python main.py validate --model best.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Training
    train_parser = subparsers.add_parser('train', help='Train RT-DETR model')
    train_parser.add_argument('--model_size', choices=['s', 'l', 'x'], default='l',
                             help='RT-DETR model size (s=small, l=large, x=extra-large)')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    train_parser.add_argument('--device', default='0', help='Device (0 for GPU, cpu for CPU)')
    
    # Inference
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', required=True, help='Model path (.pt file)')
    infer_parser.add_argument('--img', help='Image path')
    infer_parser.add_argument('--dir', help='Directory of images')
    infer_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    infer_parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    infer_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    infer_parser.add_argument('--device', default='0', help='Device')
    infer_parser.add_argument('--output', default='results', help='Output directory')
    
    # Validation
    val_parser = subparsers.add_parser('validate', help='Validate model')
    val_parser.add_argument('--model', required=True, help='Model path (.pt file)')
    val_parser.add_argument('--data', help='Data YAML path')
    
    return parser

def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("RT-DETR PCB DEFECT DETECTION")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {args.command.upper()}")
    
    detector = RTDETRDetector()
    
    try:
        if args.command == 'train':
            success = detector.train_rtdetr(args)
            if success:
                print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
                print("\nNext steps:")
                print("1. Validate: python main.py validate --model results/rtdetr_l_pcb/weights/best.pt")
                print("2. Test: python main.py infer --model results/rtdetr_l_pcb/weights/best.pt --img test.jpg")
            else:
                print("\n❌ Training failed")
                sys.exit(1)
        
        elif args.command == 'infer':
            if not args.img and not args.dir:
                print("❌ Error: Must specify --img or --dir")
                sys.exit(1)
            
            if not detector.load_model(args.model):
                sys.exit(1)
            
            if args.img:
                result = detector.run_inference(args.img, args)
                if result:
                    detector.print_detections(result)
                    detector.save_results(result, args.img, args.output)
            
            elif args.dir:
                image_dir = Path(args.dir)
                if not image_dir.exists():
                    print(f"❌ Directory not found: {args.dir}")
                    sys.exit(1)
                
                images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.png'))
                print(f"Found {len(images)} images")
                
                for img_path in images:
                    result = detector.run_inference(str(img_path), args)
                    if result:
                        detector.save_results(result, str(img_path), args.output)
        
        elif args.command == 'validate':
            success = detector.validate_model(args.model, args.data)
            if not success:
                sys.exit(1)
        
        print(f"\n✅ Operation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()