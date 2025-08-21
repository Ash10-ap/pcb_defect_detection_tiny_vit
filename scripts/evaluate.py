#!/usr/bin/env python3
"""
Evaluation script for TinyViT-YOLOv8 PCB defect detection models.

This script provides comprehensive evaluation including:
- mAP calculation at different IoU thresholds
- Per-class performance metrics
- Inference speed benchmarking
- Visualization of predictions
- Error analysis

Usage:
    python scripts/evaluate.py --model experiments/best_model.pt --data data/HRIPCB/test
    python scripts/evaluate.py --model experiments/exported_models/tinivit_yolo.onnx --data data/HRIPCB/test --format onnx
    python scripts/evaluate.py --model experiments/best_model.pt --data data/HRIPCB/test --visualize --save_dir results/
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import with proper path handling
try:
    from src.models.model_factory import create_model_from_config
    from src.datasets.pcb_dataset import PCBDataset, get_validation_transforms
    from src.utils.metrics import PCBMetrics, calculate_map
    from src.utils.export import ModelExporter
    from src.utils.inference import TinyViTYOLOInference
except ImportError:
    # Fallback for direct execution
    from models.model_factory import create_model_from_config
    from datasets.pcb_dataset import PCBDataset, get_validation_transforms
    from utils.metrics import PCBMetrics, calculate_map
    from utils.export import ModelExporter
    from utils.inference import TinyViTYOLOInference

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate TinyViT-YOLO Model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--format', type=str, default='pytorch',
                       choices=['pytorch', 'onnx', 'jit', 'quantized'],
                       help='Model format')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate prediction visualizations')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference speed benchmark')
    parser.add_argument('--num_benchmark_samples', type=int, default=100,
                       help='Number of samples for benchmarking')
    parser.add_argument('--dataset_type', type=str, default='hripcb',
                       choices=['pku', 'hripcb', 'custom'],
                       help='Type of dataset')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_arg)
        print(f"Using device: {device}")
    
    return device

def load_model(model_path: str, config: dict, device: torch.device, model_format: str):
    """Load model based on format."""
    print(f"Loading {model_format} model from: {model_path}")
    
    if model_format == 'pytorch':
        # Load PyTorch model
        if config:
            model = create_model_from_config(config['model']['name'], config['model'])
        else:
            # Try to infer model from checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_config' in checkpoint:
                model = create_model_from_config(
                    checkpoint['model_config']['name'], 
                    checkpoint['model_config']
                )
            else:
                raise ValueError("Model config not found in checkpoint and no config file provided")
        
        # Load weights
        if 'model_state_dict' in torch.load(model_path, map_location=device):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        model.eval()
        return model
        
    elif model_format == 'jit':
        # Load JIT model
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model
        
    elif model_format == 'onnx':
        # Load ONNX model
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        return session
        
    elif model_format == 'quantized':
        # Load quantized model
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model
    
    else:
        raise ValueError(f"Unsupported model format: {model_format}")

def create_dataset(data_path: str, dataset_type: str, img_size: int) -> PCBDataset:
    """Create evaluation dataset."""
    transforms = get_validation_transforms(img_size)
    
    # Determine if it's a directory or annotation file
    if os.path.isdir(data_path):
        # Directory structure
        dataset_root = data_path
        split = 'test' if os.path.exists(os.path.join(data_path, 'test')) else 'val'
    else:
        # Single annotation file
        dataset_root = os.path.dirname(data_path)
        split = os.path.splitext(os.path.basename(data_path))[0]
    
    dataset = PCBDataset(
        dataset_root=dataset_root,
        split=split,
        dataset_type=dataset_type,
        transforms=transforms,
        img_size=img_size
    )
    
    return dataset

def run_inference(model, dataloader, device: torch.device, model_format: str, args):
    """Run inference on the dataset."""
    all_predictions = []
    all_targets = []
    inference_times = []
    
    print("Running inference...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):
            if model_format == 'onnx':
                # ONNX inference
                input_name = model.get_inputs()[0].name
                images_np = images.cpu().numpy()
                
                start_time = time.time()
                outputs = model.run(None, {input_name: images_np})
                end_time = time.time()
                
                # Convert outputs back to torch tensors
                predictions = [torch.from_numpy(out) for out in outputs]
                
            else:
                # PyTorch inference
                images = images.to(device)
                
                start_time = time.time()
                predictions = model(images)
                end_time = time.time()
            
            inference_times.append((end_time - start_time) / len(images))
            
            # Post-process predictions
            batch_predictions = post_process_predictions(
                predictions, args.conf_threshold, args.iou_threshold
            )
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(targets)
    
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    fps = 1000 / avg_inference_time
    
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    return all_predictions, all_targets, {'avg_time_ms': avg_inference_time, 'fps': fps}

def post_process_predictions(predictions, conf_threshold: float, iou_threshold: float):
    """Post-process model predictions."""
    batch_predictions = []
    
    # This is a simplified post-processing
    # In practice, you'd need proper NMS and coordinate conversion
    
    if isinstance(predictions, (list, tuple)):
        # Multiple outputs (typical for YOLO)
        for i in range(len(predictions[0])):  # Batch size
            pred_dict = {
                'boxes': torch.tensor([]),
                'scores': torch.tensor([]),
                'labels': torch.tensor([]),
                'image_id': i
            }
            
            # Extract boxes, scores, labels from predictions
            # This is model-specific and would need to be implemented
            # based on your actual model output format
            
            batch_predictions.append(pred_dict)
    else:
        # Single output
        for i in range(predictions.shape[0]):  # Batch size
            pred_dict = {
                'boxes': torch.tensor([]),
                'scores': torch.tensor([]),
                'labels': torch.tensor([]),
                'image_id': i
            }
            batch_predictions.append(pred_dict)
    
    return batch_predictions

def calculate_metrics(predictions: List[Dict], targets: List[Dict], num_classes: int):
    """Calculate comprehensive evaluation metrics."""
    print("Calculating metrics...")
    
    metrics = PCBMetrics(num_classes)
    results = metrics.compute_metrics(predictions, targets)
    
    return results

def run_benchmark(model, device: torch.device, model_format: str, args):
    """Run inference speed benchmark."""
    print("Running benchmark...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    
    if model_format == 'onnx':
        dummy_input_np = dummy_input.cpu().numpy()
        input_name = model.get_inputs()[0].name
        
        # Warmup
        for _ in range(10):
            _ = model.run(None, {input_name: dummy_input_np})
        
        # Benchmark
        times = []
        for _ in range(args.num_benchmark_samples):
            start_time = time.time()
            _ = model.run(None, {input_name: dummy_input_np})
            end_time = time.time()
            times.append(end_time - start_time)
            
    else:
        dummy_input = dummy_input.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(args.num_benchmark_samples):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    fps = 1000 / avg_time
    
    benchmark_results = {
        'avg_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time,
        'fps': fps,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'num_samples': args.num_benchmark_samples
    }
    
    print(f"Benchmark Results:")
    print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Min time: {benchmark_results['min_time_ms']:.2f} ms")
    print(f"  Max time: {benchmark_results['max_time_ms']:.2f} ms")
    
    return benchmark_results

def visualize_results(predictions: List[Dict], 
                     targets: List[Dict], 
                     dataset: PCBDataset,
                     save_dir: str,
                     num_samples: int = 20):
    """Generate visualization of predictions."""
    print("Generating visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Select samples to visualize
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        try:
            image, target = dataset[idx]
            
            # Get corresponding prediction
            if idx < len(predictions):
                prediction = predictions[idx]
            else:
                continue
            
            # Convert image back to numpy for visualization
            if isinstance(image, torch.Tensor):
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                image_np = image.permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * std + mean) * 255
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            else:
                image_np = image
            
            # Visualize predictions vs ground truth
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Ground truth
            ax1.imshow(image_np)
            ax1.set_title('Ground Truth')
            ax1.axis('off')
            
            # Draw ground truth boxes
            if len(target['boxes']) > 0:
                for box, label in zip(target['boxes'], target['labels']):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, color='green', linewidth=2)
                    ax1.add_patch(rect)
                    ax1.text(x1, y1-5, f'GT: {label.item()}', 
                           color='green', fontsize=8, weight='bold')
            
            # Predictions
            ax2.imshow(image_np)
            ax2.set_title('Predictions')
            ax2.axis('off')
            
            # Draw prediction boxes
            if len(prediction['boxes']) > 0:
                for box, score, label in zip(prediction['boxes'], 
                                           prediction['scores'], 
                                           prediction['labels']):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, color='red', linewidth=2)
                    ax2.add_patch(rect)
                    ax2.text(x1, y1-5, f'Pred: {label.item()} ({score.item():.2f})', 
                           color='red', fontsize=8, weight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_{i:03d}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Failed to visualize sample {idx}: {e}")
            continue

def save_results(results: Dict, save_path: str):
    """Save evaluation results to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        else:
            return obj
    
    results_serializable = convert_types(results)
    
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to: {save_path}")

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Load configuration if provided
    config = None
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load model
    model = load_model(args.model, config, device, args.format)
    
    # Create dataset
    dataset = create_dataset(args.data, args.dataset_type, args.img_size)
    print(f"Loaded {len(dataset)} test samples")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run inference
    predictions, targets, timing_info = run_inference(
        model, dataloader, device, args.format, args
    )
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets, dataset.num_classes)
    
    # Run benchmark if requested
    benchmark_results = {}
    if args.benchmark:
        benchmark_results = run_benchmark(model, device, args.format, args)
    
    # Generate visualizations if requested
    if args.visualize:
        viz_dir = os.path.join(args.save_dir, 'visualizations')
        visualize_results(predictions, targets, dataset, viz_dir)
    
    # Compile all results
    all_results = {
        'model_path': args.model,
        'model_format': args.format,
        'dataset_info': {
            'path': args.data,
            'type': args.dataset_type,
            'num_samples': len(dataset),
            'num_classes': dataset.num_classes
        },
        'evaluation_params': {
            'conf_threshold': args.conf_threshold,
            'iou_threshold': args.iou_threshold,
            'img_size': args.img_size
        },
        'metrics': metrics,
        'timing': timing_info,
        'benchmark': benchmark_results
    }
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data} ({len(dataset)} samples)")
    print(f"mAP: {metrics.get('map', 0):.4f}")
    print(f"mAP@50: {metrics.get('map50', 0):.4f}")
    print(f"mAP@75: {metrics.get('map75', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall: {metrics.get('recall', 0):.4f}")
    print(f"F1: {metrics.get('f1', 0):.4f}")
    print(f"Average inference time: {timing_info['avg_time_ms']:.2f} ms")
    print(f"FPS: {timing_info['fps']:.2f}")
    
    # Save results
    results_path = os.path.join(args.save_dir, 'evaluation_results.json')
    save_results(all_results, results_path)
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"Results saved to: {args.save_dir}")

if __name__ == '__main__':
    main()