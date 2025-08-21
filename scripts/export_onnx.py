#!/usr/bin/env python3
"""
ONNX export script for TinyViT-YOLOv8 models with optimization.

This script exports trained models to ONNX format with various optimizations
for deployment on different platforms including edge devices.

Usage:
    python scripts/export_onnx.py --model experiments/best_model.pt --output models/tinivit_yolo.onnx
    python scripts/export_onnx.py --model experiments/best_model.pt --output models/ --optimize --quantize
    python scripts/export_onnx.py --model experiments/best_model.pt --output models/ --tensorrt --precision fp16
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import yaml
import json
import time
import numpy as np

# Import with proper path handling
try:
    from src.models.model_factory import create_model_from_config
    from src.utils.export import ModelExporter, OptimizationConfig, export_all_formats
except ImportError:
    # Fallback for direct execution
    from models.model_factory import create_model_from_config
    from utils.export import ModelExporter, OptimizationConfig, export_all_formats

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export TinyViT-YOLO to ONNX')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to PyTorch model file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path (file or directory)')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for export (cuda/cpu)')
    
    # Export formats
    parser.add_argument('--onnx', action='store_true', default=True,
                       help='Export to ONNX format')
    parser.add_argument('--jit', action='store_true',
                       help='Export to PyTorch JIT format')
    parser.add_argument('--mobile', action='store_true',
                       help='Export to mobile format')
    parser.add_argument('--tensorrt', action='store_true',
                       help='Export to TensorRT format')
    parser.add_argument('--all', action='store_true',
                       help='Export to all supported formats')
    
    # Optimization options
    parser.add_argument('--optimize', action='store_true',
                       help='Apply ONNX optimizations')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply quantization')
    parser.add_argument('--dynamic', action='store_true',
                       help='Use dynamic axes for variable input sizes')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version')
    
    # TensorRT options
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='TensorRT precision mode')
    parser.add_argument('--workspace', type=int, default=1,
                       help='TensorRT workspace size in GB')
    
    # Quantization options
    parser.add_argument('--qbackend', type=str, default='qnnpack',
                       choices=['qnnpack', 'fbgemm'],
                       help='Quantization backend')
    parser.add_argument('--qdtype', type=str, default='qint8',
                       choices=['qint8', 'quint8'],
                       help='Quantization data type')
    
    # Validation
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported models')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark exported models')
    
    return parser.parse_args()

def load_model_and_config(model_path: str, config_path: str, device: torch.device):
    """Load model and configuration."""
    print(f"Loading model from: {model_path}")
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config['model']
    else:
        # Try to load config from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            # Use default config
            model_config = {
                'name': 'tinivit_yolo_medium',
                'backbone_config': {
                    'model_name': 'tiny_vit_21m_224',
                    'pretrained': False,
                    'img_size': 640
                },
                'attention_config': {
                    'attention_type': 'cbam',
                    'use_cross_scale': True,
                    'use_residual': True
                },
                'yolo_config': {
                    'nc': 6,
                    'depth_multiple': 0.5,
                    'width_multiple': 0.5
                }
            }
    
    # Create model
    model = create_model_from_config(model_config['name'], model_config)
    
    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model type: {type(model).__name__}")
    
    return model, model_config

def create_optimization_config(args) -> OptimizationConfig:
    """Create optimization configuration from arguments."""
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
    
    config = OptimizationConfig(
        enable_quantization=args.quantize,
        quantization_backend=args.qbackend,
        quantization_dtype=getattr(torch, args.qdtype),
        
        enable_onnx_export=args.onnx,
        onnx_opset_version=args.opset,
        dynamic_axes=dynamic_axes,
        optimize_for_mobile=args.mobile,
        
        enable_tensorrt=args.tensorrt,
        tensorrt_precision=args.precision,
        
        enable_jit_trace=args.jit,
        enable_fusion=args.optimize
    )
    
    return config

def validate_exported_model(original_model, exported_path: str, 
                           example_input: torch.Tensor, model_format: str):
    """Validate that exported model produces similar outputs."""
    print(f"Validating {model_format} model...")
    
    try:
        if model_format == 'onnx':
            import onnxruntime as ort
            
            # Load ONNX model
            session = ort.InferenceSession(exported_path)
            input_name = session.get_inputs()[0].name
            
            # Run inference
            with torch.no_grad():
                original_output = original_model(example_input)
                onnx_output = session.run(None, {input_name: example_input.cpu().numpy()})
            
            # Compare outputs
            if isinstance(original_output, (list, tuple)):
                original_output = original_output[0]
            if isinstance(onnx_output, (list, tuple)):
                onnx_output = torch.from_numpy(onnx_output[0])
            else:
                onnx_output = torch.from_numpy(onnx_output)
            
            # Calculate difference
            diff = torch.abs(original_output.cpu() - onnx_output).mean()
            print(f"  Average difference: {diff:.6f}")
            
            if diff < 1e-3:
                print(f"  ✓ {model_format} validation passed")
                return True
            else:
                print(f"  ⚠ {model_format} validation warning: large difference")
                return False
                
        elif model_format == 'jit':
            # Load JIT model
            jit_model = torch.jit.load(exported_path)
            
            with torch.no_grad():
                original_output = original_model(example_input)
                jit_output = jit_model(example_input)
            
            # Compare outputs
            if isinstance(original_output, (list, tuple)):
                original_output = original_output[0]
            if isinstance(jit_output, (list, tuple)):
                jit_output = jit_output[0]
            
            diff = torch.abs(original_output - jit_output).mean()
            print(f"  Average difference: {diff:.6f}")
            
            if diff < 1e-5:
                print(f"  ✓ {model_format} validation passed")
                return True
            else:
                print(f"  ⚠ {model_format} validation warning: large difference")
                return False
                
    except Exception as e:
        print(f"  ✗ {model_format} validation failed: {e}")
        return False

def main():
    """Main export function."""
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, model_config = load_model_and_config(args.model, args.config, device)
    
    # Create example input
    example_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)
    print(f"Example input shape: {example_input.shape}")
    
    # Test model forward pass
    with torch.no_grad():
        output = model(example_input)
        if isinstance(output, (list, tuple)):
            print(f"Model output shapes: {[o.shape for o in output]}")
        else:
            print(f"Model output shape: {output.shape}")
    
    # Create optimization config
    opt_config = create_optimization_config(args)
    
    # Create exporter
    exporter = ModelExporter(model, opt_config)
    
    # Determine output directory
    if os.path.isdir(args.output):
        output_dir = args.output
        base_name = 'tinivit_yolo'
    else:
        output_dir = os.path.dirname(args.output)
        base_name = os.path.splitext(os.path.basename(args.output))[0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    exported_models = {}
    
    # Export to different formats
    try:
        if args.all or args.onnx:
            print("\n" + "="*50)
            print("EXPORTING TO ONNX")
            print("="*50)
            
            onnx_path = os.path.join(output_dir, f'{base_name}.onnx')
            exported_path = exporter.export_onnx(
                example_input, onnx_path,
                input_names=['input'],
                output_names=['output']
            )
            exported_models['onnx'] = exported_path
            
            if args.validate:
                validate_exported_model(model, exported_path, example_input, 'onnx')
        
        if args.all or args.jit:
            print("\n" + "="*50)
            print("EXPORTING TO PYTORCH JIT")
            print("="*50)
            
            jit_path = os.path.join(output_dir, f'{base_name}_jit.pt')
            exported_path = exporter.export_pytorch_jit(example_input, jit_path)
            exported_models['jit'] = exported_path
            
            if args.validate:
                validate_exported_model(model, exported_path, example_input, 'jit')
        
        if args.all or args.mobile:
            print("\n" + "="*50)
            print("EXPORTING TO MOBILE")
            print("="*50)
            
            mobile_path = os.path.join(output_dir, f'{base_name}_mobile.ptl')
            exported_path = exporter.export_mobile(example_input, mobile_path)
            exported_models['mobile'] = exported_path
        
        if args.quantize:
            print("\n" + "="*50)
            print("EXPORTING QUANTIZED MODEL")
            print("="*50)
            
            quant_path = os.path.join(output_dir, f'{base_name}_quantized.pt')
            exported_path = exporter.export_quantized(example_input, quant_path)
            exported_models['quantized'] = exported_path
        
        if args.all or args.tensorrt:
            print("\n" + "="*50)
            print("EXPORTING TO TENSORRT")
            print("="*50)
            
            if 'onnx' not in exported_models:
                # Need ONNX model first
                onnx_path = os.path.join(output_dir, f'{base_name}.onnx')
                exporter.export_onnx(example_input, onnx_path)
                exported_models['onnx'] = onnx_path
            
            trt_path = os.path.join(output_dir, f'{base_name}.trt')
            exported_path = exporter.export_tensorrt(
                exported_models['onnx'], trt_path, example_input.shape
            )
            if exported_path:
                exported_models['tensorrt'] = exported_path
        
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Benchmark if requested
    if args.benchmark and exported_models:
        print("\n" + "="*50)
        print("BENCHMARKING EXPORTED MODELS")
        print("="*50)
        
        benchmark_results = exporter.benchmark_models(
            exported_models, example_input, num_runs=100
        )
        
        print("\nBenchmark Results:")
        for model_format, results in benchmark_results.items():
            if 'error' in results:
                print(f"  {model_format}: Error - {results['error']}")
            else:
                print(f"  {model_format}:")
                print(f"    Inference time: {results['avg_inference_time_ms']:.2f} ms")
                print(f"    FPS: {results['fps']:.2f}")
                print(f"    Model size: {results['model_size_mb']:.2f} MB")
        
        # Save benchmark results
        benchmark_path = os.path.join(output_dir, 'benchmark_results.json')
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\nBenchmark results saved to: {benchmark_path}")
    
    # Save export metadata
    metadata = {
        'original_model': args.model,
        'export_config': {
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'device': str(device),
            'optimization': vars(opt_config)
        },
        'model_config': model_config,
        'exported_models': exported_models,
        'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(output_dir, 'export_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"Original model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Exported formats:")
    
    for format_name, path in exported_models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {format_name}: {path} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {format_name}: Export failed")
    
    print(f"\nMetadata saved to: {metadata_path}")
    print("\n🎉 Export completed successfully!")

if __name__ == '__main__':
    main()