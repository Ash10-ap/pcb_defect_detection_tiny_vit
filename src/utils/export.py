import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import onnx
import onnxruntime as ort
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    # Quantization settings
    enable_quantization: bool = True
    quantization_backend: str = 'qnnpack'  # 'qnnpack', 'fbgemm'
    quantization_dtype: torch.dtype = torch.qint8
    
    # ONNX export settings
    enable_onnx_export: bool = True
    onnx_opset_version: int = 11
    dynamic_axes: Optional[Dict] = None
    optimize_for_mobile: bool = False
    
    # TensorRT settings (if available)
    enable_tensorrt: bool = False
    tensorrt_precision: str = 'fp16'  # 'fp32', 'fp16', 'int8'
    
    # General optimization
    enable_jit_trace: bool = True
    enable_fusion: bool = True
    
    def __post_init__(self):
        if self.dynamic_axes is None:
            self.dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }

class ModelExporter:
    """
    Model export utility for TinyViT-YOLO deployment.
    
    Supports multiple export formats:
    - PyTorch JIT Script/Trace
    - ONNX (with optimization)
    - Quantized models
    - TensorRT (if available)
    - Mobile deployment formats
    """
    
    def __init__(self, model: nn.Module, config: OptimizationConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare model for export
        self._prepare_model()
    
    def _prepare_model(self):
        """Prepare model for export by removing training-specific components."""
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Fuse batch norm and conv layers if enabled
        if self.config.enable_fusion:
            self._fuse_model()
    
    def _fuse_model(self):
        """Fuse batch normalization with convolution layers."""
        try:
            # Basic fusion for common patterns
            torch.quantization.fuse_modules(
                self.model,
                [['conv', 'bn'], ['conv', 'bn', 'relu']],
                inplace=True
            )
            print("Model fusion completed")
        except Exception as e:
            print(f"Model fusion failed: {e}")
    
    def export_pytorch_jit(self, 
                          example_input: torch.Tensor,
                          save_path: str,
                          trace_mode: bool = True) -> str:
        """
        Export model to PyTorch JIT format.
        
        Args:
            example_input: Example input tensor for tracing
            save_path: Path to save the JIT model
            trace_mode: Use tracing instead of scripting
            
        Returns:
            Path to saved model
        """
        print("Exporting to PyTorch JIT...")
        
        with torch.no_grad():
            if trace_mode:
                # Trace the model
                traced_model = torch.jit.trace(self.model, example_input)
            else:
                # Script the model
                traced_model = torch.jit.script(self.model)
            
            # Optimize for inference
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Save model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            traced_model.save(save_path)
            
        print(f"PyTorch JIT model saved to: {save_path}")
        return save_path
    
    def export_onnx(self,
                   example_input: torch.Tensor,
                   save_path: str,
                   input_names: List[str] = ['input'],
                   output_names: List[str] = ['output']) -> str:
        """
        Export model to ONNX format.
        
        Args:
            example_input: Example input tensor
            save_path: Path to save ONNX model
            input_names: Names for input tensors
            output_names: Names for output tensors
            
        Returns:
            Path to saved ONNX model
        """
        print("Exporting to ONNX...")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                example_input,
                save_path,
                export_params=True,
                opset_version=self.config.onnx_opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=self.config.dynamic_axes,
                verbose=False
            )
        
        # Verify ONNX model
        try:
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed")
        except Exception as e:
            print(f"ONNX model verification failed: {e}")
        
        # Optimize ONNX model
        optimized_path = save_path.replace('.onnx', '_optimized.onnx')
        self._optimize_onnx_model(save_path, optimized_path)
        
        print(f"ONNX model saved to: {save_path}")
        print(f"Optimized ONNX model saved to: {optimized_path}")
        
        return optimized_path
    
    def _optimize_onnx_model(self, input_path: str, output_path: str):
        """Optimize ONNX model for inference."""
        try:
            from onnxruntime.tools import optimizer
            
            # Create optimization configuration
            opt_model = optimizer.optimize_model(
                input_path,
                model_type='bert',  # Generic optimization
                num_heads=0,
                hidden_size=0,
                optimization_options=None
            )
            
            opt_model.save_model_to_file(output_path)
            print("ONNX model optimization completed")
            
        except ImportError:
            print("ONNX optimizer not available, skipping optimization")
        except Exception as e:
            print(f"ONNX optimization failed: {e}")
    
    def export_quantized(self,
                        example_input: torch.Tensor,
                        save_path: str,
                        calibration_data: Optional[List[torch.Tensor]] = None) -> str:
        """
        Export quantized model for faster inference.
        
        Args:
            example_input: Example input tensor
            save_path: Path to save quantized model
            calibration_data: Data for post-training quantization
            
        Returns:
            Path to saved quantized model
        """
        print("Exporting quantized model...")
        
        # Set quantization backend
        torch.backends.quantized.engine = self.config.quantization_backend
        
        if calibration_data:
            # Post-training quantization
            quantized_model = self._post_training_quantization(calibration_data)
        else:
            # Dynamic quantization
            quantized_model = quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=self.config.quantization_dtype
            )
        
        # Save quantized model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(quantized_model.state_dict(), save_path)
        
        print(f"Quantized model saved to: {save_path}")
        return save_path
    
    def _post_training_quantization(self, 
                                   calibration_data: List[torch.Tensor]) -> nn.Module:
        """Perform post-training quantization."""
        # Prepare model for quantization
        self.model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibration
        print("Running calibration...")
        with torch.no_grad():
            for data in calibration_data[:100]:  # Use subset for calibration
                self.model(data.to(self.device))
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        return quantized_model
    
    def export_mobile(self,
                     example_input: torch.Tensor,
                     save_path: str) -> str:
        """
        Export model optimized for mobile deployment.
        
        Args:
            example_input: Example input tensor
            save_path: Path to save mobile model
            
        Returns:
            Path to saved mobile model
        """
        print("Exporting mobile-optimized model...")
        
        # First create JIT traced model
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, example_input)
            
            # Optimize for mobile
            from torch.utils.mobile_optimizer import optimize_for_mobile
            mobile_model = optimize_for_mobile(traced_model)
            
            # Save mobile model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            mobile_model._save_for_lite_interpreter(save_path)
        
        print(f"Mobile model saved to: {save_path}")
        return save_path
    
    def export_tensorrt(self,
                       onnx_path: str,
                       save_path: str,
                       input_shape: Tuple[int, ...]) -> str:
        """
        Export model to TensorRT format (requires TensorRT installation).
        
        Args:
            onnx_path: Path to ONNX model
            save_path: Path to save TensorRT engine
            input_shape: Input tensor shape (B, C, H, W)
            
        Returns:
            Path to saved TensorRT engine
        """
        try:
            import tensorrt as trt
            
            print("Exporting to TensorRT...")
            
            # Create TensorRT builder
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            if self.config.tensorrt_precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            elif self.config.tensorrt_precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            # Save engine
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"TensorRT engine saved to: {save_path}")
            return save_path
            
        except ImportError:
            print("TensorRT not available, skipping TensorRT export")
            return ""
        except Exception as e:
            print(f"TensorRT export failed: {e}")
            return ""
    
    def benchmark_models(self,
                        model_paths: Dict[str, str],
                        example_input: torch.Tensor,
                        num_runs: int = 100) -> Dict[str, Dict]:
        """
        Benchmark different exported models.
        
        Args:
            model_paths: Dictionary mapping format names to model paths
            example_input: Example input for benchmarking
            num_runs: Number of inference runs for timing
            
        Returns:
            Benchmark results
        """
        results = {}
        
        print("Benchmarking exported models...")
        
        for format_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
                
            print(f"Benchmarking {format_name}...")
            
            try:
                if format_name == 'pytorch_jit':
                    results[format_name] = self._benchmark_pytorch_jit(model_path, example_input, num_runs)
                elif format_name == 'onnx':
                    results[format_name] = self._benchmark_onnx(model_path, example_input, num_runs)
                elif format_name == 'quantized':
                    results[format_name] = self._benchmark_quantized(model_path, example_input, num_runs)
                
            except Exception as e:
                print(f"Benchmarking {format_name} failed: {e}")
                results[format_name] = {'error': str(e)}
        
        return results
    
    def _benchmark_pytorch_jit(self, 
                              model_path: str, 
                              example_input: torch.Tensor, 
                              num_runs: int) -> Dict:
        """Benchmark PyTorch JIT model."""
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_input)
        
        # Timing
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(example_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        
        return {
            'avg_inference_time_ms': avg_time,
            'fps': 1000 / avg_time,
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
    
    def _benchmark_onnx(self, 
                       model_path: str, 
                       example_input: torch.Tensor, 
                       num_runs: int) -> Dict:
        """Benchmark ONNX model."""
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        input_np = example_input.cpu().numpy()
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: input_np})
        
        # Timing
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, {input_name: input_np})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        
        return {
            'avg_inference_time_ms': avg_time,
            'fps': 1000 / avg_time,
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
    
    def _benchmark_quantized(self, 
                            model_path: str, 
                            example_input: torch.Tensor, 
                            num_runs: int) -> Dict:
        """Benchmark quantized model."""
        # Load quantized model
        model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        # This is a simplified benchmark for quantized models
        # In practice, you'd load the actual quantized state dict
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_input)
        
        # Timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(example_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        
        return {
            'avg_inference_time_ms': avg_time,
            'fps': 1000 / avg_time,
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
        }


def export_all_formats(model: nn.Module,
                      example_input: torch.Tensor,
                      output_dir: str,
                      model_name: str = 'tinivit_yolo') -> Dict[str, str]:
    """
    Export model to all supported formats.
    
    Args:
        model: Model to export
        example_input: Example input tensor
        output_dir: Output directory for exported models
        model_name: Base name for exported models
        
    Returns:
        Dictionary mapping format names to file paths
    """
    config = OptimizationConfig()
    exporter = ModelExporter(model, config)
    
    exported_models = {}
    
    # PyTorch JIT
    jit_path = os.path.join(output_dir, f'{model_name}_jit.pt')
    exported_models['pytorch_jit'] = exporter.export_pytorch_jit(example_input, jit_path)
    
    # ONNX
    onnx_path = os.path.join(output_dir, f'{model_name}.onnx')
    exported_models['onnx'] = exporter.export_onnx(example_input, onnx_path)
    
    # Quantized
    quant_path = os.path.join(output_dir, f'{model_name}_quantized.pt')
    exported_models['quantized'] = exporter.export_quantized(example_input, quant_path)
    
    # Mobile
    mobile_path = os.path.join(output_dir, f'{model_name}_mobile.ptl')
    exported_models['mobile'] = exporter.export_mobile(example_input, mobile_path)
    
    # TensorRT (if available)
    if config.enable_tensorrt:
        trt_path = os.path.join(output_dir, f'{model_name}.trt')
        exported_models['tensorrt'] = exporter.export_tensorrt(
            exported_models['onnx'], trt_path, example_input.shape
        )
    
    return exported_models


def test_model_export():
    """Test the model export functionality."""
    print("Testing Model Export")
    print("=" * 20)
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, 5)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    example_input = torch.randn(1, 3, 224, 224)
    
    config = OptimizationConfig()
    exporter = ModelExporter(model, config)
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test PyTorch JIT export
        jit_path = os.path.join(temp_dir, 'test_jit.pt')
        try:
            result_path = exporter.export_pytorch_jit(example_input, jit_path)
            print(f"✓ PyTorch JIT export successful: {os.path.exists(result_path)}")
        except Exception as e:
            print(f"✗ PyTorch JIT export failed: {e}")
        
        # Test ONNX export
        onnx_path = os.path.join(temp_dir, 'test.onnx')
        try:
            result_path = exporter.export_onnx(example_input, onnx_path)
            print(f"✓ ONNX export successful: {os.path.exists(result_path)}")
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
        
        # Test quantization
        quant_path = os.path.join(temp_dir, 'test_quantized.pt')
        try:
            result_path = exporter.export_quantized(example_input, quant_path)
            print(f"✓ Quantization successful: {os.path.exists(result_path)}")
        except Exception as e:
            print(f"✗ Quantization failed: {e}")
    
    print("Model export tests completed!")


if __name__ == "__main__":
    test_model_export()