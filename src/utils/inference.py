import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import onnxruntime as ort
from pathlib import Path

class TinyViTYOLOInference:
    """
    Inference engine for TinyViT-YOLO models.
    
    Supports multiple model formats:
    - PyTorch (.pt, .pth)
    - ONNX (.onnx)
    - TensorRT (.trt)
    - Mobile (.ptl)
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 img_size: int = 640):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file
            device: Device to use ('cuda', 'cpu', 'auto')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            img_size: Input image size
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Determine model format
        self.model_format = self._detect_format()
        
        # Load model
        self.model = self._load_model()
        
        # Class names (PCB defects)
        self.class_names = {
            0: 'missing_hole',
            1: 'mouse_bite', 
            2: 'open_circuit',
            3: 'short',
            4: 'spur',
            5: 'spurious_copper'
        }
        
        print(f"Loaded {self.model_format} model on {self.device}")
    
    def _detect_format(self) -> str:
        """Detect model format from file extension."""
        suffix = self.model_path.suffix.lower()
        
        if suffix in ['.pt', '.pth']:
            return 'pytorch'
        elif suffix == '.onnx':
            return 'onnx'
        elif suffix == '.trt':
            return 'tensorrt'
        elif suffix == '.ptl':
            return 'mobile'
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    def _load_model(self):
        """Load model based on format."""
        if self.model_format == 'pytorch':
            model = torch.jit.load(self.model_path, map_location=self.device)
            model.eval()
            return model
            
        elif self.model_format == 'onnx':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(self.model_path), providers=providers)
            return session
            
        elif self.model_format == 'tensorrt':
            try:
                import tensorrt as trt
                import pycuda.driver as cuda
                import pycuda.autoinit
                
                # Load TensorRT engine
                with open(self.model_path, 'rb') as f:
                    engine_data = f.read()
                
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                engine = runtime.deserialize_cuda_engine(engine_data)
                context = engine.create_execution_context()
                
                return {'engine': engine, 'context': context}
                
            except ImportError:
                raise RuntimeError("TensorRT not available")
                
        elif self.model_format == 'mobile':
            model = torch.jit.load(self.model_path, map_location=self.device)
            model.eval()
            return model
        
        else:
            raise ValueError(f"Unsupported format: {self.model_format}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, (self.img_size, self.img_size))
        
        # Normalize
        image_norm = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = (image_norm - mean) / std
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Dictionary with detection results
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        start_time = time.time()
        
        if self.model_format == 'pytorch' or self.model_format == 'mobile':
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
        elif self.model_format == 'onnx':
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: input_tensor.cpu().numpy()})
            outputs = [torch.from_numpy(out) for out in outputs]
            
        elif self.model_format == 'tensorrt':
            outputs = self._tensorrt_inference(input_tensor)
            
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Post-process
        detections = self.postprocess(outputs, image.shape)
        
        return {
            'detections': detections,
            'inference_time_ms': inference_time,
            'num_detections': len(detections['boxes'])
        }
    
    def _tensorrt_inference(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Run TensorRT inference."""
        try:
            import pycuda.driver as cuda
            
            engine = self.model['engine']
            context = self.model['context']
            
            # Allocate GPU memory
            input_shape = input_tensor.shape
            input_size = torch.numel(input_tensor) * input_tensor.element_size()
            
            d_input = cuda.mem_alloc(input_size)
            
            # Get output shape
            output_shape = engine.get_binding_shape(1)
            output_size = np.prod(output_shape) * 4  # float32
            d_output = cuda.mem_alloc(output_size)
            
            # Copy input to GPU
            cuda.memcpy_htod(d_input, input_tensor.cpu().numpy().ascontiguousarray())
            
            # Run inference
            context.execute_v2([int(d_input), int(d_output)])
            
            # Copy output back
            output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, d_output)
            
            return [torch.from_numpy(output)]
            
        except Exception as e:
            raise RuntimeError(f"TensorRT inference failed: {e}")
    
    def postprocess(self, outputs: List[torch.Tensor], original_shape: Tuple[int, int, int]) -> Dict:
        """
        Post-process model outputs to get final detections.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (H, W, C)
            
        Returns:
            Dictionary with processed detections
        """
        # This is a simplified post-processing
        # In practice, you'd need to implement proper YOLO post-processing
        # including coordinate conversion, NMS, etc.
        
        if len(outputs) == 0:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([]),
                'class_names': []
            }
        
        # Placeholder implementation
        # You would need to implement actual YOLO output parsing here
        output = outputs[0].cpu().numpy()
        
        # Extract detections above confidence threshold
        boxes = []
        scores = []
        labels = []
        class_names = []
        
        # Scale boxes back to original image size
        h_orig, w_orig = original_shape[:2]
        scale_x = w_orig / self.img_size
        scale_y = h_orig / self.img_size
        
        # Apply scaling and filtering (placeholder)
        for detection in output:
            # This is where you'd parse the actual YOLO output format
            # For now, just return empty results
            pass
        
        return {
            'boxes': np.array(boxes),
            'scores': np.array(scores),
            'labels': np.array(labels),
            'class_names': class_names
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of images in BGR format
            
        Returns:
            List of detection results
        """
        results = []
        
        for image in images:
            result = self.predict(image)
            results.append(result)
        
        return results
    
    def predict_video(self, video_path: str, output_path: str = None) -> List[Dict]:
        """
        Run inference on video frames.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            
        Returns:
            List of detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            result = self.predict(frame)
            results.append(result)
            
            # Annotate frame if saving video
            if output_path:
                annotated_frame = self.visualize_detections(frame, result['detections'])
                out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        if output_path:
            out.release()
            print(f"Annotated video saved to: {output_path}")
        
        return results
    
    def visualize_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: Detection results
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        boxes = detections['boxes']
        scores = detections['scores']
        labels = detections['labels']
        
        if len(boxes) == 0:
            return annotated
        
        # Define colors for each class
        colors = [
            (255, 0, 0),    # missing_hole: Red
            (0, 255, 0),    # mouse_bite: Green
            (0, 0, 255),    # open_circuit: Blue
            (255, 255, 0),  # short: Yellow
            (255, 0, 255),  # spur: Magenta
            (0, 255, 255),  # spurious_copper: Cyan
        ]
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get class info
            class_name = self.class_names.get(label, f'Class_{label}')
            color = colors[label % len(colors)]
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f'{class_name}: {score:.2f}'
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def benchmark(self, num_runs: int = 100) -> Dict:
        """
        Benchmark inference performance.
        
        Args:
            num_runs: Number of inference runs
            
        Returns:
            Benchmark results
        """
        # Create dummy input
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            self.predict(dummy_image)
        
        # Benchmark
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            self.predict(dummy_image)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ms
        
        return {
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'fps': 1000 / np.mean(times),
            'num_runs': num_runs
        }


def test_inference():
    """Test inference functionality."""
    print("Testing TinyViT-YOLO Inference")
    print("=" * 35)
    
    # This would require an actual model file
    # For now, just test the class instantiation
    try:
        # Create dummy model file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            # Create minimal JIT model
            class DummyModel(torch.nn.Module):
                def forward(self, x):
                    return [torch.zeros(1, 25200, 11)]
            
            dummy_model = DummyModel()
            traced = torch.jit.trace(dummy_model, torch.randn(1, 3, 640, 640))
            traced.save(f.name)
            
            # Test inference engine
            engine = TinyViTYOLOInference(f.name, device='cpu')
            
            # Test with dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = engine.predict(dummy_image)
            
            print(f"✓ Inference test passed")
            print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
            print(f"  Number of detections: {result['num_detections']}")
            
            # Clean up
            import os
            os.unlink(f.name)
            
    except Exception as e:
        print(f"✗ Inference test failed: {e}")


if __name__ == "__main__":
    test_inference()