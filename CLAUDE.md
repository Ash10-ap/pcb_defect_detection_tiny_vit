# TinyViT-YOLOv8 PCB Defect Detection - Project Memory

## Project Overview

This is a comprehensive implementation of an integrated TinyViT-YOLOv8 system for PCB defect detection, featuring:

- **TinyViT Backbone**: Lightweight Vision Transformer with pretrained weights
- **CBAM Attention Fusion**: Channel and spatial attention for enhanced small-defect detection
- **Three-Stage Training**: Foundation → Transfer Learning → Few-Shot Adaptation
- **Multi-Format Export**: PyTorch, ONNX, TensorRT, Mobile deployment
- **Comprehensive Evaluation**: mAP benchmarking, visualization, and analysis

## Architecture

```
Input → TinyViT Backbone → Channel Adapters + CBAM → YOLOv8 Neck → Detection Head
         (P3/P4/P5)        (1x1 Conv + Attention)     (PAN + FPN)    (Bbox + Cls)
```

## Project Structure

```
src/
├── models/           # TinyViT-YOLOv8 architecture
│   ├── __init__.py
│   ├── tinivit_backbone.py      # TinyViT integration
│   ├── yolo_integration.py      # Main model class
│   └── model_factory.py         # Model creation utilities
├── attention/        # Attention modules
│   ├── __init__.py
│   ├── cbam.py                  # CBAM implementation
│   ├── eca.py                   # ECA attention
│   └── attention_fusion.py     # Feature fusion
├── training/         # Multi-stage training
│   ├── __init__.py
│   ├── multi_stage_trainer.py  # Main trainer
│   ├── loss_functions.py       # SIoU, Focal losses
│   └── training_scheduler.py   # Learning rate scheduling
├── datasets/         # Data loading and augmentation
│   ├── __init__.py
│   ├── pcb_dataset.py          # Dataset classes
│   └── data_augmentation.py    # PCB-specific augmentations
└── utils/           # Utilities and helpers
    ├── __init__.py
    ├── metrics.py              # mAP calculation
    ├── export.py               # Model export utilities
    ├── inference.py            # Inference engine
    ├── checkpoint.py           # Checkpoint management
    └── visualization.py        # Result visualization
```

## Key Components

### 1. TinyViT Backbone (`src/models/tinivit_backbone.py`)
- Loads TinyViT from timm with pretrained weights
- Extracts multi-scale features (P3, P4, P5)
- Supports progressive unfreezing for transfer learning
- Output channels: [192, 384, 768] for TinyViT-21M

### 2. Attention Fusion (`src/attention/`)
- **CBAM**: Sequential channel + spatial attention
- **ECA**: Efficient channel attention alternative
- **Cross-Scale Attention**: Feature interaction across scales
- **Adaptive Fusion**: Dynamic weighting of attention mechanisms

### 3. Multi-Stage Training (`src/training/multi_stage_trainer.py`)
- **Stage 1**: Foundation training on PKU-Market-PCB
- **Stage 2**: Transfer learning on HRIPCB (3 sub-stages)
- **Stage 3**: Few-shot adaptation for novel defects
- Progressive backbone unfreezing strategy

### 4. Loss Functions (`src/training/loss_functions.py`)
- **SIoU Loss**: Advanced IoU for small objects
- **Focal Loss**: Handles class imbalance
- **Combined Loss**: Weighted combination of bbox, classification, and objectness losses

### 5. Export & Deployment (`src/utils/export.py`)
- **ONNX**: Cross-platform deployment
- **TensorRT**: NVIDIA GPU optimization
- **Quantization**: INT8 for edge devices
- **Mobile**: PyTorch Mobile format

## Training Pipeline

### Configuration File: `configs/tinivit_yolov8_pcb.yaml`
```yaml
model:
  name: "tinivit_yolo_medium"
  backbone_config:
    model_name: "tiny_vit_21m_224"
    pretrained: true
  attention_config:
    attention_type: "cbam"
    use_cross_scale: true
  yolo_config:
    nc: 6  # PCB defect classes
```

### Training Stages:
1. **Foundation (80 epochs)**: LR=1e-3, freeze backbone
2. **Transfer Early (20 epochs)**: LR=5e-4, freeze all backbone  
3. **Transfer Partial (20 epochs)**: LR=2e-4, unfreeze last 5 blocks
4. **Transfer Full (20 epochs)**: LR=1e-4, unfreeze last 10 blocks
5. **Few-Shot (30 epochs)**: LR=1e-5, freeze backbone, train attention only

## Usage Commands

### Training
```bash
# Full multi-stage training
python scripts/train_multi_stage.py --config configs/tinivit_yolov8_pcb.yaml

# Single stage training
python scripts/train_multi_stage.py --config configs/tinivit_yolov8_pcb.yaml --stage foundation

# Resume from checkpoint
python scripts/train_multi_stage.py --config configs/tinivit_yolov8_pcb.yaml --resume experiments/checkpoints/foundation_best.pt
```

### Evaluation
```bash
# Evaluate PyTorch model
python scripts/evaluate.py --model experiments/best_model.pt --data data/HRIPCB/test

# Evaluate ONNX model with visualization
python scripts/evaluate.py --model experiments/exported_models/tinivit_yolo.onnx --data data/HRIPCB/test --format onnx --visualize

# Benchmark different formats
python scripts/evaluate.py --model experiments/best_model.pt --data data/HRIPCB/test --benchmark
```

### Model Export
```bash
# Export to all formats
python scripts/export_onnx.py --model experiments/best_model.pt --output models/ --all --optimize --quantize

# Export specific formats
python scripts/export_onnx.py --model experiments/best_model.pt --output models/tinivit_yolo.onnx --onnx --tensorrt --precision fp16

# Validate exported models
python scripts/export_onnx.py --model experiments/best_model.pt --output models/ --onnx --validate --benchmark
```

## Dependencies

### Core Requirements
- torch>=2.0.0
- ultralytics>=8.0.200
- timm>=0.9.7
- albumentations>=1.3.1
- opencv-python>=4.8.0

### Export Requirements  
- onnx>=1.14.0
- onnxruntime>=1.15.0
- tensorrt (optional, for TensorRT export)

### Evaluation Requirements
- matplotlib>=3.7.0
- seaborn>=0.12.0
- scikit-learn>=1.3.0
- pycocotools>=2.0.6

## Performance Targets

- **Accuracy**: >95% mAP@0.5 on HRIPCB dataset
- **Speed**: >30 FPS on RTX 3090, >15 FPS on Jetson Xavier NX  
- **Model Size**: <50MB for edge deployment
- **Latency**: <30ms inference time

## PCB Defect Classes

1. **missing_hole**: Missing via holes
2. **mouse_bite**: Incomplete edge cuts
3. **open_circuit**: Broken traces
4. **short**: Unwanted connections
5. **spur**: Extra copper remnants
6. **spurious_copper**: Unwanted copper patches

## Research Foundation

Based on cutting-edge research:
- **TinyViT** (ECCV 2022): Efficient ViT architecture
- **YOLOv8**: State-of-the-art object detection
- **CBAM** (ECCV 2018): Convolutional attention mechanism
- **SIoU**: Improved IoU loss for small objects
- **Multi-stage training**: Progressive feature learning

## Development Notes

### Key Design Decisions:
1. **TinyViT over other ViTs**: Better speed/accuracy tradeoff for edge deployment
2. **CBAM over simpler attention**: Proven effectiveness for small object detection
3. **Three-stage training**: Addresses domain gap and few-shot scenarios
4. **SIoU loss**: Superior performance on small, densely packed objects
5. **Progressive unfreezing**: Prevents catastrophic forgetting during transfer

### Implementation Highlights:
- Full integration with Ultralytics YOLO ecosystem
- Comprehensive export pipeline for deployment
- Extensive evaluation and benchmarking tools
- PCB-specific data augmentations
- Robust checkpoint management

### Testing Strategy:
- Unit tests for all major components
- Integration tests for training pipeline
- Performance benchmarks across formats
- Validation against research baselines

## Troubleshooting

### Common Issues:
1. **CUDA OOM**: Reduce batch size, enable gradient checkpointing
2. **TinyViT loading**: Ensure timm version compatibility
3. **ONNX export**: Check dynamic axes configuration
4. **TensorRT**: Verify CUDA/TensorRT versions match

### Performance Optimization:
1. **Training**: Use mixed precision, gradient accumulation
2. **Inference**: TensorRT > ONNX > PyTorch for speed
3. **Memory**: Quantization, smaller input sizes
4. **Deployment**: Model pruning, knowledge distillation

This implementation provides a complete, production-ready system for PCB defect detection with state-of-the-art accuracy and deployment flexibility.