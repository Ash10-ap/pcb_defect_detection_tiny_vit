# TinyViT-YOLOv8 PCB Defect Detection

A state-of-the-art implementation combining TinyViT backbone with YOLOv8 for high-precision PCB defect detection, featuring attention-based feature fusion and multi-stage training pipelines.

## 🎯 Key Features

- **TinyViT Backbone**: Efficient Vision Transformer with pretrained weights
- **CBAM Attention Fusion**: Channel and spatial attention for enhanced small-defect detection
- **Multi-Stage Training**: Foundation → Transfer Learning → Few-Shot Adaptation
- **Edge Optimization**: ONNX export, quantization, and deployment support
- **Comprehensive Evaluation**: mAP benchmarking, latency analysis, and ablation studies

## 🏗️ Architecture Overview

```
Input → TinyViT Backbone → Channel Adapters + CBAM → YOLOv8 Neck → Detection Head
         (P3/P4/P5)        (1x1 Conv + Attention)     (PAN + FPN)    (Bbox + Cls)
```

## 📁 Project Structure

```
src/
├── models/           # TinyViT-YOLOv8 architecture
├── attention/        # CBAM and attention modules
├── training/         # Multi-stage training pipelines
├── datasets/         # Data loading and augmentation
└── utils/           # Utilities and helpers
configs/             # Training and model configurations
experiments/         # Experiment tracking and results
scripts/            # Training, evaluation, and deployment scripts
external/           # External dependencies (TinyViT)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit.git
   cd pcb_defect_detection_tiny_vit
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Install other requirements
   pip install -r requirements.txt
   ```

4. **Setup external dependencies:**
   ```bash
   # Clone TinyViT repository
   git clone https://github.com/wkcn/TinyViT.git external/TinyViT
   ```

5. **Verify installation:**
   ```bash
   python test_imports_minimal.py
   ```

### Training

1. **Prepare your PCB dataset** in YOLO format:
   ```
   data/
   ├── PCB_Dataset/
   │   ├── images/
   │   │   ├── train/
   │   │   ├── val/
   │   │   └── test/
   │   └── annotations/
   │       ├── train.json
   │       ├── val.json
   │       └── test.json
   ```

2. **Configure training parameters:**
   ```bash
   # Edit the configuration file
   nano configs/tinivit_yolov8_pcb.yaml
   ```

3. **Start multi-stage training:**
   ```bash
   python scripts/train_multi_stage.py --config configs/tinivit_yolov8_pcb.yaml
   ```

4. **Monitor training:**
   ```bash
   # TensorBoard
   tensorboard --logdir experiments/logs
   
   # Check saved models
   ls experiments/checkpoints/
   ```

### Evaluation

```bash
python scripts/evaluate.py \
    --model experiments/checkpoints/best_model.pt \
    --data data/PCB_Dataset/test \
    --visualize
```

### Model Export

```bash
# Export to ONNX
python scripts/export_onnx.py \
    --model experiments/checkpoints/best_model.pt \
    --optimize
```

## ⚙️ Configuration

Key configuration options in `configs/tinivit_yolov8_pcb.yaml`:

```yaml
model:
  backbone_config:
    model_name: "tiny_vit_11m_224"  # or "tiny_vit_21m_224"
    
training:
  batch_size: 16  # Adjust based on GPU memory
  
hardware:
  device: "cuda"  # or "cpu"
  use_amp: true   # Mixed precision training
```

## 🔧 PCB Defect Classes

The system detects these PCB defects:

1. `missing_hole` - Missing drill holes
2. `mouse_bite` - Incomplete edge cuts
3. `open_circuit` - Broken connections
4. `short` - Unwanted connections
5. `spur` - Extra copper traces
6. `spurious_copper` - Copper contamination

## 📊 Performance Targets

- **Accuracy**: >95% mAP@0.5 on HRIPCB dataset
- **Speed**: >30 FPS on RTX 3090, >15 FPS on Jetson Xavier NX
- **Model Size**: <50MB for edge deployment
- **Latency**: <30ms inference time

## 🔬 Research Foundation

This implementation is based on:
- **TinyViT**: Efficient Vision Transformer architecture (ECCV 2022)
- **YOLOv8**: State-of-the-art object detection framework
- **CBAM**: Convolutional Block Attention Module for feature refinement
- **PCB Detection**: Domain-specific optimizations for circuit board defects

## 🐛 Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
- Reduce `batch_size` in config
- Enable `gradient_checkpointing: true`
- Use smaller model: `tiny_vit_11m_224` instead of `tiny_vit_21m_224`

### Import Errors
- Ensure you're in the project root directory
- Check virtual environment is activated
- Verify all dependencies are installed

## 📚 Documentation

- [Training Guide](docs/TRAINING.md)
- [Evaluation Guide](docs/EVALUATION.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

- **Author**: Ash10-ap
- **Repository**: [pcb_defect_detection_tiny_vit](https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit)
- **Issues**: [GitHub Issues](https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit/issues)

## 🎉 Acknowledgments

- [TinyViT](https://github.com/wkcn/TinyViT) for the efficient backbone
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 framework
- [PyTorch](https://pytorch.org/) for the deep learning framework

---

⭐ **Star this repository if it helps your research or projects!**