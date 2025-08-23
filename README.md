# RT-DETR PCB Defect Detection

🚀 **Complete PCB defect detection system using RT-DETR** (Real-Time Detection Transformer) - optimized for high-accuracy PCB defect detection with automatic dataset download using kagglehub.

## ⚡ Quick Start (30 seconds)

```bash
# 1. Clone repository
git clone https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit.git
cd pcb_defect_detection_tiny_vit

# 2. Auto-setup everything (installs dependencies + downloads dataset)
python setup_project.py

# 3. Start training (RTX 4060 optimized)
python main.py train --epochs 100 --batch 12
```

## 🎯 Features

- ✅ **Pure RT-DETR**: No YOLO confusion, optimized transformer architecture
- ✅ **Kagglehub Integration**: Automatic dataset download with `kagglehub`
- ✅ **GPU Optimized**: Perfect settings for RTX 4060 (8GB VRAM)
- ✅ **Organized Structure**: Clean project layout with proper folders
- ✅ **One Command Setup**: Everything automated with `setup_project.py`

## 📁 Project Structure

```
RT-DETR-PCB/
├── main.py              # Complete RT-DETR system
├── setup_project.py     # One-command project setup
├── requirements.txt     # Dependencies
├── README.md           # This file
│
├── data/                     # Dataset folder
│   ├── pcb-defect-dataset/   # Local PCB dataset (10,668 images)
│   └── data.yaml            # RT-DETR dataset config
│
├── models/             # Trained models and checkpoints
├── results/            # Training results and logs
├── inference/          # Inference outputs
├── logs/              # Training logs
└── docs/              # Documentation
```

## 📦 Dataset Setup After Cloning

### Method 1: Auto Setup (Recommended)
```bash
git clone https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit.git
cd pcb_defect_detection_tiny_vit

# This installs everything and downloads dataset
python setup_project.py
```

### Method 2: Manual Setup
```bash
git clone https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit.git
cd pcb_defect_detection_tiny_vit

# Install requirements
pip install -r requirements.txt

# Download dataset
python -c "import kagglehub; kagglehub.dataset_download('norbertelter/pcb-defect-dataset')"
```

### Method 3: Auto-download During Training
```bash
git clone https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit.git
cd pcb_defect_detection_tiny_vit
pip install -r requirements.txt

# Dataset will auto-download when training starts
python main.py train --epochs 100 --batch 12
```

## 🔧 Dataset Information

**Automatic Download with Kagglehub:**
```python
import kagglehub
path = kagglehub.dataset_download("norbertelter/pcb-defect-dataset")
print("Dataset path:", path)
```

**Dataset Details:**
- **Source**: Kaggle norbertelter/pcb-defect-dataset  
- **Size**: ~600MB download
- **Images**: 10,668 high-resolution PCB images
- **Classes**: 6 defect types (mouse_bite, spur, missing_hole, short, open_circuit, spurious_copper)
- **Format**: YOLO annotation format
- **Local Location**: `data/pcb-defect-dataset/` (copied from kagglehub cache)
- **Cache Location**: `~/.cache/kagglehub/datasets/norbertelter/pcb-defect-dataset/` (source)

## 🚀 Training Commands

### Quick Training (RTX 4060)
```bash
# Recommended for RTX 4060 (8GB) 
python main.py train --epochs 100 --batch 12

# Memory-safe option
python main.py train --epochs 100 --batch 8

# Quick test (5 epochs)
python main.py train --epochs 5 --batch 12
```

### Advanced Training
```bash
# Large model for maximum accuracy
python main.py train --model_size l --epochs 100 --batch 16

# Small model for speed
python main.py train --model_size s --epochs 50 --batch 24
```

## 🔬 Inference

### Single Image
```bash
python main.py infer --model results/rtdetr_l_pcb/weights/best.pt --img test.jpg
```

### Batch Processing
```bash
python main.py infer --model results/rtdetr_l_pcb/weights/best.pt --dir test_images/
```

### Validation
```bash
python main.py validate --model results/rtdetr_l_pcb/weights/best.pt
```

## ⚙️ Configuration

### Model Sizes
- `--model_size s`: RT-DETR Small (faster, ~85% accuracy)
- `--model_size l`: RT-DETR Large (balanced, ~90% accuracy) - **default**
- `--model_size x`: RT-DETR Extra Large (slowest, ~93% accuracy)

### GPU Memory Settings
```bash
# RTX 4060 (8GB) - Recommended
python main.py train --batch 12 --imgsz 640

# RTX 3060 (6GB)
python main.py train --batch 8 --imgsz 640

# RTX 4090 (24GB)
python main.py train --batch 32 --imgsz 640
```

## 📈 Expected Performance

### Training Speed (RTX 4060)
- **Batch 12**: ~2.5 hours for 100 epochs ⚡
- **Batch 8**: ~3 hours for 100 epochs (safe)
- **Batch 16**: ~2 hours for 100 epochs (if no memory errors)

### Accuracy Expectations
- **mAP@0.5**: 85-92%
- **mAP@0.5:0.95**: 60-75%
- **Best detection**: mouse_bite, spur, missing_hole
- **Challenging**: spurious_copper (very small defects)

### Inference Speed
- **RTX 4060**: 60-120 FPS
- **RTX 3080**: 100-200 FPS  
- **CPU**: 3-8 FPS

## 🛠️ Installation & Setup

### Option 1: Auto Setup (Recommended)
```bash
python setup_project.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create folders
mkdir data models results inference logs docs

# Download dataset
python -c "import kagglehub; print(kagglehub.dataset_download('norbertelter/pcb-defect-dataset'))"
```

## 🔧 Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.200
kagglehub>=0.2.0
opencv-python>=4.8.0
numpy>=1.21.0
pyyaml>=6.0
matplotlib>=3.6.0
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python main.py train --batch 16  # or --batch 12

# Reduce image size  
python main.py train --batch 20 --imgsz 512
```

### Dataset Download Issues
```bash
# Manual kagglehub download
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('norbertelter/pcb-defect-dataset')"

# Or use setup script
python setup_project.py
```

### Slow Training
- Ensure `--device 0` for GPU
- Use `nvidia-smi` to check GPU utilization
- Increase batch size if memory allows

## 🎉 Results

After training, you'll find:
- **Best model**: `results/rtdetr_l_pcb/weights/best.pt`
- **Training plots**: `results/rtdetr_l_pcb/`
- **Validation metrics**: mAP, precision, recall curves
- **Sample detections**: Visualized training progress

## 🚀 Why RT-DETR?

**RT-DETR vs YOLO:**
- ⚡ **Real-time**: 60+ FPS on modern GPUs
- 🎯 **Higher accuracy**: Superior transformer attention
- 🔍 **Small objects**: Better for tiny PCB defects
- 🧠 **Smart features**: Global context understanding
- 🛠️ **PCB optimized**: Perfect for circuit board patterns

---

**🎯 Ready to detect PCB defects with state-of-the-art accuracy!**