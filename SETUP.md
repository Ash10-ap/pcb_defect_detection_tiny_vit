# Setup Guide - RT-DETR PCB Defect Detection

Quick setup guide for anyone cloning this repository.

## 🚀 One-Command Setup (Easiest)

```bash
git clone https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit.git
cd pcb_defect_detection_tiny_vit
python setup_project.py
```

**This will:**
- Install all Python dependencies
- Download the PCB dataset (~600MB)
- Create project folder structure
- Setup data configuration

## 🔧 Manual Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit.git
cd pcb_defect_detection_tiny_vit
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
```bash
python -c "import kagglehub; kagglehub.dataset_download('norbertelter/pcb-defect-dataset')"
```

## ⚡ Skip Setup - Auto Download

You can skip dataset setup entirely! Just start training and it will auto-download:

```bash
git clone https://github.com/Ash10-ap/pcb_defect_detection_tiny_vit.git
cd pcb_defect_detection_tiny_vit
pip install -r requirements.txt
python main.py train --epochs 100 --batch 12  # Downloads dataset automatically
```

## 📁 What Gets Created

After setup, your project structure:

```
RT-DETR-PCB/
├── main.py              # Complete RT-DETR system
├── setup_project.py     # Setup script
├── requirements.txt     # Dependencies
├── data/                     # Dataset folder
│   ├── pcb-defect-dataset/   # Local PCB dataset
│   └── data.yaml            # Dataset config
├── models/             # Trained models
├── results/            # Training outputs
└── inference/          # Inference results
```

## 💾 Dataset Location

The PCB defect dataset is stored locally in your project:
- **Local folder**: `data/pcb-defect-dataset/` (primary location used by code)
- **Kagglehub cache**: Downloaded first to cache, then copied to local folder
  - **Windows**: `C:\Users\{username}\.cache\kagglehub\datasets\norbertelter\pcb-defect-dataset\`
  - **Linux/Mac**: `~/.cache/kagglehub/datasets/norbertelter/pcb-defect-dataset/`

## 🎯 Start Training

After setup, start training with optimal settings:

```bash
# RTX 4060 (8GB) - Recommended
python main.py train --epochs 100 --batch 12

# RTX 3060 (6GB) - Safe
python main.py train --epochs 100 --batch 8

# Quick test (5 epochs)
python main.py train --epochs 5 --batch 12
```

## ❗ Troubleshooting

### Dataset Download Issues
```bash
# Manual kagglehub install
pip install kagglehub

# Try download again
python -c "import kagglehub; kagglehub.dataset_download('norbertelter/pcb-defect-dataset')"
```

### CUDA Memory Issues
- Reduce batch size: `--batch 8` or `--batch 6`
- Ensure GPU drivers are updated

### Import Errors
- Make sure you're in the project directory
- Re-run: `pip install -r requirements.txt`

## 🚀 That's It!

Your RT-DETR PCB defect detection system is ready to use!