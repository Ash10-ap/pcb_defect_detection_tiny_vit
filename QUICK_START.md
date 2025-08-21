# Quick Start Guide

Get TinyViT-YOLOv8 PCB Detection running in 5 minutes!

## 🚀 Super Quick Setup

### 1. One-Command Setup

**Windows:**
```cmd
git clone https://github.com/your-repo/tinivit-yolov8-pcb.git
cd tinivit-yolov8-pcb
python setup_environment_complete.py
```

**Linux/macOS:**
```bash
git clone https://github.com/your-repo/tinivit-yolov8-pcb.git
cd tinivit-yolov8-pcb
python3 setup_environment_complete.py
```

### 2. Activate Environment

**Windows:**
```cmd
activate.bat
```

**Linux/macOS:**
```bash
source activate.sh
```

### 3. Test Installation
```bash
python run_sample_training.py
```

## 📋 Step-by-Step Guide

### Prerequisites Check
- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] 4GB+ free disk space
- [ ] Internet connection

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-repo/tinivit-yolov8-pcb.git
   cd tinivit-yolov8-pcb
   ```

2. **Run Setup Script**
   ```bash
   python setup_environment_complete.py
   ```
   
   When prompted, choose:
   - `2` for Full installation (recommended)
   - `1` for Minimal (core only)
   - `3` for Development (if contributing)

3. **Verify Installation**
   ```bash
   # Activate environment first
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   
   # Test core functionality
   python test_imports_minimal.py
   ```

## 🎯 First Training Run

### Option 1: Sample Training (No Data Required)
```bash
# Activate environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Run sample training (uses synthetic data)
python run_sample_training.py
```

### Option 2: Real Training (Requires PCB Dataset)

1. **Prepare Your Data**
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

2. **Update Configuration**
   ```bash
   cp configs/tinivit_yolov8_pcb.yaml configs/my_config.yaml
   # Edit configs/my_config.yaml with your data paths
   ```

3. **Start Training**
   ```bash
   python scripts/train_multi_stage.py --config configs/my_config.yaml
   ```

## 📊 Monitor Training

### View Progress
```bash
# TensorBoard (if installed)
tensorboard --logdir experiments/logs

# Check saved models
ls experiments/checkpoints/
```

### Quick Evaluation
```bash
# Evaluate best model
python scripts/evaluate.py \
    --model experiments/checkpoints/best_model.pt \
    --data data/PCB_Dataset/test \
    --visualize
```

## 🔧 Common Customizations

### Change Model Size
```yaml
# In your config file
model:
  backbone_config:
    model_name: "tiny_vit_11m_224"  # Smaller, faster
    # or "tiny_vit_21m_224"         # Larger, more accurate
```

### Adjust for Your Hardware
```yaml
training:
  batch_size: 8   # Reduce if GPU memory limited
  
hardware:
  num_workers: 2  # Reduce if CPU limited
  device: "cpu"   # Force CPU if no GPU
```

### PCB-Specific Classes
```yaml
datasets:
  foundation:
    classes:
      0: "background"
      1: "missing_hole"
      2: "mouse_bite"
      3: "open_circuit"
      4: "short"
      5: "spur"
      6: "spurious_copper"
```

## 🚨 Troubleshooting

### Installation Issues
```bash
# If setup fails, try manual installation:
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

### Import Errors
```bash
# Ensure you're in the right directory and environment is active
pwd  # Should show .../tinivit-yolov8-pcb
which python  # Should show venv/bin/python
```

### Memory Issues
```bash
# Reduce batch size in config file
training:
  batch_size: 4  # Start small and increase
```

### GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Force CPU if needed
python scripts/train_multi_stage.py --device cpu --config configs/my_config.yaml
```

## 📚 Next Steps

### Explore the System
- [ ] Check out `src/` for model architecture
- [ ] Review `configs/` for training options
- [ ] Explore `scripts/` for utilities

### Customize for Your Needs
- [ ] Adapt data loading for your PCB format
- [ ] Modify classes for your defect types
- [ ] Tune hyperparameters for your dataset

### Advanced Features
- [ ] Export models: `python scripts/export_onnx.py`
- [ ] Multi-GPU training: Set `devices: [0, 1]` in config
- [ ] Experiment tracking: Enable `use_wandb: true`

## 🆘 Need Help?

1. **Check Logs:** Look in `experiments/logs/` for detailed error messages
2. **Read Documentation:** See `TROUBLESHOOTING.md` for common issues
3. **Test Components:** Run `python test_imports_minimal.py` to isolate issues
4. **Community:** Open an issue on GitHub with system info and error messages

## 🎉 Success Indicators

You know everything is working when:
- ✅ `python test_imports_minimal.py` passes
- ✅ `python run_sample_training.py` completes without errors
- ✅ Training starts and loss decreases
- ✅ Model checkpoints are saved in `experiments/`

Happy training! 🚀