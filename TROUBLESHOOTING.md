# Troubleshooting Guide

This guide covers common issues and solutions for the TinyViT-YOLOv8 PCB Detection system.

## Installation Issues

### Python Version Issues

**Problem:** `Python 3.8 or higher is required`
```bash
ERROR: Python 3.8 or higher is required
```

**Solution:**
1. Install Python 3.8+ from [python.org](https://python.org)
2. On Ubuntu/Debian: `sudo apt install python3.9 python3.9-venv`
3. On Windows: Download and install from python.org
4. Verify: `python --version` or `python3 --version`

### Virtual Environment Issues

**Problem:** Virtual environment creation fails
```bash
ERROR: Failed to create virtual environment
```

**Solutions:**
1. **Windows:** Ensure Python is in PATH and try:
   ```cmd
   python -m pip install --upgrade pip
   python -m venv venv
   ```

2. **Linux/macOS:** Install venv module:
   ```bash
   sudo apt install python3-venv  # Ubuntu/Debian
   python3 -m venv venv
   ```

3. **Permission Issues:** Check write permissions in project directory

### Dependency Installation Issues

**Problem:** `No module named 'pytorch_lightning'`
```bash
ModuleNotFoundError: No module named 'pytorch_lightning'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install specific package
pip install pytorch-lightning>=2.0.0

# Or reinstall all requirements
pip install -r requirements.txt
```

**Problem:** PyTorch installation fails
```bash
ERROR: Failed building wheel for torch
```

**Solutions:**
1. **Update pip and setuptools:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Use pre-built wheels:**
   ```bash
   # CPU only
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   
   # CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Memory issues:** Close other applications and try again

**Problem:** Ultralytics installation fails
```bash
ERROR: Failed to install ultralytics
```

**Solution:**
```bash
# Install dependencies first
pip install torch torchvision
pip install opencv-python pillow pyyaml

# Then install ultralytics
pip install ultralytics>=8.0.200
```

### TinyViT Repository Issues

**Problem:** TinyViT repository not found
```bash
Warning: TinyViT external module not found
```

**Solutions:**
1. **Manual clone:**
   ```bash
   mkdir -p external
   git clone https://github.com/wkcn/TinyViT.git external/TinyViT
   ```

2. **Check internet connection and firewall settings**

3. **Alternative download:**
   - Download ZIP from GitHub
   - Extract to `external/TinyViT/`

## Runtime Issues

### Import Errors

**Problem:** Relative import errors
```bash
ImportError: attempted relative import beyond top-level package
```

**Solution:**
1. **Run from project root:**
   ```bash
   cd /path/to/new-final-tinivit
   python scripts/train_multi_stage.py
   ```

2. **Check PYTHONPATH:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/new-final-tinivit"
   ```

**Problem:** Module not found in scripts
```bash
ModuleNotFoundError: No module named 'src.models'
```

**Solution:**
1. **Ensure proper activation:**
   ```bash
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. **Run from project root directory**

3. **Check Python path in scripts is working correctly**

### Memory Issues

**Problem:** CUDA out of memory
```bash
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Reduce batch size:**
   ```yaml
   # In config file
   training:
     batch_size: 4  # Reduce from 16
   ```

2. **Enable gradient checkpointing:**
   ```yaml
   hardware:
     gradient_checkpointing: true
   ```

3. **Use mixed precision:**
   ```yaml
   hardware:
     use_amp: true
   ```

4. **Reduce image size:**
   ```yaml
   model:
     img_size: 512  # Reduce from 640
   ```

### Training Issues

**Problem:** Loss becomes NaN
```bash
Train Loss: nan, Val mAP: 0.0000
```

**Solutions:**
1. **Reduce learning rate:**
   ```yaml
   training:
     stages:
       foundation:
         lr: 1.0e-4  # Reduce from 1.0e-3
   ```

2. **Enable gradient clipping:**
   ```yaml
   training:
     gradient_clip_val: 1.0
   ```

3. **Check data loading - ensure targets are valid**

**Problem:** Training very slow
```bash
Training taking too long per epoch
```

**Solutions:**
1. **Reduce data workers if CPU limited:**
   ```yaml
   hardware:
     num_workers: 2  # Reduce from 4
   ```

2. **Check data loading efficiency**

3. **Use smaller model variant:**
   ```yaml
   model:
     backbone_config:
       model_name: "tiny_vit_11m_224"  # Instead of 21m
   ```

## Platform-Specific Issues

### Windows Issues

**Problem:** Path issues with backslashes
```bash
FileNotFoundError: No such file or directory
```

**Solution:**
- Use forward slashes in Python code
- Use `pathlib.Path` for cross-platform compatibility

**Problem:** Long path names
```bash
The filename or extension is too long
```

**Solution:**
1. Move project to shorter path (e.g., `C:\tinivit\`)
2. Enable long path support in Windows

### macOS Issues

**Problem:** Command Line Tools missing
```bash
error: Microsoft Visual C++ 14.0 is required
```

**Solution:**
```bash
xcode-select --install
```

**Problem:** Permission denied
```bash
Permission denied: '/usr/local/...'
```

**Solution:**
```bash
# Use user installation
pip install --user package_name

# Or fix permissions
sudo chown -R $(whoami) /usr/local
```

### Linux Issues

**Problem:** Missing system dependencies
```bash
ImportError: libGL.so.1: cannot open shared object file
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# CentOS/RHEL/Fedora
sudo yum install mesa-libGL glib2 libSM libXext libXrender libgomp
```

## Performance Issues

### Slow Training

**Checklist:**
1. ✅ GPU is being used: `torch.cuda.is_available()`
2. ✅ Appropriate batch size for GPU memory
3. ✅ Data loading not bottleneck (sufficient workers)
4. ✅ Using appropriate model size
5. ✅ Mixed precision enabled if supported

### High Memory Usage

**Solutions:**
1. **Monitor GPU memory:**
   ```bash
   nvidia-smi -l 1  # Monitor every second
   ```

2. **Profile memory usage:**
   ```python
   import torch
   print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

3. **Optimize data loading:**
   - Use smaller prefetch factor
   - Reduce number of workers if RAM limited

## Getting Help

### Collecting Debug Information

When reporting issues, include:

1. **System Information:**
   ```bash
   python -c "import sys, torch; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

2. **Environment Details:**
   ```bash
   pip list | grep -E "(torch|ultralytics|timm|opencv)"
   ```

3. **Error Traceback:** Full error message with stack trace

4. **Configuration:** Relevant config file sections

### Common Commands for Debugging

```bash
# Test basic functionality
python test_imports_minimal.py

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"

# Test model creation
python -c "from src.attention.cbam import CBAM; print('CBAM import successful')"

# Verify paths
python -c "import sys; print('\\n'.join(sys.path))"
```

### Where to Get Help

1. **GitHub Issues:** For bugs and feature requests
2. **Discussions:** For questions and help
3. **Documentation:** Check README.md and code comments
4. **Community:** PyTorch and Ultralytics communities

Remember to search existing issues before creating new ones!