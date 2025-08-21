from setuptools import setup, find_packages

setup(
    name="tinivit-yolov8-pcb",
    version="1.0.0",
    description="Integrated TinyViT-YOLOv8 PCB Defect Detection System",
    author="PCB Detection Research",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.200",
        "timm>=0.9.7",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.1",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "wandb>=0.15.0",
        "scikit-learn>=1.3.0",
        "pycocotools>=2.0.6"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)