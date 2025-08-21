import torch
import torch.nn as nn

# Optional ultralytics imports
try:
    from ultralytics.nn.modules import Detect, C2f, SPPF, Conv, Concat
    from ultralytics.nn.tasks import parse_model
    from ultralytics.models.yolo.model import YOLO
    from ultralytics.utils import yaml_load
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Warning: Ultralytics not available. Some YOLO features will be disabled.")
    ULTRALYTICS_AVAILABLE = False
    
    # Define placeholder classes
    class Detect(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
    class C2f(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
    class Conv(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
    class Concat(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
    class YOLO:
        def __init__(self, *args, **kwargs):
            pass

try:
    from .tinivit_backbone import TinyViTBackbone
    from ..attention.attention_fusion import AttentionFusion, CrossScaleAttention
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.tinivit_backbone import TinyViTBackbone
    from attention.attention_fusion import AttentionFusion, CrossScaleAttention

class TinyViTYOLO(nn.Module):
    """
    Integrated TinyViT-YOLOv8 model for PCB defect detection.
    
    This model combines:
    1. TinyViT backbone for feature extraction
    2. Attention-based feature fusion (CBAM/ECA)
    3. YOLOv8 neck and detection head
    
    Architecture:
    Input → TinyViT → Attention Fusion → YOLOv8 Neck → Detection Head → Output
    """
    
    def __init__(self,
                 backbone_config={
                     'model_name': 'tiny_vit_21m_224',
                     'pretrained': True,
                     'img_size': 640
                 },
                 attention_config={
                     'attention_type': 'cbam',
                     'use_cross_scale': True,
                     'use_residual': True
                 },
                 yolo_config={
                     'nc': 6,  # Number of classes for PCB defects
                     'depth_multiple': 0.33,
                     'width_multiple': 0.25,
                 },
                 training_stage='foundation'):
        super().__init__()
        
        self.training_stage = training_stage
        self.nc = yolo_config['nc']
        
        # 1. TinyViT Backbone
        self.backbone = TinyViTBackbone(**backbone_config)
        
        # 2. Attention-based Feature Fusion
        self.attention_fusion = AttentionFusion(
            input_channels=[192, 384, 768],  # TinyViT-21M output channels
            output_channels=[128, 256, 512],  # YOLOv8 expected channels
            attention_type=attention_config['attention_type'],
            use_residual=attention_config['use_residual']
        )
        
        if attention_config['use_cross_scale']:
            self.cross_scale_attention = CrossScaleAttention([128, 256, 512])
        else:
            self.cross_scale_attention = None
            
        # 3. YOLOv8 Neck (PAN-FPN)
        self.neck = self._build_yolo_neck(yolo_config)
        
        # 4. Detection Head
        self.head = self._build_detection_head(yolo_config)
        
        # Initialize training stage settings
        self._setup_training_stage(training_stage)
        
    def _build_yolo_neck(self, config):
        """Build YOLOv8 neck with PAN-FPN structure."""
        # YOLOv8n neck configuration
        neck_modules = nn.ModuleList([
            # Upsampling path
            Conv(512, 256, 1, 1),  # P5 → 256 channels
            nn.Upsample(None, 2, 'nearest'),
            Concat(1),  # Concat P4
            C2f(512, 256, 1, False),  # 256 + 256 → 256
            
            Conv(256, 128, 1, 1),  # P4 → 128 channels
            nn.Upsample(None, 2, 'nearest'),
            Concat(1),  # Concat P3
            C2f(256, 128, 1, False),  # 128 + 128 → 128
            
            # Downsampling path
            Conv(128, 128, 3, 2),  # P3 → downsample
            Concat(1),  # Concat with P4
            C2f(256, 256, 1, False),  # 128 + 256 → 256
            
            Conv(256, 256, 3, 2),  # P4 → downsample
            Concat(1),  # Concat with P5
            C2f(512, 512, 1, False),  # 256 + 512 → 512
        ])
        
        return neck_modules
    
    def _build_detection_head(self, config):
        """Build YOLOv8 detection head."""
        nc = config['nc']
        anchors = 1  # YOLOv8 uses anchor-free detection
        
        # Detection head for 3 scales: P3, P4, P5
        head = Detect(nc, anchors, [128, 256, 512])
        
        return head
    
    def _setup_training_stage(self, stage):
        """Configure model for different training stages."""
        if stage == 'foundation':
            # Stage 1: Freeze backbone, train adapters + neck/head
            self.backbone.freeze_backbone(freeze_stages=10)
            
        elif stage == 'transfer_early':
            # Stage 2a: Freeze backbone, train neck/head only
            self.backbone.freeze_backbone(freeze_stages=12)
            
        elif stage == 'transfer_partial':
            # Stage 2b: Unfreeze last 5 ViT blocks
            self.backbone.freeze_backbone(freeze_stages=7)
            self.backbone.unfreeze_last_n_stages(5)
            
        elif stage == 'transfer_full':
            # Stage 2c: Unfreeze last 10 ViT blocks
            self.backbone.freeze_backbone(freeze_stages=2)
            self.backbone.unfreeze_last_n_stages(10)
            
        elif stage == 'few_shot':
            # Stage 3: Fine-tune only attention + head
            self.backbone.freeze_backbone(freeze_stages=12)
            # Attention modules remain trainable
            
        elif stage == 'full_finetune':
            # Full model fine-tuning
            for param in self.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass through the integrated model.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Detection outputs from YOLOv8 head
        """
        # 1. Extract multi-scale features from TinyViT
        backbone_features = self.backbone(x)  # [P3, P4, P5]
        
        # 2. Apply attention-based feature fusion
        enhanced_features = self.attention_fusion(backbone_features)
        
        # 3. Apply cross-scale attention if enabled
        if self.cross_scale_attention is not None:
            enhanced_features = self.cross_scale_attention(enhanced_features)
        
        # 4. Pass through YOLOv8 neck
        p3, p4, p5 = enhanced_features
        
        # Neck forward pass (simplified)
        # Upsampling path
        p5_up = self.neck[0](p5)  # Conv 512→256
        p5_up = self.neck[1](p5_up)  # Upsample
        p4_concat = self.neck[2]([p5_up, p4])  # Concat
        p4_out = self.neck[3](p4_concat)  # C2f
        
        p4_up = self.neck[4](p4_out)  # Conv 256→128
        p4_up = self.neck[5](p4_up)  # Upsample
        p3_concat = self.neck[6]([p4_up, p3])  # Concat
        p3_out = self.neck[7](p3_concat)  # C2f
        
        # Downsampling path
        p3_down = self.neck[8](p3_out)  # Conv+downsample
        p4_concat2 = self.neck[9]([p3_down, p4_out])  # Concat
        p4_final = self.neck[10](p4_concat2)  # C2f
        
        p4_down = self.neck[11](p4_final)  # Conv+downsample
        p5_concat = self.neck[12]([p4_down, p5])  # Concat
        p5_final = self.neck[13](p5_concat)  # C2f
        
        # 5. Detection head
        neck_outputs = [p3_out, p4_final, p5_final]
        detections = self.head(neck_outputs)
        
        return detections
    
    def set_training_stage(self, stage):
        """Change training stage and update frozen parameters."""
        self.training_stage = stage
        self._setup_training_stage(stage)
        
        print(f"Model set to training stage: {stage}")
        
        # Print trainable parameters info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")


class TinyViTYOLOWrapper(YOLO):
    """
    Wrapper class that integrates TinyViT-YOLO with Ultralytics YOLO interface.
    
    This allows using the familiar YOLO training and inference API while
    using our custom TinyViT backbone.
    """
    
    def __init__(self, model_config=None, task='detect'):
        # Initialize with a standard YOLOv8 model first
        super().__init__('yolov8n.pt', task=task)
        
        # Replace the model with our custom TinyViT-YOLO
        if model_config is None:
            model_config = {
                'backbone_config': {'model_name': 'tiny_vit_21m_224', 'pretrained': True},
                'attention_config': {'attention_type': 'cbam', 'use_cross_scale': True},
                'yolo_config': {'nc': 6}
            }
        
        self.model = TinyViTYOLO(**model_config)
        
    def set_training_stage(self, stage):
        """Set the training stage for multi-stage training."""
        self.model.set_training_stage(stage)


def create_tinivit_yolo(model_size='21m', num_classes=6, attention_type='cbam'):
    """
    Factory function to create TinyViT-YOLO models.
    
    Args:
        model_size: '11m' or '21m' for TinyViT model size
        num_classes: Number of detection classes
        attention_type: 'cbam', 'eca', or 'enhanced_cbam'
        
    Returns:
        TinyViTYOLO model instance
    """
    backbone_config = {
        'model_name': f'tiny_vit_{model_size}_224',
        'pretrained': True,
        'img_size': 640
    }
    
    attention_config = {
        'attention_type': attention_type,
        'use_cross_scale': True,
        'use_residual': True
    }
    
    yolo_config = {
        'nc': num_classes,
        'depth_multiple': 0.33,
        'width_multiple': 0.25 if model_size == '11m' else 0.5
    }
    
    model = TinyViTYOLO(
        backbone_config=backbone_config,
        attention_config=attention_config,
        yolo_config=yolo_config
    )
    
    return model


def test_tinivit_yolo():
    """Test the TinyViT-YOLO model."""
    model = create_tinivit_yolo(model_size='21m', num_classes=6)
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        outputs = model(x)
        
    if isinstance(outputs, (list, tuple)):
        for i, out in enumerate(outputs):
            print(f"Output {i} shape: {out.shape}")
    else:
        print(f"Output shape: {outputs.shape}")
    
    # Test training stage changes
    for stage in ['foundation', 'transfer_early', 'transfer_partial', 'few_shot']:
        print(f"\n--- Testing stage: {stage} ---")
        model.set_training_stage(stage)


if __name__ == "__main__":
    test_tinivit_yolo()