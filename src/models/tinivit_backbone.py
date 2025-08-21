import torch
import torch.nn as nn
import sys
import os
import timm

# Try to import TinyViT from external directory
try:
    tiny_vit_path = os.path.join(os.path.dirname(__file__), '../../external/TinyViT')
    if tiny_vit_path not in sys.path:
        sys.path.append(tiny_vit_path)
    from models.tiny_vit import TinyViT
    TINY_VIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TinyViT external module not found: {e}. Using timm models only.")
    TINY_VIT_AVAILABLE = False

class TinyViTBackbone(nn.Module):
    """
    TinyViT backbone for object detection with multi-scale feature extraction.
    
    Based on the official TinyViT implementation with modifications for:
    - Multi-scale feature extraction (P3, P4, P5)
    - Integration with YOLO detection heads
    - Pretrained weight loading from timm
    """
    
    def __init__(self, 
                 model_name='tiny_vit_21m_224',
                 pretrained=True,
                 img_size=640,
                 num_classes=1000,
                 drop_rate=0.0,
                 drop_path_rate=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.model_name = model_name
        
        # Load TinyViT from timm with pretrained weights
        if pretrained:
            self.backbone = timm.create_model(
                f'{model_name}.dist_in22k_ft_in1k',
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3, 4]  # Extract features at multiple scales
            )
        else:
            # Create TinyViT from scratch
            self.backbone = self._create_tiny_vit(
                model_name, img_size, num_classes, drop_rate, drop_path_rate
            )
        
        # Get feature dimensions for each scale
        self.feature_info = self.backbone.feature_info
        self.out_channels = [info['num_chs'] for info in self.feature_info]
        
        # Typical TinyViT-21M output channels: [96, 192, 384, 768]
        # We'll use the last 3 for P3, P4, P5: [192, 384, 768]
        self.fpn_channels = self.out_channels[-3:]
        
    def _create_tiny_vit(self, model_name, img_size, num_classes, drop_rate, drop_path_rate):
        """Create TinyViT model from local implementation."""
        if not TINY_VIT_AVAILABLE:
            raise ImportError("TinyViT module not available. Please clone the external repository.")
            
        if model_name == 'tiny_vit_21m_224':
            model = TinyViT(
                img_size=img_size,
                in_chans=3,
                num_classes=num_classes,
                embed_dims=[96, 192, 384, 768],
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.0,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
            )
        elif model_name == 'tiny_vit_11m_224':
            model = TinyViT(
                img_size=img_size,
                in_chans=3,
                num_classes=num_classes,
                embed_dims=[64, 128, 256, 448],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 8, 14],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.0,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model
        
    def forward(self, x):
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            List of feature tensors [P3, P4, P5] with shapes:
            - P3: (B, 192, H/8, W/8)
            - P4: (B, 384, H/16, W/16)  
            - P5: (B, 768, H/32, W/32)
        """
        features = self.backbone(x)
        
        # Return the last 3 feature maps for P3, P4, P5
        # Skip the first feature map (too high resolution for detection)
        return features[-3:]
    
    def freeze_backbone(self, freeze_stages=10):
        """
        Freeze early transformer blocks for transfer learning.
        
        Args:
            freeze_stages: Number of stages to freeze from the beginning
        """
        if hasattr(self.backbone, 'stages'):
            for i, stage in enumerate(self.backbone.stages):
                if i < freeze_stages:
                    for param in stage.parameters():
                        param.requires_grad = False
        elif hasattr(self.backbone, 'blocks'):
            for i, block in enumerate(self.backbone.blocks):
                if i < freeze_stages:
                    for param in block.parameters():
                        param.requires_grad = False
                        
    def unfreeze_last_n_stages(self, n_stages=5):
        """
        Unfreeze the last n transformer stages for fine-tuning.
        
        Args:
            n_stages: Number of stages to unfreeze from the end
        """
        if hasattr(self.backbone, 'stages'):
            total_stages = len(self.backbone.stages)
            for i, stage in enumerate(self.backbone.stages):
                if i >= (total_stages - n_stages):
                    for param in stage.parameters():
                        param.requires_grad = True
        elif hasattr(self.backbone, 'blocks'):
            total_blocks = len(self.backbone.blocks)
            for i, block in enumerate(self.backbone.blocks):
                if i >= (total_blocks - n_stages):
                    for param in block.parameters():
                        param.requires_grad = True


def test_tiny_vit_backbone():
    """Test the TinyViT backbone implementation."""
    model = TinyViTBackbone(pretrained=False)
    x = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        features = model(x)
        
    print(f"Input shape: {x.shape}")
    for i, feat in enumerate(features):
        print(f"P{i+3} feature shape: {feat.shape}")
    
    print(f"Output channels: {model.fpn_channels}")


if __name__ == "__main__":
    test_tiny_vit_backbone()