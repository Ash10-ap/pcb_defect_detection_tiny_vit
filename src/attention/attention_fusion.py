import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .cbam import CBAM, EnhancedCBAM
    from .eca import ECA, ECAWithSpatial
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from attention.cbam import CBAM, EnhancedCBAM
    from attention.eca import ECA, ECAWithSpatial

class AttentionFusion(nn.Module):
    """
    Attention-based feature fusion module for integrating TinyViT features
    with YOLOv8 detection pipeline.
    
    This module acts as the bridge between TinyViT backbone outputs and
    YOLOv8 neck, applying attention mechanisms to enhance small defect detection.
    """
    
    def __init__(self, 
                 input_channels=[192, 384, 768],
                 output_channels=[128, 256, 512],
                 attention_type='cbam',
                 use_residual=True):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.attention_type = attention_type
        self.use_residual = use_residual
        
        # Channel adapters: 1x1 conv to match YOLOv8 expected channels
        self.channel_adapters = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        
        for in_ch, out_ch in zip(input_channels, output_channels):
            # 1x1 convolution for channel adaptation
            adapter = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            )
            self.channel_adapters.append(adapter)
            
            # Attention module selection
            if attention_type == 'cbam':
                attention = CBAM(out_ch)
            elif attention_type == 'enhanced_cbam':
                attention = EnhancedCBAM(out_ch, use_residual=use_residual)
            elif attention_type == 'eca':
                attention = ECA(out_ch)
            elif attention_type == 'eca_spatial':
                attention = ECAWithSpatial(out_ch)
            else:
                raise ValueError(f"Unsupported attention type: {attention_type}")
                
            self.attention_modules.append(attention)
    
    def forward(self, tinivit_features):
        """
        Forward pass applying channel adaptation and attention.
        
        Args:
            tinivit_features: List of feature tensors from TinyViT backbone
                             [P3, P4, P5] with channels [192, 384, 768]
        
        Returns:
            List of processed features with YOLOv8-compatible channels
            [P3, P4, P5] with channels [128, 256, 512]
        """
        enhanced_features = []
        
        for feat, adapter, attention in zip(tinivit_features, 
                                           self.channel_adapters,
                                           self.attention_modules):
            # Apply channel adaptation
            adapted = adapter(feat)
            
            # Apply attention mechanism
            enhanced = attention(adapted)
            
            enhanced_features.append(enhanced)
        
        return enhanced_features


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention module that allows features from different scales
    to interact and enhance each other. Particularly useful for detecting
    defects of varying sizes in PCB images.
    """
    
    def __init__(self, channels_list=[128, 256, 512], reduction=16):
        super().__init__()
        self.channels_list = channels_list
        self.num_scales = len(channels_list)
        
        # Cross-scale interaction modules
        self.cross_scale_modules = nn.ModuleList()
        
        for i in range(self.num_scales):
            for j in range(self.num_scales):
                if i != j:
                    # Create module for interaction between scale i and scale j
                    module = nn.Sequential(
                        nn.Conv2d(channels_list[j], channels_list[i] // reduction, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels_list[i] // reduction, channels_list[i], 1),
                        nn.Sigmoid()
                    )
                    self.cross_scale_modules.append(module)
    
    def forward(self, features):
        """
        Args:
            features: List of feature tensors [P3, P4, P5]
        
        Returns:
            List of enhanced feature tensors with cross-scale interactions
        """
        enhanced_features = []
        
        for i, feat_i in enumerate(features):
            h_i, w_i = feat_i.shape[2:]
            enhanced_feat = feat_i
            
            for j, feat_j in enumerate(features):
                if i != j:
                    # Resize feat_j to match feat_i spatial dimensions
                    feat_j_resized = F.interpolate(
                        feat_j, size=(h_i, w_i), 
                        mode='bilinear', align_corners=False
                    )
                    
                    # Compute cross-scale attention
                    idx = i * (self.num_scales - 1) + (j if j < i else j - 1)
                    attention_weight = self.cross_scale_modules[idx](feat_j_resized)
                    
                    # Apply attention to enhance current scale features
                    enhanced_feat = enhanced_feat + feat_i * attention_weight
            
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive feature fusion that dynamically weights different attention
    mechanisms based on the input characteristics.
    """
    
    def __init__(self, 
                 input_channels=[192, 384, 768],
                 output_channels=[128, 256, 512]):
        super().__init__()
        
        self.fusion_cbam = AttentionFusion(
            input_channels, output_channels, attention_type='cbam'
        )
        self.fusion_eca = AttentionFusion(
            input_channels, output_channels, attention_type='eca_spatial'
        )
        
        # Adaptive weighting modules
        self.weight_modules = nn.ModuleList()
        for out_ch in output_channels:
            weight_module = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, 2, 1),
                nn.Softmax(dim=1)
            )
            self.weight_modules.append(weight_module)
    
    def forward(self, tinivit_features):
        """
        Adaptively fuse CBAM and ECA attention mechanisms.
        """
        cbam_features = self.fusion_cbam(tinivit_features)
        eca_features = self.fusion_eca(tinivit_features)
        
        fused_features = []
        
        for cbam_feat, eca_feat, weight_module in zip(
            cbam_features, eca_features, self.weight_modules
        ):
            # Compute adaptive weights
            combined = cbam_feat + eca_feat
            weights = weight_module(combined)  # Shape: (B, 2, 1, 1)
            
            # Apply weighted fusion
            fused = weights[:, 0:1] * cbam_feat + weights[:, 1:2] * eca_feat
            fused_features.append(fused)
        
        return fused_features


def test_attention_fusion():
    """Test attention fusion modules."""
    batch_size = 2
    
    # Simulate TinyViT backbone outputs
    p3 = torch.randn(batch_size, 192, 80, 80)  # 1/8 scale
    p4 = torch.randn(batch_size, 384, 40, 40)  # 1/16 scale
    p5 = torch.randn(batch_size, 768, 20, 20)  # 1/32 scale
    
    tinivit_features = [p3, p4, p5]
    
    print("Input feature shapes:")
    for i, feat in enumerate(tinivit_features):
        print(f"  P{i+3}: {feat.shape}")
    
    # Test basic attention fusion
    fusion = AttentionFusion()
    enhanced_features = fusion(tinivit_features)
    
    print("\nEnhanced feature shapes:")
    for i, feat in enumerate(enhanced_features):
        print(f"  P{i+3}: {feat.shape}")
    
    # Test cross-scale attention
    cross_scale = CrossScaleAttention()
    cross_enhanced = cross_scale(enhanced_features)
    
    print("\nCross-scale enhanced shapes:")
    for i, feat in enumerate(cross_enhanced):
        print(f"  P{i+3}: {feat.shape}")
    
    # Test adaptive fusion
    adaptive = AdaptiveFeatureFusion()
    adaptive_features = adaptive(tinivit_features)
    
    print("\nAdaptive fusion shapes:")
    for i, feat in enumerate(adaptive_features):
        print(f"  P{i+3}: {feat.shape}")
    
    print("\nAll attention fusion modules working correctly!")


if __name__ == "__main__":
    test_attention_fusion()