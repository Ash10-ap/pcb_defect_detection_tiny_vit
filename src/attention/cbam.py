import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    
    Applies global average pooling and global max pooling, then uses
    a shared MLP to compute channel attention weights.
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    
    Applies channel-wise average and max pooling, then uses
    a 7x7 convolution to compute spatial attention weights.
    """
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequential application of channel attention followed by spatial attention.
    Proven effective for small object detection and PCB defect localization.
    
    Reference: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention
        out = x * self.channel_attention(x)
        
        # Apply spatial attention
        out = out * self.spatial_attention(out)
        
        return out


class EnhancedCBAM(nn.Module):
    """
    Enhanced CBAM with residual connections and normalization.
    
    Adds batch normalization and residual connections for better
    gradient flow and training stability in deep networks.
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        if use_residual:
            self.bn = nn.BatchNorm2d(channels)
            
    def forward(self, x):
        identity = x
        
        # Apply channel attention
        out = x * self.channel_attention(x)
        
        # Apply spatial attention  
        out = out * self.spatial_attention(out)
        
        if self.use_residual:
            out = self.bn(out + identity)
            
        return out


class MultiScaleCBAM(nn.Module):
    """
    Multi-scale CBAM that processes features at different spatial scales
    before combining them. Useful for multi-scale object detection.
    """
    
    def __init__(self, channels, scales=[1, 2, 4], reduction=16):
        super().__init__()
        self.scales = scales
        self.cbam_modules = nn.ModuleList([
            CBAM(channels, reduction) for _ in scales
        ])
        
        # Fusion layer to combine multi-scale features
        self.fusion = nn.Conv2d(channels * len(scales), channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        _, _, h, w = x.shape
        multi_scale_features = []
        
        for scale, cbam in zip(self.scales, self.cbam_modules):
            if scale == 1:
                feat = cbam(x)
            else:
                # Downsample, apply attention, then upsample
                pooled = F.avg_pool2d(x, scale, scale)
                attended = cbam(pooled)
                feat = F.interpolate(attended, size=(h, w), mode='bilinear', align_corners=False)
            
            multi_scale_features.append(feat)
        
        # Concatenate and fuse multi-scale features
        concatenated = torch.cat(multi_scale_features, dim=1)
        fused = self.fusion(concatenated)
        fused = self.bn(fused)
        
        return fused


def test_cbam_modules():
    """Test CBAM attention modules."""
    batch_size, channels, height, width = 2, 256, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Test basic CBAM
    cbam = CBAM(channels)
    out_cbam = cbam(x)
    print(f"CBAM output shape: {out_cbam.shape}")
    
    # Test enhanced CBAM
    enhanced_cbam = EnhancedCBAM(channels)
    out_enhanced = enhanced_cbam(x)
    print(f"Enhanced CBAM output shape: {out_enhanced.shape}")
    
    # Test multi-scale CBAM
    ms_cbam = MultiScaleCBAM(channels)
    out_ms = ms_cbam(x)
    print(f"Multi-scale CBAM output shape: {out_ms.shape}")
    
    print("All CBAM modules working correctly!")


if __name__ == "__main__":
    test_cbam_modules()