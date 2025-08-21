import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) Module.
    
    A lightweight alternative to CBAM's channel attention that uses
    1D convolution for channel interaction with adaptive kernel size.
    
    Reference: "ECA-Net: Efficient Channel Attention for Deep CNNs" (CVPR 2020)
    """
    
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        
        # Adaptive kernel size based on channel dimension
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling
        y = self.avg_pool(x)
        
        # 1D convolution along channel dimension
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Apply sigmoid and multiply with input
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class ECAWithSpatial(nn.Module):
    """
    ECA module combined with spatial attention for enhanced performance.
    
    Combines the efficiency of ECA for channel attention with
    spatial attention for better small object detection.
    """
    
    def __init__(self, channels, gamma=2, b=1, spatial_kernel=7):
        super().__init__()
        
        self.eca = ECA(channels, gamma, b)
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel, 
                                     padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Apply ECA channel attention
        out = self.eca(x)
        
        # Apply spatial attention
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_out))
        
        return out * spatial_weight


def test_eca_modules():
    """Test ECA attention modules."""
    batch_size, channels, height, width = 2, 256, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Test basic ECA
    eca = ECA(channels)
    out_eca = eca(x)
    print(f"ECA output shape: {out_eca.shape}")
    
    # Test ECA with spatial
    eca_spatial = ECAWithSpatial(channels)
    out_eca_spatial = eca_spatial(x)
    print(f"ECA with spatial output shape: {out_eca_spatial.shape}")
    
    print("All ECA modules working correctly!")


if __name__ == "__main__":
    test_eca_modules()