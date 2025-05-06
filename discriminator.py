import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ScaleDiscriminator(nn.Module):
    """Single-scale discriminator with spectral normalization"""
    def __init__(self, channels=32, kernel_size=5, stride=2, use_spectral_norm=True):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else lambda x: x
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)),
            norm_f(nn.Conv1d(channels, channels*2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)),
            norm_f(nn.Conv1d(channels*2, channels*4, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)),
            norm_f(nn.Conv1d(channels*4, channels*8, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)),
            norm_f(nn.Conv1d(channels*8, channels*16, kernel_size=kernel_size, stride=1, padding=kernel_size//2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(channels*16, 1, kernel_size=3, padding=1))
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform (B, 1, T) or (B, T)
        
        Returns:
            Tuple[Tensor, List[Tensor]]: (final output, list of features)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension if needed (B, T) -> (B, 1, T)
            
        feature_maps = []
        
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            feature_maps.append(x)
            
        x = self.conv_post(x)
        feature_maps.append(x)
        
        return x, feature_maps

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for waveform using WGAN approach"""
    def __init__(self, scales=3, channels=32, kernel_size=5, stride=2, use_spectral_norm=True):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(channels, kernel_size, stride, use_spectral_norm) 
            for _ in range(scales)
        ])
        
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
            for _ in range(scales - 1)
        ])
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform (B, T) or (B, 1, T)
        
        Returns:
            Tuple[List[Tensor], List[List[Tensor]]]: (outputs, features)
                outputs: List of discriminator outputs at each scale
                features: List of feature maps from each scale
        """
        outputs = []
        features = []
        
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling[i-1](x)
                
            disc_output, disc_features = discriminator(x)
            outputs.append(disc_output)
            features.append(disc_features)
            
        return outputs, features
