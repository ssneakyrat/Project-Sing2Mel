import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralProcessor(nn.Module):
    """
    Handles spectral tilt and harmonic scaling for vocal quality control.
    """
    def __init__(self, num_harmonics=100, input_channels=128):
        super(SpectralProcessor, self).__init__()
        self.num_harmonics = num_harmonics
        
        # Create harmonic indices tensor (registered as buffer)
        harmonic_indices = torch.arange(1, num_harmonics + 1).float().view(1, -1, 1)
        self.register_buffer('harmonic_indices', harmonic_indices)
        
        # Global spectral tilt network (controls overall bright/dark quality)
        self.spectral_tilt_net = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Allow positive and negative tilt (-1 to 1)
        )
        
        # Harmonic-specific scaling network (direct control over harmonic groups)
        self.harmonic_scaling_net = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, num_harmonics, kernel_size=3, padding=1),
            nn.Sigmoid()  # 0-1 range for per-harmonic scaling
        )
        
    def forward(self, condition, harmonic_amplitudes):
        """
        Apply spectral processing to harmonic amplitudes.
        
        Args:
            condition: Conditioning features [B, input_channels, T]
            harmonic_amplitudes: Harmonic amplitudes [B, num_harmonics, T]
            
        Returns:
            Enhanced harmonic amplitudes with spectral processing [B, num_harmonics, T]
        """
        # Generate spectral tilt parameter
        spectral_tilt = self.spectral_tilt_net(condition)  # [B, 1, T]
        
        # Generate per-harmonic scaling factors
        harmonic_scaling = self.harmonic_scaling_net(condition)  # [B, num_harmonics, T]
        
        # Apply spectral tilt at frame rate
        tilt_factor = torch.exp(-0.1 * spectral_tilt * self.harmonic_indices)  # Exponential tilt
        enhanced_amplitudes = harmonic_amplitudes * tilt_factor
        
        # Apply per-harmonic scaling at frame rate
        enhanced_amplitudes = enhanced_amplitudes * harmonic_scaling
        
        return enhanced_amplitudes