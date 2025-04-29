import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HarmonicProcessor(nn.Module):
    
    def __init__(self, num_harmonics=100, input_channels=128):
        super(HarmonicProcessor, self).__init__()
        self.num_harmonics = num_harmonics
        
        # Learnable harmonic amplitude network
        self.harmonic_amp_net = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, num_harmonics, kernel_size=3, padding=1),
            nn.Softplus()  # Ensure positive amplitudes with smooth gradient
        )
        
    def forward(self, condition):
        """
        Generate harmonic amplitudes from conditioning.
        
        Args:
            condition: Conditioning features [B, input_channels, T]
            f0: Optional fundamental frequency (not used in base implementation)
                but included for API compatibility
                
        Returns:
            harmonic_amplitudes: Harmonic amplitudes [B, num_harmonics, T]
        """
        harmonic_amplitudes = self.harmonic_amp_net(condition)  # [B, num_harmonics, T]
        return harmonic_amplitudes