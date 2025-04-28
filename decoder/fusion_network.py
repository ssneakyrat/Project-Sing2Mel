import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FusionNetwork(nn.Module):
    """Combines multiple waveform components with learned weights"""
    def __init__(self, num_components=2):
        super(FusionNetwork, self).__init__()
        self.num_components = num_components
        
        # Weight predictor network - time-varying weights for each component
        self.weight_net = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, num_components, kernel_size=3, padding=1),
            nn.Softmax(dim=1)  # Weights sum to 1 across components
        )
        
        # Final adjustment network - for post-processing the combined signal
        self.final_adjust = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh()  # Ensure output is in [-1, 1] range
        )
        
    def forward(self, components, condition):
        """
        Args:
            components: List of waveform components, each [B, audio_length]
            condition: Conditioning information [B, C, T]
        Returns:
            fused_signal: Combined waveform [B, audio_length]
        """
        batch_size, _, time_steps = condition.shape
        audio_length = components[0].shape[1]
        
        # Predict component weights
        weights = self.weight_net(condition)  # [B, num_components, T]
        
        # Upsample weights to audio sample rate
        weights = F.interpolate(
            weights, 
            size=audio_length, 
            mode='linear', 
            align_corners=False
        )  # [B, num_components, audio_length]
        
        # Combine components - each gets its own time-varying weight
        fused_signal = torch.zeros_like(components[0])
        min_len = min(comp.shape[-1] for comp in components)
        fused_signal = fused_signal[:, :min_len]
        
        for i in range(self.num_components):
            # Truncate each component to the minimum length
            truncated_component = components[i][:, :min_len]
            # Ensure weights are also truncated if necessary (assuming weights have the same last dimension)
            truncated_weight = weights[:, i, :min_len]
            fused_signal += truncated_component * truncated_weight
        
        # Final adjustment to improve the signal
        fused_signal = fused_signal.unsqueeze(1)  # [B, 1, audio_length]
        fused_signal = self.final_adjust(fused_signal)
        fused_signal = fused_signal.squeeze(1)  # [B, audio_length]
        
        return fused_signal