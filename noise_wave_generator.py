import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoiseWaveGenerator(nn.Module):
    """Advanced noise wave generator for breath-like sound synthesis with voicing control"""
    def __init__(self, n_mels=80, hop_length=240, hidden_channels=256):
        super(NoiseWaveGenerator, self).__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        
        # Initial expansion from mel+f0+voicing to hidden channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(n_mels + 2, hidden_channels, kernel_size=1),  # +2 for f0 and voicing
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # Residual blocks with dilated convolutions
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, dilation=1),
            ResidualBlock(hidden_channels, dilation=2),
            ResidualBlock(hidden_channels, dilation=4),
            ResidualBlock(hidden_channels, dilation=8)
        ])
        
        # Upsampling layers to reach waveform resolution
        # We need to upsample by factor of hop_length
        # Using 3 layers: 4x -> 4x -> 15x (or adjusted to match exactly)
        self.upsampling_blocks = nn.ModuleList()
        
        # Determine upsampling factors
        upsample_factors = self._get_upsample_factors(hop_length)
        in_channels = hidden_channels
        
        for factor in upsample_factors:
            self.upsampling_blocks.append(
                UpsamplingBlock(in_channels, in_channels // 2, factor)
            )
            in_channels = in_channels // 2
        
        # Final convolution to single channel
        self.final_conv = nn.Conv1d(in_channels, 1, kernel_size=7, padding=3)
        
        # Modified gating mechanism to incorporate voicing
        self.gate_conv = nn.Sequential(
            nn.Conv1d(n_mels + 2, 32, kernel_size=3, padding=1),  # +2 for f0 and voicing
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # High-pass filter for breath-like characteristics
        self.high_pass = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        # Initialize as high-pass filter
        with torch.no_grad():
            self.high_pass.weight.fill_(0)
            self.high_pass.weight[0, 0, 0] = -0.5
            self.high_pass.weight[0, 0, 1] = 1.0
            self.high_pass.weight[0, 0, 2] = -0.5
    
    def forward(self, mel, f0, voicing):
        """
        Args:
            mel: Mel spectrogram (B, n_mels, T)
            f0: Fundamental frequency (B, 1, T)
            voicing: Voicing prediction (B, 1, T)
        Returns:
            noise_wave: Noise waveform (B, T * hop_length)
        """
        B, _, T = mel.shape
        target_length = T * self.hop_length
        
        # Concatenate mel, f0, and voicing
        x = torch.cat([mel, f0, voicing], dim=1)  # (B, n_mels+2, T)
        
        # Initial expansion
        x_expanded = self.input_conv(x)  # (B, hidden_channels, T)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x_expanded = block(x_expanded)
        
        # Upsampling to waveform resolution
        x_upsampled = x_expanded
        for block in self.upsampling_blocks:
            x_upsampled = block(x_upsampled)
        
        # Final convolution to get waveform
        noise_wave = self.final_conv(x_upsampled)  # (B, 1, upsampled_length)
        
        # Ensure exact length match with target
        if noise_wave.shape[-1] != target_length:
            noise_wave = F.interpolate(
                noise_wave, 
                size=target_length, 
                mode='linear', 
                align_corners=False
            )
        
        # Apply high-pass filtering for breath characteristics
        noise_wave = self.high_pass(noise_wave)
        
        # Apply gating mechanism with voicing information
        gate = self.gate_conv(x)  # (B, 1, T)
        gate = F.interpolate(gate, size=target_length, mode='linear', align_corners=False)
        
        # Invert voicing for noise (more noise in unvoiced regions)
        voicing_upsampled = F.interpolate(voicing, size=target_length, mode='linear', align_corners=False)
        noise_gate = gate * (1 - voicing_upsampled)  # More noise where voicing is low
        
        noise_wave = noise_wave * noise_gate
        
        # Squeeze to (B, T*hop_length)
        noise_wave = noise_wave.squeeze(1)
        
        return noise_wave
    
    def _get_upsample_factors(self, target_factor):
        """Calculate upsampling factors to reach target_factor"""
        factors = []
        remaining = target_factor
        
        # We'll use factors close to powers of 2 for efficiency
        while remaining > 1:
            if remaining >= 8:
                factors.append(8)
                remaining = remaining / 8
            elif remaining >= 4:
                factors.append(4)
                remaining = remaining / 4
            elif remaining >= 2:
                factors.append(2)
                remaining = remaining / 2
            else:
                factors.append(int(math.ceil(remaining)))
                break
        
        # Adjust last factor to match exactly
        product = 1
        for f in factors:
            product *= f
        
        if product != target_factor and factors:
            factors[-1] = int(factors[-1] * target_factor / product)
        
        return factors


class ResidualBlock(nn.Module):
    """Residual block with dilated convolution"""
    def __init__(self, channels, dilation=1, kernel_size=3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, 
            dilation=dilation, padding=dilation * (kernel_size - 1) // 2
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, 
            dilation=dilation, padding=dilation * (kernel_size - 1) // 2
        )
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class UpsamplingBlock(nn.Module):
    """Upsampling block using transposed convolution"""
    def __init__(self, in_channels, out_channels, upsample_factor):
        super(UpsamplingBlock, self).__init__()
        
        # Calculate kernel size and stride for desired upsampling
        # For even upsampling factors: kernel_size = 2*factor, stride = factor
        # For odd upsampling factors: kernel_size = 2*factor-1, stride = factor
        kernel_size = 2 * upsample_factor if upsample_factor % 2 == 0 else 2 * upsample_factor - 1
        
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=upsample_factor,
            padding=(kernel_size - upsample_factor) // 2,
            output_padding=0 if upsample_factor % 2 == 0 else 0
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x