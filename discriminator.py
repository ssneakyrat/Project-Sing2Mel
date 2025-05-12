import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfModulatedBatchNorm1d(nn.Module):
    """Self-modulated batch normalization for 1D data"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False, eps=eps, momentum=momentum)
        self.gamma_conv = nn.Conv1d(num_features, num_features, kernel_size=1, padding=0, bias=True)
        self.beta_conv = nn.Conv1d(num_features, num_features, kernel_size=1, padding=0, bias=True)
        
    def forward(self, x):
        # Apply regular BatchNorm without affine transformation
        x_normalized = self.bn(x)
        
        # Generate gamma and beta for self-modulation
        gamma = self.gamma_conv(x)
        beta = self.beta_conv(x)
        
        # Apply modulation
        return gamma * x_normalized + beta

class SelfModulatedBatchNorm2d(nn.Module):
    """Self-modulated batch normalization for 2D data (spectrograms)"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
        self.gamma_conv = nn.Conv2d(num_features, num_features, kernel_size=1, padding=0, bias=True)
        self.beta_conv = nn.Conv2d(num_features, num_features, kernel_size=1, padding=0, bias=True)
        
    def forward(self, x):
        # Apply regular BatchNorm without affine transformation
        x_normalized = self.bn(x)
        
        # Generate gamma and beta for self-modulation
        gamma = self.gamma_conv(x)
        beta = self.beta_conv(x)
        
        # Apply modulation
        return gamma * x_normalized + beta

class SkipLayerExcitation(nn.Module):
    """Skip-layer excitation module as used in FastGAN"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x, skip):
        b, c, _ = x.size()
        # Get the dimensions of skip for correct reshaping
        b_skip, c_skip, _ = skip.size()
        y = self.avg_pool(skip).view(b_skip, c_skip)
        # If channel dimensions differ, project skip features to match x's channels
        if c_skip != c:
            # Simple projection via a fully connected layer
            projection = nn.Linear(c_skip, c, bias=False).to(x.device)
            y = projection(y)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SkipLayerExcitation2D(nn.Module):
    """Skip-layer excitation module for 2D data (spectrograms)"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x, skip):
        b, c, _, _ = x.size()
        # Get the dimensions of skip for correct reshaping
        b_skip, c_skip, _, _ = skip.size()
        y = self.avg_pool(skip).view(b_skip, c_skip)
        # If channel dimensions differ, project skip features to match x's channels
        if c_skip != c:
            # Simple projection via a fully connected layer
            projection = nn.Linear(c_skip, c, bias=False).to(x.device)
            y = projection(y)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class WaveDiscriminatorBlock(nn.Module):
    """Basic building block for the waveform discriminator"""
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=4, padding=7):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = SelfModulatedBatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class SpectrogramDiscriminatorBlock(nn.Module):
    """Basic building block for the spectrogram discriminator"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = SelfModulatedBatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class SelfAttention(nn.Module):
    """Self-attention layer for 2D data"""
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Project to query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention to values
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Apply residual connection with learnable weight
        return self.gamma * out + x

class WaveformDiscriminator(nn.Module):
    """Multi-scale discriminator operating on raw waveform"""
    def __init__(self, scales=[1, 0.5, 0.25], base_channels=64, max_channels=1024):
        super().__init__()
        self.scales = scales
        self.discriminators = nn.ModuleList()
        
        for scale in scales:
            # Create a discriminator for each scale
            layers = []
            # Initial layer
            layers.append(nn.Conv1d(1, base_channels, kernel_size=15, stride=4, padding=7))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Increasing channel dimensions
            current_channels = base_channels
            for i in range(4):  # Four downsampling layers
                next_channels = min(current_channels * 2, max_channels)
                layers.append(WaveDiscriminatorBlock(current_channels, next_channels))
                current_channels = next_channels
            
            # Final classification layer
            layers.append(nn.Conv1d(current_channels, 1, kernel_size=3, stride=1, padding=1))
            
            self.discriminators.append(nn.ModuleList(layers))
            
        # Skip-layer excitation modules
        self.skip_excite = SkipLayerExcitation(current_channels)
            
    def forward(self, x):
        """
        Forward pass through all discriminator scales
        
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            outputs: List of discriminator outputs
            features: List of intermediate features for feature matching loss
        """
        outputs = []
        features = []
        
        for scale_idx, scale in enumerate(self.scales):
            # Downsample if needed
            if scale != 1:
                # Simple linear interpolation for downsampling
                curr_x = F.interpolate(x, scale_factor=scale, mode='linear', align_corners=False)
            else:
                curr_x = x
                
            # Forward pass through this scale's discriminator
            disc = self.discriminators[scale_idx]
            skip_features = []
            
            for i, layer in enumerate(disc[:-1]):  # All layers except final
                curr_x = layer(curr_x)
                # Save intermediate features for skip connections and feature matching
                if i % 2 == 0 and i > 0:  # Save features after activation
                    skip_features.append(curr_x)
                    features.append(curr_x)
            
            # Apply skip-layer excitation on final features if we have skip features
            if len(skip_features) > 0:
                # Use the earliest skip feature for excitation
                curr_x = self.skip_excite(curr_x, skip_features[0])
                
            # Final classification layer
            outputs.append(disc[-1](curr_x))
            
        return outputs, features

class SpectrogramDiscriminator(nn.Module):
    """Discriminator operating on mel spectrograms"""
    def __init__(self, n_mels, base_channels=32, max_channels=512):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Initial layer
        self.layers.append(nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Increasing channel dimensions
        current_channels = base_channels
        for i in range(4):  # Four downsampling layers
            next_channels = min(current_channels * 2, max_channels)
            self.layers.append(SpectrogramDiscriminatorBlock(current_channels, next_channels))
            current_channels = next_channels
            
            # Add self-attention after the second block
            if i == 1:
                self.layers.append(SelfAttention(current_channels))
        
        # Skip-layer excitation
        self.skip_excite = SkipLayerExcitation2D(current_channels)
        
        # Final classification layer
        self.layers.append(nn.Conv2d(current_channels, 1, kernel_size=3, stride=1, padding=1))
        
    def forward(self, x):
        """
        Forward pass through the spectrogram discriminator
        
        Args:
            x: Input mel spectrogram [B, 1, n_mels, T]
            
        Returns:
            output: Discriminator output
            features: Intermediate features for feature matching loss
        """
        features = []
        skip_features = []
        
        for i, layer in enumerate(self.layers[:-1]):  # All layers except final
            x = layer(x)
            # Save intermediate features for skip connections and feature matching
            if i % 2 == 0 and i > 0 and not isinstance(layer, SelfAttention):  # Save features after activation
                skip_features.append(x)
                features.append(x)
        
        # Apply skip-layer excitation on final features
        if len(skip_features) > 0:
            # Use the earliest skip feature for excitation
            x = self.skip_excite(x, skip_features[0])
            
        # Final classification layer
        output = self.layers[-1](x)
        
        return output, features

class FastGANDiscriminator(nn.Module):
    """Combined waveform and spectrogram discriminator"""
    def __init__(self, n_mels, sample_rate, mel_transform=None):
        super().__init__()
        self.wave_disc = WaveformDiscriminator()
        self.spec_disc = SpectrogramDiscriminator(n_mels)
        self.mel_transform = mel_transform
        self.sample_rate = sample_rate
        
    def forward(self, wave):
        """
        Forward pass through both discriminators
        
        Args:
            wave: Input waveform [B, 1, T]
            
        Returns:
            outputs: Dictionary with all discriminator outputs
            features: Dictionary with all features for feature matching
        """
        outputs = {}
        features = {}
        
        # Waveform discriminator
        wave_outputs, wave_features = self.wave_disc(wave)
        outputs['wave'] = wave_outputs
        features['wave'] = wave_features
        
        # Generate mel spectrogram if needed
        if self.mel_transform is not None:
            # Create a mel spectrogram from the waveform
            mel = self.mel_transform(wave.squeeze(1))  # [B, n_mels, T]
            mel = torch.log(mel + 1e-7)  # Log mel
            mel = mel.unsqueeze(1)  # [B, 1, n_mels, T]
            
            # Spectrogram discriminator
            spec_output, spec_features = self.spec_disc(mel)
            outputs['spec'] = spec_output
            features['spec'] = spec_features
        
        return outputs, features