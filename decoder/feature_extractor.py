import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ConvBlock(nn.Module):
    """Basic convolutional block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=dilation * (kernel_size - 1) // 2, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=dilation * (kernel_size - 1) // 2, dilation=dilation
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection if dimensions don't match
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x, 0.1)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x + residual, 0.1)
        
        return x

class F0RefinementNetwork(nn.Module):
    def __init__(
        self,
        n_mels=80,
        hidden_dim=256,
        f0_embed_dim=64,
        n_blocks=3,
        output_dim=1
    ):
        super(F0RefinementNetwork, self).__init__()
        
        # Mel encoder
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # F0 embedding
        self.f0_encoder = nn.Sequential(
            nn.Conv1d(1, f0_embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(f0_embed_dim),
            nn.LeakyReLU(0.1),
            nn.Conv1d(f0_embed_dim, f0_embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(f0_embed_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim + f0_embed_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Refinement blocks
        self.refinement_blocks = nn.ModuleList([
            ConvBlock(
                hidden_dim, hidden_dim, 
                kernel_size=3, dilation=2**i % 5 + 1
            ) for i in range(n_blocks)
        ])
        
        # Output heads
        self.f0_correction_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=3, padding=1),
            nn.Tanh()  # Output scaled to [-1, 1] for additive correction
        )
        
        self.voicing_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range for voicing probability
        )
        
        # Additional features output (matching the original FeatureExtractor)
        self.features_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim // 2, 4, kernel_size=3, padding=1),
            nn.Tanh()  # Additional features in [-1, 1] range
        )
        
    def forward(self, mel, f0):
        # Handle different possible input shapes for mel
        if len(mel.shape) == 3:
            if mel.shape[1] != 80:  # If mel is [B, T, n_mels]
                mel = mel.transpose(1, 2)  # Convert to [B, n_mels, T]
        else:
            raise ValueError(f"Expected mel to have 3 dimensions, got shape {mel.shape}")
        
        # Ensure f0 has shape [B, 1, T]
        if len(f0.shape) == 2:
            f0 = f0.unsqueeze(1)
        elif f0.shape[1] != 1:
            raise ValueError(f"Expected f0 to have shape [B, 1, T], got {f0.shape}")
        
        # Extract features from mel and f0
        mel_features = self.mel_encoder(mel)
        f0_embedding = self.f0_encoder(f0)
        
        # Fuse features
        fused = torch.cat([mel_features, f0_embedding], dim=1)
        fused = self.fusion(fused)
        
        # Process through refinement blocks
        x = fused
        for block in self.refinement_blocks:
            x = block(x)
        
        # Generate outputs
        f0_correction = self.f0_correction_head(x)
        voicing = self.voicing_head(x)
        features = self.features_head(x)
        
        # Apply correction to input f0
        # Scaling factor to limit the magnitude of corrections
        # You can adjust this value based on your data characteristics
        correction_scale = 0.2  
        refined_f0 = f0 + (f0 * f0_correction * correction_scale)
        
        return refined_f0, voicing, features

class FeatureExtractor(nn.Module):
    """Combines the original FeatureExtractor with the F0RefinementNetwork"""
    def __init__(self, n_mels=80, hidden_dim=256, sample_rate=24000, hop_length=240, n_fft=2048):
        super(FeatureExtractor, self).__init__()
        
        # F0 refinement network
        self.f0_refinement = F0RefinementNetwork(
            n_mels=n_mels,
            hidden_dim=hidden_dim
        )
    
    def forward(self, mel, f0):
        # Get refined f0, voicing, and features
        refined_f0, voicing, features = self.f0_refinement(mel, f0)
        
        return refined_f0, voicing, features
