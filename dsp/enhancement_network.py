import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv1d(dim, dim, 1)
        self.key = nn.Conv1d(dim, dim, 1)
        self.value = nn.Conv1d(dim, dim, 1)
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight for attention
        
    def forward(self, x):
        # x shape: [B, C, T]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # [B, T, T]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)  # [B, C, T]
        
        # Residual connection with learnable weight
        return x + self.gamma * out


class EnhancementNetwork(nn.Module):
    def __init__(self, fft_size=1024, hop_length=240, hidden_size=128, condition_dim=256):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length

        # Conditioning projection for parameter predictor features with normalization
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim),  # Add normalization at input
            nn.Linear(condition_dim, hidden_size),
            nn.LayerNorm(hidden_size),    # Add normalization after projection
            nn.LeakyReLU(0.1)
        )
        
        # Simplified conditioning - single projection to modulation values
        self.condition_modulation = nn.Sequential(
            nn.Linear(hidden_size, self.fft_size//2 + 1),
            nn.Tanh()  # Constrain to [-1, 1]
        )
        
        # Conditioning strength - gradually increase during training
        self.condition_strength = nn.Parameter(torch.tensor(0.01))  # Start very small
        
        # Network to enhance magnitude spectrum with attention mechanism
        self.mag_enhance = nn.Sequential(
            nn.Conv1d(self.fft_size//2 + 1, hidden_size, 3, padding=1),
            nn.InstanceNorm1d(hidden_size),  # Normalization for better stability
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),  # Dilated convolution for wider receptive field
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, self.fft_size//2 + 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Phase correction network (predicts phase adjustments)
        self.phase_enhance = nn.Sequential(
            nn.Conv1d(self.fft_size//2 + 1, hidden_size//2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size//2, hidden_size//2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size//2, self.fft_size//2 + 1, 3, padding=1),
            nn.Tanh()  # Output range [-1, 1] for phase adjustments
        )
        
        # Cross-attention between magnitude and phase
        self.cross_attention = SelfAttentionBlock(self.fft_size//2 + 1)
        
        # Final projection layer to combine features
        self.final_proj = nn.Sequential(
            nn.Conv1d(self.fft_size//2 + 1, hidden_size, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, self.fft_size//2 + 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, condition=None):
        """
        Forward pass with optional conservative conditioning from parameter predictor.
        
        Args:
            x: Input complex spectrogram [B, F, T]
            condition: Conditioning features from parameter predictor [B, T, condition_dim]
                       (Hidden features providing phonetic and singer context)
        
        Returns:
            enhanced_complex: Enhanced complex spectrogram [B, F, T]
        """
        x_stft = x
        
        # Get magnitude and phase
        mag = torch.abs(x_stft)  # [B, F, T]
        phase = torch.angle(x_stft)  # [B, F, T]
        
        # Apply conditioning if provided - now with a more conservative approach
        if condition is not None:
            # Apply LayerNorm before projection (within the Sequential module)
            condition_features = self.condition_projection(condition)  # [B, T, hidden_size]
            
            # Generate modulation with constrained range
            modulation = self.condition_modulation(condition_features)  # [B, T, F]
            
            # Reshape modulation to match magnitude shape [B, F, T]
            modulation = modulation.permute(0, 2, 1)
            
            # Apply very conservative additive conditioning with learnable strength
            # Use clipped strength parameter to ensure it stays positive but small
            strength = torch.clamp(self.condition_strength, 0.0, 0.3)
            
            # Apply as a residual connection - original mag + small modulation
            mag_delta = modulation * strength * 0.1  # Further reduce the effect
            mag = mag + mag_delta  # Simple additive conditioning
        
        # Process magnitude with enhancement network (unchanged)
        mag_features = self.mag_enhance(mag)
        enhanced_mag = mag_features * mag  # Apply as multiplicative scaling
        
        # Process phase with correction network (unchanged)
        phase_adj = self.phase_enhance(mag)  # Use magnitude as input for phase adjustment
        phase_adj = phase_adj / (torch.max(torch.abs(phase_adj)) + 1e-6)  # Normalize to [-1, 1]
        enhanced_phase = phase + phase_adj 
        
        # Apply cross-attention to refined features (unchanged)
        # This allows magnitude and phase processing to inform each other
        attended_mag = self.cross_attention(enhanced_mag)
        
        # Final magnitude refinement (unchanged)
        final_mag = self.final_proj(attended_mag) * enhanced_mag
        
        # Convert back to time domain (unchanged)
        enhanced_complex = torch.polar(final_mag, enhanced_phase)
        
        return enhanced_complex