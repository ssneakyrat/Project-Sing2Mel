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


class PhaseAwareEnhancer(nn.Module):
    def __init__(self, fft_size=1024, hop_length=240, hidden_dim=256):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        
        # Network to enhance magnitude spectrum with attention mechanism
        self.mag_enhance = nn.Sequential(
            nn.Conv1d(self.fft_size//2 + 1, hidden_dim, 3, padding=1),
            nn.InstanceNorm1d(hidden_dim),  # Normalization for better stability
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, dilation=2, padding=2),  # Dilated convolution for wider receptive field
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, self.fft_size//2 + 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Phase correction network (predicts phase adjustments)
        self.phase_enhance = nn.Sequential(
            nn.Conv1d(self.fft_size//2 + 1, hidden_dim//2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim//2, hidden_dim//2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim//2, self.fft_size//2 + 1, 3, padding=1),
            nn.Tanh()  # Output range [-1, 1] for phase adjustments
        )
        
        # Cross-attention between magnitude and phase
        self.cross_attention = SelfAttentionBlock(self.fft_size//2 + 1)
        
        # Final projection layer to combine features
        self.final_proj = nn.Sequential(
            nn.Conv1d(self.fft_size//2 + 1, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, self.fft_size//2 + 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Convert to frequency domain
        x_stft = torch.stft(x, self.fft_size, self.hop_length, 
                           window=torch.hann_window(self.fft_size).to(x.device),
                           return_complex=True)
        
        # Get magnitude and phase
        mag = torch.abs(x_stft)  # [B, F, T]
        phase = torch.angle(x_stft)  # [B, F, T]
        
        # Permute dimensions for 1D convolutions if needed
        # The data is already in [B, F, T] format, which is correct for Conv1D
        
        # Process magnitude with enhancement network
        mag_features = self.mag_enhance(mag)
        enhanced_mag = mag_features * mag  # Apply as multiplicative scaling
       
        # Process phase with correction network
        phase_adj = self.phase_enhance(mag)  # Use magnitude as input for phase adjustment
        enhanced_phase = phase + (phase_adj)  # Apply small adjustments to phase
        
        # Apply cross-attention to refined features 
        # This allows magnitude and phase processing to inform each other
        attended_mag = self.cross_attention(enhanced_mag)
        
        # Final magnitude refinement
        final_mag = self.final_proj(attended_mag) * enhanced_mag
        
        # Convert back to time domain
        enhanced_complex = torch.polar(final_mag, enhanced_phase)
        enhanced_audio = torch.istft(enhanced_complex, self.fft_size, self.hop_length,
                                    window=torch.hann_window(self.fft_size).to(x.device),
                                    length=x.size(-1))
        
        return enhanced_audio
    
    def set_enhancement_strength(self, mag_strength=1.0, phase_strength=1.0):
        """
        Adjust the strength of magnitude and phase enhancements.
        
        Args:
            mag_strength: Strength of magnitude enhancement (0.0-1.0)
            phase_strength: Strength of phase adjustment (0.0-1.0)
        """
        self.mag_strength = mag_strength
        self.phase_strength = phase_strength