import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HarmonicGenerator(nn.Module):
    def __init__(self, n_mels=80, sample_rate=22050, f_min=0, f_max=8000, k_nearest=5):
        super(HarmonicGenerator, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.k_nearest = k_nearest  # Number of nearest mel bins to consider
        
        # Pre-compute mel frequencies and their mel-scale values
        mel_freqs = self.mel_frequencies(n_mels, f_min, f_max)
        self.register_buffer('mel_freqs', torch.tensor(mel_freqs))
        self.register_buffer('mel_freqs_mel', self.hz_to_mel(self.mel_freqs))
        
        # Pre-compute normalization factors
        self.register_buffer('f_norm', torch.tensor(1.0 / f_max))
        self.register_buffer('mel_spacing', self.mel_freqs_mel[1] - self.mel_freqs_mel[0])
        
        # Optimized network architecture
        input_dim = 1 + k_nearest + 2 * k_nearest  # f0 + k mel indicators + 2*k sinusoidal
        
        self.f0_transform = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_mels)
        )
        
        # Skip connection for residual learning
        self.residual_proj = nn.Linear(input_dim, n_mels)
    
    @staticmethod
    def hz_to_mel(frequencies):
        """Vectorized Hz to mel conversion"""
        return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
    
    @staticmethod
    def mel_to_hz(mels):
        """Vectorized mel to Hz conversion"""
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)
    
    def mel_frequencies(self, n_mels, f_min, f_max):
        """Generate mel scale frequencies"""
        min_mel = self.hz_to_mel(torch.tensor(f_min)).item()
        max_mel = self.hz_to_mel(torch.tensor(f_max)).item()
        mels = torch.linspace(min_mel, max_mel, n_mels)
        return self.mel_to_hz(mels)
    
    def find_k_nearest_mels(self, f0_mel):
        """Find k nearest mel bins using efficient broadcasting"""
        # f0_mel: [batch, seq_len]
        # mel_freqs_mel: [n_mels]
        
        # Compute distances efficiently
        distances = torch.abs(
            f0_mel.unsqueeze(-1) - self.mel_freqs_mel.unsqueeze(0).unsqueeze(0)
        )  # [batch, seq_len, n_mels]
        
        # Find k nearest indices
        k_nearest_values, k_nearest_indices = torch.topk(
            distances, 
            k=self.k_nearest, 
            dim=-1, 
            largest=False
        )  # both: [batch, seq_len, k_nearest]
        
        return k_nearest_indices, k_nearest_values
    
    def generate_sparse_mel_encodings(self, f0):
        """Generate sparse mel-aware encodings"""
        batch_size, seq_len = f0.shape
        
        # Convert f0 to mel scale
        f0_mel = self.hz_to_mel(f0)
        
        # Find k nearest mel bins
        k_nearest_indices, k_nearest_distances = self.find_k_nearest_mels(f0_mel)
        
        # Adaptive sigma based on mel spacing
        sigma = self.mel_spacing * 0.5
        
        # Create sparse mel indicators
        mel_indicator = torch.exp(-k_nearest_distances**2 / (2 * sigma**2))
        
        # Generate sparse sinusoidal encodings
        phase = 2 * np.pi * f0.unsqueeze(-1) / self.sample_rate
        
        # Get frequencies for nearest mel bins
        nearest_mel_freqs = torch.gather(
            self.mel_freqs.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1),
            dim=2,
            index=k_nearest_indices
        )
        
        # Compute phase for nearest mel bins only
        phase_mel = phase * nearest_mel_freqs
        
        # Generate sin and cos encodings for nearest mel bins
        sin_enc = torch.sin(phase_mel)
        cos_enc = torch.cos(phase_mel)
        
        # Normalize f0
        f0_normalized = f0.unsqueeze(-1) * self.f_norm
        
        # Concatenate sparse features
        encodings = torch.cat([
            f0_normalized,     # Original f0 (normalized)
            mel_indicator,     # Sparse mel bin indicators
            sin_enc,          # Sparse sine encodings
            cos_enc           # Sparse cosine encodings
        ], dim=-1)
        
        return encodings, k_nearest_indices
    
    def forward(self, f0):
        """Transform f0 conditions to mel spectrogram dimension"""
        # Generate sparse mel-aware encodings
        encodings, nearest_indices = self.generate_sparse_mel_encodings(f0)
        
        # Transform to mel spectrogram dimension with residual connection
        main_out = self.f0_transform(encodings)
        residual = self.residual_proj(encodings)
        
        # Combine with residual connection
        predicted_mel = main_out + residual
        
        return predicted_mel