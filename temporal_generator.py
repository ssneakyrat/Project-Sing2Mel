import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn

class TemporalGenerator(nn.Module):
    def __init__(self, num_phonemes, n_mels=80, hidden_dim=256):
        super().__init__()
        
        # Phoneme embedding
        self.phoneme_encoder = nn.Embedding(num_phonemes + 1, 64)
        
        # Lightweight temporal convolutional network
        self.tcn = nn.Sequential(
            # Local features
            nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            # Medium-range features
            nn.Conv1d(128, 256, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            # Long-range features
            nn.Conv1d(256, hidden_dim, kernel_size=3, padding=8, dilation=8),
            nn.ReLU()
        )
        
        # Efficient mel projection
        self.mel_proj = nn.Conv1d(hidden_dim, n_mels, kernel_size=1)
        
    def forward(self, phoneme_seq):
        # [batch, seq_len] -> [batch, seq_len, 64]
        x = self.phoneme_encoder(phoneme_seq)
        
        # Transpose for conv1d: [batch, 64, seq_len]
        x = x.transpose(1, 2)
        
        # Apply temporal convolutions
        x = self.tcn(x)
        
        # Project to mel: [batch, n_mels, seq_len]
        mel = self.mel_proj(x)
        
        # Transpose back: [batch, seq_len, n_mels]
        mel = mel.transpose(1, 2)
        
        return mel