import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConditioningNetwork(nn.Module):
    """Processes input features (mel, f0, phoneme) to generate conditioning vectors"""
    def __init__(self, n_mels=80, num_phonemes=100, hidden_dim=128):
        super(ConditioningNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        
        # Process mel spectrogram
        self.mel_conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        # Process F0
        self.f0_embed = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        # Process phoneme
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, hidden_dim // 4)
        
        # Combine all features
        self.combined_conv = nn.Sequential(
            nn.Conv1d(hidden_dim + hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, mel, f0, phoneme_seq):
        """
        Args:
            mel: Mel spectrogram [B, T, n_mels] or [B, n_mels, T]
            f0: Fundamental frequency contour [B, T]
            phoneme_seq: Phoneme sequence [B, T]
        Returns:
            condition: Conditioning vector [B, hidden_dim, T]
        """
        batch_size = mel.shape[0]
        
        # Check and fix mel dimensions if needed
        if mel.size(1) == self.n_mels:
            # Already in [B, n_mels, T] format
            pass
        else:
            # Convert from [B, T, n_mels] to [B, n_mels, T]
            mel = mel.transpose(1, 2)
        
        # Get time dimension from mel
        time_steps = mel.size(2)
        
        # Process mel
        mel_features = self.mel_conv(mel)  # [B, hidden_dim, T]
        
        # Process F0 - ensure it's in [B, 1, T] format
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)  # [B, 1, T]
        
        # Ensure f0 has the same sequence length as mel
        if f0.size(2) != time_steps:
            f0 = F.interpolate(f0, size=time_steps, mode='linear', align_corners=False)
            
        f0_features = self.f0_embed(f0)  # [B, hidden_dim//4, T]
        
        # Process phoneme - ensure it has the same sequence length as mel
        if phoneme_seq.size(1) != time_steps:
            # Interpolate phoneme indices (this isn't ideal but works for training)
            phoneme_seq_float = phoneme_seq.float().unsqueeze(1)  # [B, 1, phoneme_seq_len]
            phoneme_seq_resized = F.interpolate(
                phoneme_seq_float, 
                size=time_steps, 
                mode='nearest'
            ).squeeze(1).long()  # [B, T]
        else:
            phoneme_seq_resized = phoneme_seq
            
        # Embed phonemes
        phoneme_features = self.phoneme_embed(phoneme_seq_resized)  # [B, T, hidden_dim//4]
        phoneme_features = phoneme_features.transpose(1, 2)  # [B, hidden_dim//4, T]
        
        # Combine features
        combined = torch.cat([mel_features, f0_features, phoneme_features], dim=1)
        condition = self.combined_conv(combined)
        
        return condition