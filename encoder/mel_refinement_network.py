import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MelRefinementNetwork(nn.Module):
    """
    Network to refine predicted mel spectrograms using linear layers with conditioning.
    """
    def __init__(self, n_mels, phoneme_embed_dim=128, singer_embed_dim=16, language_embed_dim=8, hidden_dim=256):
        super(MelRefinementNetwork, self).__init__()
        
        # Store dimensions for conditioning
        self.n_mels = n_mels
        self.phoneme_embed_dim = phoneme_embed_dim
        self.singer_embed_dim = singer_embed_dim
        self.language_embed_dim = language_embed_dim
        
        # Calculate total conditioning dimension
        self.conditioning_dim = phoneme_embed_dim + singer_embed_dim + language_embed_dim + 1  # +1 for f0
        
        # Simple architecture with linear layers and conditioning
        self.refine_net = nn.Sequential(
            nn.Linear(n_mels + self.conditioning_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mels)
        )
        
    def forward(self, mel, f0, phoneme_emb, singer_emb, language_emb):
        """
        Refine the predicted mel spectrogram with conditioning.
        
        Args:
            mel: Predicted mel spectrogram [B, T, n_mels]
            f0: Fundamental frequency trajectory [B, T]
            phoneme_emb: Phoneme embeddings [B, T, phoneme_dim]
            singer_emb: Singer embeddings [B, singer_dim]
            language_emb: Language embeddings [B, language_dim]
        
        Returns:
            Refined mel spectrogram [B, T, n_mels]
        """
        batch_size, n_frames, n_mels = mel.shape
        
        # Expand global conditioning to match time dimension
        singer_emb_expanded = singer_emb.unsqueeze(1).expand(-1, n_frames, -1)  # [B, T, singer_dim]
        language_emb_expanded = language_emb.unsqueeze(1).expand(-1, n_frames, -1)  # [B, T, language_dim]
        
        # Process f0
        f0 = f0.unsqueeze(-1)  # [B, T, 1]
        
        # Concatenate mel with conditioning
        conditioning = torch.cat([f0, phoneme_emb, singer_emb_expanded, language_emb_expanded], dim=-1)
        mel_with_conditioning = torch.cat([mel, conditioning], dim=-1)  # [B, T, n_mels + conditioning_dim]
        
        # Reshape for linear layer processing
        mel_reshaped = mel_with_conditioning.reshape(-1, mel_with_conditioning.size(-1))
        
        # Apply refinement
        refined_mel = self.refine_net(mel_reshaped)
        
        # Reshape back to original dimensions
        refined_mel = refined_mel.reshape(batch_size, n_frames, n_mels)
        
        # Add residual connection to learn refinements rather than entire spectrogram
        return mel + refined_mel