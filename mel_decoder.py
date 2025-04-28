import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from decoder.conditioning_network import ConditioningNetwork
from decoder.harmonic_synthesizer import HarmonicSynthesizer

class MelDecoder(nn.Module):
    """
    Optimized DDSP-based vocal synthesis model with harmonic, noise, and transient components
    - Improved memory efficiency
    - Vectorized operations
    - Consolidated processing
    - Reduced redundant computations
    """
    def __init__(self, num_phonemes, num_singers, num_languages, 
                 n_mels=80, hop_length=240, num_harmonics=100, sample_rate=24000):
        super(MelDecoder, self).__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # Embedding layers (efficient implementation)
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, 64)
        self.singer_embed = nn.Embedding(num_singers, 32)
        self.language_embed = nn.Embedding(num_languages, 32)
        
        # Conditioning network
        conditioning_hidden_dim = 128
        self.conditioning_network = ConditioningNetwork(
            n_mels=n_mels,
            num_phonemes=num_phonemes,
            hidden_dim=conditioning_hidden_dim
        )
        
        # Determine the actual output dimension from conditioning network
        # This depends on the ConditioningNetwork implementation
        # Default is typically 128, but could be different
        conditioning_output_dim = getattr(self.conditioning_network, 'output_dim', conditioning_hidden_dim)
        
        # Calculate total conditioning dimensions including potential embeddings
        total_conditioning_dims = conditioning_output_dim
        if hasattr(self, 'singer_embed'):
            total_conditioning_dims += self.singer_embed.embedding_dim  # 32
        if hasattr(self, 'language_embed'):
            total_conditioning_dims += self.language_embed.embedding_dim  # 32
        
        # DDSP Components with correct input channels
        self.harmonic_synth = HarmonicSynthesizer(
            sample_rate=sample_rate,
            hop_length=hop_length,
            num_harmonics=num_harmonics,
            input_channels=total_conditioning_dims  # Now accounts for all embeddings
        )
        
        # Precompute audio length multiplier for efficiency
        self.register_buffer('hop_length_tensor', torch.tensor(hop_length, dtype=torch.float))
        
    def forward(self, mel, f0, phoneme_seq, singer_id=None, language_id=None):
        """
        Args:
            mel: Mel spectrogram [B, T, n_mels] or [B, n_mels, T]
            f0: Fundamental frequency contour [B, T]
            phoneme_seq: Phoneme sequence [B, T]
            singer_id: Singer identity [B] (optional)
            language_id: Language identity [B] (optional)
        Returns:
            audio: Generated audio waveform [B, audio_length]
        """
        # Get batch size and sequence length
        batch_size = mel.shape[0]
        
        # Normalize inputs for better consistency
        # Standardize mel format to [B, n_mels, T]
        if mel.size(1) == self.n_mels:
            time_steps = mel.size(2)  # Already in [B, n_mels, T] format
        else:
            time_steps = mel.size(1)  # [B, T, n_mels] format
            mel = mel.transpose(1, 2)  # Convert to [B, n_mels, T]
        
        # Ensure f0 and phoneme_seq have matching time dimensions
        assert f0.size(1) == time_steps, "F0 time dimension must match mel spectrogram"
        assert phoneme_seq.size(1) == time_steps, "Phoneme sequence time dimension must match mel spectrogram"
        
        # Calculate target audio length once using precomputed tensor
        # Using float calculations before converting to integer for better stability
        audio_length = (time_steps * self.hop_length_tensor).long()
        
        # Generate conditioning information from inputs (vectorized)
        condition = self.conditioning_network(mel, f0, phoneme_seq)
        
        # Efficiently apply singer and language conditioning if provided
        if singer_id is not None:
            singer_embed = self.singer_embed(singer_id)  # [B, 32]
            # Expand to condition's time dimension and combine
            singer_embed = singer_embed.unsqueeze(2).expand(-1, -1, condition.size(2))
            condition = torch.cat([condition, singer_embed], dim=1)
            
        if language_id is not None:
            language_embed = self.language_embed(language_id)  # [B, 32]
            # Expand to condition's time dimension and combine
            language_embed = language_embed.unsqueeze(2).expand(-1, -1, condition.size(2))
            condition = torch.cat([condition, language_embed], dim=1)
        
        # Generate harmonic component using optimized synthesizer
        harmonic_signal = self.harmonic_synth(f0, condition, audio_length)
        
        return harmonic_signal
        
    def set_device(self, device):
        """
        Helper method to move all model components to the specified device
        """
        self.to(device)
        return self