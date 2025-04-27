import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from harmonic_generator import HarmonicGenerator
from temporal_generator import TemporalGenerator
from noise_generator import NoiseGenerator
from mel_fusion import MelFusion

    
class SingingVoiceModel(nn.Module):
    def __init__(self, num_phonemes, num_singers, num_languages, n_mels=80, 
                 sample_rate=22050, f_min=0, f_max=8000, k_nearest=5, n_noise_bands=4):
        super(SingingVoiceModel, self).__init__()
        
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        
        # Embedding layers
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, 64)
        self.singer_embed = nn.Embedding(num_singers, 32)
        self.language_embed = nn.Embedding(num_languages, 32)
        
        # Optimized signal generator with sparse mel-aware encoding
        self.temporal_generator = HarmonicGenerator(
            n_mels=n_mels, 
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            k_nearest=k_nearest
        )
        
        # New melodic generator
        self.harmonic_generator = TemporalGenerator(
            num_phonemes=num_phonemes,
            n_mels=n_mels
        )
        
        # Efficient noise generator
        self.noise_generator = NoiseGenerator(n_mels=n_mels, n_bands=n_noise_bands)
        
        self.fusion_layer = MelFusion(
            n_mels=n_mels,
            reduction=4,
            attention_heads=1
        )
        
    def forward(self, phoneme_seq, f0, singer_id, language_id):
        # Generate signal from f0 using the OptimizedSignalGenerator
        temporal_mel = self.temporal_generator(f0)
        
        # Generate melodic features from phonemes
        harmonic_mel = self.harmonic_generator(phoneme_seq)
        
        # Generate noise component
        noise_mel = self.noise_generator(temporal_mel, harmonic_mel)
        
        combined_mel = self.fusion_layer(temporal_mel, harmonic_mel, noise_mel)
        
        return combined_mel, temporal_mel, harmonic_mel, noise_mel