import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from decoder.conditioning_network import ConditioningNetwork
from decoder.fusion_network import FusionNetwork
from decoder.harmonic_synthesizer import HarmonicSynthesizer
from decoder.noise_generator import FilteredNoiseGenerator

class MelDecoder(nn.Module):
    """DDSP-based vocal synthesis model with harmonic, noise, and transient components"""
    def __init__(self, num_phonemes, num_singers, num_languages, 
                 n_mels=80, hop_length=240, num_harmonics=100, sample_rate=24000):
        super(MelDecoder, self).__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # Embedding layers (kept for potential future use)
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, 64)
        self.singer_embed = nn.Embedding(num_singers, 32)
        self.language_embed = nn.Embedding(num_languages, 32)
        
        # Conditioning network - processes inputs into a unified representation
        self.conditioning_network = ConditioningNetwork(
            n_mels=n_mels,
            num_phonemes=num_phonemes,
            hidden_dim=128
        )
        
        # DDSP Components
        self.harmonic_synth = HarmonicSynthesizer(
            sample_rate=sample_rate,
            hop_length=hop_length,
            num_harmonics=num_harmonics
        )
        
        self.noise_gen = FilteredNoiseGenerator(
            n_fft=1024,
            hop_length=hop_length,
            sample_rate=sample_rate
        )
        
        # Fusion network - combines waveforms with learned weights
        self.fusion_network = FusionNetwork(num_components=2)  # Updated to handle 3 components
        
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
        
        # Determine time steps based on mel shape
        if mel.size(1) == self.n_mels:
            time_steps = mel.size(2)  # [B, n_mels, T] format
        else:
            time_steps = mel.size(1)  # [B, T, n_mels] format
        
        # Calculate target audio length
        audio_length = time_steps * self.hop_length
        
        # Generate conditioning information from inputs
        condition = self.conditioning_network(mel, f0, phoneme_seq)
        
        # Generate harmonic component (pitched/tonal sounds)
        harmonic_signal = self.harmonic_synth(f0, condition, audio_length)
        
        # Generate noise component (unpitched sounds, breath, turbulence)
        noise_signal = self.noise_gen(condition, audio_length)
        
        # Combine components using fusion network
        components = [harmonic_signal, noise_signal]
        audio = self.fusion_network(components, condition)
        
        return audio