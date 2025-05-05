import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from direct_feature_encoder import DirectFeatureEncoder
from decoder.wave_generator_oscillator import WaveGeneratorOscillator
from decoder.core import scale_function, frequency_filter, upsample
from decoder.enhancement_network import PhaseAwareEnhancer
    
class DirectAudio(nn.Module):
    """
    Lightweight DDSP-based singing voice synthesis model with a unified
    neural network for direct control parameter generation.
    """
    def __init__(self, 
                 num_phonemes, 
                 num_singers, 
                 num_languages,
                 hop_length=240, 
                 sample_rate=24000,
                 num_harmonics=80, 
                 num_mag_harmonic=256,
                 num_mag_noise=80,
                 hidden_dim=256,
                 num_heads=4,
                 num_transformer_layers=3,
                 num_conv_blocks=3):
        super(DirectAudio, self).__init__()
        
        # Basic parameters
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        
        # Define embedding dimensions
        self.phoneme_embed_dim = 128
        self.singer_embed_dim = 16
        self.language_embed_dim = 8
        
        # Embedding layers
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, self.phoneme_embed_dim)
        self.singer_embed = nn.Embedding(num_singers, self.singer_embed_dim)
        self.language_embed = nn.Embedding(num_languages, self.language_embed_dim)   

        # Register buffers for use in forward pass
        self.register_buffer("sampling_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(hop_length))
        
        # Define control parameter output configuration
        self.output_splits = {
            'harmonic_magnitude': num_mag_harmonic,
            'noise_magnitude': num_mag_noise
        }

        # Initialize the unified DirectFeatureEncoder
        self.direct_encoder = DirectFeatureEncoder(
            output_splits=self.output_splits,
            phoneme_embed_dim=self.phoneme_embed_dim,
            singer_embed_dim=self.singer_embed_dim,
            language_embed_dim=self.language_embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            num_conv_blocks=num_conv_blocks
        )

        # Harmonic Synthesizer parameters
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, num_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        # Initialize harmonic synthesizer
        self.harmonic_synthesizer = WaveGeneratorOscillator(
            sample_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)

        # Audio enhancement network
        self.refinement = PhaseAwareEnhancer()

    def forward(self, f0, phoneme_seq, singer_id, language_id, initial_phase=None):
        """
        Forward pass with direct control parameter generation.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices)
            language_id: Language IDs [B] (indices)
            initial_phase: Optional initial phase for the harmonic oscillator
            
        Returns:
            Audio signal [B, T*hop_length], control parameters dict
        """
        batch_size, n_frames = f0.shape[0], f0.shape[1]
        
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]

        # Ensure f0 has correct shape
        f0_unsqueeze = f0.unsqueeze(2)  # [B, T, 1]
        
        # Get control parameters directly from the unified encoder
        ctrls = self.direct_encoder(f0_unsqueeze, phoneme_emb, singer_emb, language_emb)

        # Process harmonic and noise parameters
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # Process F0 - make sure it's in Hz and properly shaped
        f0_unsqueeze = torch.clamp(f0_unsqueeze, min=0.0, max=1000.0)
        f0_unsqueeze[f0_unsqueeze < 80] = 0 + 1e-7  # Set unvoiced regions to 0

        # Upsample f0 to match audio sample rate
        pitch = upsample(f0_unsqueeze, self.block_size)

        # Generate harmonic component
        harmonic, final_phase = self.harmonic_synthesizer(pitch, initial_phase)
        
        # Apply frequency filtering to harmonic component
        harmonic = frequency_filter(harmonic, src_param)

        # Generate and filter noise component
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(noise, noise_param)
        
        # Combine harmonic and noise components
        signal = harmonic + noise       

        # Apply refinement network
        signal = self.refinement(signal)

        # Return audio output and control parameters
        return signal, final_phase