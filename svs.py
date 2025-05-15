import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoder.latent_encoder import LatentEncoder
from decoder.feature_extractor import FeatureExtractor
from decoder.harmonic_oscillator import HarmonicOscillator
from decoder.core import scale_function, frequency_filter, upsample
from decoder.vocal_filter import vocal_frequency_filter
from decoder.phaser_network import PhaseAwareEnhancer

# Modified SVS class with MelEncoder integration
class SVS(nn.Module):
    """
    Lightweight DDSP-based singing voice synthesis model with separated
    expressive control prediction and signal processing components.
    """
    def __init__(self, 
                 num_phonemes, 
                 num_singers, 
                 num_languages,
                 n_mels, 
                 hop_length, 
                 sample_rate,
                 num_harmonics, 
                 num_mag_harmonic,
                 num_mag_noise,
                 ):
        super(SVS, self).__init__()
        
        # Basic parameters
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        self.num_mag_harmonic = num_mag_harmonic
        self.num_mag_noise = num_mag_noise
        
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
        
        # Define feature extractor output splits
        split_map = {
            'harmonic_magnitude': num_mag_harmonic,
            'noise_magnitude': num_mag_noise
        }

        # Initialize feature extractor with proper dimensions
        self.feature_extractor = FeatureExtractor(
            input_channel=n_mels,
            output_splits=split_map,
            phoneme_dim=self.phoneme_embed_dim,
            singer_dim=self.singer_embed_dim,
            language_dim=self.language_embed_dim
        )

        # Harmonic Synthesizer parameters
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, num_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        # Initialize harmonic synthesizer
        self.harmonic_synthesizer = HarmonicOscillator(
            sample_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)
        
        # Initialize mel encoder
        self.encoder = LatentEncoder(
            n_mels=n_mels,
            phoneme_embed_dim=self.phoneme_embed_dim,
            singer_embed_dim=self.singer_embed_dim,
            language_embed_dim=self.language_embed_dim
        )

        self.harmonic_phaser = PhaseAwareEnhancer(hidden_dim=512)
        self.noise_phaser = PhaseAwareEnhancer(hidden_dim=256)

    def forward(self, f0, phoneme_seq, singer_id, language_id, initial_phase=None):
        """
        Forward pass with separated expressive control and signal processing.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices)
            language_id: Language IDs [B] (indices)
            mel: Optional mel-spectrogram [B, T, n_mels] (if None, it will be predicted)
            initial_phase: Optional initial phase for the harmonic oscillator
            
        Returns:
            Audio signal [B, T*hop_length], expressive parameters dict
        """
        batch_size, n_frames = f0.shape[0], f0.shape[1]
        
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]

        # Prepare f0 for mel encoder
        f0_unsqueeze = f0.unsqueeze(2)  # [B, T, 1]
        
        # Generate mel spectrogram if not provided
        predicted_mel = self.encoder(f0_unsqueeze, phoneme_emb, singer_emb, language_emb)

        # Get control parameters from feature extractor
        ctrls = self.feature_extractor(predicted_mel, f0, phoneme_emb, singer_emb, language_emb)

        # Process harmonic and noise parameters
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # Process F0 - make sure it's in Hz and properly shaped
        f0_unsqueeze = torch.clamp(f0_unsqueeze, min=0.0, max=1000.0)
        f0_unsqueeze[f0_unsqueeze < 80] = 0 + 1e-7  # Set unvoiced regions to 0

        # upsample
        pitch = upsample(f0_unsqueeze, self.block_size)
        
        # harmonic
        harmonic, final_phase = self.harmonic_synthesizer(pitch, initial_phase)
        harmonic = vocal_frequency_filter(
            harmonic, 
            src_param, 
            gender="neutral",  # Or dynamically set based on singer
            formant_emphasis=False,
            vocal_range_boost=False,
            breathiness=0,
            multi_resolution=False
        )

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = vocal_frequency_filter(
            noise, 
            noise_param, 
            gender="neutral",  # Or dynamically set based on singer
            formant_emphasis=False,
            vocal_range_boost=False,
            breathiness=0,
            multi_resolution=False
        )
        
        harmonic = self.harmonic_phaser(harmonic)
        noise = self.noise_phaser(noise)

        signal = harmonic + noise

        # Return both the audio output and the expressive parameters
        return signal, predicted_mel