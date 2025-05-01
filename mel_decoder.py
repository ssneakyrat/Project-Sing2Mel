import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from decoder.feature_extractor import FeatureExtractor
from decoder.expressive_control import ExpressiveControl  # Import optimized version
from decoder.harmonic_oscillator import HarmonicOscillator
from decoder.wave_generator_oscillator import WaveGeneratorOscillator
from decoder.core import scale_function, frequency_filter

class MelDecoder(nn.Module):
    """
    Lightweight DDSP-based singing voice synthesis model with optimized expressive control.
    """
    def __init__(self, 
                 num_phonemes, 
                 num_singers, 
                 num_languages,
                 n_mels=80, 
                 hop_length=240, 
                 sample_rate=24000,
                 num_harmonics=150, 
                 num_mag_harmonic=256,
                 num_mag_noise=80,
                 n_fft=1024):
        super(MelDecoder, self).__init__()
        
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
        self.harmonic_synthesizer = WaveGeneratorOscillator(
            sample_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)
        
        # Initialize optimized expressive control
        self.expressive_control = ExpressiveControl(input_dim=256, sample_rate=sample_rate)

    def forward(self, mel, f0, phoneme_seq, singer_id, language_id, initial_phase=None):
        """
        Optimized forward pass with combined expressive control processing.
        
        Args:
            mel: Mel-spectrogram [B, T, n_mels]
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices)
            language_id: Language IDs [B] (indices)
            initial_phase: Optional initial phase for the harmonic oscillator
            
        Returns:
            Audio signal [B, T*hop_length]
        """
        batch_size, n_frames = mel.shape[0], mel.shape[1]
        
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]

        # Get control parameters from feature extractor
        ctrls = self.feature_extractor(mel, f0, phoneme_emb, singer_emb, language_emb)

        # Process harmonic and noise parameters
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # Use harmonic magnitude as conditioning for expressive control
        conditioning = ctrls['harmonic_magnitude']  # [B, T, 256]

        # Process F0 - make sure it's in Hz and properly shaped
        f0_unsqueeze = f0.unsqueeze(2)  # [B, T, 1]
        f0_unsqueeze = torch.clamp(f0_unsqueeze, min=0.0, max=1000.0)
        f0_unsqueeze[f0_unsqueeze < 80] = 0 + 1e-7  # Set unvoiced regions to 0
        pitch = f0_unsqueeze
        
        # Get expressive control parameters
        expressive_params = self.expressive_control(conditioning)

        # Create time indices for vibrato calculation - more efficient
        time_idx = torch.arange(n_frames, device=mel.device).float().unsqueeze(0).expand(batch_size, -1) / 100.0
        
        # OPTIMIZATION: Apply vibrato at frame rate first
        # Ensure time_idx matches the shape of f0 and expressive parameters
        if time_idx.shape[1] != n_frames or time_idx.shape[1] != expressive_params['vibrato_rate'].shape[1]:
            time_idx = torch.arange(n_frames, device=mel.device).float().unsqueeze(0).expand(batch_size, -1) / 100.0
        
        # Apply vibrato to F0 at frame rate
        f0_with_vibrato = self.expressive_control.apply_vibrato(
            f0_unsqueeze, time_idx, expressive_params
        )
        
        # Upsample to audio rate for synthesis
        f0_upsampled = F.interpolate(
            f0_with_vibrato.transpose(1, 2), 
            size=n_frames * self.hop_length, 
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        # Generate harmonic component
        harmonic, final_phase = self.harmonic_synthesizer(f0_upsampled.squeeze(1), initial_phase)
        harmonic = frequency_filter(harmonic, src_param)

        # Generate noise component
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(noise, noise_param)
        
        # OPTIMIZATION: Apply all expressive effects in one combined operation
        # This replaces separate calls to apply_tension, apply_vocal_fry, and apply_breathiness
        # The process_audio method now handles all shape alignment internally
        output, _ = self.expressive_control.process_audio(
            harmonic, noise, f0_with_vibrato, time_idx, expressive_params
        )
            
        return output