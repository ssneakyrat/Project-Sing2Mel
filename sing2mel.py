import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from decoder.core import upsample
from decoder.wave_generator_oscillator import WaveGeneratorOscillator
from encoder.mel_encoder import MelEncoder

# Noise conditioner network
class NoiseConditioner(nn.Module):
    def __init__(self, input_dim):
        super(NoiseConditioner, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1 to control noise magnitude
        )
    
    def forward(self, conditioning_features):
        return self.network(conditioning_features)
    
# Modified SVS class with MelEncoder integration
class Sing2Mel(nn.Module):
    """
    Lightweight DDSP-based singing voice synthesis model with separated
    expressive control prediction and signal processing components.
    """
    def __init__(self, 
                 num_phonemes, 
                 num_singers, 
                 num_languages,
                 n_mels=80, 
                 hop_length=240, 
                 sample_rate=24000,
                 num_harmonics=80, 
                 num_mag_harmonic=256,
                 num_mag_noise=80,
                 ):
        super(Sing2Mel, self).__init__()
        
        self.register_buffer("block_size", torch.tensor(hop_length))
        
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
        
        # Harmonic Synthesizer parameters
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, num_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        # Initialize harmonic synthesizer
        self.harmonic_synthesizer = WaveGeneratorOscillator(
            sample_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)

        # Initialize mel encoder
        self.mel_encoder = MelEncoder(
            n_mels=n_mels,
            n_harmonics=num_harmonics,
            sample_rate=sample_rate,
            phoneme_embed_dim=self.phoneme_embed_dim,
            singer_embed_dim=self.singer_embed_dim,
            language_embed_dim=self.language_embed_dim
        )
        
        # Initialize noise conditioner
        self.noise_conditioner = NoiseConditioner(
            self.phoneme_embed_dim + self.singer_embed_dim + self.language_embed_dim
        )

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
        """
        batch_size, n_frames = f0.shape[0], f0.shape[1]
        
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]

        # Prepare f0 for mel encoder
        f0_unsqueeze = f0.unsqueeze(2)  # [B, T, 1]

        # Process F0 - make sure it's in Hz and properly shaped
        f0_unsqueeze = torch.clamp(f0_unsqueeze, min=0.0, max=1000.0)
        f0_unsqueeze[f0_unsqueeze < 80] = 0 + 1e-7  # Set unvoiced regions to 0

        # upsample
        pitch = upsample(f0_unsqueeze, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthesizer(pitch, initial_phase)
        
        # Get global conditioning features for noise
        phoneme_features = phoneme_emb.mean(dim=1)  # Average over time dimension [B, phoneme_dim]
        conditioning_features = torch.cat([phoneme_features, singer_emb, language_emb], dim=1)  # [B, total_dim]

        # Get noise conditioning factor
        noise_conditioning = self.noise_conditioner(conditioning_features)  # [B, 1]
        
        # Generate base noise and apply conditioning
        base_noise = torch.rand_like(harmonic) * 2 - 1
        conditioned_noise = base_noise * noise_conditioning.view(batch_size, 1)
        
        # Add conditioned noise to harmonic
        harmonic = harmonic + conditioned_noise

        predicted_mel = self.mel_encoder(harmonic, f0, phoneme_emb, singer_emb, language_emb)

        # Return audio output, latent mel and final_phase
        return predicted_mel