import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from decoder.feature_extractor import FeatureExtractor
from decoder.expressive_control import ExpressiveControl  # Import parameter predictor
from decoder.signal_processor import SignalProcessor  # Import new signal processor
from decoder.wave_generator_oscillator import WaveGeneratorOscillator
from decoder.core import scale_function, frequency_filter

class MelEncoder(nn.Module):
    """
    Encodes fundamental frequency and linguistic features into mel spectrograms.
    Uses simple linear layers to predict mel spectrograms from f0, phoneme, singer, and language embeddings.
    """
    def __init__(self, 
                 n_mels=80, 
                 phoneme_embed_dim=128, 
                 singer_embed_dim=16, 
                 language_embed_dim=8,
                 hidden_dim=512):
        super(MelEncoder, self).__init__()
        
        # Input dimensions
        self.n_mels = n_mels
        self.phoneme_embed_dim = phoneme_embed_dim
        self.singer_embed_dim = singer_embed_dim
        self.language_embed_dim = language_embed_dim
        
        # Calculate total input dimension
        f0_dim = 8  # Projected dimension for f0
        total_input_dim = phoneme_embed_dim + singer_embed_dim + language_embed_dim + f0_dim
        
        # F0 projection layer
        self.f0_projection = nn.Linear(1, f0_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
        ])
        
        # Residual connections
        self.residual_projections = nn.ModuleList([
            nn.Linear(total_input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, n_mels)
        
    def forward(self, f0, phoneme_emb, singer_emb, language_emb):
        """
        Forward pass to generate mel spectrogram from linguistic features.
        
        Args:
            f0: Fundamental frequency trajectory [B, T, 1]
            phoneme_emb: Phoneme embeddings [B, T, phoneme_embed_dim]
            singer_emb: Singer embeddings [B, singer_embed_dim]
            language_emb: Language embeddings [B, language_embed_dim]
            
        Returns:
            Predicted mel spectrogram [B, T, n_mels]
        """
        batch_size, seq_len = f0.shape[0], f0.shape[1]
        
        # Project f0
        f0_proj = self.f0_projection(f0)  # [B, T, f0_dim]
        
        # Expand singer and language embeddings to match sequence length
        singer_emb_expanded = singer_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, singer_embed_dim]
        language_emb_expanded = language_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, language_embed_dim]
        
        # Concatenate all inputs
        encoder_input = torch.cat([phoneme_emb, f0_proj, singer_emb_expanded, language_emb_expanded], dim=-1)
        
        # Process through encoder layers with residual connections
        x = encoder_input
        residual = self.residual_projections[0](x)
        x = self.encoder_layers[0](x) + residual
        
        residual = self.residual_projections[1](x)
        x = self.encoder_layers[1](x) + residual
        
        x = self.encoder_layers[2](x) + x  # Self-residual for the last layer
        
        # Project to mel spectrogram
        mel_pred = torch.sigmoid(self.output_projection(x))  # Apply sigmoid to constrain values
        
        return mel_pred

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
                 n_mels=80, 
                 hop_length=240, 
                 sample_rate=24000,
                 num_harmonics=80, 
                 num_mag_harmonic=256,
                 num_mag_noise=80,
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
        self.harmonic_synthesizer = WaveGeneratorOscillator(
            sample_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)
        
        # Initialize expressive control for parameter prediction
        self.expressive_control = ExpressiveControl(input_dim=256, sample_rate=sample_rate)
        
        # Initialize signal processor for audio effects
        self.signal_processor = SignalProcessor(sample_rate=sample_rate)
        
        # Initialize mel encoder
        self.mel_encoder = MelEncoder(
            n_mels=n_mels,
            phoneme_embed_dim=self.phoneme_embed_dim,
            singer_embed_dim=self.singer_embed_dim,
            language_embed_dim=self.language_embed_dim
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
        predicted_mel = self.mel_encoder(f0_unsqueeze, phoneme_emb, singer_emb, language_emb)

        # Get control parameters from feature extractor
        ctrls = self.feature_extractor(predicted_mel, f0, phoneme_emb, singer_emb, language_emb)

        # Process harmonic and noise parameters
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # Use harmonic magnitude as conditioning for expressive control
        conditioning = ctrls['harmonic_magnitude']  # [B, T, 256]

        # Process F0 - make sure it's in Hz and properly shaped
        f0_unsqueeze = torch.clamp(f0_unsqueeze, min=0.0, max=1000.0)
        f0_unsqueeze[f0_unsqueeze < 80] = 0 + 1e-7  # Set unvoiced regions to 0
        
        # Get expressive control parameters using the parameter predictor
        expressive_params = self.expressive_control(conditioning)

        # Create time indices for vibrato calculation
        time_idx = torch.arange(n_frames, device=predicted_mel.device).float().unsqueeze(0).expand(batch_size, -1) / 100.0
        
        # Apply vibrato to F0 at frame rate using the signal processor
        f0_with_vibrato = self.signal_processor.apply_vibrato(
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
        
        # Apply all audio effects using the signal processor
        output, _ = self.signal_processor.process_audio(
            harmonic, noise, f0_with_vibrato, time_idx, expressive_params
        )
            
        # Return both the audio output and the expressive parameters
        return output, expressive_params, predicted_mel