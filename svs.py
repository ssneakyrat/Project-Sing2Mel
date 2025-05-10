import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from dsp.harmonic_generator import HarmonicGenerator
from dsp.parameter_predictor import ParameterPredictor
from dsp.vocal_filter import VocalFilter
from dsp.noise_generator import NoiseGenerator

class SVS(nn.Module):
    def __init__(self, 
                 num_phonemes, 
                 num_singers, 
                 num_languages,
                 n_mels=80, 
                 n_harmonics=80,
                 n_noise_bands=8,
                 hop_length=240, 
                 win_length=1024,
                 n_fft=1024,
                 sample_rate=24000
                 ):
        super(SVS, self).__init__()
        
        # Basic parameters
        self.n_mels = n_mels
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        
        # Define embedding dimensions
        self.phoneme_embed_dim = 128
        self.singer_embed_dim = 16
        self.language_embed_dim = 8
        
        # Embedding layers
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, self.phoneme_embed_dim)
        self.singer_embed = nn.Embedding(num_singers, self.singer_embed_dim)
        self.language_embed = nn.Embedding(num_languages, self.language_embed_dim)
        
        # STFT Generator with specified number of harmonics
        self.stft_generator = HarmonicGenerator(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            sample_rate=self.sample_rate,
            n_harmonics=self.n_harmonics
        )
        
        # Create mel transform for converting STFT to mel
        self.register_buffer(
            'mel_basis',
            torch.from_numpy(
                torchaudio.transforms.MelScale(
                    n_mels=n_mels,
                    sample_rate=sample_rate,
                    f_min=0,
                    f_max=sample_rate/2,
                    n_stft=n_fft//2+1,
                ).fb.T.numpy()
            ).float()
        )
        
        # Updated parameter predictor with register awareness and more formants
        self.parameter_predictor = ParameterPredictor(
            phoneme_dim=self.phoneme_embed_dim,
            singer_dim=self.singer_embed_dim,
            language_dim=self.language_embed_dim,
            hidden_dim=256,
            num_formants=8,  # Increased from 5 to 8 to model higher formants
            num_harmonics=self.n_harmonics,
            n_noise_bands=self.n_noise_bands,
            use_lstm=True
        )
        
        # Add vocal filter
        self.vocal_filter = VocalFilter(
            n_fft=self.n_fft,
            sample_rate=self.sample_rate,
            use_parallel_filters=True,
            filter_mode='resonator'
        )
        
        # Add noise generator
        self.noise_generator = NoiseGenerator(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            sample_rate=self.sample_rate,
            n_bands=self.n_noise_bands
        )
        
    def forward(self, f0, phoneme_seq, singer_id, language_id, initial_phase=None):
        """
        Forward pass with dynamic harmonic amplitudes, formant filtering, and noise components.
        Now with register-aware formant adaptation and enhanced high-frequency modeling.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices)
            language_id: Language IDs [B] (indices)
            initial_phase: Optional initial phase
            
        Returns:
            signal: Audio signal [B, T_audio]
            predicted_mel: Mel-spectrogram [B, T, n_mels]
            mixed_stft: Combined harmonic and noise STFT [B, F, T]
            model_params: Dictionary with model parameters and losses
        """
        batch_size, n_frames = f0.shape[0], f0.shape[1]
        device = f0.device
        
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]
        
        # Predict parameters for both harmonic and noise components
        # Now includes register classification and formant regularization
        params = self.parameter_predictor(
            f0, 
            phoneme_emb, 
            singer_emb, 
            language_emb
        )
        
        # Generate harmonic STFT using STFTGenerator with dynamic harmonic amplitudes
        harmonic_stft = self.stft_generator(f0, params['harmonic_amplitudes'])
        
        # Extract formant filter parameters 
        filter_params = {
            'frequencies': params['frequencies'],
            'bandwidths': params['bandwidths'],
            'amplitudes': params['amplitudes']
        }
        
        # Apply formant filtering to the harmonic STFT
        filtered_stft = self.vocal_filter(harmonic_stft, filter_params)
        
        # Generate noise STFT using NoiseGenerator
        noise_params = {
            'noise_gain': params['noise_gain'],
            'spectral_shape': params['spectral_shape'],
            'voiced_mix': params['voiced_mix']
        }
        noise_stft = self.noise_generator(noise_params)
        
        # Mix harmonic and noise components
        # voiced_mix controls the ratio: 1 = fully voiced, 0 = fully unvoiced
        voiced_mix = params['voiced_mix'].transpose(1, 2)  # [B, 1, T]
        
        # Mix with proper broadcasting
        mixed_stft = filtered_stft * voiced_mix + noise_stft * (1.0 - voiced_mix)
        
        # Convert mixed STFT to audio using torch's inverse STFT
        window = torch.hann_window(self.win_length).to(device)
        signal = torch.istft(
            mixed_stft,
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=False
        )
        
        # Get magnitude of mixed STFT for mel conversion
        stft_mag = torch.abs(mixed_stft)  # Shape: [B, F, T]
        
        # Apply mel transformation
        mel_basis = self.mel_basis.to(device).transpose(0, 1)  # Shape: [F, M]
        
        # Apply matrix multiplication using einsum for better clarity
        predicted_mel = torch.einsum('bft,fm->bmt', stft_mag, mel_basis)
        
        # Apply log transformation for better dynamic range
        predicted_mel = torch.log(torch.clamp(predicted_mel, min=1e-5))
        
        # Transpose mel to expected [B, T, M] shape
        predicted_mel = predicted_mel.transpose(1, 2)
        
        # Create a dictionary of model parameters for analysis and visualization
        model_params = {
            'register_weights': params['register_weights'],
            'formant_frequencies': params['frequencies'],
            'formant_bandwidths': params['bandwidths'],
            'formant_amplitudes': params['amplitudes'],
            'regularization_loss': params['regularization_loss']
        }
        
        # Return all relevant outputs
        return signal, predicted_mel, mixed_stft, model_params