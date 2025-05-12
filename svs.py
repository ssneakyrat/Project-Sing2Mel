import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

from dsp.parameter_predictor import ParameterPredictor
from dsp.harmonic_generator import HarmonicGenerator

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
                 sample_rate=24000,
                 predict_adsr=True
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
        self.predict_adsr = predict_adsr
        
        # Define embedding dimensions
        self.phoneme_embed_dim = 128
        self.singer_embed_dim = 16
        self.language_embed_dim = 8
        
        # Embedding layers
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, self.phoneme_embed_dim)
        self.singer_embed = nn.Embedding(num_singers, self.singer_embed_dim)
        self.language_embed = nn.Embedding(num_languages, self.language_embed_dim)
        
        # Add parameter predictor with harmonic amplitude and noise parameter prediction
        self.parameter_predictor = ParameterPredictor(
            phoneme_dim=self.phoneme_embed_dim,
            singer_dim=self.singer_embed_dim,
            language_dim=self.language_embed_dim,
            hidden_dim=256,
            num_formants=5,
            num_harmonics=self.n_harmonics,
            n_noise_bands=self.n_noise_bands,
            predict_adsr=predict_adsr
        )
        
        # Add harmonic generator
        self.harmonic_generator = HarmonicGenerator(
            n_harmonics=self.n_harmonics,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            use_adsr=True
        )
        
        # Mel spectrogram transform for spectral loss computation
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=20.0,
            f_max=sample_rate // 2,
            n_mels=n_mels,
            power=1.0,  # Use amplitude spectrogram, not power
            norm="slaney",
            mel_scale="slaney"
        )

    def forward(self, f0, phoneme_seq, singer_id, language_id, note_on_frames=None, note_off_frames=None, duration=None, initial_phase=None):
        """
        Forward pass with dynamic harmonic amplitudes, formant filtering, and noise components.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices)
            language_id: Language IDs [B] (indices)
            note_on_frames: Frame indices for note onset [B, num_notes] or None
            note_off_frames: Frame indices for note offset [B, num_notes] or None
            duration: Optional duration in samples
            initial_phase: Optional initial phase
        """
        batch_size, n_frames = f0.shape
        device = f0.device
        
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]
        
        # Predict parameters for both harmonic and noise components
        params = self.parameter_predictor(
            f0, 
            phoneme_emb, 
            singer_emb, 
            language_emb
        )
        
        # Generate harmonic component
        harmonic_signal = self.harmonic_generator(
            f0,
            params,
            duration=duration,
            note_on_frames=note_on_frames,
            note_off_frames=note_off_frames,
            initial_phase=initial_phase
        )
        
        # TODO: Generate noise component
        # For now, we'll just use a placeholder
        if duration is None:
            duration = (n_frames + 1) * self.hop_length
            
        noise_signal = torch.zeros_like(harmonic_signal)
        
        # Mix harmonic and noise components based on voiced_mix parameter
        # Since we've already applied voiced_mix in the harmonic generator,
        # we need to apply (1 - voiced_mix) to the noise component
        
        # Interpolate voiced_mix to audio sample rate
        voiced_mix = params['voiced_mix']  # [B, T, 1]
        voiced_mix_transposed = voiced_mix.transpose(1, 2)  # [B, 1, T]
        
        interpolated_voice_mix = F.interpolate(
            voiced_mix_transposed,
            size=duration,
            mode='linear',
            align_corners=False
        )  # [B, 1, duration]
        
        # Apply noise gain and (1 - voiced_mix) to noise
        noise_gain = params['noise_gain']  # [B, T, 1]
        noise_gain_transposed = noise_gain.transpose(1, 2)  # [B, 1, T]
        
        interpolated_noise_gain = F.interpolate(
            noise_gain_transposed,
            size=duration,
            mode='linear',
            align_corners=False
        )  # [B, 1, duration]
        
        # Scale noise by gain and unvoiced factor (1 - voiced_mix)
        scaled_noise = noise_signal * interpolated_noise_gain.squeeze(1) * (1 - interpolated_voice_mix.squeeze(1))
        
        # Combine harmonic and noise signals
        signal = harmonic_signal + scaled_noise
        
        # Apply audio normalization to prevent clipping
        signal = torch.tanh(signal)
        
        # Generate mel spectrogram for loss computation
        with torch.no_grad():
            predicted_mel = self.mel_transform(signal)
            # Apply log scaling
            predicted_mel = torch.log(torch.clamp(predicted_mel, min=1e-5))
        
        # Return all relevant outputs
        return signal, predicted_mel