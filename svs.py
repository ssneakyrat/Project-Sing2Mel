import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from dsp.stft_generator import STFTGenerator
from dsp.parameter_predictor import ParameterPredictor
from dsp.vocal_filter import VocalFilter

class SVS(nn.Module):
    def __init__(self, 
                 num_phonemes, 
                 num_singers, 
                 num_languages,
                 n_mels=80, 
                 hop_length=240, 
                 win_length=1024,
                 n_fft=1024,
                 sample_rate=24000
                 ):
        super(SVS, self).__init__()
        
        # Basic parameters
        self.n_mels = n_mels
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
        
        # STFT Generator
        self.stft_generator = STFTGenerator(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            sample_rate=self.sample_rate
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
        
        # Add parameter predictor
        self.parameter_predictor = ParameterPredictor(
            phoneme_dim=self.phoneme_embed_dim,
            singer_dim=self.singer_embed_dim,
            language_dim=self.language_embed_dim,
            hidden_dim=256,
            num_formants=5,
            use_lstm=True
        )
        
        # Add vocal filter
        self.vocal_filter = VocalFilter(
            n_fft=self.n_fft,
            sample_rate=self.sample_rate,
            use_parallel_filters=True,
            filter_mode='resonator'
        )
        
    def forward(self, f0, phoneme_seq, singer_id, language_id, initial_phase=None):
        """
        Forward pass with formant filtering.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices)
            language_id: Language IDs [B] (indices)
            initial_phase: Optional initial phase
            
        Returns:
            Dictionary containing:
                'signal': Audio signal [B, T_audio]
                'mel': Mel-spectrogram [B, T, n_mels]
                'stft_original': Original STFT [B, F, T]
                'stft_filtered': Filtered STFT [B, F, T]
                'filter_params': Filter parameters dict
        """
        batch_size, n_frames = f0.shape[0], f0.shape[1]
        device = f0.device
        
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]
        
        # Generate STFT using STFTGenerator (using only f0)
        original_stft = self.stft_generator(f0)
        
        # Predict formant filter parameters
        filter_params = self.parameter_predictor(
            f0, 
            phoneme_emb, 
            singer_emb, 
            language_emb
        )
        
        # Apply formant filtering to the STFT
        filtered_stft = self.vocal_filter(original_stft, filter_params)
        
        # Convert filtered STFT to audio using torch's inverse STFT
        window = torch.hann_window(self.win_length).to(device)
        signal = torch.istft(
            filtered_stft, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=False
        )
        
        # Get magnitude of filtered STFT for mel conversion
        stft_mag = torch.abs(filtered_stft)  # Shape: [B, F, T]
        
        # Apply mel transformation
        mel_basis = self.mel_basis.to(device).transpose(0, 1)  # Shape: [F, M]
        
        # Apply matrix multiplication using einsum for better clarity
        # [B, F, T] x [F, M] -> [B, M, T]
        predicted_mel = torch.einsum('bft,fm->bmt', stft_mag, mel_basis)
        
        # Apply log transformation for better dynamic range
        predicted_mel = torch.log(torch.clamp(predicted_mel, min=1e-5))
        
        # Transpose mel to expected [B, T, M] shape
        predicted_mel = predicted_mel.transpose(1, 2)
        
        # Return all relevant outputs
        return signal, predicted_mel, filtered_stft