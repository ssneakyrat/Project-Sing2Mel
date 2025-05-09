import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from dsp.stft_generator import STFTGenerator

# Modified SVS class with MelEncoder and STFTGenerator integration
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
        
    # This is the modified section of the forward method in svs.py
    def forward(self, f0, phoneme_seq, singer_id, language_id, initial_phase=None):
        """
        Forward pass with simplified STFT generation.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices)
            language_id: Language IDs [B] (indices)
            initial_phase: Optional initial phase (not used in this simplified version)
            
        Returns:
            Audio signal [B, T*hop_length], mel-spectrogram [B, T, n_mels], STFT [B, F, T]
        """
        batch_size, n_frames = f0.shape[0], f0.shape[1]
        
        # Apply embeddings (kept for future conditioning)
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]
        
        # Generate STFT using simplified STFTGenerator (using only f0)
        predicted_stft = self.stft_generator(f0)
        
        # Convert STFT to audio using torch's inverse STFT
        signal = torch.istft(
            predicted_stft, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(f0.device),
            return_complex=False
        )
        
        # Get magnitude of STFT for mel conversion
        stft_mag = torch.abs(predicted_stft)  # Shape should be [B, F, T]
        
        # Reshape if stft_mag is flattened (this fixes the dimension mismatch)
        if stft_mag.dim() == 2:
            F = self.n_fft // 2 + 1  # n_freqs
            B = batch_size
            T = stft_mag.shape[1]  # n_frames
            stft_mag = stft_mag.view(B, F, T)  # Reshape to [B, F, T]
        
        # Transpose mel_basis from [M, F] to [F, M] for correct multiplication
        mel_basis = self.mel_basis.to(f0.device).transpose(0, 1)  # Shape [F, M]
        
        # Apply matrix multiplication for each batch element
        # [B, F, T] x [F, M] -> [B, M, T]
        # Use einsum for clearer dimension handling
        predicted_mel = torch.einsum('bft,fm->bmt', stft_mag, mel_basis)
        
        # Apply log transformation for better dynamic range
        predicted_mel = torch.log(torch.clamp(predicted_mel, min=1e-5))
        
        return signal, predicted_mel, predicted_stft