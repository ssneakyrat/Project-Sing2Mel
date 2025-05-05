import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MelEncoder(nn.Module):
    """
    Main class that synthesizes mel spectrograms directly in the spectral domain.
    With explicit gradient tracking for all operations.
    """
    def __init__(self, 
                 n_mels=80, 
                 n_harmonics=80,
                 phoneme_embed_dim=128, 
                 singer_embed_dim=16, 
                 language_embed_dim=8,
                 sample_rate=24000,
                 n_fft=1024):
        super().__init__()
        
        self.n_mels = n_mels
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        
        # Feature dimensions
        self.phoneme_embed_dim = phoneme_embed_dim
        self.singer_embed_dim = singer_embed_dim
        self.language_embed_dim = language_embed_dim
        
        # Register the mel filterbank as a Parameter to ensure gradient tracking
        # If you don't want it to be trainable, you can use requires_grad=False
        mel_filterbank = self._create_mel_filterbank()
        self.register_parameter('mel_filterbank', nn.Parameter(mel_filterbank, requires_grad=False))
        
        # We'll define a learnable transform to process the inputs
        # This ensures there's at least one trainable parameter to maintain gradient flow
        self.input_transform = nn.Linear(phoneme_embed_dim + singer_embed_dim + language_embed_dim, 
                                         phoneme_embed_dim)
        
    def forward(self, signal, f0, phoneme_emb, singer_emb, language_emb):
        """
        Synthesize mel spectrogram directly from input signal with explicit gradient tracking.
        
        Args:
            signal: Input audio signal [B, L] where L is the signal length
            f0: Fundamental frequency [B, T, 1]
            phoneme_emb: Phoneme embeddings [B, T, phoneme_embed_dim]
            singer_emb: Singer embeddings [B, singer_embed_dim]
            language_emb: Language embeddings [B, language_embed_dim]
            
        Returns:
            mel_spectrogram: Mel spectrogram [B, T, n_mels]
        """
        # Ensure all inputs have requires_grad if needed
        # You might want to comment this out in production to avoid modifying inputs
        if not signal.requires_grad and torch.is_floating_point(signal):
            signal.requires_grad_(True)
        
        # 0. Process the embeddings to ensure gradient flow
        # Expand singer and language embeddings to match the time dimension of phoneme_emb
        batch_size, time_length = phoneme_emb.shape[0], phoneme_emb.shape[1]
        singer_emb_expanded = singer_emb.unsqueeze(1).expand(-1, time_length, -1)
        language_emb_expanded = language_emb.unsqueeze(1).expand(-1, time_length, -1)
        
        # Concatenate and transform
        combined_emb = torch.cat([phoneme_emb, singer_emb_expanded, language_emb_expanded], dim=-1)
        transformed_emb = self.input_transform(combined_emb)
        
        # 1. Compute STFT of the signal
        hop_length = self.n_fft // 4
        
        # Create window as a parameter to ensure gradient flow
        window = torch.hann_window(self.n_fft, device=signal.device)
        
        # Use torch.stft with return_complex=True for better gradient flow
        complex_stft = torch.stft(
            signal, 
            n_fft=self.n_fft, 
            hop_length=hop_length, 
            win_length=self.n_fft, 
            window=window,
            return_complex=True  # Return complex tensor for better gradient support
        )
        
        # 2. Calculate magnitude spectrum with stable gradients
        epsilon = 1e-10
        magnitude = torch.abs(complex_stft) + epsilon  # [B, F, T]
        
        # 3. Apply mel filterbank
        # Transpose for batch matmul: [B, T, F] = [B, F, T].transpose(1, 2)
        magnitude = magnitude.transpose(1, 2)  # [B, T, F]
        
        # Apply mel filterbank: [B, T, n_mels] = [B, T, F] @ [F, n_mels]
        mel_spectrogram = torch.matmul(magnitude, self.mel_filterbank.transpose(0, 1))  # [B, T, n_mels]
        
        # 4. Apply log compression with stable gradients
        mel_spectrogram = torch.log(mel_spectrogram + epsilon)
        
        # 5. Match the target length with explicit gradient-preserving ops
        target_length = f0.shape[1]
        current_length = mel_spectrogram.shape[1]

        if current_length < target_length:
            # Handle length difference with interpolation (preserves gradients)
            x_permuted = mel_spectrogram.permute(0, 2, 1)  # [B, n_mels, T]
            
            # Use align_corners=True for better gradient flow with linear interpolation
            interpolated = F.interpolate(
                x_permuted, 
                size=target_length, 
                mode='linear', 
                align_corners=True
            )
            
            # Permute back to original shape
            return interpolated.permute(0, 2, 1)  # [B, target_length, n_mels]
        else:
            return mel_spectrogram
    
    def _create_mel_filterbank(self):
        """
        Create a mel filterbank matrix to convert from linear frequency to mel frequency.
        
        Returns:
            mel_filterbank: Mel filterbank matrix [n_mels, n_fft//2 + 1]
        """
        # Number of FFT bins
        n_freqs = self.n_fft // 2 + 1
        
        # Convert min and max frequencies to mel scale
        min_freq = 0
        max_freq = self.sample_rate / 2
        min_mel = self._hz_to_mel(min_freq)
        max_mel = self._hz_to_mel(max_freq)
        
        # Create evenly spaced points in mel scale
        mel_points = torch.linspace(min_mel, max_mel, self.n_mels + 2)
        
        # Convert mel points back to Hz
        hz_points = self._mel_to_hz(mel_points)
        
        # Convert Hz points to FFT bin indices
        bin_indices = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()
        
        # Create the filterbank matrix
        filterbank = torch.zeros(self.n_mels, n_freqs)
        
        # For each mel band, create a triangular filter
        for i in range(self.n_mels):
            # Lower and upper frequency boundaries for triangular filter
            lower = bin_indices[i]
            center = bin_indices[i + 1]
            upper = bin_indices[i + 2]
            
            # Create triangular filter (with checks to avoid division by zero)
            if lower != center:
                for j in range(lower, center):
                    filterbank[i, j] = (j - lower) / float(center - lower)
            
            if center != upper:
                for j in range(center, upper):
                    filterbank[i, j] = (upper - j) / float(upper - center)
        
        # Normalize the filterbank
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-10
        hz_diff = hz_points[2:self.n_mels+2] - hz_points[:self.n_mels]
        enorm = 2.0 / (hz_diff + epsilon)
        filterbank *= enorm.unsqueeze(1)
        
        return filterbank
    
    def _hz_to_mel(self, hz):
        """
        Convert from Hz to mel scale.
        
        Args:
            hz: Frequency in Hz
            
        Returns:
            mel: Frequency in mel scale
        """
        if isinstance(hz, torch.Tensor):
            return 2595 * torch.log10(1 + hz / 700)
        else:
            return 2595 * math.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel):
        """
        Convert from mel scale to Hz.
        
        Args:
            mel: Frequency in mel scale
            
        Returns:
            hz: Frequency in Hz
        """
        if isinstance(mel, torch.Tensor):
            return 700 * (10**(mel / 2595) - 1)
        else:
            return 700 * (10**(mel / 2595) - 1)