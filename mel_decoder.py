import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MelDecoder(nn.Module):
    """
    Lightweight DDSP-based singing voice synthesis model.
    Converts mel-spectrograms to audio guided by F0, phonemes, singer, and language.
    Optimized for better performance with improved temporal modeling.
    """
    def __init__(self, 
                 num_phonemes, 
                 num_singers, 
                 num_languages,
                 n_mels=80, 
                 hop_length=240, 
                 sample_rate=24000,
                 num_harmonics=48, 
                 n_fft=1024):
        super(MelDecoder, self).__init__()
        
        # Basic parameters
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        self.n_fft = n_fft
        
        # Define embedding dimensions (kept small for efficiency)
        self.phoneme_embed_dim = 64
        self.singer_embed_dim = 16
        self.language_embed_dim = 8
        
        # Embedding layers
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, self.phoneme_embed_dim)
        self.singer_embed = nn.Embedding(num_singers, self.singer_embed_dim)
        self.language_embed = nn.Embedding(num_languages, self.language_embed_dim)
        
        # Feature extraction layers
        self.mel_conv = nn.Conv1d(n_mels, 64, kernel_size=3, padding=1)
        self.f0_conv = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        
        # Calculate combined feature dimension
        combined_dim = (64 +                     # mel features
                       8 +                       # f0 features
                       self.phoneme_embed_dim +  # phoneme embedding
                       self.singer_embed_dim +   # singer embedding
                       self.language_embed_dim)  # language embedding
        
        # Conditioning network
        self.conditioning = nn.Sequential(
            nn.Conv1d(combined_dim, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(96, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        
        # IMPROVED: Replace GRU with bidirectional GRU for better temporal modeling
        # Using same hidden size but bidirectional, so output is doubled
        self.hidden_size = 64
        self.gru = nn.GRU(
            96, 
            self.hidden_size, 
            batch_first=False,
            bidirectional=True
        )
        
        # IMPROVED: Add a projection layer to combine bidirectional outputs
        # This helps with information flow between directions
        self.bi_projection = nn.Sequential(
            nn.Conv1d(self.hidden_size * 2, self.hidden_size, kernel_size=1),
            nn.LeakyReLU(0.1)
        )
        
        # IMPROVED: Add temporal context integration
        # This helps capture longer dependencies for smoother transitions
        self.context_layer = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, padding=2, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, padding=4, dilation=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1)
        )
        
        # Synthesis parameter generators - using updated hidden size
        self.harmonic_amplitudes = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.num_harmonics, kernel_size=3, padding=1),
            nn.Softplus(),
        )
        
        # Spectral filter for noise component
        self.noise_filter = nn.Sequential(
            nn.Conv1d(self.hidden_size, n_fft//2+1, kernel_size=3, padding=1),
            nn.Softplus(),
        )
        
        # Mixing parameter (harmonic vs. noise balance)
        self.harmonic_mix = nn.Sequential(
            nn.Conv1d(self.hidden_size, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  
        )
        
        # Initialize window for STFT
        self.register_buffer(
            'window', 
            torch.hann_window(self.n_fft)
        )
    
    def _harmonics_phase_corrected(self, f0):
        """
        Generate phase-corrected harmonics based on F0.
        Ensures phase continuity across frames.
        Optimized implementation with vectorized operations.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
        
        Returns:
            Phase-corrected harmonic frequencies [B, num_harmonics, T*hop_length]
        """
        batch_size, n_frames = f0.shape
        audio_length = n_frames * self.hop_length
        device = f0.device
        
        # Convert f0 from Hz to angular frequency (radians per sample)
        omega = 2 * math.pi * f0 / self.sample_rate  # [B, T]
        
        # Pre-calculate harmonic indices (1, 2, 3, ..., num_harmonics)
        harmonic_indices = torch.arange(1, self.num_harmonics + 1, device=device).float()
        harmonic_indices = harmonic_indices.view(1, -1, 1)  # [1, num_harmonics, 1]
        
        # Upsample omega to sample-level - use standard interpolation
        # This is simpler and still efficient
        omega_upsampled = F.interpolate(
            omega.unsqueeze(1), 
            size=audio_length, 
            mode='linear', 
            align_corners=False
        ).squeeze(1)  # [B, T*hop_length]
        
        # Create cumulative phase by integrating omega (accumulated phase)
        phase = torch.zeros(batch_size, audio_length, device=device)
        phase[:, 1:] = torch.cumsum(omega_upsampled[:, :-1], dim=1)
        
        # Expand phase for broadcasting with harmonics
        phase = phase.unsqueeze(1)  # [B, 1, T*hop_length]
        
        # Calculate all harmonic phases in one vectorized operation
        # This avoids redundant calculations and memory allocations
        harmonic_phases = phase * harmonic_indices  # [B, num_harmonics, T*hop_length]
        
        # Generate phase-correct sinusoids
        sinusoids = torch.sin(harmonic_phases)  # [B, num_harmonics, T*hop_length]
        
        return sinusoids
    
    def _filter_noise(self, noise_filter):
        """
        Apply spectral filtering to noise using efficient FFT-based approach.
        Optimized implementation using PyTorch's built-in STFT/ISTFT.
        
        Args:
            noise_filter: Filter shapes in frequency domain [B, n_fft//2+1, T]
            
        Returns:
            Filtered noise [B, T*hop_length]
        """
        batch_size, n_freq, n_frames = noise_filter.shape
        device = noise_filter.device
        
        # Generate white noise at target length
        audio_length = n_frames * self.hop_length
        noise = torch.randn(batch_size, audio_length, device=device)
        
        # Compute STFT using PyTorch's optimized implementation
        noise_stft = torch.stft(
            noise,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            normalized=False,
            onesided=True
        )  # [B, F, T] where F = n_fft//2+1
        
        # Always interpolate noise_filter to exactly match STFT time frames
        stft_frames = noise_stft.shape[2]
        adjusted_filter = F.interpolate(
            noise_filter,
            size=stft_frames,
            mode='linear',
            align_corners=False
        )
        
        # Apply frequency-domain filtering using broadcasting
        # For complex tensors in PyTorch, we need to be careful about dimensions
        # noise_stft is [B, F, T] complex tensor
        # We need adjusted_filter to be [B, F, T] real tensor
        filtered_stft = noise_stft * adjusted_filter
        
        # Convert back to time domain using inverse STFT
        filtered_output = torch.istft(
            filtered_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            normalized=False,
            onesided=True,
            length=audio_length
        )  # [B, T*hop_length]
        
        return filtered_output
    
    def forward(self, mel, f0, phoneme_seq, singer_id, language_id):
        """
        Forward pass of the lightweight DDSP singer model.
        Now with improved temporal modeling for better voice quality.
        
        Args:
            mel: Mel-spectrogram [B, T, n_mels]
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T]
            singer_id: Singer IDs [B]
            language_id: Language IDs [B]
            
        Returns:
            Audio signal [B, T*hop_length]
        """
        batch_size, seq_length = mel.shape[0], mel.shape[1]
        
        # Transpose mel to match Conv1D expected shape [B, n_mels, T]
        mel = mel.transpose(1, 2)
        
        # Extract features from inputs
        mel_features = F.leaky_relu(self.mel_conv(mel), 0.1)  # [B, 64, T]
        
        f0_expanded = f0.unsqueeze(1)  # [B, 1, T]
        f0_features = F.leaky_relu(self.f0_conv(f0_expanded), 0.1)  # [B, 8, T]
        
        # Process phoneme sequence - assuming phoneme_seq is [B, T]
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_embed_dim]
        phoneme_emb = phoneme_emb.transpose(1, 2)  # [B, phoneme_embed_dim, T]
        
        # Get singer and language embeddings
        singer_emb = self.singer_embed(singer_id).unsqueeze(2).expand(-1, -1, seq_length)
        language_emb = self.language_embed(language_id).unsqueeze(2).expand(-1, -1, seq_length)
        
        # Combine all features
        combined = torch.cat([
            mel_features, f0_features, phoneme_emb, singer_emb, language_emb
        ], dim=1)  # [B, combined_dim, T]
        
        # Extract conditioning features
        cond = self.conditioning(combined)  # [B, 96, T]
        
        # IMPROVED: Apply bidirectional temporal modeling
        cond_temporal = cond.transpose(1, 2).transpose(0, 1)  # [T, B, 96]
        gru_out, _ = self.gru(cond_temporal)  # [T, B, hidden_size*2] (bidirectional)
        gru_out = gru_out.transpose(0, 1).transpose(1, 2)  # [B, hidden_size*2, T]
        
        # IMPROVED: Project bidirectional features to original dimension
        gru_projected = self.bi_projection(gru_out)  # [B, hidden_size, T]
        
        # IMPROVED: Apply temporal context integration for smoother transitions
        temporal_features = self.context_layer(gru_projected) + gru_projected  # [B, hidden_size, T]
        
        # Generate synthesis parameters
        harmonic_amps = self.harmonic_amplitudes(temporal_features)  # [B, num_harmonics, T]
        noise_filters = self.noise_filter(temporal_features)  # [B, n_fft//2+1, T]
        mix_param = self.harmonic_mix(temporal_features)  # [B, 1, T]
        
        # Generate raw harmonics (sine waves with phase correction)
        harmonic_waves = self._harmonics_phase_corrected(f0)  # [B, num_harmonics, T*hop_length]
        
        # Calculate target audio length
        audio_length = harmonic_waves.shape[2]
        
        # Efficient batch upsampling of all control parameters at once
        # Stack parameters along a new dimension for a single upsampling operation
        control_params = torch.cat([
            harmonic_amps,  # [B, num_harmonics, T]
            mix_param,      # [B, 1, T]
        ], dim=1)  # [B, num_harmonics+1, T]
        
        # Single upsampling operation
        control_params_upsampled = F.interpolate(
            control_params,
            size=audio_length,
            mode='linear',
            align_corners=False
        )  # [B, num_harmonics+1, T*hop_length]
        
        # Split upsampled parameters
        harmonic_amps_upsampled = control_params_upsampled[:, :self.num_harmonics]  # [B, num_harmonics, T*hop_length]
        mix_upsampled = control_params_upsampled[:, -1:]  # [B, 1, T*hop_length]
        
        # Apply amplitudes to each harmonic and sum
        # Use torch.bmm for more efficient batch matrix multiplication
        harmonic_waves_flat = harmonic_waves.transpose(1, 2)  # [B, T*hop_length, num_harmonics]
        harmonic_amps_flat = harmonic_amps_upsampled.transpose(1, 2)  # [B, T*hop_length, num_harmonics]
        
        # Element-wise multiplication and sum along last dimension
        harmonic_signal = torch.sum(harmonic_waves_flat * harmonic_amps_flat, dim=2)  # [B, T*hop_length]
        
        # Generate and filter noise
        noise_signal = self._filter_noise(noise_filters)  # [B, T*hop_length]
        
        # Mix harmonic and noise components
        mix_upsampled = mix_upsampled.squeeze(1)  # [B, T*hop_length]
        
        # Ensure all tensors have same length (handle edge cases)
        min_length = min(harmonic_signal.shape[1], noise_signal.shape[1], mix_upsampled.shape[1])
        harmonic_signal = harmonic_signal[:, :min_length]
        noise_signal = noise_signal[:, :min_length]
        mix_upsampled = mix_upsampled[:, :min_length]
        
        # Final mixing
        audio_signal = mix_upsampled * harmonic_signal + (1 - mix_upsampled) * noise_signal
        
        return audio_signal