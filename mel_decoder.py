import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MelDecoder(nn.Module):
    """
    Lightweight DDSP-based singing voice synthesis model.
    Converts mel-spectrograms to audio guided by F0, phonemes, singer, and language.
    Optimized for better performance with improved source-filter vocal model.
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
        
        # Temporal modeling with CAUSAL convolutions instead of bidirectional GRU
        self.hidden_size = 64
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(96, self.hidden_size, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=4, dilation=4),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1),
        )
        
        # Simplified glottal source model
        self.glottal_pulse_params = nn.Conv1d(self.hidden_size, 3, kernel_size=3, padding=1)
        
        # Combined frequency domain processing parameters
        self.unified_filter_params = nn.Sequential(
            nn.Conv1d(self.hidden_size, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, n_fft//2+1, kernel_size=3, padding=1),
            #nn.Softplus(),
        )
        
        # Component mixture balance
        self.component_mix = nn.Sequential(
            nn.Conv1d(self.hidden_size, 3, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )
        
        # Precomputed values
        self.register_buffer('window', torch.hann_window(self.n_fft))
        self.register_buffer('formant_centers', torch.tensor([10, 20, 36, 56, 80], dtype=torch.float32))
        self.register_buffer('freq_bins', torch.arange(0, self.n_fft//2+1, dtype=torch.float32))
        
        # Precomputed coefficients for glottal source
        self.register_buffer('opening_factors', 
                           torch.linspace(0, math.pi, steps=int(0.9 * self.hop_length)))
        self.register_buffer('closing_factors', 
                           1 - torch.linspace(0, 1, steps=int(0.1 * self.hop_length))**2)
    
    def _generate_glottal_source(self, f0, glottal_params):
        """
        Simplified glottal source generation with optimized operations.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            glottal_params: Simplified parameters [B, 3, T]
            
        Returns:
            Glottal source waveform [B, T*hop_length]
        """
        batch_size, n_frames = f0.shape
        audio_length = n_frames * self.hop_length
        device = f0.device
        
        # Convert f0 to phase increments
        phase_increment = 2 * math.pi * f0 / self.sample_rate
        
        # Simplified glottal parameters
        open_quotient = torch.sigmoid(glottal_params[:, 0]) * 0.5 + 0.25  # [B, T]
        spectral_tilt = torch.sigmoid(glottal_params[:, 1]) * 0.5         # [B, T]
        shimmer = torch.sigmoid(glottal_params[:, 2]) * 0.05              # [B, T]
        
        # Accumulate phase efficiently
        phase_increments = phase_increment.unsqueeze(2).expand(-1, -1, self.hop_length)
        phase_increments = phase_increments.reshape(batch_size, -1)  # [B, T*hop_length]
        phase = torch.zeros_like(phase_increments)
        phase[:, 1:] = torch.cumsum(phase_increments[:, :-1], dim=1)
        
        # Generate simplified glottal pulses using precomputed factors
        t = phase % (2 * math.pi)
        
        # Create glottal pulse shape with vectorized operations
        open_mask = t < open_quotient.unsqueeze(2).expand(-1, -1, self.hop_length).reshape(batch_size, -1)
        
        # Simplified glottal pulse
        pulse = torch.where(open_mask, 
                          torch.sin(t / open_quotient.unsqueeze(2).expand(-1, -1, self.hop_length).reshape(batch_size, -1)),
                          0)
        
        # Apply shimmer
        shimmer_factor = 1.0 + shimmer.unsqueeze(2).expand(-1, -1, self.hop_length).reshape(batch_size, -1) * (torch.rand_like(pulse) - 0.5)
        pulse = pulse * shimmer_factor
        
        return pulse
    
    def _apply_unified_filtering(self, source, frication_noise, aspiration_noise, filter_params):
        """
        Unified frequency domain processing - combines all filtering operations.
        
        Args:
            source: Glottal source signal [B, T*hop_length]
            frication_noise: Unfiltered frication noise [B, T*hop_length]
            aspiration_noise: Unfiltered aspiration noise [B, T*hop_length]
            filter_params: Combined filter parameters [B, n_fft//2+1, T]
            
        Returns:
            Processed signals [B, T*hop_length, 3] - source, frication, aspiration
        """
        batch_size, audio_length = source.shape
        
        # Stack all sources for batch processing
        min_len = min(source.shape[1], frication_noise.shape[1], aspiration_noise.shape[1])
        source = source[:, :min_len]
        frication_noise = frication_noise[:, :min_len]
        aspiration_noise = aspiration_noise[:, :min_len]
        all_signals = torch.stack([source, frication_noise, aspiration_noise], dim=1)  # [B, 3, T*hop_length]
        
        # Compute STFT on all signals at once
        all_stft = torch.stft(
            all_signals.reshape(-1, audio_length),  # [B*3, T*hop_length]
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            normalized=False,
            onesided=True
        ).reshape(batch_size, 3, self.n_fft//2+1, -1)  # [B, 3, F, T']
        
        # Expand filter parameters
        expanded_filters = filter_params.unsqueeze(1).expand(-1, 3, -1, -1)
        
        # Optional: Modify filters for different components
        filter_mods = torch.ones_like(expanded_filters)
        filter_mods[:, 0] *= 1.2  # Enhance vocal content
        filter_mods[:, 1] *= 0.8  # Reduce aspiration
        filter_mods[:, 2] *= 0.6  # Reduce frication
        expanded_filters = expanded_filters * filter_mods
        
        # Apply filtering
        min_len = min(all_stft.shape[-1], expanded_filters.shape[-1])
        all_stft = all_stft[:, :, :, :min_len]
        expanded_filters = expanded_filters[:, :, :, :min_len]
        filtered_stft = all_stft * expanded_filters
        
        # Convert back to time domain
        filtered_signals = torch.istft(
            filtered_stft.reshape(-1, self.n_fft//2+1, filtered_stft.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            normalized=False,
            onesided=True,
            length=audio_length
        ).reshape(batch_size, 3, audio_length)  # [B, 3, T*hop_length]
        
        return filtered_signals
    
    def _generate_noise(self, batch_size, audio_length, device, correlation=0.0):
        """
        Optimized noise generation.
        
        Args:
            batch_size: Batch size
            audio_length: Length of audio
            device: Device to use
            correlation: Temporal correlation factor (0=white noise, >0=colored noise)
            
        Returns:
            Noise signal [B, T*hop_length]
        """
        noise = torch.randn(batch_size, audio_length, device=device)
        
        if correlation > 0:
            # Simplified temporal correlation using IIR filter
            noise = torch.nn.functional.conv1d(
                noise.unsqueeze(1),
                torch.tensor([[[correlation, 1-correlation]]]).to(device),
                padding=1
            ).squeeze(1)
        
        return noise
    
    def forward(self, mel, f0, phoneme_seq, singer_id, language_id):
        """
        Optimized forward pass.
        
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
        device = mel.device
        
        # Transpose mel to match Conv1D expected shape [B, n_mels, T]
        mel = mel.transpose(1, 2)
        
        # Extract features from inputs
        mel_features = F.leaky_relu(self.mel_conv(mel), 0.1)  # [B, 64, T]
        
        f0_expanded = f0.unsqueeze(1)  # [B, 1, T]
        f0_features = F.leaky_relu(self.f0_conv(f0_expanded), 0.1)  # [B, 8, T]
        
        # Process phoneme sequence
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
        
        # Apply temporal modeling with causal convolutions
        temporal_features = self.temporal_conv(cond)  # [B, hidden_size, T]
        
        # Generate source-filter model parameters
        glottal_params = self.glottal_pulse_params(temporal_features)  # [B, 3, T]
        filter_params = self.unified_filter_params(temporal_features)  # [B, n_fft//2+1, T]
        
        # Get component mixing parameters
        mix_params = self.component_mix(temporal_features)  # [B, 3, T]
        
        # Generate glottal source signal
        glottal_source = self._generate_glottal_source(f0, glottal_params)  # [B, T*hop_length]
        
        # Generate noise components efficiently
        audio_length = glottal_source.shape[1]
        frication_noise = self._generate_noise(batch_size, audio_length, device, correlation=0.0)
        aspiration_noise = self._generate_noise(batch_size, audio_length, device, correlation=0.8)
        
        # Apply unified filtering to all components
        filtered_signals = self._apply_unified_filtering(
            glottal_source, frication_noise, aspiration_noise, filter_params)  # [B, 3, T*hop_length]
        
        # Upsample mixing parameters efficiently
        mix_upsampled = F.interpolate(
            mix_params,
            size=audio_length,
            mode='linear',
            align_corners=False
        )  # [B, 3, T*hop_length]
        
        # Separate mix parameters
        source_mix = mix_upsampled[:, 0]
        frication_mix = mix_upsampled[:, 1]
        aspiration_mix = mix_upsampled[:, 2]
        
        # Final mixing of all components
        audio_signal = (source_mix * filtered_signals[:, 0] + 
                       frication_mix * filtered_signals[:, 1] + 
                       aspiration_mix * filtered_signals[:, 2])
        
        return audio_signal