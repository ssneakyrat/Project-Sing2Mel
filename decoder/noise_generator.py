import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoiseGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Add filter parameters
        self.filter_size = 64
        self.register_buffer("filter_window", torch.hann_window(self.filter_size))

    def frequency_dependent_filter(self, noise, f0, sample_rate=24000):
        """
        Apply frequency-dependent filtering to noise based on f0.
        
        Args:
            noise: Input noise to be filtered
            f0: Fundamental frequency values
            sample_rate: Audio sample rate
            
        Returns:
            Filtered noise with f0-dependent spectral characteristics
        """
        batch_size = noise.shape[0]
        
        # Ensure f0 is properly shaped
        if f0.dim() > 2:
            f0 = f0.squeeze(-1)  # [B, T]
        
        # Normalize f0 for filter coefficient calculation (0 to 1 range)
        f0_norm = torch.clamp(f0, 0, 1000)  # Safety clamp
        f0_norm = (f0_norm - 80) / 920  # Map 80-1000 Hz to 0-1
        f0_norm = torch.clamp(f0_norm, 0, 1)  # Ensure 0-1 range
        
        # Convert to frequency domain using STFT
        noise_stft = torch.stft(
            noise, 
            n_fft=self.filter_size, 
            hop_length=self.filter_size // 4,
            window=self.filter_window,
            return_complex=True
        )  # [B, F, T]
        
        # Get the resulting time dimension after STFT
        stft_time_dim = noise_stft.shape[2]
        
        # Interpolate f0_norm to match STFT time dimension
        f0_interp = F.interpolate(
            f0_norm.unsqueeze(1), 
            size=stft_time_dim, 
            mode='linear', 
            align_corners=False
        ).squeeze(1)  # [B, T_stft]
        
        # Create frequency-dependent filter based on f0
        freq_bins = noise_stft.shape[1]
        
        # Create frequency axis (0 to 1)
        freqs = torch.linspace(0, 1, freq_bins, device=noise.device)
        
        # Expanding dimensions for broadcasting
        freqs = freqs.view(1, -1, 1)  # [1, F, 1]
        f0_interp = f0_interp.view(batch_size, 1, -1)  # [B, 1, T_stft]
        
        # Spectral tilt: higher f0 = more high freq, lower f0 = more low freq
        filter_shape = torch.pow(freqs, 1.0 - f0_interp)  # [B, F, T_stft]
        
        # Apply the filter
        filtered_stft = noise_stft * filter_shape
        
        # Back to time domain
        filtered_noise = torch.istft(
            filtered_stft,
            n_fft=self.filter_size,
            hop_length=self.filter_size // 4,
            window=self.filter_window,
            length=noise.shape[1]
        )
        
        return filtered_noise

    def forward(self, harmonic, noise_param, f0):
        """
        Generate noise with characteristics dependent on f0.
        
        Args:
            harmonic: Harmonic component of the signal [B, T*hop_length]
            noise_param: Noise magnitude parameters [B, T, num_mag_noise]
            f0: Fundamental frequency [B, T*hop_length, 1] or [B, T*hop_length]
            
        Returns:
            Noise component shaped like harmonic input
        """
        # Base noise - random values between -1 and 1
        base_noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        
        # Ensure f0 has compatible shape with harmonic for masking
        if f0.dim() > harmonic.dim():
            # If f0 is [B, T*hop_length, 1] and harmonic is [B, T*hop_length]
            f0_shaped = f0.squeeze(-1)
        else:
            f0_shaped = f0
            
        voiced_mask = (f0_shaped > 80).float()
        
        # Different noise profiles for voiced vs unvoiced segments
        unvoiced_noise = base_noise * 1.2  # Slightly amplified for unvoiced
        
        # Apply frequency-dependent filtering to voiced noise
        voiced_noise = self.frequency_dependent_filter(base_noise * 0.7, f0_shaped)
        
        # Combine using the voiced/unvoiced mask
        noise = voiced_noise * voiced_mask + unvoiced_noise * (1 - voiced_mask)
        
        return noise