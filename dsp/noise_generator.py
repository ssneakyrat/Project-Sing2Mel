import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NoiseGenerator(nn.Module):
    """
    Generates spectrally shaped noise for singing voice synthesis.
    
    This module creates noise components to model breathiness, aspiration,
    and unvoiced consonants in the synthesized voice.
    """
    def __init__(self, 
                 n_fft=1024, 
                 hop_length=240, 
                 win_length=1024, 
                 sample_rate=24000,
                 n_bands=8):  # Number of frequency bands for spectral shaping
        super(NoiseGenerator, self).__init__()
        # STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        
        # Number of frequency bins in STFT
        self.n_freqs = n_fft // 2 + 1
        
        # Pre-calculate frequency band centers (mel-scaled)
        # This gives better control over perceptually important regions
        min_mel = 0
        max_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
        mel_points = torch.linspace(min_mel, max_mel, n_bands + 2)
        
        # Convert back to Hz
        f_pts = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Convert Hz to FFT bin indices
        fft_bins = torch.floor(f_pts / sample_rate * n_fft).int()
        fft_bins = torch.clamp(fft_bins, 0, self.n_freqs - 1)
        
        # Register band edges as buffer
        self.register_buffer('band_edges', fft_bins)
        
        # Create filter bank using optimized method
        filters = self._create_filter_bank(fft_bins, n_bands, self.n_freqs)
        
        # Register filter bank
        self.register_buffer('filter_bank', filters)
        
        # Pre-compute and cache filter_bank expanded dimensions for forward pass (optimization 5)
        filter_bank_expanded = filters.unsqueeze(0).unsqueeze(-1)
        self.register_buffer('filter_bank_expanded', filter_bank_expanded)
        
    def _create_filter_bank(self, fft_bins, n_bands, n_freqs):
        """
        Create triangular filter bank in a vectorized way (optimization 2).
        
        Args:
            fft_bins: Frequency bin indices for band edges
            n_bands: Number of frequency bands
            n_freqs: Number of frequency bins
            
        Returns:
            filter_bank: Triangular filters [n_bands, n_freqs]
        """
        # Pre-allocate filter bank tensor
        filters = torch.zeros(n_bands, n_freqs)
        
        # Create frequency bin indices tensor
        freq_indices = torch.arange(n_freqs).unsqueeze(0).expand(n_bands, -1)
        
        for i in range(n_bands):
            start, center, end = fft_bins[i], fft_bins[i+1], fft_bins[i+2]
            
            # Create masks for the rising and falling edges
            rising_mask = (freq_indices[i] >= start) & (freq_indices[i] < center)
            falling_mask = (freq_indices[i] >= center) & (freq_indices[i] < end)
            
            # Calculate normalized distances for rising and falling parts
            if center > start:
                rising_norm = (freq_indices[i][rising_mask] - start) / float(center - start)
                filters[i, rising_mask] = rising_norm
                
            if end > center:
                falling_norm = 1.0 - (freq_indices[i][falling_mask] - center) / float(end - center)
                filters[i, falling_mask] = falling_norm
        
        return filters
        
    def forward(self, noise_params):
        """
        Generate spectrally shaped noise in STFT domain (optimized version 1).
        
        Args:
            noise_params: Dictionary containing:
                - 'noise_gain': Overall noise amplitude [B, T, 1]
                - 'spectral_shape': Spectral envelope [B, T, n_bands]
                - 'voiced_mix': Mix ratio between voiced/unvoiced [B, T, 1]
                
        Returns:
            Complex STFT representation of noise [B, F, T]
        """
        batch_size = noise_params['noise_gain'].shape[0]
        n_frames = noise_params['noise_gain'].shape[1]
        device = noise_params['noise_gain'].device
        
        # Generate complex white noise (optimization 3)
        noise_complex = torch.randn(batch_size, self.n_freqs, n_frames, 2, device=device)
        noise_complex = torch.view_as_complex(noise_complex)
        
        # Apply spectral shaping using the filter bank
        spectral_shape = noise_params['spectral_shape'].transpose(1, 2)  # [B, n_bands, T]
        
        # Vectorized approach - reshape dimensions for broadcasting (optimization 1)
        # Use pre-computed filter_bank_expanded: [1, n_bands, F, 1]
        
        # [B, n_bands, T] -> [B, n_bands, 1, T]
        spectral_shape = spectral_shape.unsqueeze(2)
        
        # [B, F, T] -> [B, 1, F, T]
        noise_complex = noise_complex.unsqueeze(1)
        
        # Apply all filters at once through broadcasting
        # [B, 1, F, T] * ([1, n_bands, F, 1] * [B, n_bands, 1, T]) -> [B, n_bands, F, T]
        shaped_bands = noise_complex * (self.filter_bank_expanded * spectral_shape)
        
        # Sum across the band dimension
        # [B, n_bands, F, T] -> [B, F, T]
        shaped_noise = shaped_bands.sum(dim=1)
        
        # Apply overall noise gain
        # [B, T, 1] -> [B, 1, T]
        noise_gain = noise_params['noise_gain'].transpose(1, 2)
        
        # Scale noise by gain
        scaled_noise = shaped_noise * noise_gain
        
        return scaled_noise
        
    def generate_time_domain(self, noise_params):
        """
        Alternative implementation: Generate noise directly in time domain.
        
        This method is not used in the current implementation, but provided
        as an alternative approach.
        
        Args:
            noise_params: Dictionary containing noise parameters
                
        Returns:
            Time-domain audio signal [B, N]
        """
        batch_size = noise_params['noise_gain'].shape[0]
        n_frames = noise_params['noise_gain'].shape[1]
        device = noise_params['noise_gain'].device
        
        # Calculate total audio length
        total_samples = (n_frames - 1) * self.hop_length + self.win_length
        
        # Generate white noise
        noise = torch.randn(batch_size, total_samples, device=device)
        
        # Create time points for each sample and frame
        sample_times = torch.arange(total_samples, device=device) / self.sample_rate
        frame_times = torch.arange(n_frames, device=device) * self.hop_length / self.sample_rate
        
        # Create interpolation indices (optimization 4)
        prev_frame_idx, next_frame_idx, interp_weights = self._create_interpolation_indices(
            frame_times, sample_times)
        
        # Interpolate noise gain
        noise_gain_frames = noise_params['noise_gain'].squeeze(-1)  # [B, T]
        gain_prev = noise_gain_frames[:, prev_frame_idx]  # [B, N]
        gain_next = noise_gain_frames[:, next_frame_idx]  # [B, N]
        gain_interp = gain_prev + interp_weights * (gain_next - gain_prev)  # [B, N]
        
        # Apply time-varying gain
        noise = noise * gain_interp
        
        # Note: For proper spectral shaping in time domain, 
        # we would need to implement a time-varying filter, which
        # is more complex and less efficient than frequency-domain shaping
        
        return noise
    
    def _create_interpolation_indices(self, frame_times, sample_times):
        """
        Create indices for efficient interpolation between frames (optimization 4).
        
        Args:
            frame_times: Time points for each frame [T]
            sample_times: Time points for each sample [N]
            
        Returns:
            prev_frame_idx: Index of previous frame for each sample [N]
            next_frame_idx: Index of next frame for each sample [N]
            interp_weights: Interpolation weights for each sample [N]
        """
        n_frames = frame_times.shape[0]
        n_samples = sample_times.shape[0]
        
        # Find the index of the last frame that is <= each sample time
        # This is more efficient than creating the full time difference matrix
        prev_frame_idx = torch.searchsorted(frame_times, sample_times) - 1
        
        # Handle edge cases
        prev_frame_idx = torch.clamp(prev_frame_idx, 0, n_frames - 2)
        next_frame_idx = prev_frame_idx + 1
        
        # Get frame times for interpolation
        prev_times = frame_times[prev_frame_idx]
        next_times = frame_times[next_frame_idx]
        
        # Calculate interpolation weights
        time_deltas = next_times - prev_times
        # Add small epsilon to avoid division by zero
        time_deltas = torch.clamp(time_deltas, min=1e-7)
        
        interp_weights = (sample_times - prev_times) / time_deltas
        interp_weights = torch.clamp(interp_weights, 0.0, 1.0)
        
        return prev_frame_idx, next_frame_idx, interp_weights