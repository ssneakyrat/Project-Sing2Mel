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
        
        # Create band filters (triangular filters in mel space)
        # These will be used to shape the noise spectrum
        filters = torch.zeros(n_bands, self.n_freqs)
        
        for i in range(n_bands):
            # Create triangular filter for each band
            filt = torch.zeros(self.n_freqs)
            
            # Rising edge
            start, center, end = fft_bins[i], fft_bins[i+1], fft_bins[i+2]
            
            # Handle case where bins may be the same due to rounding
            if center > start:
                ramp_up = torch.linspace(0, 1, center - start)
                filt[start:center] = ramp_up
            
            # Falling edge
            if end > center:
                ramp_down = torch.linspace(1, 0, end - center)
                filt[center:end] = ramp_down
            
            filters[i] = filt
        
        # Register filter bank
        self.register_buffer('filter_bank', filters)
        
    def forward(self, noise_params):
        """
        Generate spectrally shaped noise in STFT domain.
        
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
        
        # For each frame, generate white noise and shape its spectrum
        
        # Generate complex white noise in the frequency domain
        # Shape: [B, F, T]
        noise_real = torch.randn(batch_size, self.n_freqs, n_frames, device=device)
        noise_imag = torch.randn(batch_size, self.n_freqs, n_frames, device=device)
        noise_complex = torch.complex(noise_real, noise_imag)
        
        # Apply spectral shaping using the filter bank
        spectral_shape = noise_params['spectral_shape'].transpose(1, 2)  # [B, n_bands, T]
        
        # Reshape for matrix multiplication
        # Filters: [n_bands, F], Spectral shape: [B, n_bands, T]
        # Target: [B, F, T]
        
        # Step 1: Apply each band filter to the whole spectrum
        # For each band, we have a filter shape [F] that we apply to the noise
        shaped_noise = torch.zeros_like(noise_complex)
        
        # Iterate through bands and apply spectral shaping
        for b in range(self.n_bands):
            # Get the filter for this band [F]
            band_filter = self.filter_bank[b].unsqueeze(0).unsqueeze(-1)  # [1, F, 1]
            
            # Get the gain for this band across all frames [B, 1, T]
            band_gain = spectral_shape[:, b:b+1, :]  # [B, 1, T]
            
            # Apply the filter with its gain to the noise
            # [1, F, 1] * [B, 1, T] -> [B, F, T]
            shaped_band = noise_complex * (band_filter * band_gain)
            
            # Add to the total shaped noise
            shaped_noise = shaped_noise + shaped_band
        
        # Apply overall noise gain
        # [B, T, 1] -> [B, 1, T]
        noise_gain = noise_params['noise_gain'].transpose(1, 2)
        
        # Scale noise by gain - broadcasting will apply to each frequency bin
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
        
        # Create interpolation indices
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
        Create indices for efficient interpolation between frames.
        Identical to the method in HarmonicGenerator.
        
        Args:
            frame_times: Time points for each frame [T]
            sample_times: Time points for each sample [N]
            
        Returns:
            prev_frame_idx: Index of previous frame for each sample [N]
            next_frame_idx: Index of next frame for each sample [N]
            interp_weights: Interpolation weights for each sample [N]
        """
        n_frames = frame_times.shape[0]
        device = frame_times.device
        
        # Create sample times matrix of shape [N, 1]
        sample_times = sample_times.unsqueeze(1)
        
        # Create frame times matrix of shape [1, T]
        frame_times_mat = frame_times.unsqueeze(0)
        
        # Calculate time differences: [N, T]
        # Each row represents all frame time differences for a single sample time
        time_diffs = sample_times - frame_times_mat
        
        # Create mask of valid (non-negative) differences: [N, T]
        # For each sample, all frames that occurred earlier or at the same time will be 1
        valid_frames = (time_diffs >= 0).float()
        
        # For each sample, calculate the latest valid frame
        # Sum valid_frames across dim 1 and subtract 1 to get the index
        prev_frame_idx = torch.sum(valid_frames, dim=1).long() - 1
        
        # Handle edge cases for first and last frames
        prev_frame_idx = torch.clamp(prev_frame_idx, 0, n_frames - 2)
        next_frame_idx = prev_frame_idx + 1
        
        # Get actual frame times for interpolation
        prev_times = frame_times[prev_frame_idx]
        next_times = frame_times[next_frame_idx]
        
        # Calculate interpolation weights
        time_deltas = next_times - prev_times
        time_deltas = torch.clamp(time_deltas, min=1e-7)  # Avoid division by zero
        
        interp_weights = (sample_times.squeeze(1) - prev_times) / time_deltas
        interp_weights = torch.clamp(interp_weights, 0.0, 1.0)
        
        return prev_frame_idx, next_frame_idx, interp_weights