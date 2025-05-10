import torch
import torch.nn as nn
import torchaudio
import math

class HarmonicGenerator(nn.Module):
    def __init__(self, 
                 n_fft=1024, 
                 hop_length=240, 
                 win_length=1024, 
                 sample_rate=24000,
                 n_harmonics=80):
        super(HarmonicGenerator, self).__init__()
        # STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        
        # Pre-compute harmonic multipliers
        self.register_buffer('harmonic_multipliers', 
                             torch.arange(1, n_harmonics + 1).float())
        
        # Pre-compute default harmonic amplitudes
        default_distribution = torch.tensor([
            1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125
        ][:self.n_harmonics])
        self.register_buffer('default_harmonic_amps', default_distribution)
        
    def forward(self, f0, harmonic_amplitudes=None):
        """
        Generate audio by synthesizing harmonics in time domain, then converting to STFT.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            harmonic_amplitudes: Dynamic harmonic amplitude values [B, T, n_harmonics]
                If None, use default falloff pattern.
            
        Returns:
            Complex STFT representation [B, F, T] where F = n_fft//2 + 1
        """
        batch_size, n_frames = f0.shape
        device = f0.device
        
        # Set up harmonic amplitudes
        if harmonic_amplitudes is None:
            harmonic_amplitudes = self.default_harmonic_amps.view(1, 1, -1).expand(
                batch_size, n_frames, self.n_harmonics)
        
        # 1. Generate time-domain signal
        audio = self._generate_harmonics_vectorized(f0, harmonic_amplitudes)
        
        # 2. Add noise if desired
        #if self.noise_level > 0:
        #    audio = audio + torch.randn_like(audio) * self.noise_level
        
        # 3. Convert to STFT domain
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=device),
            return_complex=True
        )
        
        # Trim to match expected number of frames
        return stft[:, :, :n_frames]
    
    def _generate_harmonics_vectorized(self, f0, harmonic_amplitudes):
        """
        Fully vectorized harmonic signal generation.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            harmonic_amplitudes: Harmonic amplitude values [B, T, n_harmonics]
            
        Returns:
            Audio waveform [B, N] where N is the number of audio samples
        """
        batch_size, n_frames = f0.shape
        device = f0.device
        
        # Calculate total audio samples
        total_samples = (n_frames - 1) * self.hop_length + self.win_length
        
        # Create time points for each sample
        sample_times = torch.arange(total_samples, device=device) / self.sample_rate
        frame_times = torch.arange(n_frames, device=device) * self.hop_length / self.sample_rate
        
        # Process each batch in parallel
        audio = torch.zeros(batch_size, total_samples, device=device)
        
        # Vectorized interpolation for all batches at once
        # This avoids the need for a batch loop
        
        # Create indices for batch dimension to use in advanced indexing
        batch_indices = torch.arange(batch_size, device=device)
        
        # Create index tensors for sample-based lookup
        # These allow us to map from each audio sample to the appropriate frame
        prev_frame_idx, next_frame_idx, interp_weights = self._create_interpolation_indices(
            frame_times, sample_times)
        
        # Setup for interpolating f0 values
        # Shape: [B, N]
        f0_prev = f0[:, prev_frame_idx]
        f0_next = f0[:, next_frame_idx]
        f0_interp = f0_prev + interp_weights * (f0_next - f0_prev)
        f0_interp = torch.clamp(f0_interp, min=0.0)
        
        # Calculate phase by integrating frequency
        phase = torch.cumsum(2 * math.pi * f0_interp / self.sample_rate, dim=1)
        
        # Get harmonic amplitudes for each sample
        # First interpolate all harmonics at once
        # Shape: [B, N, H]
        h_amps_prev = harmonic_amplitudes[:, prev_frame_idx, :]
        h_amps_next = harmonic_amplitudes[:, next_frame_idx, :]
        h_amps_interp = h_amps_prev + interp_weights.unsqueeze(-1) * (h_amps_next - h_amps_prev)
        
        # Vectorized harmonic synthesis - calculate all harmonics at once
        # We use broadcasting to create all harmonic phases
        # Shape: [B, N, H]
        harmonic_phases = phase.unsqueeze(-1) * self.harmonic_multipliers.view(1, 1, -1)
        
        # Generate all harmonics at once
        # Shape: [B, N, H]
        harmonic_signals = h_amps_interp * torch.sin(harmonic_phases)
        
        # Sum all harmonics
        # Shape: [B, N]
        audio = torch.sum(harmonic_signals, dim=2)
        
        # Normalize audio (without loop)
        max_vals = torch.max(torch.abs(audio), dim=1, keepdim=True)[0]
        max_vals = torch.clamp(max_vals, min=1e-7)
        audio = audio / max_vals
        
        return audio
    
    def _create_interpolation_indices(self, frame_times, sample_times):
        """
        Create indices for efficient interpolation.
        
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