import torch
import torch.nn as nn
import torchaudio

class STFTGenerator(nn.Module):
    def __init__(self, 
                 n_fft=1024, 
                 hop_length=240, 
                 win_length=1024, 
                 sample_rate=24000,
                 n_harmonics=8):
        super(STFTGenerator, self).__init__()
        # STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        
        # Remove fixed harmonic_distribution parameter
        # Keep the noise level parameter
        self.noise_level = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
    def forward(self, f0, harmonic_amplitudes=None):
        """
        Generate simplified STFT with harmonic and noise components.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            harmonic_amplitudes: Dynamic harmonic amplitude values [B, T, n_harmonics]
                If None, use default falloff pattern.
            
        Returns:
            Complex STFT representation [B, F, T] where F = n_fft//2 + 1
        """
        batch_size, n_frames = f0.shape
        n_freqs = self.n_fft // 2 + 1
        device = f0.device
        
        # If harmonic_amplitudes is not provided, use default falloff pattern
        if harmonic_amplitudes is None:
            # Create default harmonic amplitudes with falloff pattern
            default_distribution = torch.tensor(
                [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125][:self.n_harmonics],
                device=device
            )
            # Expand to [B, T, n_harmonics]
            harmonic_amplitudes = default_distribution.view(1, 1, -1).expand(batch_size, n_frames, -1)
        
        # 1. Generate harmonic components with dynamic amplitudes
        harmonic_stft = self._generate_simple_harmonics(f0, n_freqs, harmonic_amplitudes)
        
        # 2. Generate simple noise component in frequency domain
        noise_stft = self._generate_simple_noise(batch_size, n_frames, n_freqs)
        
        # 3. Combine components
        combined_stft = harmonic_stft + noise_stft
        
        return combined_stft
        
    def _generate_simple_harmonics(self, f0, n_freqs, harmonic_amplitudes):
        """
        Differentiable harmonic generation using soft binning and dynamic amplitudes.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            n_freqs: Number of frequency bins
            harmonic_amplitudes: Harmonic amplitude values [B, T, n_harmonics]
        """
        batch_size, n_frames = f0.shape
        device = f0.device
        
        # Initialize empty STFT tensor
        harmonic_stft_real = torch.zeros((batch_size, n_freqs, n_frames), device=device)
        harmonic_stft_imag = torch.zeros((batch_size, n_freqs, n_frames), device=device)
        
        # Create meshgrid for all bin indices
        all_bins = torch.arange(n_freqs, device=device).view(1, -1, 1)  # [1, F, 1]
        
        # Vectorized approach: create tensors for all harmonics at once
        # Create harmonic numbers tensor
        harmonic_numbers = torch.arange(1, self.n_harmonics + 1, device=device).float()
        
        # Calculate all harmonic frequencies at once
        # [B, T, 1] * [n_harmonics] -> [B, T, n_harmonics]
        all_harmonic_freqs = f0.unsqueeze(-1) * harmonic_numbers.view(1, 1, -1)
        
        # Calculate bin indices for all harmonics
        # [B, T, n_harmonics]
        all_bin_indices = all_harmonic_freqs * self.n_fft / self.sample_rate
        
        # Process each harmonic (still need a loop, but we're more vectorized inside)
        for h in range(self.n_harmonics):
            # Get bin indices for this harmonic [B, T]
            bin_indices_float = all_bin_indices[:, :, h]
            
            # Reshape for broadcasting
            bin_indices_float = bin_indices_float.unsqueeze(1)  # [B, 1, T]
            
            # Compute distance from each bin center (soft binning)
            distance = torch.abs(all_bins - bin_indices_float)  # [B, F, T]
            
            # Use a triangular window for soft binning
            window_width = 1.0
            weight = torch.clamp(1.0 - distance / window_width, min=0.0)
            
            # Apply amplitude from harmonic_amplitudes
            # harmonic_amplitudes shape is [B, T, n_harmonics]
            # Reshape to [B, 1, T] for broadcasting with weight [B, F, T]
            amplitude = harmonic_amplitudes[:, :, h].unsqueeze(1)  # [B, 1, T]
            
            # Apply contribution
            contribution = weight * amplitude
            
            # Add contribution to the STFT real part
            harmonic_stft_real += contribution
        
        # Combine real and imaginary parts into complex STFT
        harmonic_stft = torch.complex(harmonic_stft_real, harmonic_stft_imag)
        return harmonic_stft
        
    def _generate_simple_noise(self, batch_size, n_frames, n_freqs):
        """
        Generate simple white noise in the STFT domain.
        """
        device = next(self.parameters()).device
        
        # Generate random complex values for noise
        noise_real = torch.randn(batch_size, n_freqs, n_frames, device=device) * self.noise_level
        noise_imag = torch.randn(batch_size, n_freqs, n_frames, device=device) * self.noise_level
        
        noise_stft = torch.complex(noise_real, noise_imag)
        return noise_stft