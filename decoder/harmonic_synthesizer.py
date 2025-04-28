import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HarmonicSynthesizer(nn.Module):
    """
    Generates harmonic components based on F0 with efficiency optimizations:
    1. Vectorized harmonic generation
    2. Optimized convolution architecture
    3. Efficient upsampling
    4. Dynamic harmonic selection
    5. Phase calculation caching
    """
    def __init__(self, sample_rate=24000, hop_length=240, num_harmonics=100):
        super(HarmonicSynthesizer, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.num_harmonics = num_harmonics
        
        # Optimized amplitude predictor using depthwise separable convolutions
        self.harmonic_amplitude_net = nn.Sequential(
            # Depthwise + Pointwise (separable convolution)
            nn.Conv1d(128, 128, kernel_size=3, padding=1, groups=128),  # Depthwise
            nn.Conv1d(128, 256, kernel_size=1),  # Pointwise
            nn.LeakyReLU(0.1),
            
            # Second separable convolution
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=256),  # Depthwise
            nn.Conv1d(256, 256, kernel_size=1),  # Pointwise
            nn.LeakyReLU(0.1),
            
            # Final layer
            nn.Conv1d(256, num_harmonics, kernel_size=1),  # Pointwise only
            nn.Softplus()
        )
        
        # Register buffers for caching phase calculations
        self.register_buffer('phase_cache', None)
        self.register_buffer('last_f0', None)
        # Store audio length as a scalar tensor
        self.register_buffer('last_audio_length', torch.tensor(0, dtype=torch.long))
        
    def forward(self, f0, condition, audio_length):
        """
        Args:
            f0: Fundamental frequency contour [B, T]
            condition: Conditioning information [B, C, T]
            audio_length: Target audio length in samples
        Returns:
            harmonic_signal: Generated harmonic signal [B, audio_length]
        """
        batch_size, time_steps = f0.shape
        device = f0.device
        
        # Predict harmonic amplitudes
        harmonic_amplitudes = self.harmonic_amplitude_net(condition)  # [B, num_harmonics, T]
        
        # Upsample f0 to audio sample rate
        f0_upsampled = self._efficient_upsample(f0.unsqueeze(1), audio_length).squeeze(1)  # [B, audio_length]
        
        # Check if we can reuse cached phase calculation
        can_use_cache = (
            self.phase_cache is not None 
            and self.last_f0 is not None
            and self.last_audio_length.item() == audio_length
            and f0.shape == self.last_f0.shape  # Check shapes match first
            and torch.allclose(f0, self.last_f0, atol=1e-5)
        )
        
        if not can_use_cache:
            # Calculate instantaneous phase: integrate frequency over time
            # Convert from Hz to radians per sample
            omega = 2 * math.pi * f0_upsampled / self.sample_rate  # [B, audio_length]
            phase = torch.cumsum(omega, dim=1)  # [B, audio_length]
            
            # Update cache
            self.phase_cache = phase
            self.last_f0 = f0.clone()
            self.last_audio_length = torch.tensor(audio_length, dtype=torch.long, device=device)
        else:
            phase = self.phase_cache
            
        # Dynamic harmonic selection to avoid computing unnecessary harmonics
        # Calculate max possible harmonic based on Nyquist criterion
        min_f0 = torch.clamp_min(f0_upsampled, 20.0)  # Prevent division by zero with minimum f0
        max_harmonic_indices = torch.floor(self.sample_rate / (2 * min_f0)).long()
        max_harmonic = max_harmonic_indices.min().item()
        actual_harmonics = min(max_harmonic, self.num_harmonics)
        
        # Create harmonic indices tensor for vectorized computation
        harmonic_indices = torch.arange(1, actual_harmonics + 1, device=device).float()
        
        # Upsample amplitudes to audio length - only for harmonics we'll actually use
        harmonic_amps = self._efficient_upsample(
            harmonic_amplitudes[:, :actual_harmonics, :], 
            audio_length
        )  # [B, actual_harmonics, audio_length]
        
        # Initialize output signal
        harmonic_signal = torch.zeros(batch_size, audio_length, device=device)
        
        # Simpler approach for harmonic generation - process one harmonic at a time
        # but use vectorized operations within each harmonic
        for h in range(1, actual_harmonics + 1):
            # Nyquist frequency limit check to prevent aliasing
            nyquist_mask = (h * f0_upsampled < self.sample_rate / 2).float()
            
            # Get amplitude for this harmonic
            harmonic_amp = harmonic_amps[:, h-1, :]  # [B, audio_length]
            
            # Generate sine wave for this harmonic
            harmonic_phase = phase * h
            harmonic = harmonic_amp * torch.sin(harmonic_phase) * nyquist_mask
            
            harmonic_signal += harmonic
            
        return harmonic_signal
        
    def _efficient_upsample(self, tensor, target_len):
        """More efficient upsampling with reduced memory footprint"""
        # For small tensors, interpolate is fine
        if tensor.shape[2] * target_len < 1e7:  # Heuristic threshold
            return F.interpolate(
                tensor, 
                size=target_len, 
                mode='linear', 
                align_corners=False
            )
        
        # For large tensors, use block processing
        batch_size, channels, time_steps = tensor.shape
        scale_factor = target_len / time_steps
        
        # Process in time blocks to save memory
        result = torch.zeros(batch_size, channels, target_len, device=tensor.device)
        block_size = min(1000, time_steps)  # Process 1000 frames at a time
        
        for block_start in range(0, time_steps, block_size):
            block_end = min(block_start + block_size, time_steps)
            block = tensor[:, :, block_start:block_end]
            
            # Calculate corresponding output indices
            out_start = int(block_start * scale_factor)
            out_end = min(int(block_end * scale_factor), target_len)
            
            # Interpolate this block
            upsampled_block = F.interpolate(
                block,
                size=out_end - out_start,
                mode='linear',
                align_corners=False
            )
            
            # Insert into result
            result[:, :, out_start:out_end] = upsampled_block
            
        return result