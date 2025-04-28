import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VoiceQualityNetwork(nn.Module):
    """
    Optimized network to model voice quality characteristics like jitter, shimmer and breathiness.
    - Improved parameter prediction with grouped convolutions
    - Batched noise generation
    - Vectorized operations for applying effects
    """
    def __init__(self, hidden_dim=128, input_channels=None):
        super(VoiceQualityNetwork, self).__init__()
        
        # If input_channels is not provided, use hidden_dim
        if input_channels is None:
            input_channels = hidden_dim
            
        # Network to predict jitter, shimmer, and breathiness using depthwise separable convolutions
        self.quality_predictor = nn.Sequential(
            # First depthwise separable convolution (more efficient)
            nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),  # Depthwise
            nn.Conv1d(input_channels, 64, kernel_size=1),  # Pointwise
            nn.LeakyReLU(0.1),
            
            # Second depthwise separable convolution
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=64),  # Depthwise
            nn.Conv1d(64, 32, kernel_size=1),  # Pointwise
            nn.LeakyReLU(0.1),
            
            # Final pointwise convolution
            nn.Conv1d(32, 3, kernel_size=1),  # Jitter, shimmer, breathiness
            nn.Sigmoid()  # Normalized parameters
        )
        
        # Register buffer for noise generation (reusable)
        self.register_buffer('noise_cache', None)
        self.register_buffer('last_shape', None)
        
    def forward(self, condition, audio_length):
        """
        Args:
            condition: Conditioning information [B, C, T]
            audio_length: Length of the audio to generate
        Returns:
            quality_params: Voice quality parameters at audio rate
                - jitter: Phase variation [B, 1, audio_length]
                - shimmer: Amplitude variation [B, 1, audio_length]
                - breathiness: Noise level [B, 1, audio_length]
        """
        batch_size, channels, time_steps = condition.shape
        device = condition.device
        
        # Predict quality parameters at frame rate
        quality_params_frames = self.quality_predictor(condition)  # [B, 3, T]
        
        # Upsample to audio rate with efficient upsampling
        quality_params = self._efficient_upsample(
            quality_params_frames,
            audio_length
        )  # [B, 3, audio_length]
        
        # Scale parameters to appropriate ranges (vectorized)
        jitter = quality_params[:, 0:1, :] * 0.05  # Small phase variations (0-0.05)
        shimmer = quality_params[:, 1:2, :] * 0.15  # Amplitude variations (0-0.15)
        breathiness = quality_params[:, 2:3, :] * 0.3  # Breathiness level (0-0.3)
        
        return jitter, shimmer, breathiness
    
    def _efficient_upsample(self, tensor, target_len):
        """
        Memory-efficient upsampling with block processing for large tensors
        """
        # For small tensors, use direct interpolation
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
        
    def _generate_cached_noise(self, shape, device):
        """
        Generate noise tensors with caching for reuse when possible
        """
        # Check if we can reuse the cached noise
        if (self.noise_cache is not None and 
            self.last_shape is not None and
            torch.all(torch.tensor(shape, device=device) == self.last_shape.to(device))):
            return self.noise_cache.to(device)
        
        # Otherwise, generate new noise and cache it
        noise = torch.randn(*shape, device=device)
        self.noise_cache = noise
        self.last_shape = torch.tensor(shape, device=device)
        
        return noise
        
    def apply_voice_qualities(self, harmonic_signal, phase, jitter, shimmer, breathiness):
        """
        Apply voice quality effects to the harmonic signal with vectorized operations
        
        Args:
            harmonic_signal: Base harmonic signal [B, audio_length]
            phase: Phase information [B, audio_length]
            jitter: Phase variation [B, 1, audio_length]
            shimmer: Amplitude variation [B, 1, audio_length]
            breathiness: Noise level [B, 1, audio_length]
        Returns:
            enhanced_signal: Signal with added voice qualities [B, audio_length]
        """
        batch_size, audio_length = harmonic_signal.shape
        device = harmonic_signal.device
        
        # Generate all random variations at once using cached noise when possible
        # Shape: [B, 3, audio_length] for jitter, shimmer, and breathiness noise
        all_noise = self._generate_cached_noise((batch_size, 3, audio_length), device)
        
        # Extract individual noise components
        jitter_noise = all_noise[:, 0] * jitter.squeeze(1)
        shimmer_envelope = 1.0 + (all_noise[:, 1] * shimmer.squeeze(1))
        breath_noise = all_noise[:, 2] * breathiness.squeeze(1)
        
        # Apply all effects in a single vectorized operation
        enhanced_signal = harmonic_signal * shimmer_envelope + breath_noise
        
        return enhanced_signal
        
    def generate_correlated_noise(self, base_signal, correlation_factor, noise_level):
        """
        Generate noise that's partially correlated with the base signal
        This adds more natural variation tied to the signal content
        
        Args:
            base_signal: Reference signal to correlate with [B, audio_length]
            correlation_factor: How much to correlate (0-1) [B, 1, audio_length]
            noise_level: Overall noise amplitude [B, 1, audio_length]
        Returns:
            correlated_noise: Partially signal-correlated noise [B, audio_length]
        """
        batch_size, audio_length = base_signal.shape
        device = base_signal.device
        
        # Generate pure noise
        pure_noise = torch.randn_like(base_signal)
        
        # Apply envelope following for correlation
        # Use the absolute value of the base signal as a rough envelope
        signal_envelope = torch.abs(base_signal)
        
        # Normalize the envelope
        max_env = torch.max(signal_envelope, dim=1, keepdim=True)[0] + 1e-5
        normalized_envelope = signal_envelope / max_env
        
        # Mix pure noise with envelope-shaped noise according to correlation factor
        correlated_noise = (
            (1 - correlation_factor.squeeze(1)) * pure_noise + 
            correlation_factor.squeeze(1) * pure_noise * normalized_envelope
        )
        
        # Apply overall noise level
        scaled_noise = correlated_noise * noise_level.squeeze(1)
        
        return scaled_noise