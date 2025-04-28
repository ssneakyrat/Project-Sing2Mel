import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FormantNetwork(nn.Module):
    """
    Network to model vocal formants that define vowel sounds and voice quality.
    Predicts formant frequencies and bandwidths, then applies formant filtering
    to harmonic amplitudes.
    """
    def __init__(self, num_formants=5, hidden_dim=128):
        super(FormantNetwork, self).__init__()
        
        # Network to predict formant frequencies and bandwidths
        self.formant_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, num_formants * 2, kernel_size=1)  # Predict frequency and bandwidth
        )
        
        self.num_formants = num_formants
        
    def forward(self, condition, f0, harmonic_amplitudes):
        """
        Args:
            condition: Conditioning information [B, C, T]
            f0: Fundamental frequency [B, T]
            harmonic_amplitudes: Raw harmonic amplitudes [B, num_harmonics, T]
        Returns:
            shaped_amplitudes: Formant-shaped harmonic amplitudes [B, num_harmonics, T]
        """
        batch_size, num_harmonics, time_steps = harmonic_amplitudes.shape
        device = harmonic_amplitudes.device
        
        # Predict formant frequencies and bandwidths
        formant_params = self.formant_predictor(condition)  # [B, num_formants*2, T]
        
        # Split into frequencies and bandwidths
        formant_freqs = formant_params[:, :self.num_formants, :]  # [B, num_formants, T]
        # Apply sigmoid and scale to reasonable frequency range (200-3500 Hz)
        formant_freqs = 200 + 3300 * torch.sigmoid(formant_freqs)
        
        formant_bandwidths = formant_params[:, self.num_formants:, :]  # [B, num_formants, T]
        # Apply softplus to ensure positive bandwidths (50-200 Hz range)
        formant_bandwidths = 50 + 150 * F.softplus(formant_bandwidths)
        
        # Create harmonic frequency tensor
        # First, repeat f0 for each harmonic
        f0_expanded = f0.unsqueeze(1)  # [B, 1, T]
        harmonic_indices = torch.arange(1, num_harmonics + 1, device=device).reshape(1, -1, 1)  # [1, num_harmonics, 1]
        harmonic_freqs = harmonic_indices * f0_expanded  # [B, num_harmonics, T]
        
        # Apply formant filtering
        shaped_amplitudes = self._apply_formant_filter(
            harmonic_amplitudes, harmonic_freqs, formant_freqs, formant_bandwidths)
        
        return shaped_amplitudes
        
    def _apply_formant_filter(self, harmonic_amplitudes, harmonic_freqs, formant_freqs, formant_bandwidths):
        """
        Apply formant filtering to harmonic amplitudes
        Args:
            harmonic_amplitudes: [B, num_harmonics, T]
            harmonic_freqs: [B, num_harmonics, T]
            formant_freqs: [B, num_formants, T]
            formant_bandwidths: [B, num_formants, T]
        Returns:
            shaped_amplitudes: [B, num_harmonics, T]
        """
        batch_size, num_harmonics, time_steps = harmonic_amplitudes.shape
        _, num_formants, _ = formant_freqs.shape
        
        # Initialize formant gains
        formant_gain = torch.ones_like(harmonic_amplitudes)
        
        # For each formant, calculate its effect on each harmonic
        for i in range(num_formants):
            # Get current formant frequency and bandwidth
            f_freq = formant_freqs[:, i:i+1, :]  # [B, 1, T]
            f_bw = formant_bandwidths[:, i:i+1, :]  # [B, 1, T]
            
            # Calculate resonance response using simplified formant model
            # This is a resonance factor based on distance from formant center
            numerator = f_bw ** 2
            denominator = (harmonic_freqs - f_freq) ** 2 + numerator
            resonance = numerator / torch.clamp(denominator, min=1e-5)
            
            # Accumulate the effect of this formant
            formant_gain = formant_gain * (0.8 + 0.2 * resonance)
        
        # Apply formant gains to harmonic amplitudes
        shaped_amplitudes = harmonic_amplitudes * formant_gain
        
        return shaped_amplitudes


class VoiceQualityNetwork(nn.Module):
    """
    Network to model voice quality characteristics like jitter, shimmer and breathiness.
    These micro-variations add naturalness to synthesized vocal sounds.
    """
    def __init__(self, hidden_dim=128):
        super(VoiceQualityNetwork, self).__init__()
        
        # Network to predict jitter, shimmer, and breathiness
        self.quality_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 3, kernel_size=1),  # Jitter, shimmer, breathiness
            nn.Sigmoid()  # Normalized parameters
        )
        
    def forward(self, condition, audio_length):
        """
        Args:
            condition: Conditioning information [B, C, T]
            audio_length: Length of the audio to generate
        Returns:
            quality_params: Voice quality parameters [B, 3, audio_length]
                - jitter: Phase variation [B, 1, audio_length]
                - shimmer: Amplitude variation [B, 1, audio_length]
                - breathiness: Noise level [B, 1, audio_length]
        """
        batch_size, channels, time_steps = condition.shape
        device = condition.device
        
        # Predict quality parameters at frame rate
        quality_params_frames = self.quality_predictor(condition)  # [B, 3, T]
        
        # Upsample to audio rate
        quality_params = F.interpolate(
            quality_params_frames,
            size=audio_length,
            mode='linear',
            align_corners=False
        )  # [B, 3, audio_length]
        
        # Scale parameters to appropriate ranges
        jitter = quality_params[:, 0:1, :] * 0.05  # Small phase variations (0-0.05)
        shimmer = quality_params[:, 1:2, :] * 0.15  # Amplitude variations (0-0.15)
        breathiness = quality_params[:, 2:3, :] * 0.3  # Breathiness level (0-0.3)
        
        return jitter, shimmer, breathiness
        
    def apply_voice_qualities(self, harmonic_signal, phase, jitter, shimmer, breathiness):
        """
        Apply voice quality effects to the harmonic signal
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
        
        # Generate random phase variations for jitter
        jitter_noise = torch.randn(batch_size, audio_length, device=device) * jitter.squeeze(1)
        
        # Generate random amplitude variations for shimmer
        shimmer_envelope = 1.0 + (torch.randn(batch_size, audio_length, device=device) * shimmer.squeeze(1))
        
        # Generate breath noise
        breath_noise = torch.randn(batch_size, audio_length, device=device) * breathiness.squeeze(1)
        
        # Apply shimmer to the harmonic signal
        shimmer_signal = harmonic_signal * shimmer_envelope
        
        # Add breath noise
        enhanced_signal = shimmer_signal + breath_noise
        
        return enhanced_signal


class HarmonicSynthesizer(nn.Module):
    """
    Generates harmonic components based on F0 with efficiency optimizations 
    and enhanced vocal characteristics:
    1. Formant modeling for realistic vowel sounds
    2. Voice quality modeling for natural micro-variations
    3. Vectorized harmonic generation
    4. Optimized convolution architecture
    5. Efficient upsampling
    6. Dynamic harmonic selection
    7. Phase calculation caching
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
        
        # Add new vocal-specific networks
        self.formant_network = FormantNetwork(num_formants=5, hidden_dim=128)
        self.voice_quality_network = VoiceQualityNetwork(hidden_dim=128)
        
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
        
        # Apply formant shaping to harmonic amplitudes
        shaped_amplitudes = self.formant_network(condition, f0, harmonic_amplitudes)
        
        # Generate voice quality parameters
        jitter, shimmer, breathiness = self.voice_quality_network(condition, audio_length)
        
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
        
        # Apply jitter to phase - small random variations for naturalness
        jittered_phase = phase + jitter.squeeze(1) * torch.randn_like(phase)
        
        # Dynamic harmonic selection to avoid computing unnecessary harmonics
        # Calculate max possible harmonic based on Nyquist criterion
        min_f0 = torch.clamp_min(f0_upsampled, 20.0)  # Prevent division by zero with minimum f0
        max_harmonic_indices = torch.floor(self.sample_rate / (2 * min_f0)).long()
        max_harmonic = max_harmonic_indices.min().item()
        actual_harmonics = min(max_harmonic, self.num_harmonics)
        
        # Upsample amplitudes to audio length - only for harmonics we'll actually use
        harmonic_amps = self._efficient_upsample(
            shaped_amplitudes[:, :actual_harmonics, :], 
            audio_length
        )  # [B, actual_harmonics, audio_length]
        
        # Initialize output signal
        harmonic_signal = torch.zeros(batch_size, audio_length, device=device)
        
        # Generate harmonic components
        for h in range(1, actual_harmonics + 1):
            # Nyquist frequency limit check to prevent aliasing
            nyquist_mask = (h * f0_upsampled < self.sample_rate / 2).float()
            
            # Get amplitude for this harmonic
            harmonic_amp = harmonic_amps[:, h-1, :]  # [B, audio_length]
            
            # Generate sine wave for this harmonic using jittered phase
            harmonic_phase = jittered_phase * h
            harmonic = harmonic_amp * torch.sin(harmonic_phase) * nyquist_mask
            
            harmonic_signal += harmonic
        
        # Apply voice quality effects (shimmer and breathiness)
        enhanced_signal = self.voice_quality_network.apply_voice_qualities(
            harmonic_signal, phase, jitter, shimmer, breathiness
        )
        
        return enhanced_signal
        
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