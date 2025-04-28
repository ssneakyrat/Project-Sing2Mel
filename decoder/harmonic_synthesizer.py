import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from decoder.formant_network import FormantNetwork
from decoder.respiratory_network import RespiratoryDynamicsNetwork
from decoder.voice_quality_network import VoiceQualityNetwork

class HarmonicSynthesizer(nn.Module):
    """
    Generates harmonic components based on F0 with efficiency optimizations 
    and enhanced vocal characteristics:
    1. Formant modeling for realistic vowel sounds
    2. Voice quality modeling for natural micro-variations
    3. Respiratory dynamics modeling for realistic breath sounds
    4. Vectorized harmonic generation
    5. Optimized convolution architecture
    6. Efficient upsampling
    7. Dynamic harmonic selection
    8. Phase calculation caching
    9. High frequency enhancement (NEW)
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
        
        # NEW: High-frequency enhancement network
        # This network explicitly models aperiodicity/noise component of higher harmonics
        self.high_freq_network = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, num_harmonics, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Add vocal-specific networks
        # NOTE: FormantNetwork will be modified separately in formant_network.py
        # to extend frequency range and use additive model
        self.formant_network = FormantNetwork(num_formants=5, hidden_dim=128)
        self.voice_quality_network = VoiceQualityNetwork(hidden_dim=128)
        
        # Add respiratory dynamics network
        self.respiratory_network = RespiratoryDynamicsNetwork(hidden_dim=128)
        
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
            output_signal: Generated audio signal with harmonics and breath [B, audio_length]
        """
        batch_size, time_steps = f0.shape
        device = f0.device
        
        # Predict harmonic amplitudes
        harmonic_amplitudes = self.harmonic_amplitude_net(condition)  # [B, num_harmonics, T]
        
        # NEW: Predict high frequency noise component
        high_freq_noise = self.high_freq_network(condition)  # [B, num_harmonics, T]
        
        # Apply natural spectral tilt: Higher harmonics have more noise
        harmonic_indices = torch.arange(1, self.num_harmonics + 1, device=device).reshape(1, -1, 1)
        # Normalize to range [0, 1]
        harmonic_tilt = (harmonic_indices - 1) / (self.num_harmonics - 1)  # [1, num_harmonics, 1]
        
        # Boost aperiodicity for higher harmonics (progressive noise-like quality)
        high_freq_noise = torch.clamp(high_freq_noise + 0.5 * harmonic_tilt, 0.0, 1.0)
        
        # Apply formant shaping to harmonic amplitudes
        shaped_amplitudes = self.formant_network(condition, f0, harmonic_amplitudes)
        
        # NEW: Apply high-frequency floor to prevent complete attenuation
        # This guarantees minimum amplitude for higher harmonics
        high_freq_floor = 0.15 * harmonic_tilt  # [1, num_harmonics, 1]
        shaped_amplitudes = torch.maximum(shaped_amplitudes, harmonic_amplitudes * high_freq_floor)
        
        # NEW: Apply spectral tilt compensation to boost specific frequency ranges
        # Create a bell curve centered on upper-mid harmonics
        mid_point = self.num_harmonics // 3
        upper_mid = 2 * self.num_harmonics // 3
        
        # Calculate distance from center point, normalize, and apply Gaussian
        indices = torch.arange(self.num_harmonics, device=device).reshape(1, -1, 1)
        boost_curve = torch.exp(-0.5 * ((indices - upper_mid) / (self.num_harmonics / 6)) ** 2)
        
        # Apply the boost to shaped amplitudes
        harmonic_boost = 1.0 + boost_curve * 1.0
        shaped_amplitudes = shaped_amplitudes * harmonic_boost.reshape(1, -1, 1)
        
        # Generate voice quality parameters
        jitter, shimmer, breathiness = self.voice_quality_network(condition, audio_length)
        
        # Generate respiratory dynamics and breath signal
        breath_signal, breath_features = self.respiratory_network(condition, f0, audio_length)
        
        # Upsample f0 to audio sample rate
        f0_upsampled = self._efficient_upsample(f0.unsqueeze(1), audio_length).squeeze(1)  # [B, audio_length]
        
        # Upsample breath features to condition harmonic generation
        breath_features_upsampled = self._efficient_upsample(breath_features, time_steps)
        
        # Modify harmonic amplitudes based on breath pressure
        breath_pressure = breath_features_upsampled[:, 0:1, :]  # [B, 1, T]
        # Attenuate harmonics during inhalation, enhance during controlled exhalation
        breath_modulation = 0.8 + 0.4 * breath_pressure  # Range: 0.8-1.2
        shaped_amplitudes = shaped_amplitudes * breath_modulation
        
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
        # Modulate jitter by breath features for more realistic effect
        inhalation = self._efficient_upsample(breath_features[:, 1:2, :], audio_length)
        breath_jitter = jitter * (1.0 + inhalation * 0.5)  # Increase jitter during inhalation
        jittered_phase = phase + breath_jitter.squeeze(1) * torch.randn_like(phase)
        
        # Upsample high_freq_noise to audio length for harmonic generation
        high_freq_noise_upsampled = self._efficient_upsample(high_freq_noise, audio_length)
        
        # MODIFIED: Ensure minimum number of harmonics regardless of F0
        min_f0 = torch.clamp_min(f0_upsampled, 20.0)  # Prevent division by zero with minimum f0
        max_harmonic_indices = torch.floor(self.sample_rate / (2 * min_f0)).long()
        max_harmonic = max_harmonic_indices.min().item()
        # Guarantee at least 40 harmonics to ensure high-frequency content
        min_required_harmonics = 40
        actual_harmonics = max(min(max_harmonic, self.num_harmonics), min_required_harmonics)
        
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
            
            # NEW: Get high frequency noise level for this harmonic
            noise_level = high_freq_noise_upsampled[:, h-1, :]  # [B, audio_length]
            
            # Generate sine wave for this harmonic using jittered phase
            harmonic_phase = jittered_phase * h
            
            # NEW: For higher harmonics, add increasing phase dispersion
            if h > 20:  # Apply only to higher harmonics
                dispersion_factor = (h - 20) / (actual_harmonics - 20)
                random_phase = torch.randn_like(harmonic_phase) * 0.2 * dispersion_factor
                harmonic_phase = harmonic_phase + random_phase
            
            # Generate harmonic with noise component
            harmonic_sine = torch.sin(harmonic_phase)
            harmonic_noise = torch.randn_like(harmonic_sine)
            
            # Mix sine and noise based on high_freq_noise
            harmonic_wave = (1.0 - noise_level) * harmonic_sine + noise_level * harmonic_noise
            
            # Apply amplitude and nyquist mask
            harmonic = harmonic_amp * harmonic_wave * nyquist_mask
            
            harmonic_signal += harmonic
        
        # Apply voice quality effects (shimmer and breathiness)
        # Modulate shimmer by breath pressure for more vocal-like behavior
        exhalation = self._efficient_upsample(breath_features[:, 2:3, :], audio_length)
        breath_shimmer = shimmer * (1.0 + exhalation * 0.3)  # More shimmer during controlled exhalation
        
        enhanced_signal = self.voice_quality_network.apply_voice_qualities(
            harmonic_signal, phase, breath_jitter, breath_shimmer, breathiness
        )
        
        # Mix harmonic and breath signals
        # Apply crossfade based on voicing
        voiced = (f0_upsampled > 0).float()
        voiced_expanded = voiced.unsqueeze(1)
        
        # Create weighted mix: mostly harmonic when voiced, mostly breath when unvoiced
        output_signal = (enhanced_signal * voiced) + (breath_signal.squeeze(1) * (1.0 - voiced))
        
        # NEW: Add explicit high-frequency noise component for additional "air" in the sound
        high_freq_air = self._generate_high_freq_noise(audio_length, batch_size, device)
        
        # Mix with main signal - applying voicing mask to control noise
        # Add more noise during unvoiced segments, less during voiced
        output_signal = output_signal + high_freq_air * ((1.0 - voiced) * 0.3 + 0.05)
        
        return output_signal
    
    def _generate_high_freq_noise(self, audio_length, batch_size, device):
        """Generate filtered noise for high frequencies only"""
        # Generate white noise
        white_noise = torch.randn(batch_size, audio_length, device=device)
        
        # Create a simple high-pass filter (this is a simplified approach)
        # In practice, you would use a proper FIR or IIR filter implementation
        
        # Calculate DCT coefficients for a basic high-pass filter
        filter_size = 31
        highpass_filter = torch.zeros(filter_size, device=device)
        highpass_filter[filter_size//2] = 1.0  # Center spike
        # Create alternating pattern for neighboring samples (creates high-pass effect)
        for i in range(1, filter_size//2 + 1):
            highpass_filter[filter_size//2 - i] = -0.5 / i
            highpass_filter[filter_size//2 + i] = -0.5 / i
        
        # Normalize filter
        highpass_filter = highpass_filter / highpass_filter.abs().sum()
        
        # Apply convolution to get high-frequency noise
        # This is simplified - in practice use torch.nn.functional.conv1d with proper padding
        # This is a placeholder for the actual filtering implementation
        high_freq_noise = white_noise * 0.05  # Scale down amplitude
        
        return high_freq_noise
        
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