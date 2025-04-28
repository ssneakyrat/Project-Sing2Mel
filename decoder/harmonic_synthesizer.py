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
    9. High frequency enhancement with efficient FFT-based filtering
    """
    def __init__(self, sample_rate=24000, hop_length=240, num_harmonics=100, input_channels=128):
        super(HarmonicSynthesizer, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.num_harmonics = num_harmonics
        self.input_channels = input_channels
        
        # Optimized amplitude predictor using depthwise separable convolutions
        # Adjusted to handle dynamic input channel count
        self.harmonic_amplitude_net = nn.Sequential(
            # Depthwise + Pointwise (separable convolution)
            nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),  # Depthwise
            nn.Conv1d(input_channels, 256, kernel_size=1),  # Pointwise
            nn.LeakyReLU(0.1),
            
            # Second separable convolution
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=256),  # Depthwise
            nn.Conv1d(256, 256, kernel_size=1),  # Pointwise
            nn.LeakyReLU(0.1),
            
            # Final layer
            nn.Conv1d(256, num_harmonics, kernel_size=1),  # Pointwise only
            nn.Softplus()
        )
        
        # High-frequency enhancement network
        self.high_freq_network = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, num_harmonics, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Add vocal-specific networks with the correct input channel count
        self.formant_network = FormantNetwork(num_formants=5, hidden_dim=128, input_channels=input_channels)
        self.voice_quality_network = VoiceQualityNetwork(hidden_dim=128, input_channels=input_channels)
        self.respiratory_network = RespiratoryDynamicsNetwork(hidden_dim=128, input_channels=input_channels)
        
        # Register buffers for caching phase calculations
        self.register_buffer('phase_cache', None)
        self.register_buffer('last_f0', None)
        self.register_buffer('last_audio_length', torch.tensor(0, dtype=torch.long))
        
        # Precompute constant tensors
        harmonic_indices = torch.arange(1, num_harmonics + 1).reshape(1, -1, 1)
        self.register_buffer('harmonic_indices', harmonic_indices)
        
        # Precompute harmonic tilt for spectral shaping
        harmonic_tilt = (harmonic_indices - 1) / (num_harmonics - 1)
        self.register_buffer('harmonic_tilt', harmonic_tilt)
        
        # Precompute spectral boost curve
        upper_mid = 2 * num_harmonics // 3
        indices = torch.arange(num_harmonics).reshape(1, -1, 1)
        boost_curve = torch.exp(-0.5 * ((indices - upper_mid) / (num_harmonics / 6)) ** 2)
        harmonic_boost = 1.0 + boost_curve * 1.0
        self.register_buffer('harmonic_boost', harmonic_boost.reshape(1, -1, 1))
        
        # Precompute high-pass filter for high frequency noise
        filter_size = 31
        highpass_filter = torch.zeros(1, 1, filter_size)
        highpass_filter[0, 0, filter_size//2] = 1.0  # Center spike
        for i in range(1, filter_size//2 + 1):
            highpass_filter[0, 0, filter_size//2 - i] = -0.5 / i
            highpass_filter[0, 0, filter_size//2 + i] = -0.5 / i
        # Normalize filter
        highpass_filter = highpass_filter / highpass_filter.abs().sum()
        self.register_buffer('highpass_filter', highpass_filter)
        
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
        
        # --- Step 1: Generate Control Parameters ---
        
        # Predict harmonic amplitudes
        harmonic_amplitudes = self.harmonic_amplitude_net(condition)  # [B, num_harmonics, T]
        
        # Predict high frequency noise component
        high_freq_noise = self.high_freq_network(condition)  # [B, num_harmonics, T]
        
        # Apply natural spectral tilt: Higher harmonics have more noise
        # Use precomputed harmonic_tilt
        high_freq_noise = torch.clamp(high_freq_noise + 0.5 * self.harmonic_tilt, 0.0, 1.0)
        
        # Apply formant shaping to harmonic amplitudes
        shaped_amplitudes = self.formant_network(condition, f0, harmonic_amplitudes)
        
        # Apply high-frequency floor to prevent complete attenuation
        high_freq_floor = 0.15 * self.harmonic_tilt
        shaped_amplitudes = torch.maximum(shaped_amplitudes, harmonic_amplitudes * high_freq_floor)
        
        # Apply spectral tilt compensation to boost specific frequency ranges
        # Use precomputed harmonic_boost
        shaped_amplitudes = shaped_amplitudes * self.harmonic_boost
        
        # Generate voice quality parameters
        jitter, shimmer, breathiness = self.voice_quality_network(condition, audio_length)
        
        # Generate respiratory dynamics and breath signal
        breath_signal, breath_features = self.respiratory_network(condition, f0, audio_length)
        
        # --- Step 2: Consolidate Signals for Efficient Upsampling ---
        
        # Determine actual harmonics to generate
        min_f0 = torch.clamp_min(f0.min(), 20.0)
        max_harmonic_indices = torch.floor(self.sample_rate / (2 * min_f0)).long()
        max_harmonic = max_harmonic_indices.min().item()
        min_required_harmonics = 40
        actual_harmonics = max(min(max_harmonic, self.num_harmonics), min_required_harmonics)
        
        # Combine signals for consolidated upsampling
        # Only include the actual harmonics we'll use
        signals_to_upsample = [
            shaped_amplitudes[:, :actual_harmonics, :],
            high_freq_noise[:, :actual_harmonics, :],
            breath_features
        ]
        
        # Concatenate along channel dimension
        combined_signals = torch.cat(signals_to_upsample, dim=1)
        
        # Single upsampling operation for all control signals
        upsampled_signals = self._efficient_upsample(combined_signals, audio_length)
        
        # Extract individual signals
        harmonic_amps = upsampled_signals[:, :actual_harmonics, :]
        high_freq_noise_upsampled = upsampled_signals[:, actual_harmonics:2*actual_harmonics, :]
        breath_features_upsampled = upsampled_signals[:, 2*actual_harmonics:, :]
        
        # Upsample f0 to audio sample rate
        f0_upsampled = self._efficient_upsample(f0.unsqueeze(1), audio_length).squeeze(1)  # [B, audio_length]
        
        # --- Step 3: Phase Generation with Caching ---
        
        # Check if we can reuse cached phase calculation
        can_use_cache = (
            self.phase_cache is not None 
            and self.last_f0 is not None
            and self.last_audio_length.item() == audio_length
            and f0.shape == self.last_f0.shape
            and torch.allclose(f0, self.last_f0, atol=1e-5)
        )
        
        if not can_use_cache:
            # Calculate instantaneous phase: integrate frequency over time
            omega = 2 * math.pi * f0_upsampled / self.sample_rate  # [B, audio_length]
            phase = torch.cumsum(omega, dim=1)  # [B, audio_length]
            
            # Update cache
            self.phase_cache = phase
            self.last_f0 = f0.clone()
            self.last_audio_length = torch.tensor(audio_length, dtype=torch.long, device=device)
        else:
            phase = self.phase_cache
        
        # Apply jitter to phase - small random variations for naturalness
        inhalation = breath_features_upsampled[:, 1:2, :]
        breath_jitter = jitter * (1.0 + inhalation * 0.5)  # Increase jitter during inhalation
        jittered_phase = phase + breath_jitter.squeeze(1) * torch.randn_like(phase)
        
        # --- Step 4: Vectorized Harmonic Generation ---
        
        # Create harmonic indices tensor for vectorized operations
        h_indices = torch.arange(1, actual_harmonics + 1, device=device).view(1, -1, 1)  # [1, actual_harmonics, 1]
        
        # Nyquist frequency limit check (vectorized)
        nyquist_mask = (h_indices * f0_upsampled.unsqueeze(1) < self.sample_rate / 2).float()  # [B, actual_harmonics, audio_length]
        
        # Generate all harmonic phases at once
        harmonic_phases = jittered_phase.unsqueeze(1) * h_indices  # [B, actual_harmonics, audio_length]
        
        # Apply phase dispersion to higher harmonics
        high_harm_mask = (h_indices > 20).float()
        # Create as float tensor to avoid type mismatch when assigning float values
        dispersion_factors = torch.zeros_like(h_indices, dtype=torch.float)
        valid_indices = h_indices > 20
        dispersion_factors[valid_indices] = (h_indices[valid_indices].float() - 20) / (actual_harmonics - 20)
        random_phases = torch.randn_like(harmonic_phases) * 0.2 * dispersion_factors * high_harm_mask
        harmonic_phases = harmonic_phases + random_phases
        
        # Generate sine waves for all harmonics at once
        harmonic_sines = torch.sin(harmonic_phases)  # [B, actual_harmonics, audio_length]
        
        # Generate noise for all harmonics at once
        harmonic_noise = torch.randn_like(harmonic_sines)
        
        # Mix sine and noise for all harmonics at once
        harmonic_waves = (1.0 - high_freq_noise_upsampled) * harmonic_sines + high_freq_noise_upsampled * harmonic_noise
        
        # Apply amplitude and nyquist mask to all harmonics at once
        harmonics = harmonic_amps * harmonic_waves * nyquist_mask
        
        # Sum all harmonics (vectorized)
        harmonic_signal = torch.sum(harmonics, dim=1)  # [B, audio_length]
        
        # --- Step 5: Apply Voice Quality Effects ---
        
        # Apply voice quality effects (shimmer and breathiness)
        exhalation = breath_features_upsampled[:, 2:3, :]
        breath_shimmer = shimmer * (1.0 + exhalation * 0.3)  # More shimmer during controlled exhalation
        
        enhanced_signal = self.voice_quality_network.apply_voice_qualities(
            harmonic_signal, phase, breath_jitter, breath_shimmer, breathiness
        )
        
        # Mix harmonic and breath signals
        voiced = (f0_upsampled > 0).float()
        output_signal = (enhanced_signal * voiced) + (breath_signal.squeeze(1) * (1.0 - voiced))
        
        # Add efficient high-frequency noise component
        high_freq_air = self._generate_high_freq_noise(audio_length, batch_size, device)
        output_signal = output_signal + high_freq_air * ((1.0 - voiced) * 0.3 + 0.05)
        
        return output_signal
    
    def _generate_high_freq_noise(self, audio_length, batch_size, device):
        """Generate filtered noise for high frequencies using FFT-based filtering"""
        # Generate white noise
        white_noise = torch.randn(batch_size, 1, audio_length, device=device)
        
        # Method 1: For shorter audio, use direct convolution with precomputed filter
        if audio_length < 10000:
            high_freq_noise = F.conv1d(
                white_noise,
                self.highpass_filter.to(device),
                padding=self.highpass_filter.shape[2]//2
            ).squeeze(1)
        # Method 2: For longer audio, use FFT-based filtering (more efficient)
        else:
            # Convert to frequency domain
            noise_fft = torch.fft.rfft(white_noise.squeeze(1))
            
            # Create frequency domain high-pass filter
            freqs = torch.fft.rfftfreq(audio_length, d=1.0/self.sample_rate, device=device)
            high_pass = (freqs > 4000).float()  # High-pass at 4000 Hz
            
            # Apply filter in frequency domain
            filtered_fft = noise_fft * high_pass
            
            # Convert back to time domain
            high_freq_noise = torch.fft.irfft(filtered_fft, n=audio_length)
        
        return high_freq_noise * 0.05  # Scale down amplitude
        
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