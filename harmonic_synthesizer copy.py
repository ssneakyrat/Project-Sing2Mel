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


class ArticulationNoiseNetwork(nn.Module):
    """
    Network to model consonant-specific noise characteristics and articulation transitions.
    Handles different noise types including:
    1. Fricative noise (for s, sh, f, v sounds)
    2. Plosive bursts (for p, t, k sounds)
    3. Transient articulation effects
    4. Vocal tract constriction noise
    """
    def __init__(self, hidden_dim=128, num_noise_bands=24):
        super(ArticulationNoiseNetwork, self).__init__()
        
        self.num_noise_bands = num_noise_bands
        
        # Network to predict noise characteristics
        self.noise_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, num_noise_bands + 3, kernel_size=1)  # Bands + timing parameters
        )
        
        # Spectral shaping for different consonant types
        self.spectral_shaper = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 4, kernel_size=1),  # Different noise types (fricative, plosive, etc.)
            nn.Softmax(dim=1)
        )
        
        # Filter bank for noise shaping
        self.filter_bank = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=31, padding=15, bias=False)
            for _ in range(num_noise_bands)
        ])
        
        # Initialize filters with different frequency responses
        self._init_filter_bank()
        
        # Noise type specific filters
        self.noise_type_filters = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=63, padding=31, bias=False),  # Fricative (high-pass)
            nn.Conv1d(1, 1, kernel_size=63, padding=31, bias=False),  # Plosive (band-pass with attack)
            nn.Conv1d(1, 1, kernel_size=63, padding=31, bias=False),  # Nasal (low-pass)
            nn.Conv1d(1, 1, kernel_size=63, padding=31, bias=False)   # Transition (dynamic)
        ])
        
        # Initialize noise type filters
        self._init_noise_type_filters()
        
    def _init_filter_bank(self):
        """Initialize the filter bank with bandpass filters of different frequencies"""
        for i, filter_module in enumerate(self.filter_bank):
            # Create bandpass filter coefficients
            center_freq = 100 + (i * 7000 / self.num_noise_bands)  # From 100Hz to ~7.1kHz
            bandwidth = 50 + (i * 100 / self.num_noise_bands)      # Increases with frequency
            
            # Simple sinc filter initialization (approximating bandpass)
            kernel_size = filter_module.weight.shape[2]
            half_kernel = kernel_size // 2
            
            for k in range(kernel_size):
                if k == half_kernel:
                    filter_module.weight.data[0, 0, k] = 1.0
                else:
                    # Approximate bandpass filter
                    dist = k - half_kernel
                    filter_module.weight.data[0, 0, k] = math.sin(2 * math.pi * center_freq * dist / 24000) / (math.pi * dist + 1e-8)
                    
            # Apply window function to smooth filter
            window = torch.hann_window(kernel_size)
            filter_module.weight.data[0, 0, :] *= window
            
            # Normalize
            filter_module.weight.data[0, 0, :] /= filter_module.weight.data[0, 0, :].abs().sum() + 1e-8
    
    def _init_noise_type_filters(self):
        """Initialize filters for different consonant noise types"""
        # Fricative (high-pass filter)
        kernel_size = self.noise_type_filters[0].weight.shape[2]
        half_kernel = kernel_size // 2
        
        # High-pass for fricatives (s, sh, f, etc.)
        self.noise_type_filters[0].weight.data[0, 0, half_kernel] = 1.0
        for k in range(kernel_size):
            if k != half_kernel:
                dist = k - half_kernel
                # High frequency emphasis
                self.noise_type_filters[0].weight.data[0, 0, k] = -0.85 * math.sin(math.pi * dist / kernel_size) / (math.pi * dist + 1e-8)
        
        # Plosive (band-pass with attack)
        self.noise_type_filters[1].weight.data[0, 0, half_kernel] = 1.0
        for k in range(kernel_size):
            if k != half_kernel:
                dist = k - half_kernel
                # Mid frequency emphasis with quick decay
                self.noise_type_filters[1].weight.data[0, 0, k] = math.sin(math.pi * dist / (kernel_size/2)) / (math.pi * dist + 1e-8) * math.exp(-abs(dist)/5)
        
        # Nasal (low-pass filter)
        for k in range(kernel_size):
            dist = k - half_kernel
            # Low frequency emphasis
            self.noise_type_filters[2].weight.data[0, 0, k] = math.sin(math.pi * dist / (kernel_size*2)) / (max(1, math.pi * dist))
            
        # Transition (mixed characteristics)
        for k in range(kernel_size):
            dist = k - half_kernel
            # Mixed frequency response
            self.noise_type_filters[3].weight.data[0, 0, k] = math.sin(math.pi * dist / (kernel_size/4)) / (max(1, math.pi * dist)) * math.exp(-abs(dist)/10)
            
        # Apply windows and normalize all filters
        for i in range(4):
            window = torch.hann_window(kernel_size)
            self.noise_type_filters[i].weight.data[0, 0, :] *= window
            self.noise_type_filters[i].weight.data[0, 0, :] /= self.noise_type_filters[i].weight.data[0, 0, :].abs().sum() + 1e-8
    
    def forward(self, condition, audio_length):
        """
        Args:
            condition: Conditioning information [B, C, T]
            audio_length: Length of the audio to generate
        Returns:
            articulation_noise: Shaped noise for articulation [B, audio_length]
            noise_gate: Gating envelope to control noise application [B, audio_length]
        """
        batch_size, channels, time_steps = condition.shape
        device = condition.device
        
        # Predict noise band amplitudes and timing parameters
        noise_params = self.noise_predictor(condition)  # [B, num_bands+3, T]
        
        # Split parameters
        band_amplitudes = torch.sigmoid(noise_params[:, :self.num_noise_bands, :])  # [B, num_bands, T]
        attack_rate = torch.sigmoid(noise_params[:, -3, :]) * 10  # Controls attack speed (0-10)
        release_rate = torch.sigmoid(noise_params[:, -2, :]) * 5   # Controls release speed (0-5)
        noise_intensity = torch.sigmoid(noise_params[:, -1, :])    # Overall intensity (0-1)
        
        # Predict spectral shaping for different consonant types
        noise_type_weights = self.spectral_shaper(condition)  # [B, 4, T]
        
        # Upsample to audio rate
        band_amplitudes_upsampled = F.interpolate(
            band_amplitudes, size=audio_length, mode='linear', align_corners=False
        )
        noise_type_weights_upsampled = F.interpolate(
            noise_type_weights, size=audio_length, mode='linear', align_corners=False
        )
        noise_intensity_upsampled = F.interpolate(
            noise_intensity.unsqueeze(1), size=audio_length, mode='linear', align_corners=False
        ).squeeze(1)
        attack_rate_upsampled = F.interpolate(
            attack_rate.unsqueeze(1), size=audio_length, mode='linear', align_corners=False
        ).squeeze(1)
        release_rate_upsampled = F.interpolate(
            release_rate.unsqueeze(1), size=audio_length, mode='linear', align_corners=False
        ).squeeze(1)
        
        # Generate base white noise
        white_noise = torch.randn(batch_size, 1, audio_length, device=device)
        
        # Apply each noise type filter weighted by its predicted importance
        filtered_noise_types = torch.zeros(batch_size, 1, audio_length, device=device)
        for i in range(4):  # For each noise type
            noise_type_filter = self.noise_type_filters[i]
            noise_weight = noise_type_weights_upsampled[:, i, :].unsqueeze(1)  # [B, 1, audio_length]
            
            # Apply the specific filter
            filtered_type = noise_type_filter(white_noise)
            
            # Weight by the predicted importance
            filtered_noise_types += filtered_type * noise_weight
        
        # Generate filtered noise for each band
        shaped_noise = torch.zeros(batch_size, audio_length, device=device)
        
        for i in range(self.num_noise_bands):
            # Extract band amplitude
            band_amp = band_amplitudes_upsampled[:, i, :]  # [B, audio_length]
            
            # Apply band filter
            filtered_band = self.filter_bank[i](filtered_noise_types).squeeze(1)  # [B, audio_length]
            
            # Apply amplitude modulation
            shaped_noise += filtered_band * band_amp
        
        # Create dynamic gate for noise (for plosives and transitions)
        noise_gate = self._generate_noise_gate(
            noise_intensity_upsampled, attack_rate_upsampled, release_rate_upsampled, audio_length, device
        )
        
        # Apply gate to shaped noise
        articulation_noise = shaped_noise * noise_gate
        
        return articulation_noise, noise_gate
    
    def _generate_noise_gate(self, intensity, attack_rate, release_rate, audio_length, device):
        """
        Generate dynamic envelope for controlling noise application.
        Creates attack/release patterns to simulate plosives and articulation transitions.
        
        Args:
            intensity: Base noise intensity [B, audio_length]
            attack_rate: Attack speed parameter [B, audio_length]
            release_rate: Release speed parameter [B, audio_length]
            audio_length: Length of the audio
            device: Device to create tensors on
            
        Returns:
            noise_gate: Dynamic envelope for noise [B, audio_length]
        """
        batch_size = intensity.shape[0]
        
        # Create smoothed version of intensity for finding transition points
        smoothed = F.avg_pool1d(
            intensity.unsqueeze(1), 
            kernel_size=min(641, audio_length), 
            stride=1, 
            padding=min(320, audio_length//2)
        ).squeeze(1)
        
        # Find points where intensity increases (potential plosive onsets)
        intensity_diff = F.pad(intensity[:, 1:] - intensity[:, :-1], (1, 0))
        
        # Create gate envelope that enhances transitions
        noise_gate = intensity.clone()
        
        # Enhance attack portions (where intensity increases rapidly)
        attack_mask = (intensity_diff > 0.1).float()
        enhanced_attack = torch.clamp(intensity + attack_mask * attack_rate * 0.3, 0, 1)
        
        # Apply soft attack enhancement where intensity is increasing
        increasing = (intensity_diff > 0).float()
        noise_gate = noise_gate * (1 - increasing) + enhanced_attack * increasing
        
        # Apply additional smoothing to create natural envelope
        if audio_length > 4:
            noise_gate = F.pad(noise_gate.unsqueeze(1), (2, 2), mode='reflect')
            noise_gate = F.avg_pool1d(noise_gate, kernel_size=5, stride=1).squeeze(1)
        
        return noise_gate


class HarmonicSynthesizer(nn.Module):
    """
    Generates harmonic components based on F0 with efficiency optimizations 
    and enhanced vocal characteristics:
    1. Formant modeling for realistic vowel sounds
    2. Voice quality modeling for natural micro-variations
    3. Consonant and articulation noise modeling
    4. Vectorized harmonic generation
    5. Optimized convolution architecture
    6. Efficient upsampling
    7. Dynamic harmonic selection
    8. Phase calculation caching
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
        
        # Add vocal-specific networks
        self.formant_network = FormantNetwork(num_formants=5, hidden_dim=128)
        self.voice_quality_network = VoiceQualityNetwork(hidden_dim=128)
        
        # Add the new articulation noise network
        self.articulation_network = ArticulationNoiseNetwork(hidden_dim=128)
        
        # Add a mixer network to balance harmonic and noise components
        self.component_mixer = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 2, kernel_size=1),  # Balance between harmonic and noise components
            nn.Softmax(dim=1)  # Ensures the weights sum to 1
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
            vocal_signal: Generated vocal signal with harmonics and articulation noise [B, audio_length]
        """
        batch_size, time_steps = f0.shape
        device = f0.device
        
        # Predict harmonic amplitudes
        harmonic_amplitudes = self.harmonic_amplitude_net(condition)  # [B, num_harmonics, T]
        
        # Apply formant shaping to harmonic amplitudes
        shaped_amplitudes = self.formant_network(condition, f0, harmonic_amplitudes)
        
        # Generate voice quality parameters
        jitter, shimmer, breathiness = self.voice_quality_network(condition, audio_length)
        
        # Generate articulation noise components
        articulation_noise, noise_gate = self.articulation_network(condition, audio_length)
        
        # Predict component mixing weights
        component_weights = self.component_mixer(condition)  # [B, 2, T]
        
        # Upsample component weights to audio sample rate
        component_weights_upsampled = F.interpolate(
            component_weights, size=audio_length, mode='linear', align_corners=False
        )  # [B, 2, audio_length]
        
        # Extract weights
        harmonic_weight = component_weights_upsampled[:, 0, :]  # [B, audio_length]
        noise_weight = component_weights_upsampled[:, 1, :]     # [B, audio_length]
        
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
        enhanced_harmonic_signal = self.voice_quality_network.apply_voice_qualities(
            harmonic_signal, phase, jitter, shimmer, breathiness
        )
        
        # Mix harmonic signal and articulation noise based on predicted weights
        vocal_signal = (
            enhanced_harmonic_signal * harmonic_weight + 
            articulation_noise * noise_weight
        )
        
        return vocal_signal
        
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