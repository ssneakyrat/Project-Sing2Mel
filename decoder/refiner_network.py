import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RefinementNetwork(nn.Module):
    """
    Enhanced RefinementNetwork for singing vocals using DDSP principles.
    Optimized for vocal synthesis with formant enhancement, dynamic control,
    and sibilance management while maintaining numerical stability.
    """
    def __init__(self, input_channels=128, fft_size=1024, hop_factor=4, sample_rate=24000):
        super(RefinementNetwork, self).__init__()
        self.input_channels = input_channels
        self.fft_size = fft_size
        self.hop_length = fft_size // hop_factor
        self.sample_rate = sample_rate
        
        # --- Basic controls from original RefinementNetwork ---
        self.output_gain = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output gain control (0-1)
        )
        
        # Wet/dry balance
        self.wet_dry = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Sigmoid()  # 0-1 range
        )
        
        # --- New vocal-specific controls ---
        
        # 1. Formant enhancement (5 formants)
        self.formant_enhancement = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 5, kernel_size=1),  # 5 formant regions
            nn.Tanh()  # -1 to 1 range for formant emphasis
        )
        
        # 2. Vocal-specific filter bank (8 bands)
        self.vocal_filter_bank = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 8, kernel_size=1),  # 8 bands tuned to vocal frequencies
            nn.Tanh()  # -1 to 1 range for cut/boost
        )
        
        # 3. Dynamic range control (compressor params)
        self.comp_threshold = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()  # 0-1 range for threshold mapping
        )
        
        self.comp_ratio = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()  # 0-1 range mapped to 1:1-8:1
        )
        
        # 4. Sibilance control (de-esser)
        self.sibilance_processor = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Amount of reduction
        )
        
        # 5. Vowel-Consonant detector
        self.vowel_consonant_detector = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),  # Wider context
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()  # 0 = consonant, 1 = vowel
        )
        
        # 6. Vibrato enhancement
        self.vibrato_enhance = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),  # Wider kernel for temporal patterns
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Tanh()  # For vibrato depth adjustment
        )
        
        # --- Create specialized filters ---
        
        # Formant center frequencies (Hz) - standard defaults
        formant_centers = torch.tensor([500, 1500, 2500, 3500, 4500], dtype=torch.float32)
        self.register_buffer('formant_centers', formant_centers.view(1, -1, 1))
        
        # Formant bandwidths (Hz)
        formant_bandwidths = torch.tensor([100, 200, 300, 400, 500], dtype=torch.float32)
        self.register_buffer('formant_bandwidths', formant_bandwidths.view(1, -1, 1))
        
        # Create vocal-specific filter bands
        self.register_buffer('filter_bands', self._create_vocal_filter_bands())
        
        # Create sibilance band (5-8kHz region)
        nyquist = fft_size // 2
        sibilance_band = torch.zeros(nyquist + 1)
        sib_start = int(5000 / (sample_rate / 2) * nyquist)
        sib_end = int(8000 / (sample_rate / 2) * nyquist)
        sibilance_band[sib_start:sib_end] = 1.0
        self.register_buffer('sibilance_band', sibilance_band.unsqueeze(0).unsqueeze(0))
        
        # Register window for STFT
        self.register_buffer('window', torch.hann_window(fft_size))
    
    def _create_vocal_filter_bands(self):
        """Create filter bands specifically tuned for vocal frequencies"""
        nyquist = self.fft_size // 2
        num_bins = nyquist + 1
        
        # Vocal-focused frequency bands (in Hz)
        band_centers = [
            120,    # Sub-fundamental
            240,    # Fundamental
            500,    # First formant region
            1100,   # Second formant region
            2500,   # Third formant region
            3500,   # Fourth formant region
            5000,   # Sibilance/presence
            8000    # Air/brightness
        ]
        
        # Convert to bin indices
        band_indices = [min(nyquist, int(freq / (self.sample_rate / 2) * nyquist)) for freq in band_centers]
        
        # Create filter bands with smooth overlaps
        bands = torch.zeros(8, num_bins)
        
        for i in range(8):
            center = band_indices[i]
            
            # Calculate width based on psychoacoustic principles (wider at higher frequencies)
            if i < 2:
                width = max(center // 2, 1)  # Narrow for fundamentals
            elif i < 4:
                width = max(center // 3, 1)  # Medium for low formants
            else:
                width = max(center // 4, 1)  # Wider for high frequencies
            
            # Create band with smooth edges
            start = max(0, center - width)
            end = min(num_bins, center + width)
            
            # Linear ramp up
            if start > 0:
                ramp_length = min(width, center - start)
                if ramp_length > 0:  # Add safety check
                    ramp_up = torch.linspace(0, 1, ramp_length)
                    bands[i, start:start+ramp_length] = ramp_up
            
            # Center peak
            bands[i, center] = 1.0
            
            # Linear ramp down
            if end < num_bins:
                ramp_length = min(width, end - center - 1)
                if ramp_length > 0:  # Add safety check
                    ramp_down = torch.linspace(1, 0, ramp_length)
                    bands[i, center+1:center+1+ramp_length] = ramp_down
        
        # Normalize so bands sum to ~1.0 across all bands
        band_sum = bands.sum(dim=0, keepdim=True)
        band_sum[band_sum < 0.1] = 1.0  # Avoid division by small numbers
        bands = bands / band_sum
        
        return bands.unsqueeze(0)  # [1, 8, F]
    
    def apply_formants(self, mag_db, formant_controls, stft_time):
        """Apply formant enhancement to magnitude spectrum"""
        batch_size = mag_db.shape[0]
        num_bins = mag_db.shape[1]
        
        # Interpolate formant controls to match STFT time steps
        if formant_controls.shape[2] != stft_time:
            formant_controls = F.interpolate(
                formant_controls, size=stft_time, mode='linear', align_corners=False
            )
        
        # Create frequency axis (normalized 0-1)
        freq_axis = torch.linspace(0, 1, num_bins, device=mag_db.device)
        
        # Create formant filters
        formant_filters = torch.zeros((batch_size, num_bins, stft_time), device=mag_db.device)
        
        # For each formant
        for i in range(5):  # 5 formants
            # Get center frequency (normalized 0-1)
            center_freq = self.formant_centers[:, i] / (self.sample_rate / 2)
            center_freq = center_freq.clamp(0.01, 0.99)  # Safety clamp
            
            # Get bandwidth (normalized 0-1)
            bandwidth = self.formant_bandwidths[:, i] / (self.sample_rate / 2)
            bandwidth = bandwidth.clamp(0.01, 0.3)  # Safety clamp
            
            # Expand dimensions for broadcasting
            freq = freq_axis.view(1, -1, 1)  # [1, F, 1]
            center = center_freq.view(-1, 1, 1)  # [B, 1, 1]
            bw = bandwidth.view(-1, 1, 1)  # [B, 1, 1]
            
            # Create resonance filter
            resonance = 1.0 / (1.0 + ((freq - center) / (bw / 2)) ** 2)
            
            # Get formant control for this formant
            control = formant_controls[:, i:i+1]  # [B, 1, T]
            
            # Apply control to filter
            formant_effect = resonance * control.transpose(1, 2)  # [B, F, T]
            formant_filters += formant_effect
        
        # Limit enhancement range to ±6 dB
        formant_filters = formant_filters.clamp(-6.0, 6.0)
        
        # Apply to magnitude spectrum
        enhanced_mag_db = mag_db + formant_filters
        
        return enhanced_mag_db
    
    def apply_vocal_eq(self, mag_db, filter_controls, stft_time):
        """Apply vocal-specific EQ to magnitude spectrum"""
        # Interpolate filter controls to match STFT time steps
        if filter_controls.shape[2] != stft_time:
            filter_controls = F.interpolate(
                filter_controls, size=stft_time, mode='linear', align_corners=False
            )
        
        # Reshape filter controls for batched matrix op
        filter_weights = filter_controls.unsqueeze(2)  # [B, 8, 1, T]
        
        # Apply filters (weighted sum of filter bands)
        filter_bands_expanded = self.filter_bands.unsqueeze(-1)  # [1, 8, F, 1]
        filter_effect = torch.sum(filter_bands_expanded * filter_weights, dim=1)  # [B, F, T]
        
        # Scale to reasonable EQ range (±6 dB)
        filter_effect = filter_effect * 6.0
        
        # Apply to magnitude spectrum
        filtered_mag_db = mag_db + filter_effect
        
        return filtered_mag_db
    
    def apply_sibilance_control(self, mag_db, sibilance_amount, stft_time):
        """Apply de-essing to control sibilance"""
        # Interpolate sibilance amount to match STFT time steps
        if sibilance_amount.shape[2] != stft_time:
            sibilance_amount = F.interpolate(
                sibilance_amount, size=stft_time, mode='linear', align_corners=False
            )
        
        # Scale amount for reasonable range (max -12dB reduction)
        sibilance_amount = sibilance_amount.clamp(0.0, 1.0)
        reduction_db = -12.0 * sibilance_amount
        
        # Apply reduction only to sibilance frequency band
        sibilance_mask = self.sibilance_band.expand(-1, -1, stft_time)
        reduction_db = reduction_db * sibilance_mask
        
        # Apply reduction to magnitude spectrum
        processed_mag_db = mag_db + reduction_db
        
        return processed_mag_db
    
    def enhance_vibrato(self, phase, vibrato_control, stft_time):
        """Enhance natural vibrato through phase modulation"""
        # Interpolate vibrato control to match STFT time steps
        if vibrato_control.shape[2] != stft_time:
            vibrato_control = F.interpolate(
                vibrato_control, size=stft_time, mode='linear', align_corners=False
            )
        
        # Scale vibrato control to a reasonable range (±0.3 rad max)
        vibrato_strength = vibrato_control.clamp(-1.0, 1.0) * 0.3
        
        # Apply a 5-6Hz modulation (common vibrato rate) to the phase
        time_indices = torch.arange(stft_time, device=phase.device).float()
        vibrato_freq = torch.tensor(2 * math.pi * 5.5 / (self.sample_rate / self.hop_length), device=phase.device)
        
        # Create modulation signal
        mod_signal = torch.sin(vibrato_freq * time_indices).view(1, 1, -1)
        
        # Focus on mid-frequency region where vibrato is most noticeable
        # Create a band-pass filter focusing on 200-2000Hz
        num_bins = phase.shape[1]
        vibrato_mask = torch.zeros(num_bins, device=phase.device)
        
        # Convert Hz to bin indices
        low_bin = max(1, int(200 / (self.sample_rate / 2) * (num_bins - 1)))
        high_bin = min(num_bins - 1, int(2000 / (self.sample_rate / 2) * (num_bins - 1)))
        
        # Create smooth ramp for vibrato band
        vibrato_mask[low_bin:high_bin] = 1.0
        
        # Apply modulation to phase, scaled by control signal and mask
        vibrato_mod = (vibrato_strength * mod_signal).unsqueeze(1)  # [B, 1, 1, T]
        vibrato_mask = vibrato_mask.view(1, -1, 1)  # [1, F, 1]
        
        enhanced_phase = phase + (vibrato_mod * vibrato_mask)
        
        return enhanced_phase
    
    def apply_compression(self, signal, threshold, ratio, audio_length):
        """Apply dynamic range compression to the audio signal"""
        # Ensure control signals match audio length
        if threshold.shape[1] != audio_length:
            threshold = F.interpolate(threshold.unsqueeze(1), size=audio_length, 
                                    mode='linear', align_corners=False).squeeze(1)
        
        if ratio.shape[1] != audio_length:
            ratio = F.interpolate(ratio.unsqueeze(1), size=audio_length, 
                                mode='linear', align_corners=False).squeeze(1)
        
        # Scale parameters to appropriate ranges
        threshold_db = -40 * threshold - 10  # Map 0-1 to -50 to -10 dB
        ratio_val = 1 + 7 * ratio  # Map 0-1 to ratio 1:1-8:1
        
        # Fixed time constants for stability
        attack_time = 0.005  # 5ms
        release_time = 0.050  # 50ms
        
        # Convert attack/release times to coefficients
        attack_coef = torch.exp(-1.0 / (attack_time * self.sample_rate))
        release_coef = torch.exp(-1.0 / (release_time * self.sample_rate))
        
        # Calculate envelope (approximate RMS)
        envelope = torch.abs(signal)
        
        # Convert to dB
        envelope_db = 20 * torch.log10(torch.clamp(envelope, min=1e-5))
        
        # Calculate gain reduction
        gain_reduction_db = torch.minimum(
            torch.zeros_like(envelope_db),
            (threshold_db - envelope_db) * (1 - 1/ratio_val)
        )
        
        # Apply attack/release smoothing
        smoothed_gain_db = torch.zeros_like(gain_reduction_db)
        
        # First sample
        smoothed_gain_db[:, 0] = gain_reduction_db[:, 0]
        
        # Remaining samples - apply smoothing
        for i in range(1, audio_length):
            # Determine whether to use attack or release coefficient
            is_attack = gain_reduction_db[:, i] < smoothed_gain_db[:, i-1]
            coef = torch.where(is_attack, attack_coef, release_coef)
            
            # Apply smoothing
            smoothed_gain_db[:, i] = coef * smoothed_gain_db[:, i-1] + (1 - coef) * gain_reduction_db[:, i]
        
        # Convert gain reduction from dB to linear
        smoothed_gain_linear = 10 ** (smoothed_gain_db / 20)
        
        # Apply gain reduction
        compressed_signal = signal * smoothed_gain_linear
        
        return compressed_signal
    
    def forward(self, signal, condition):
        """
        Apply vocal-specific refinement to the audio signal.
        
        Args:
            signal: Input audio [B, T]
            condition: Conditioning signal [B, C, T_cond]
            
        Returns:
            Processed audio [B, T]
        """
        # Ensure input is valid
        signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        condition = torch.nan_to_num(condition, nan=0.0, posinf=0.0, neginf=0.0)
        
        batch_size = signal.shape[0]
        audio_length = signal.shape[1]
        
        # Get all control parameters
        gain = self.output_gain(condition)  # [B, 1, T_cond]
        formant_controls = self.formant_enhancement(condition)  # [B, 5, T_cond]
        filter_controls = self.vocal_filter_bank(condition)  # [B, 8, T_cond]
        comp_thresh = self.comp_threshold(condition)  # [B, 1, T_cond]
        comp_ratio = self.comp_ratio(condition)  # [B, 1, T_cond]
        sibilance_amount = self.sibilance_processor(condition)  # [B, 1, T_cond]
        vowel_consonant = self.vowel_consonant_detector(condition)  # [B, 1, T_cond]
        vibrato_control = self.vibrato_enhance(condition)  # [B, 1, T_cond]
        mix = self.wet_dry(condition)  # [B, 1, T_cond]
        
        # Apply conservative limits to control parameters
        gain = torch.clamp(gain, 0.5, 1.5)
        mix = torch.clamp(mix, 0.3, 0.7)
        
        # Ensure signal is 2D [B, T]
        if signal.dim() == 3:
            signal = signal.squeeze(1)
        
        # Preserve original signal for wet/dry mixing
        original_signal = signal.clone()
        
        # Reshape control params to match audio if needed
        if gain.shape[2] != audio_length:
            gain = F.interpolate(gain, size=audio_length, mode='linear', align_corners=False)
            mix = F.interpolate(mix, size=audio_length, mode='linear', align_corners=False)
        
        try:
            # --- STFT-based processing ---
            stft = torch.stft(
                signal,
                n_fft=self.fft_size,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            
            # Handle NaNs in STFT output
            if torch.isnan(stft).any() or torch.isinf(stft).any():
                # If STFT failed, return original signal with gain
                return torch.clamp(original_signal * gain.squeeze(1), -1.0, 1.0)
            
            # Get magnitude and phase
            mag = torch.abs(stft)  # [B, F, T]
            phase = torch.angle(stft)
            
            # Convert magnitude to dB
            mag_db = 20 * torch.log10(torch.clamp(mag, min=1e-5))
            
            # Get STFT time steps
            stft_time = mag.shape[2]
            
            # --- Apply vocal-specific processing ---
            
            # 1. Formant enhancement
            enhanced_mag_db = self.apply_formants(mag_db, formant_controls, stft_time)
            
            # 2. Vocal-specific EQ
            filtered_mag_db = self.apply_vocal_eq(enhanced_mag_db, filter_controls, stft_time)
            
            # 3. Sibilance control (de-essing)
            processed_mag_db = self.apply_sibilance_control(filtered_mag_db, sibilance_amount, stft_time)
            
            # 4. Vibrato enhancement
            processed_phase = self.enhance_vibrato(phase, vibrato_control, stft_time)
            
            # Convert processed mag_db back to linear
            processed_mag = 10 ** (processed_mag_db / 20)
            
            # Reconstruct complex STFT
            output_stft = processed_mag * torch.exp(1j * processed_phase)
            
            # Convert back to time domain
            processed = torch.istft(
                output_stft,
                n_fft=self.fft_size,
                hop_length=self.hop_length,
                window=self.window,
                length=audio_length
            )
            
            # 5. Dynamic range compression
            processed = self.apply_compression(
                processed,
                comp_thresh.squeeze(1),
                comp_ratio.squeeze(1),
                audio_length
            )
            
            # Check for NaNs in output
            if torch.isnan(processed).any() or torch.isinf(processed).any():
                # Fallback to original if processing failed
                processed = original_signal
            
            # Apply output gain
            processed = processed * gain.squeeze(1)
            
            # Apply wet/dry mix
            output = (mix.squeeze(1) * processed) + ((1.0 - mix.squeeze(1)) * original_signal)
            
            # Final sanity check and clamping
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            output = torch.clamp(output, -1.0, 1.0)
            
        except Exception as e:
            # Complete fallback if any error occurs
            output = original_signal * gain.squeeze(1)
            output = torch.clamp(output, -1.0, 1.0)
        
        return output