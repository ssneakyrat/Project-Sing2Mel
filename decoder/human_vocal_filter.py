import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class HumanVocalFilter(nn.Module):
    """
    Specialized filter for human vocal characteristics, extending the functionality
    of the frequency_filter function while optimizing specifically for vocals.
    """
    def __init__(self, 
                 sample_rate=24000,
                 formant_emphasis=True,
                 vocal_range_boost=True,
                 breathiness=0.3,
                 gender="neutral",
                 articulation=0.5):    # 0.0 (staccato) to 1.0 (legato)
        super(HumanVocalFilter, self).__init__()
        
        self.sample_rate = sample_rate
        self.formant_emphasis = formant_emphasis
        self.vocal_range_boost = vocal_range_boost
        self.breathiness = breathiness
        self.gender = gender
        self.articulation = articulation  # New parameter for articulation control
        
        # Initialize formant regions (in Hz) - average values that can be adjusted
        # Format: (F1, F2, F3, F4)
        self.formant_regions = {
            "male": (500, 1500, 2500, 3500),
            "female": (550, 1650, 2750, 3850),
            "neutral": (525, 1575, 2625, 3675),
            "child": (650, 1750, 2850, 3950)
        }
        
        # Initialize vocal range
        self.vocal_ranges = {
            "male": (80, 700),     # Bass/Baritone/Tenor
            "female": (160, 1100), # Alto/Soprano
            "neutral": (100, 900),
            "child": (200, 1200)
        }
        
        # Q factors for formant regions (controls bandwidth of formant emphasis)
        self.formant_q = nn.Parameter(
            torch.tensor([7.0, 6.0, 5.0, 4.0]), 
            requires_grad=False
        )
        
        # Register vocal profile as buffer
        self._create_vocal_profile()
    
    def _create_vocal_profile(self):
        """Create vocal profile buffers for faster computation"""
        # Initialize formant profiles based on gender
        formants = self.formant_regions[self.gender]
        vocal_range = self.vocal_ranges[self.gender]
        
        # Register formants as buffer
        self.register_buffer("formants", torch.tensor(formants).float())
        
        # Register vocal range as buffer
        self.register_buffer("vocal_range", torch.tensor(vocal_range).float())
        
        # Create nyquist frequency buffer
        self.register_buffer("nyquist", torch.tensor(self.sample_rate / 2).float())
    
    def _apply_window_to_impulse_response(self, impulse_response, window_size=0, causal=False):
        """Apply a window to an impulse response and put in causal form."""
        # If IR is in causal form, put it in zero-phase form.
        if causal:
            impulse_response = torch.fftshift(impulse_response, dims=-1)
        
        # Get a window for better time/frequency resolution than rectangular.
        # Window defaults to IR size, cannot be bigger.
        ir_size = int(impulse_response.size(-1))
        if (window_size <= 0) or (window_size > ir_size):
            window_size = ir_size
        window = nn.Parameter(torch.hann_window(window_size), requires_grad=False).to(impulse_response.device)
        
        # Zero pad the window and put in in zero-phase form.
        padding = ir_size - window_size
        if padding > 0:
            half_idx = (window_size + 1) // 2
            window = torch.cat([window[half_idx:],
                               torch.zeros([padding], device=window.device),
                               window[:half_idx]], dim=0)
        else:
            window = window.roll((window.size(-1)+1)//2, -1)
        
        # Apply the window, to get new IR (both in zero-phase form).
        window = window.unsqueeze(0)
        impulse_response = impulse_response * window
        
        # Put IR in causal form and trim zero padding.
        if padding > 0:
            first_half_start = (ir_size - (half_idx - 1)) + 1
            second_half_end = half_idx + 1
            impulse_response = torch.cat([impulse_response[..., first_half_start:],
                                         impulse_response[..., :second_half_end]],
                                         dim=-1)
        else:
            impulse_response = impulse_response.roll((impulse_response.size(-1)+1)//2, -1)
        
        return impulse_response
    
    def _frequency_impulse_response(self, magnitudes, window_size=0):
        """Get windowed impulse responses using the frequency sampling method with vocal enhancements."""
        # Get the IR (zero-phase form).
        magnitudes = torch.complex(magnitudes, torch.zeros_like(magnitudes))
        impulse_response = torch.fft.irfft(magnitudes)
        
        # Window and put in causal form.
        impulse_response = self._apply_window_to_impulse_response(
            impulse_response, window_size=window_size or impulse_response.size(-1)
        )
        return impulse_response
    
    def _get_fft_size(self, frame_size, ir_size, power_of_2=True):
        """Calculate final size for efficient FFT."""
        convolved_frame_size = ir_size + frame_size - 1
        if power_of_2:
            # Next power of 2.
            fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
        else:
            # Use 5-smooth number for non-TPU hardware (more flexible)
            # For simplicity, we'll stick with power of 2 here
            fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
        return fft_size
    
    def _crop_and_compensate_delay(self, audio_out, audio_size, ir_size, padding='same', delay_compensation=-1):
        """Crop and shift the output audio after convolution."""
        # Compensate for the group delay of the filter.
        total_size = audio_size + ir_size - 1
        
        # Default to constant group delay of filter.
        if delay_compensation < 0:
            delay_compensation = (ir_size - 1) // 2
        
        # Crop the output.
        if padding == 'valid':
            return audio_out
        elif padding == 'same':
            return audio_out[..., delay_compensation:delay_compensation + audio_size]
        else:
            raise ValueError(f'Invalid padding option: {padding}')
    
    def _fft_convolve(self, audio, impulse_response, padding='same', delay_compensation=-1):
        """Filter audio with frames of time-varying impulse responses."""
        # Add a frame dimension to impulse response if it doesn't have one.
        ir_shape = impulse_response.size()
        if len(ir_shape) == 2:
            impulse_response = impulse_response.unsqueeze(1)
            ir_shape = impulse_response.size()
        
        # Get shapes of audio and impulse response.
        batch_size_ir, n_ir_frames, ir_size = ir_shape
        batch_size, audio_size = audio.size()
        
        # Validate that batch sizes match.
        if batch_size != batch_size_ir:
            raise ValueError(f'Batch size of audio ({batch_size}) and impulse response ({batch_size_ir}) must be the same.')
        
        # Cut audio into frames.
        frame_size = int(np.ceil(audio_size / n_ir_frames))
        hop_size = frame_size
        
        # Convert audio to proper shape for frame processing
        audio = audio.unsqueeze(1)
        if frame_size != audio_size:
            filters = torch.eye(frame_size, device=audio.device).unsqueeze(1)
            audio_frames = F.conv1d(audio, filters, stride=hop_size).transpose(1, 2)
            n_audio_frames = audio_frames.size(1)
        else:
            audio_frames = audio
        
        # Check that number of frames match.
        n_audio_frames = int(audio_frames.shape[1])
        
        if n_audio_frames != n_ir_frames:
            raise ValueError(
                f'Number of Audio frames ({n_audio_frames}) and impulse response frames ({n_ir_frames}) do not match.'
            )
        
        # Pad and FFT the audio and impulse responses.
        fft_size = self._get_fft_size(frame_size, ir_size, power_of_2=True)
        
        audio_fft = torch.fft.rfft(audio_frames, fft_size)
        ir_fft = torch.fft.rfft(impulse_response, fft_size)
        
        # Multiply the FFTs (same as convolution in time).
        audio_ir_fft = torch.multiply(audio_fft, ir_fft)
        
        # Take the IFFT to resynthesize audio.
        audio_frames_out = torch.fft.irfft(audio_ir_fft)
        
        if frame_size != audio_size:
            overlap_add_filter = torch.eye(audio_frames_out.size(-1), device=audio.device).unsqueeze(1)
            output_signal = F.conv_transpose1d(
                audio_frames_out.transpose(1, 2),
                overlap_add_filter,
                stride=frame_size,
                padding=0
            ).squeeze(1)
        else:
            output_signal = self._crop_and_compensate_delay(
                audio_frames_out.squeeze(1),
                audio_size,
                ir_size,
                padding,
                delay_compensation
            )
        
        return output_signal[..., :frame_size * n_ir_frames]
    
    def _apply_formant_emphasis(self, magnitudes):
        """Apply formant emphasis to the magnitude spectrum."""
        if not self.formant_emphasis:
            return magnitudes
        
        # Get shape information
        batch_size, n_frames, n_frequencies = magnitudes.shape
        
        # Create frequency axis
        freq_axis = torch.linspace(0, self.nyquist, n_frequencies, device=magnitudes.device)
        
        # Initialize formant emphasis mask
        formant_mask = torch.ones_like(freq_axis)
        
        # Apply resonant peaks at formant frequencies
        for i, formant in enumerate(self.formants):
            # Skip if formant is above nyquist
            if formant >= self.nyquist:
                continue
                
            # Calculate bandwidth based on Q factor
            bandwidth = formant / self.formant_q[i]
            
            # Create resonant peak for this formant
            numerator = (bandwidth/2)**2
            peaks = numerator / ((freq_axis - formant)**2 + numerator)
            
            # Scale the peaks - higher formants get progressively less emphasis
            scale_factor = 1.0 - (i * 0.15)  # 1.0, 0.85, 0.7, 0.55
            formant_mask += scale_factor * peaks
        
        # Normalize and apply gain control
        formant_mask = 0.7 + (0.3 * formant_mask / formant_mask.max())
        
        # Expand mask to match magnitudes shape
        formant_mask = formant_mask.unsqueeze(0).unsqueeze(0).expand_as(magnitudes)
        
        # Apply the formant mask to magnitudes
        enhanced_magnitudes = magnitudes * formant_mask
        
        return enhanced_magnitudes
    
    def _apply_vocal_range_boost(self, magnitudes):
        """Apply boost to the vocal frequency range."""
        if not self.vocal_range_boost:
            return magnitudes
        
        # Get shape information
        batch_size, n_frames, n_frequencies = magnitudes.shape
        
        # Create frequency axis
        freq_axis = torch.linspace(0, self.nyquist, n_frequencies, device=magnitudes.device)
        
        # Initialize vocal range mask
        vocal_mask = torch.ones_like(freq_axis)
        
        # Extract vocal range
        vocal_min, vocal_max = self.vocal_range
        
        # Create vocal range boost using smooth transition
        # Boost fundamental and early harmonics
        low_shelf = 1.0 / (1.0 + torch.exp(-(freq_axis - vocal_min) * 0.05))
        high_shelf = 1.0 / (1.0 + torch.exp((freq_axis - vocal_max) * 0.01))
        
        # Combine shelves to create a bandpass-like effect
        vocal_mask = 1.0 + 0.3 * (low_shelf * high_shelf)
        
        # Expand mask to match magnitudes shape
        vocal_mask = vocal_mask.unsqueeze(0).unsqueeze(0).expand_as(magnitudes)
        
        # Apply the vocal mask to magnitudes
        boosted_magnitudes = magnitudes * vocal_mask
        
        return boosted_magnitudes
    
    def _apply_breathiness(self, magnitudes):
        """Apply breathiness effect to high frequencies."""
        if self.breathiness <= 0:
            return magnitudes
        
        # Get shape information
        batch_size, n_frames, n_frequencies = magnitudes.shape
        
        # Create frequency axis
        freq_axis = torch.linspace(0, self.nyquist, n_frequencies, device=magnitudes.device)
        
        # Create breathy mask - more energy in higher frequencies
        breath_freq = 2500  # Typical breath noise frequency center
        breath_mask = 1.0 / (1.0 + torch.exp(-(freq_axis - breath_freq) * 0.005))
        
        # Scale the breath effect by breathiness parameter
        breath_mask = 1.0 + self.breathiness * breath_mask
        
        # Expand mask to match magnitudes shape
        breath_mask = breath_mask.unsqueeze(0).unsqueeze(0).expand_as(magnitudes)
        
        # Apply the breath mask to magnitudes
        breathy_magnitudes = magnitudes * breath_mask
        
        return breathy_magnitudes
    
    def _apply_articulation(self, magnitudes):
        """
        Apply articulation control to the magnitude spectrum.
        
        Articulation affects how notes transition and the onset/offset characteristics:
        - Low articulation (staccato): sharper attacks, quicker decays
        - High articulation (legato): smoother transitions, longer sustains
        
        Args:
            magnitudes: Magnitude spectrum [batch, n_frames, n_frequencies]
            
        Returns:
            Modified magnitude spectrum with articulation effects
        """
        # Skip if articulation is at neutral setting (0.5)
        if abs(self.articulation - 0.5) < 0.01:
            return magnitudes
        
        # Get shape information
        batch_size, n_frames, n_frequencies = magnitudes.shape
        
        if n_frames <= 1:
            return magnitudes  # No articulation possible with single frame
        
        # Create a temporal smoothing kernel based on articulation level
        # Higher articulation (legato) = more smoothing across frames
        kernel_size = min(n_frames, max(3, int(self.articulation * 9)))
        if kernel_size % 2 == 0:  # Make sure kernel size is odd
            kernel_size += 1
            
        # Create smoothing kernel - more weight in center for neutral articulation
        # More uniform for legato, more centered for staccato
        if self.articulation > 0.5:  # Legato - smoother transitions
            # Create a smoother kernel for legato
            alpha = (self.articulation - 0.5) * 2  # Map 0.5-1.0 to 0.0-1.0
            sigma = 0.5 + 1.5 * alpha  # Larger sigma = more smoothing
            
            # Gaussian kernel for smooth transitions
            kernel = torch.zeros(kernel_size, device=magnitudes.device)
            center = kernel_size // 2
            for i in range(kernel_size):
                kernel[i] = math.exp(-0.5 * ((i - center) / sigma) ** 2)
            kernel = kernel / kernel.sum()  # Normalize
            
        else:  # Staccato - sharper transitions
            # Create a more peaked kernel for staccato
            alpha = (0.5 - self.articulation) * 2  # Map 0.0-0.5 to 1.0-0.0
            exponent = 1.0 + 4.0 * alpha  # Higher exponent = more peaked
            
            # Power function kernel for sharper transitions
            kernel = torch.zeros(kernel_size, device=magnitudes.device)
            center = kernel_size // 2
            for i in range(kernel_size):
                # Create a peaked distribution
                norm_dist = abs(i - center) / (kernel_size // 2)
                kernel[i] = (1.0 - norm_dist) ** exponent
            kernel = kernel / kernel.sum()  # Normalize
        
        # Reshape kernel for 1D convolution along time dimension
        kernel = kernel.view(1, 1, kernel_size)
        
        # Apply different smoothing strategies based on frequency bands
        # Low frequencies (fundamentals) - preserve more attack/decay characteristics
        # High frequencies (harmonics/transients) - more affected by articulation
        
        # Create frequency band weights - higher frequencies more affected by articulation
        freq_weights = torch.linspace(0.5, 1.0, n_frequencies, device=magnitudes.device)
        
        # Apply temporal smoothing to each frequency band
        smoothed_mags = torch.zeros_like(magnitudes)
        
        # Process each batch item separately
        for b in range(batch_size):
            # Prepare 3D input for conv1d: [channels, frequency, time]
            # We process each frequency band as a separate channel
            mag_for_conv = magnitudes[b].permute(1, 0).unsqueeze(1)  # [n_frequencies, 1, n_frames]
            
            # Apply 1D convolution along time dimension
            # Use 'same' padding to maintain frame count
            padding = kernel_size // 2
            smoothed = F.conv1d(mag_for_conv, kernel, padding=padding)
            
            # Reshape back to original dimensions
            smoothed = smoothed.squeeze(1).permute(1, 0)  # [n_frames, n_frequencies]
            
            # Blend original and smoothed based on frequency-dependent weights
            # and overall articulation parameter
            # High articulation = more smoothing, especially for high frequencies
            blend_factor = freq_weights.unsqueeze(0)  # [1, n_frequencies]
            
            # Apply the articulation effect
            smoothed_mags[b] = magnitudes[b] * (1 - blend_factor) + smoothed * blend_factor
        
        # Apply attack and decay modifications based on articulation
        if self.articulation < 0.5:  # Staccato - enhance attack, shorten decay
            # Calculate the amount of attack/decay modification
            mod_factor = (0.5 - self.articulation) * 2  # 0.0-0.5 → 0.0-1.0
            
            # Create attack emphasis (first few frames)
            attack_frames = max(1, int(n_frames * 0.1))  # Use 10% of frames for attack
            if attack_frames > 1:
                attack_curve = torch.linspace(1.0 + mod_factor, 1.0, attack_frames, device=magnitudes.device)
                attack_curve = attack_curve.unsqueeze(0).unsqueeze(2)  # [1, attack_frames, 1]
                
                # Apply attack emphasis
                smoothed_mags[:, :attack_frames, :] *= attack_curve
            
            # Create decay shortening (later frames)
            decay_start = max(attack_frames, int(n_frames * 0.5))  # Start decay at 50% of frames
            if decay_start < n_frames - 1:
                decay_length = n_frames - decay_start
                decay_curve = torch.linspace(1.0, 1.0 - 0.7 * mod_factor, decay_length, device=magnitudes.device)
                decay_curve = decay_curve.unsqueeze(0).unsqueeze(2)  # [1, decay_length, 1]
                
                # Apply decay shortening
                smoothed_mags[:, decay_start:, :] *= decay_curve
        
        return smoothed_mags
    
    def _enhance_magnitudes(self, magnitudes):
        """Apply all vocal enhancements to magnitude spectrum."""
        # Apply formant emphasis
        magnitudes = self._apply_formant_emphasis(magnitudes)
        
        # Apply vocal range boost
        magnitudes = self._apply_vocal_range_boost(magnitudes)
        
        # Apply breathiness
        magnitudes = self._apply_breathiness(magnitudes)
        
        # Apply articulation control
        magnitudes = self._apply_articulation(magnitudes)
        
        return magnitudes
    
    def forward(self, audio, magnitudes, window_size=0, padding='same'):
        """
        Filter audio with a finite impulse response filter optimized for human vocals.
        
        Args:
            audio: Input audio. Tensor of shape [batch, audio_timesteps].
            magnitudes: Frequency transfer curve. Float32 Tensor of shape 
                [batch, n_frames, n_frequencies] or [batch, n_frequencies].
            window_size: Size of the window to apply in the time domain.
                If window_size < 1, it defaults to n_frequencies.
            padding: Either 'valid' or 'same'. For 'same' the final output
                has the same size as the input audio.
                
        Returns:
            Filtered audio optimized for vocal characteristics.
        """
        # Apply vocal-specific enhancements to magnitude spectrum
        enhanced_magnitudes = self._enhance_magnitudes(magnitudes)
        
        # Get impulse response
        impulse_response = self._frequency_impulse_response(
            enhanced_magnitudes, window_size=window_size
        )
        
        # Apply FFT convolution
        return self._fft_convolve(audio, impulse_response, padding=padding)


# Function that works as a drop-in replacement for the original frequency_filter
def vocal_frequency_filter(audio, magnitudes, window_size=0, padding='same', 
                          gender="neutral", formant_emphasis=True, 
                          vocal_range_boost=True, breathiness=0.3,
                          sample_rate=24000, articulation=0.5):
    """
    A drop-in replacement for frequency_filter that specializes in human vocals.
    
    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        magnitudes: Frequency transfer curve. Float32 Tensor of shape 
            [batch, n_frames, n_frequencies] or [batch, n_frequencies].
        window_size: Size of the window to apply in the time domain.
        padding: Either 'valid' or 'same'.
        gender: One of "male", "female", "neutral", or "child".
        formant_emphasis: Whether to emphasize vocal formants.
        vocal_range_boost: Whether to boost the vocal frequency range.
        breathiness: Amount of breathiness to add (0.0 to 1.0).
        sample_rate: Audio sample rate.
        articulation: Amount of articulation control (0.0 to 1.0):
                      0.0 = Very staccato (sharp, detached notes)
                      0.5 = Neutral articulation
                      1.0 = Very legato (smooth, connected notes)
        
    Returns:
        Filtered audio optimized for vocal characteristics.
    """
    # Create the vocal filter
    vocal_filter = HumanVocalFilter(
        sample_rate=sample_rate,
        formant_emphasis=formant_emphasis,
        vocal_range_boost=vocal_range_boost,
        breathiness=breathiness,
        gender=gender,
        articulation=articulation
    ).to(audio.device)
    
    # Apply the filter
    return vocal_filter(audio, magnitudes, window_size, padding)