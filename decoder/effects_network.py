import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EffectsNetwork(nn.Module):
    """
    DDSP-based audio effects processor with modulation, time, and spectral effects.
    Uses differentiable implementations of classic audio effects with shared backbone.
    
    Features:
    - Parameter-efficient shared backbone for feature extraction
    - Physics-informed implementation of classic audio effects
    - Adaptive parameter generation for each effect
    - Effect mixing and routing capabilities
    - Time-synchronized modulation sources
    """
    def __init__(self, input_channels=64, sample_rate=24000):
        super(EffectsNetwork, self).__init__()
        self.input_channels = input_channels
        self.sample_rate = sample_rate
        
        # Shared feature backbone - reduces parameter count significantly
        self.feature_backbone = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, groups=4),  # Depthwise separable conv
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # --- Modulation sources ---
        self.modulation_params = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 12, kernel_size=1)  # 3 LFOs x 4 params (rate, depth, waveform, phase)
        )
        
        # --- Time-based effects parameters ---
        # Delay and Reverb
        self.time_params = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 14, kernel_size=1)  # Delay: 4 params, Reverb: 10 params
        )
        
        # --- Modulation effects parameters ---
        # Flanger, Chorus, Phaser
        self.mod_effects_params = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 12, kernel_size=1)  # 4 params each for flanger, chorus, phaser
        )
        
        # --- Amplitude effects parameters ---
        # Tremolo, Vibrato
        self.amp_effects_params = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 8, kernel_size=1)  # 4 params each for tremolo and vibrato
        )
        
        # --- Effect routing and mixing ---
        self.routing_params = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 14, kernel_size=1)  # 7 effect levels, 7 routing positions
        )
        
        # Initialize effect buffer templates
        self._init_effect_buffers()
    
    def _init_effect_buffers(self):
        """Initialize buffers for efficient processing of effects"""
        # Maximum delay time (2 seconds)
        max_delay_samples = int(2.0 * self.sample_rate)
        self.register_buffer('delay_buffer_length', torch.tensor(max_delay_samples))
        
        # Allpass filter coefficients for phaser
        allpass_freqs = torch.tensor([200, 400, 800, 1600, 3200, 6400], dtype=torch.float32)
        allpass_freqs = allpass_freqs / (self.sample_rate / 2)  # Normalize
        self.register_buffer('allpass_freqs', allpass_freqs)
        
        # Reverb decay times per frequency band
        reverb_bands = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], dtype=torch.float32)
        self.register_buffer('reverb_bands', reverb_bands)
        
        # LFO waveform templates (sine, triangle, square)
        t = torch.linspace(0, 2*math.pi, 1024, dtype=torch.float32)
        sine = torch.sin(t)
        triangle = 1 - 2 * torch.abs((t / math.pi) % 2 - 1)
        square = torch.sign(torch.sin(t))
        
        self.register_buffer('lfo_sine', sine)
        self.register_buffer('lfo_triangle', triangle)
        self.register_buffer('lfo_square', square)
    
    def generate_lfo(self, rate, depth, waveform, phase, n_samples):
        """
        Generate a low-frequency oscillator signal
        
        Args:
            rate: LFO rate in Hz [B, 1, T_control]
            depth: LFO modulation depth [B, 1, T_control]
            waveform: LFO waveform type (0=sine, 1=triangle, 2=square) [B, 1, T_control]
            phase: LFO phase offset (0-1) [B, 1, T_control]
            n_samples: Number of audio samples to generate
            
        Returns:
            LFO signal [B, n_samples]
        """
        batch_size = rate.shape[0]
        
        # Interpolate control signals to audio rate
        rate_audio = F.interpolate(rate, size=n_samples, mode='linear', align_corners=False)
        depth_audio = F.interpolate(depth, size=n_samples, mode='linear', align_corners=False)
        waveform_audio = F.interpolate(waveform, size=n_samples, mode='nearest')
        phase_audio = F.interpolate(phase, size=n_samples, mode='linear', align_corners=False)
        
        # Create time vector (normalized 0-1)
        t = torch.arange(n_samples, device=rate.device).float() / self.sample_rate
        
        # Compute phase increment based on rate
        phase_inc = rate_audio.squeeze(1) / self.sample_rate
        
        # Accumulate phase
        cumulative_phase = torch.cumsum(phase_inc, dim=1)
        
        # Add initial phase offset
        total_phase = (cumulative_phase + phase_audio.squeeze(1)) % 1.0
        
        # Scale to 0-1024 for table lookup
        index = (total_phase * 1024).long() % 1024
        
        # Initialize LFO output
        lfo_output = torch.zeros((batch_size, n_samples), device=rate.device)
        
        # Apply waveform selection via table lookup
        for b in range(batch_size):
            # Generate masks for each waveform type
            sine_mask = (waveform_audio[b, 0] < 0.33).float()
            tri_mask = ((waveform_audio[b, 0] >= 0.33) & (waveform_audio[b, 0] < 0.67)).float()
            square_mask = (waveform_audio[b, 0] >= 0.67).float()
            
            # Apply masks and combine waveforms
            lfo_value = (
                sine_mask * self.lfo_sine[index[b]] + 
                tri_mask * self.lfo_triangle[index[b]] + 
                square_mask * self.lfo_square[index[b]]
            )
            
            # Apply depth
            lfo_output[b] = lfo_value * depth_audio[b, 0]
        
        return lfo_output
    
    def apply_delay(self, audio, delay_time, feedback, filter_freq, wet_dry, n_samples):
        """
        Apply delay/echo effect with feedback and filtering
        
        Args:
            audio: Input audio [B, T]
            delay_time: Delay time in seconds [B, 1, T_control]
            feedback: Feedback amount 0-1 [B, 1, T_control]
            filter_freq: Lowpass filter cutoff (0-1) [B, 1, T_control]
            wet_dry: Wet/dry mix (0-1) [B, 1, T_control]
            n_samples: Number of audio samples
            
        Returns:
            Processed audio [B, T]
        """
        batch_size = audio.shape[0]
        
        # Interpolate parameters to audio rate
        delay_time_s = F.interpolate(delay_time, size=n_samples, mode='linear', align_corners=False)
        feedback_gain = F.interpolate(feedback, size=n_samples, mode='linear', align_corners=False)
        filter_freq_norm = F.interpolate(filter_freq, size=n_samples, mode='linear', align_corners=False)
        wet_dry_mix = F.interpolate(wet_dry, size=n_samples, mode='linear', align_corners=False)
        
        # Convert delay time to samples
        delay_samples = (delay_time_s * self.sample_rate).long().clamp(1, self.delay_buffer_length - 1)
        
        # Initialize output and delay buffer
        output = torch.zeros_like(audio)
        delay_buffer = torch.zeros((batch_size, self.delay_buffer_length.item()), device=audio.device)
        
        # Previous filtered sample for simple one-pole filter
        prev_y = torch.zeros((batch_size), device=audio.device)
        
        # Convert filter frequency to coefficient (0-1)
        # Simple one-pole lowpass filter with alpha = dt * 2π * freq
        dt = 1.0 / self.sample_rate
        
        # Process sample by sample (this is the most computationally intensive part)
        for i in range(n_samples):
            # Compute delay indices (varying delay time)
            delay_idx = (i - delay_samples[:, 0, i]).clamp(0, self.delay_buffer_length - 1)
            
            # Feedback path with filtering
            for b in range(batch_size):
                # Get delayed sample
                delayed_sample = delay_buffer[b, delay_idx[b]]
                
                # Apply simple one-pole lowpass filter to feedback
                alpha = filter_freq_norm[b, 0, i] * 0.9  # Scale to 0-0.9 for stability
                prev_y[b] = alpha * delayed_sample + (1.0 - alpha) * prev_y[b]
                
                # Compute next sample (input + filtered feedback)
                next_sample = audio[b, i] + feedback_gain[b, 0, i] * prev_y[b]
                
                # Store in delay buffer
                buffer_idx = i % self.delay_buffer_length.item()
                delay_buffer[b, buffer_idx] = next_sample
                
                # Mix dry/wet
                output[b, i] = (1.0 - wet_dry_mix[b, 0, i]) * audio[b, i] + wet_dry_mix[b, 0, i] * delayed_sample
        
        return output
    
    def apply_reverb(self, audio, room_size, damping, width, freeze, wet_dry, n_samples):
        """
        Apply reverb effect using feedback delay network
        
        Args:
            audio: Input audio [B, T]
            room_size: Room size parameter (0-1) [B, 1, T_control]
            damping: High frequency damping (0-1) [B, 1, T_control]
            width: Stereo width (ignored for mono) [B, 1, T_control]
            freeze: Freeze control - infinite reverb when 1 [B, 1, T_control]
            wet_dry: Wet/dry mix (0-1) [B, 1, T_control]
            n_samples: Number of audio samples
            
        Returns:
            Processed audio [B, T]
        """
        batch_size = audio.shape[0]
        
        # Interpolate parameters to audio rate
        room_size_audio = F.interpolate(room_size, size=n_samples, mode='linear', align_corners=False)
        damping_audio = F.interpolate(damping, size=n_samples, mode='linear', align_corners=False)
        freeze_audio = F.interpolate(freeze, size=n_samples, mode='linear', align_corners=False)
        wet_dry_mix = F.interpolate(wet_dry, size=n_samples, mode='linear', align_corners=False)
        
        # Simplified 4-line feedback delay network (FDN)
        # Using fixed prime-number delay lengths for good diffusion
        delay_lengths = [1427, 1637, 1871, 2053]  # Prime numbers for good diffusion
        
        # Scale delay lengths based on room size
        max_room_scale = 1.5
        min_room_scale = 0.5
        
        # Initialize delay lines
        delay_lines = []
        for length in delay_lengths:
            delay_lines.append(torch.zeros((batch_size, length), device=audio.device))
        
        # FDN matrix (Hadamard matrix for efficient mixing)
        fdn_matrix = torch.tensor([
            [ 1,  1,  1,  1],
            [ 1, -1,  1, -1],
            [ 1,  1, -1, -1],
            [ 1, -1, -1,  1]
        ], device=audio.device).float() * 0.5  # Normalize
        
        # Damping filters state
        damping_mem = torch.zeros((batch_size, 4), device=audio.device)
        
        # Output buffer
        output = torch.zeros_like(audio)
        
        # Process sample by sample
        for i in range(n_samples):
            # Get current parameters
            room = room_size_audio[:, 0, i].unsqueeze(1)  # [B, 1]
            damp = damping_audio[:, 0, i].unsqueeze(1)  # [B, 1]
            freeze_factor = freeze_audio[:, 0, i].unsqueeze(1)  # [B, 1]
            
            # Calculate effective feedback (higher when frozen)
            effective_feedback = room * (1.0 - freeze_factor) + freeze_factor * 0.999
            
            # Create a column vector to hold the current outputs of the delay lines
            delay_outputs = torch.zeros((batch_size, 4), device=audio.device)
            
            # Read from delay lines and apply damping
            for j in range(4):
                # Get correct read index for this delay line
                read_idx = i % delay_lengths[j]
                
                # Read from delay line
                for b in range(batch_size):
                    # Read from buffer
                    delay_out = delay_lines[j][b, read_idx]
                    
                    # Apply damping (simplified lowpass)
                    damping_mem[b, j] = delay_out * damp[b, 0] + (1.0 - damp[b, 0]) * damping_mem[b, j]
                    
                    # Store damped output
                    delay_outputs[b, j] = damping_mem[b, j]
            
            # Apply FDN matrix for diffusion
            diffused = torch.matmul(delay_outputs, fdn_matrix)  # [B, 4]
            
            # Compute next sample with input and feedback
            reverb_out = torch.sum(delay_outputs, dim=1)  # Mix all delay lines
            
            # Write back to delay lines with feedback and input
            for j in range(4):
                # Get correct write index for this delay line
                write_idx = (i + 1) % delay_lengths[j]
                
                # Write to delay line
                for b in range(batch_size):
                    delay_lines[j][b, write_idx] = audio[b, i] * 0.2 + diffused[b, j] * effective_feedback[b, 0]
            
            # Apply wet/dry mix to output
            for b in range(batch_size):
                output[b, i] = (1.0 - wet_dry_mix[b, 0, i]) * audio[b, i] + wet_dry_mix[b, 0, i] * reverb_out[b]
        
        return output
    
    def apply_flanger(self, audio, rate, depth, feedback, wet_dry, lfo_signal, n_samples):
        """
        Apply flanger effect with LFO modulation
        
        Args:
            audio: Input audio [B, T]
            rate: LFO rate parameter (already used to generate lfo_signal)
            depth: Delay modulation depth (ms) [B, 1, T_control]
            feedback: Feedback amount (-1 to 1) [B, 1, T_control]
            wet_dry: Wet/dry mix (0-1) [B, 1, T_control]
            lfo_signal: Pre-generated LFO signal [B, T]
            n_samples: Number of audio samples
            
        Returns:
            Processed audio [B, T]
        """
        batch_size = audio.shape[0]
        
        # Interpolate parameters to audio rate
        depth_audio = F.interpolate(depth, size=n_samples, mode='linear', align_corners=False)
        feedback_audio = F.interpolate(feedback, size=n_samples, mode='linear', align_corners=False)
        wet_dry_mix = F.interpolate(wet_dry, size=n_samples, mode='linear', align_corners=False)
        
        # Convert depth from ms to samples (1-20ms typical for flanger)
        max_depth_samples = int(0.02 * self.sample_rate)  # 20ms
        depth_samples = depth_audio * max_depth_samples
        
        # Initialize output and delay buffer
        output = torch.zeros_like(audio)
        delay_buffer = torch.zeros((batch_size, max_depth_samples + 1), device=audio.device)
        
        # Process sample by sample
        for i in range(n_samples):
            # Compute modulated delay length using LFO
            delay_length = ((lfo_signal[:, i] + 1.0) * 0.5 * depth_samples[:, 0, i]).long().clamp(1, max_depth_samples-1)
            
            # Process each batch
            for b in range(batch_size):
                # Get buffer write index
                write_idx = i % (max_depth_samples + 1)
                
                # Get buffer read index
                read_idx = (write_idx - delay_length[b]) % (max_depth_samples + 1)
                
                # Get delayed sample
                delayed_sample = delay_buffer[b, read_idx]
                
                # Compute feedback sample
                next_sample = audio[b, i] + feedback_audio[b, 0, i] * delayed_sample
                
                # Write to buffer
                delay_buffer[b, write_idx] = next_sample
                
                # Mix original with delayed for classic flanging sound
                output[b, i] = (1.0 - wet_dry_mix[b, 0, i]) * audio[b, i] + wet_dry_mix[b, 0, i] * delayed_sample
        
        return output
    
    def apply_chorus(self, audio, rate, depth, voices, wet_dry, lfo_signal, n_samples):
        """
        Apply chorus effect with multiple voices and LFO modulation
        
        Args:
            audio: Input audio [B, T]
            rate: LFO rate parameter (already used to generate lfo_signal)
            depth: Delay modulation depth (ms) [B, 1, T_control]
            voices: Number of chorus voices (1-4) [B, 1, T_control]
            wet_dry: Wet/dry mix (0-1) [B, 1, T_control]
            lfo_signal: Pre-generated LFO signal [B, T]
            n_samples: Number of audio samples
            
        Returns:
            Processed audio [B, T]
        """
        batch_size = audio.shape[0]
        
        # Interpolate parameters to audio rate
        depth_audio = F.interpolate(depth, size=n_samples, mode='linear', align_corners=False)
        voices_audio = F.interpolate(voices, size=n_samples, mode='linear', align_corners=False)
        wet_dry_mix = F.interpolate(wet_dry, size=n_samples, mode='linear', align_corners=False)
        
        # Set chorus parameters
        max_chorus_voices = 4
        max_delay_samples = int(0.05 * self.sample_rate)  # 50ms max for chorus
        
        # Create voice-specific phase offsets (equally spaced)
        voice_phases = torch.linspace(0, 2*math.pi, max_chorus_voices+1, device=audio.device)[:-1]
        
        # Initialize output and delay buffer
        output = torch.zeros_like(audio)
        delay_buffer = torch.zeros((batch_size, max_delay_samples + 1), device=audio.device)
        
        # Generate separate LFOs for each voice with phase offsets
        voice_lfo_signals = []
        for v in range(max_chorus_voices):
            # Shift the lfo by phase offset
            phase_offset = voice_phases[v] / (2 * math.pi)
            shifted_lfo = torch.roll(lfo_signal, shifts=int(phase_offset * n_samples), dims=1)
            voice_lfo_signals.append(shifted_lfo)
        
        # Process sample by sample
        for i in range(n_samples):
            # Determine active voices (rounded to nearest integer)
            active_voices = (voices_audio[:, 0, i] * max_chorus_voices).round().long().clamp(1, max_chorus_voices)
            
            # Initialize chorus output
            chorus_out = torch.zeros(batch_size, device=audio.device)
            
            # Process each voice
            for v in range(max_chorus_voices):
                # Only process active voices
                voice_mask = (v < active_voices).float()
                
                # Modulate delay time with LFO
                delay_mod = ((voice_lfo_signals[v][:, i] + 1.0) * 0.5)
                delay_time = (10 + depth_audio[:, 0, i] * 40) / 1000.0  # 10-50ms
                delay_samples = (delay_time * self.sample_rate * delay_mod).long().clamp(1, max_delay_samples-1)
                
                # Process each batch
                for b in range(batch_size):
                    if v < active_voices[b]:
                        # Get buffer indices
                        write_idx = i % (max_delay_samples + 1)
                        read_idx = (write_idx - delay_samples[b]) % (max_delay_samples + 1)
                        
                        # Get delayed sample for this voice
                        delayed_sample = delay_buffer[b, read_idx]
                        
                        # Accumulate for this voice (with slight detune)
                        detune_factor = 1.0 + (v - active_voices[b]/2) * 0.002  # ±0.2% detune
                        chorus_out[b] += delayed_sample * detune_factor * voice_mask[b]
            
            # Write to delay buffer
            for b in range(batch_size):
                write_idx = i % (max_delay_samples + 1)
                delay_buffer[b, write_idx] = audio[b, i]
            
            # Normalize chorus output by number of voices
            for b in range(batch_size):
                if active_voices[b] > 0:
                    chorus_out[b] = chorus_out[b] / active_voices[b]
                
                # Apply wet/dry mix
                output[b, i] = (1.0 - wet_dry_mix[b, 0, i]) * audio[b, i] + wet_dry_mix[b, 0, i] * chorus_out[b]
        
        return output
    
    def apply_phaser(self, audio, rate, depth, stages, feedback, lfo_signal, n_samples):
        """
        Apply phaser effect with allpass filters
        
        Args:
            audio: Input audio [B, T]
            rate: LFO rate parameter (already used to generate lfo_signal)
            depth: Modulation depth [B, 1, T_control]
            stages: Number of allpass stages (2-12) [B, 1, T_control]
            feedback: Feedback amount (-0.9 to 0.9) [B, 1, T_control]
            lfo_signal: Pre-generated LFO signal [B, T]
            n_samples: Number of audio samples
            
        Returns:
            Processed audio [B, T]
        """
        batch_size = audio.shape[0]
        
        # Interpolate parameters to audio rate
        depth_audio = F.interpolate(depth, size=n_samples, mode='linear', align_corners=False)
        stages_audio = F.interpolate(stages, size=n_samples, mode='linear', align_corners=False)
        feedback_audio = F.interpolate(feedback, size=n_samples, mode='linear', align_corners=False)
        
        # Determine number of stages to use (2-12)
        max_stages = 12
        
        # Initialize state for allpass filters (for each stage)
        allpass_states = torch.zeros((batch_size, max_stages), device=audio.device)
        
        # Initialize output and feedback memory
        output = torch.zeros_like(audio)
        feedback_mem = torch.zeros(batch_size, device=audio.device)
        
        # Base and min/max frequencies for allpass filters in normalized frequency (0-1)
        min_freq = 200 / (self.sample_rate / 2)  # 200 Hz normalized
        max_freq = 8000 / (self.sample_rate / 2)  # 8000 Hz normalized
        
        # Process sample by sample
        for i in range(n_samples):
            # Compute modulated filter frequencies
            mod_val = (lfo_signal[:, i] + 1.0) * 0.5  # 0-1 range
            sweep_range = max_freq - min_freq
            mod_freq = min_freq + mod_val * sweep_range * depth_audio[:, 0, i]
            
            # Get actual stages to use (rounded to integer)
            actual_stages = (2 + stages_audio[:, 0, i] * 10).round().long().clamp(2, max_stages)
            
            # Process each batch
            for b in range(batch_size):
                # Apply feedback
                in_sample = audio[b, i] + feedback_mem[b] * feedback_audio[b, 0, i].clamp(-0.9, 0.9)
                
                # Apply allpass chain (first-order allpass filters)
                out_sample = in_sample
                for s in range(actual_stages[b]):
                    # Compute allpass coefficient
                    stage_freq = mod_freq[b] * (1.0 + s * 0.25)  # Spread frequencies for each stage
                    alpha = (math.sin(math.pi * stage_freq) - 1) / (math.sin(math.pi * stage_freq) + 1)
                    
                    # Apply allpass filter
                    mem = allpass_states[b, s]
                    allpass_states[b, s] = out_sample + alpha * (mem - out_sample)
                    out_sample = alpha * (allpass_states[b, s] - mem) + mem
                
                # Store output for feedback
                feedback_mem[b] = out_sample
                
                # Apply phaser mix (sum with inverted signal for phasing effect)
                output[b, i] = 0.5 * (in_sample + out_sample)
        
        return output
    
    def apply_tremolo(self, audio, rate, depth, shape, symmetry, lfo_signal, n_samples):
        """
        Apply tremolo effect (amplitude modulation)
        
        Args:
            audio: Input audio [B, T]
            rate: LFO rate parameter (already used to generate lfo_signal)
            depth: Modulation depth [B, 1, T_control]
            shape: Modulation shape (0=sine, 1=triangle, 2=square) [B, 1, T_control]
            symmetry: Waveform symmetry/skew [B, 1, T_control]
            lfo_signal: Pre-generated LFO signal [B, T]
            n_samples: Number of audio samples
            
        Returns:
            Processed audio [B, T]
        """
        batch_size = audio.shape[0]
        
        # Interpolate parameters to audio rate
        depth_audio = F.interpolate(depth, size=n_samples, mode='linear', align_corners=False)
        
        # Initialize output
        output = torch.zeros_like(audio)
        
        # Apply tremolo effect
        for i in range(n_samples):
            # Get modulator value (convert -1 to 1 range to 0 to 1)
            mod_val = (lfo_signal[:, i] + 1.0) * 0.5
            
            # Apply depth control
            tremolo_amount = 1.0 - depth_audio[:, 0, i] * mod_val
            
            # Apply amplitude modulation
            output[:, i] = audio[:, i] * tremolo_amount
        
        return output
    
    def apply_vibrato(self, audio, rate, depth, shape, symmetry, lfo_signal, n_samples):
        """
        Apply vibrato effect (pitch/frequency modulation)
        
        Args:
            audio: Input audio [B, T]
            rate: LFO rate parameter (already used to generate lfo_signal)
            depth: Modulation depth (cents) [B, 1, T_control]
            shape: Modulation shape (0=sine, 1=triangle, 2=square) [B, 1, T_control]
            symmetry: Waveform symmetry/skew [B, 1, T_control]
            lfo_signal: Pre-generated LFO signal [B, T]
            n_samples: Number of audio samples
            
        Returns:
            Processed audio [B, T]
        """
        batch_size = audio.shape[0]
        
        # Interpolate parameters to audio rate
        depth_audio = F.interpolate(depth, size=n_samples, mode='linear', align_corners=False)
        
        # Set maximum depth (in samples)
        max_depth_samples = int(0.01 * self.sample_rate)  # 10ms
        
        # Initialize output and delay buffer (for variable delay)
        output = torch.zeros_like(audio)
        delay_buffer = torch.zeros((batch_size, max_depth_samples + 1), device=audio.device)
        
        # Process sample by sample
        for i in range(n_samples):
            # Calculate variable delay using LFO
            # Map LFO (-1 to 1) to delay range (0 to max_depth)
            vibrato_delay = ((lfo_signal[:, i] + 1.0) * 0.5 * depth_audio[:, 0, i] * max_depth_samples).long().clamp(0, max_depth_samples-1)
            
            # Write input to delay buffer
            for b in range(batch_size):
                write_idx = i % (max_depth_samples + 1)
                delay_buffer[b, write_idx] = audio[b, i]
                
                # Read with variable delay for vibrato effect
                read_idx = (write_idx - vibrato_delay[b]) % (max_depth_samples + 1)
                output[b, i] = delay_buffer[b, read_idx]
        
        return output
    
    def forward(self, audio, condition):
        """
        Apply audio effects to the input signal based on conditioning
        
        Args:
            audio: Input audio [B, T]
            condition: Conditioning signal [B, C, T_cond]
            
        Returns:
            Processed audio [B, T]
        """
        batch_size = audio.shape[0]
        n_samples = audio.shape[1]
        
        # Ensure signal is 2D [B, T]
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        # Store original signal for effect chain processing
        original_audio = audio.clone()
        
        # Extract shared features
        features = self.feature_backbone(condition)
        
        # Get all parameters from shared features
        mod_params = self.modulation_params(features)  # [B, 12, T_cond]
        time_params = self.time_params(features)  # [B, 14, T_cond]
        mod_effect_params = self.mod_effects_params(features)  # [B, 12, T_cond]
        amp_effect_params = self.amp_effects_params(features)  # [B, 8, T_cond]
        routing_params = self.routing_params(features)  # [B, 14, T_cond]
        
        # Extract modulation source parameters (3 LFOs)
        lfo_rates = []
        lfo_depths = []
        lfo_shapes = []
        lfo_phases = []
        
        for i in range(3):
            # Rate: 0.1-20 Hz
            lfo_rates.append(torch.sigmoid(mod_params[:, i*4:i*4+1]) * 19.9 + 0.1)
            # Depth: 0-1
            lfo_depths.append(torch.sigmoid(mod_params[:, i*4+1:i*4+2]))
            # Shape: 0-1 (will be mapped to waveform)
            lfo_shapes.append(torch.sigmoid(mod_params[:, i*4+2:i*4+3]))
            # Phase: 0-1
            lfo_phases.append(torch.sigmoid(mod_params[:, i*4+3:i*4+4]))
        
        # Generate LFO signals
        lfo_signals = []
        for i in range(3):
            lfo_signals.append(self.generate_lfo(
                lfo_rates[i], lfo_depths[i], lfo_shapes[i], lfo_phases[i], n_samples
            ))
        
        # Extract time-based effect parameters
        # Delay parameters
        delay_time = torch.sigmoid(time_params[:, 0:1]) * 2.0  # 0-2 seconds
        delay_feedback = torch.sigmoid(time_params[:, 1:2])    # 0-1
        delay_filter = torch.sigmoid(time_params[:, 2:3])      # 0-1 (filter cutoff)
        delay_mix = torch.sigmoid(time_params[:, 3:4])         # 0-1 (wet/dry)
        
        # Reverb parameters
        reverb_size = torch.sigmoid(time_params[:, 4:5])       # 0-1 (room size)
        reverb_damp = torch.sigmoid(time_params[:, 5:6])       # 0-1 (damping)
        reverb_width = torch.sigmoid(time_params[:, 6:7])      # 0-1 (stereo width)
        reverb_freeze = torch.sigmoid(time_params[:, 7:8])     # 0-1 (freeze)
        reverb_mix = torch.sigmoid(time_params[:, 8:9])        # 0-1 (wet/dry)
        # Additional reverb parameters not used in this implementation
        
        # Extract modulation effect parameters
        # Flanger parameters
        flanger_depth = torch.sigmoid(mod_effect_params[:, 0:1]) * 10.0  # 0-10ms
        flanger_feedback = torch.tanh(mod_effect_params[:, 1:2]) * 0.9   # -0.9 to 0.9
        flanger_mix = torch.sigmoid(mod_effect_params[:, 2:3])           # 0-1 (wet/dry)
        
        # Chorus parameters
        chorus_depth = torch.sigmoid(mod_effect_params[:, 3:4]) * 30.0   # 0-30ms
        chorus_voices = torch.sigmoid(mod_effect_params[:, 4:5]) * 3.0 + 1.0  # 1-4 voices
        chorus_mix = torch.sigmoid(mod_effect_params[:, 5:6])            # 0-1 (wet/dry)
        
        # Phaser parameters
        phaser_depth = torch.sigmoid(mod_effect_params[:, 6:7])          # 0-1
        phaser_stages = torch.sigmoid(mod_effect_params[:, 7:8])         # 0-1 (mapped to 2-12)
        phaser_feedback = torch.tanh(mod_effect_params[:, 8:9]) * 0.9    # -0.9 to 0.9
        phaser_mix = torch.sigmoid(mod_effect_params[:, 9:10])           # 0-1 (wet/dry)
        
        # Extract amplitude effect parameters
        # Tremolo parameters
        tremolo_depth = torch.sigmoid(amp_effect_params[:, 0:1])         # 0-1
        tremolo_shape = torch.sigmoid(amp_effect_params[:, 1:2])         # 0-1 (waveform)
        tremolo_symmetry = torch.sigmoid(amp_effect_params[:, 2:3])      # 0-1 (skew)
        
        # Vibrato parameters
        vibrato_depth = torch.sigmoid(amp_effect_params[:, 4:5])         # 0-1
        vibrato_shape = torch.sigmoid(amp_effect_params[:, 5:6])         # 0-1 (waveform)
        vibrato_symmetry = torch.sigmoid(amp_effect_params[:, 6:7])      # 0-1 (skew)
        
        # Extract routing parameters
        effect_levels = torch.sigmoid(routing_params[:, 0:7])            # 0-1 for each effect
        effect_order = torch.softmax(routing_params[:, 7:14], dim=1)     # Relative ordering
        
        # Apply effects in dynamic order based on effect_order
        # First, put all effects and their levels in a list
        effects_data = [
            (0, effect_levels[:, 0:1], lambda x: self.apply_delay(x, delay_time, delay_feedback, delay_filter, delay_mix, n_samples)),
            (1, effect_levels[:, 1:2], lambda x: self.apply_reverb(x, reverb_size, reverb_damp, reverb_width, reverb_freeze, reverb_mix, n_samples)),
            (2, effect_levels[:, 2:3], lambda x: self.apply_flanger(x, lfo_rates[0], flanger_depth, flanger_feedback, flanger_mix, lfo_signals[0], n_samples)),
            (3, effect_levels[:, 3:4], lambda x: self.apply_chorus(x, lfo_rates[1], chorus_depth, chorus_voices, chorus_mix, lfo_signals[1], n_samples)),
            (4, effect_levels[:, 4:5], lambda x: self.apply_phaser(x, lfo_rates[2], phaser_depth, phaser_stages, phaser_feedback, lfo_signals[2], n_samples)),
            (5, effect_levels[:, 5:6], lambda x: self.apply_tremolo(x, lfo_rates[0], tremolo_depth, tremolo_shape, tremolo_symmetry, lfo_signals[0], n_samples)),
            (6, effect_levels[:, 6:7], lambda x: self.apply_vibrato(x, lfo_rates[1], vibrato_depth, vibrato_shape, vibrato_symmetry, lfo_signals[1], n_samples))
        ]
        
        # Sort effects by their order weights
        effect_priorities = torch.argsort(effect_order, dim=1, descending=True)
        
        # Apply effects in determined order
        processed_audio = original_audio.clone()
        
        for b in range(batch_size):
            current_audio = processed_audio[b:b+1].clone()
            
            # Apply effects in order for this batch item
            for idx in range(7):  # 7 effects
                effect_idx = effect_priorities[b, idx].item()
                effect_level = effect_levels[b:b+1, effect_idx:effect_idx+1]
                
                # If effect level is significant, apply it
                if effect_level.mean() > 0.05:
                    # Get effect function
                    effect_func = effects_data[effect_idx][2]
                    
                    # Apply effect
                    effect_output = effect_func(current_audio)
                    
                    # Mix based on effect level
                    current_audio = current_audio * (1 - effect_level.mean()) + effect_output * effect_level.mean()
            
            # Update the output for this batch
            processed_audio[b:b+1] = current_audio
        
        # Final normalization and safety checks
        if torch.isnan(processed_audio).any() or torch.isinf(processed_audio).any():
            # Fallback to original if processing failed
            processed_audio = original_audio
        
        # Clamp to prevent clipping
        processed_audio = torch.clamp(processed_audio, -1.0, 1.0)
        
        return processed_audio