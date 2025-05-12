import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class HarmonicGenerator(nn.Module):
    """
    Optimized version of the HarmonicGenerator for singing voice synthesis.
    
    This module takes fundamental frequency (f0) and parameters to generate harmonic signals
    with appropriate amplitude modulation over time, with improved performance.
    """
    def __init__(self, 
                 n_harmonics=80, 
                 sample_rate=24000, 
                 hop_length=240,
                 use_adsr=True):
        """
        Initialize the OptimizedHarmonicGenerator.
        
        Args:
            n_harmonics: Number of harmonics to generate
            sample_rate: Audio sample rate in Hz
            hop_length: Number of samples between frames
            use_adsr: Whether to use ADSR envelopes for amplitude shaping
        """
        super(HarmonicGenerator, self).__init__()
        
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.use_adsr = use_adsr
        
        # Harmonic indices (1, 2, 3, ..., n_harmonics)
        self.register_buffer('harmonic_indices', torch.arange(1, n_harmonics + 1).float())
        
        # Precomputed constants
        self.register_buffer('TWO_PI', torch.tensor(2 * math.pi))
        self.register_buffer('sample_rate_tensor', torch.tensor(sample_rate, dtype=torch.float))
        
        # Default ADSR parameters (used when not provided by parameter predictor)
        self.default_attack = 0.01   # 10ms
        self.default_decay = 0.05    # 50ms
        self.default_sustain = 0.7   # 70% of peak
        self.default_release = 0.1   # 100ms
        
    def generate_sine_vectorized(self, frequencies, phases, duration):
        """
        Vectorized sine wave generation for improved performance.
        
        Args:
            frequencies: Frequencies for each harmonic [B, T, n_harmonics]
            phases: Initial phases [B, n_harmonics] or None
            duration: Number of samples to generate
            
        Returns:
            Sine waves for each harmonic [B, n_harmonics, duration]
        """
        batch_size, n_frames, _ = frequencies.shape
        device = frequencies.device
        
        # Initialize output tensor
        output = torch.zeros(batch_size, self.n_harmonics, duration, device=device)
        
        # Initialize phases if not provided
        if phases is None:
            phases = torch.zeros(batch_size, self.n_harmonics, device=device)
        
        # Sample indices and frame indices tensors
        sample_indices = torch.arange(0, duration, device=device)
        frame_positions = torch.arange(0, n_frames, device=device) * self.hop_length
        
        # For each frame, determine which samples it covers
        for i in range(n_frames):
            # Current frame's frequencies
            frame_freqs = frequencies[:, i, :]  # [B, n_harmonics]
            
            # Start and end sample for this frame
            start_sample = frame_positions[i]
            end_sample = frame_positions[i] + self.hop_length if i < n_frames - 1 else duration
            
            if start_sample >= duration:
                break
                
            # Time slice for this frame (samples since start of audio)
            t_frame = sample_indices[start_sample:end_sample].float() / self.sample_rate_tensor
            
            # Generate sine wave for this frame (vectorized across batch and harmonics)
            # Phase term: 2πft + φ
            t_frame_expanded = t_frame.view(1, 1, -1)  # [1, 1, samples]
            freq_expanded = frame_freqs.unsqueeze(-1)  # [B, n_harmonics, 1]
            phase_expanded = phases.unsqueeze(-1)  # [B, n_harmonics, 1]
            
            phase_term = self.TWO_PI * freq_expanded * t_frame_expanded + phase_expanded
            sine_frame = torch.sin(phase_term)
            
            # Add to output (in-place operation)
            output[:, :, start_sample:end_sample] = sine_frame
            
            # Update phases for the next frame to ensure continuity
            if i < n_frames - 1:
                # Calculate phase at the end of current frame
                end_phase = self.TWO_PI * frame_freqs * (self.hop_length / self.sample_rate)
                phases = (phases + end_phase) % self.TWO_PI
        
        return output
    
    def apply_adsr_vectorized(self, signal, note_on_frames, note_off_frames=None, adsr_params=None):
        """
        Vectorized ADSR envelope application for improved performance.
        
        Args:
            signal: Input signal [B, n_harmonics, N]
            note_on_frames: Frame indices for note onset [B, num_notes]
            note_off_frames: Frame indices for note offset [B, num_notes] or None
            adsr_params: ADSR parameters [B, T, 4] or None (uses defaults if None)
            
        Returns:
            Signal with ADSR envelope applied [B, n_harmonics, N]
        """
        if not self.use_adsr:
            return signal
            
        batch_size, _, signal_length = signal.shape
        device = signal.device
        n_frames = signal_length // self.hop_length + 1
        
        # Create envelope buffer
        envelope = torch.zeros(batch_size, 1, signal_length, device=device)
        
        # Sample indices tensor for vectorized operations
        sample_indices = torch.arange(0, signal_length, device=device)
        
        # For each batch and note
        for b in range(batch_size):
            for n in range(note_on_frames.shape[1]):
                # Note start and end in samples
                note_start = note_on_frames[b, n] * self.hop_length
                
                if note_off_frames is not None:
                    note_end = note_off_frames[b, n] * self.hop_length
                else:
                    note_end = signal_length
                
                if note_start >= signal_length or note_start < 0:
                    continue
                    
                if note_end > signal_length:
                    note_end = signal_length
                
                # Get ADSR parameters - either from provided params or defaults
                if adsr_params is not None:
                    # Use parameters at note onset
                    onset_frame = min(note_on_frames[b, n], n_frames - 1)
                    attack = adsr_params[b, onset_frame, 0].item()
                    decay = adsr_params[b, onset_frame, 1].item()
                    sustain = adsr_params[b, onset_frame, 2].item()
                    release = adsr_params[b, onset_frame, 3].item()
                else:
                    # Use default parameters
                    attack = self.default_attack
                    decay = self.default_decay
                    sustain = self.default_sustain
                    release = self.default_release
                
                # Convert times to samples
                attack_samples = int(attack * self.sample_rate)
                decay_samples = int(decay * self.sample_rate)
                release_samples = int(release * self.sample_rate)
                
                # Create a sample position relative to note start
                relative_pos = sample_indices - note_start
                
                # Attack phase (vectorized)
                attack_end = note_start + attack_samples
                attack_mask = (relative_pos >= 0) & (relative_pos < attack_samples)
                if attack_mask.any():
                    attack_values = relative_pos.float() / (attack_samples or 1)
                    envelope[b, 0, attack_mask] = attack_values[attack_mask]
                
                # Decay phase (vectorized)
                decay_end = attack_end + decay_samples
                decay_mask = (relative_pos >= attack_samples) & (relative_pos < attack_samples + decay_samples)
                if decay_mask.any():
                    decay_pos = relative_pos[decay_mask] - attack_samples
                    decay_values = 1.0 - (1.0 - sustain) * (decay_pos.float() / (decay_samples or 1))
                    envelope[b, 0, decay_mask] = decay_values
                
                # Sustain phase (vectorized)
                sustain_mask = (relative_pos >= attack_samples + decay_samples) & (relative_pos < note_end - note_start)
                if sustain_mask.any():
                    envelope[b, 0, sustain_mask] = sustain
                
                # Release phase (vectorized)
                release_pos = sample_indices - note_end
                release_mask = (release_pos >= 0) & (release_pos < release_samples)
                if release_mask.any():
                    release_values = sustain * (1.0 - (release_pos[release_mask].float() / (release_samples or 1)))
                    envelope[b, 0, release_mask] = release_values
        
        # Apply envelope to signal
        return signal * envelope
    
    def fast_interpolate(self, x, target_len):
        """
        Fast linear interpolation for 3D tensors.
        
        Args:
            x: Input tensor [B, C, T]
            target_len: Target length
            
        Returns:
            Interpolated tensor [B, C, target_len]
        """
        # Use F.interpolate with optimized settings
        return F.interpolate(
            x,
            size=target_len,
            mode='linear',
            align_corners=False
        )
    
    def forward(self, f0, params, duration=None, note_on_frames=None, note_off_frames=None, initial_phase=None):
        """
        Generate harmonic content based on f0 and predicted parameters.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            params: Dictionary of parameters from ParameterPredictor
                - 'harmonic_amplitudes': Amplitudes for each harmonic [B, T, n_harmonics]
                - 'voiced_mix': Voicing factor [B, T, 1]
                - 'adsr_params': ADSR envelope parameters [B, T, 4] (optional)
            duration: Duration in samples (if None, calculated from f0 and hop_length)
            note_on_frames: Frame indices for note onset [B, num_notes] or None
            note_off_frames: Frame indices for note offset [B, num_notes] or None
            initial_phase: Initial phase for oscillators [B, n_harmonics] or None
            
        Returns:
            Harmonic component of the synthesized signal [B, N]
        """
        batch_size, n_frames = f0.shape
        device = f0.device
        
        # Calculate duration if not provided
        if duration is None:
            duration = (n_frames + 1) * self.hop_length
        
        # Get harmonic amplitudes and voiced mix factor from params
        harmonic_amplitudes = params['harmonic_amplitudes']  # [B, T, n_harmonics]
        voiced_mix = params['voiced_mix']  # [B, T, 1]
        
        # Get ADSR parameters if available
        adsr_params = params.get('adsr_params', None)  # [B, T, 4] or None
        
        # Expand f0 to get frequencies for all harmonics (vectorized)
        # [B, T, 1] * [1, 1, n_harmonics] = [B, T, n_harmonics]
        harmonic_frequencies = f0.unsqueeze(-1) * self.harmonic_indices.view(1, 1, -1)
        
        # Generate sine waves for all harmonics (vectorized)
        harmonic_signals = self.generate_sine_vectorized(harmonic_frequencies, initial_phase, duration)  # [B, n_harmonics, N]
        
        # Reshape harmonic amplitudes for broadcasting
        # From [B, T, n_harmonics] to [B, n_harmonics, T]
        harmonic_amps_transposed = harmonic_amplitudes.transpose(1, 2)
        
        # Interpolate amplitudes to match signal length with optimized interpolation
        # From [B, n_harmonics, T] to [B, n_harmonics, N]
        interpolated_amps = self.fast_interpolate(harmonic_amps_transposed, duration)
        
        # Similarly interpolate voiced_mix factor
        # From [B, T, 1] to [B, 1, N]
        voiced_mix_transposed = voiced_mix.transpose(1, 2)
        interpolated_voice_mix = self.fast_interpolate(voiced_mix_transposed, duration)
        
        # Apply harmonic amplitudes (element-wise multiplication)
        weighted_harmonics = harmonic_signals * interpolated_amps
        
        # Apply ADSR envelope if specified
        if self.use_adsr and note_on_frames is not None:
            weighted_harmonics = self.apply_adsr_vectorized(weighted_harmonics, note_on_frames, note_off_frames, adsr_params)
        
        # Sum all harmonics
        summed_harmonics = weighted_harmonics.sum(dim=1)  # [B, N]
        
        # Apply voiced mix factor
        final_output = summed_harmonics * interpolated_voice_mix.squeeze(1)
        
        return final_output


class JitOptimizedHarmonicGenerator(HarmonicGenerator):
    """
    Further optimized version of the HarmonicGenerator that uses TorchScript JIT compilation
    for critical functions to improve performance.
    """
    
    def __init__(self, n_harmonics=80, sample_rate=24000, hop_length=240, use_adsr=True):
        """Initialize the JIT-optimized HarmonicGenerator."""
        super(JitOptimizedHarmonicGenerator, self).__init__(
            n_harmonics=n_harmonics,
            sample_rate=sample_rate,
            hop_length=hop_length,
            use_adsr=use_adsr
        )
        
        # Compile critical functions with TorchScript
        self._fast_sine = torch.jit.script(self._sine_kernel)
        
    @staticmethod
    def _sine_kernel(freq, phase, t, two_pi):
        """JIT-optimized sine wave kernel."""
        return torch.sin(two_pi * freq * t + phase)
    
    def generate_sine_vectorized(self, frequencies, phases, duration):
        """
        JIT-optimized sine wave generation.
        """
        batch_size, n_frames, _ = frequencies.shape
        device = frequencies.device
        
        # Initialize output tensor
        output = torch.zeros(batch_size, self.n_harmonics, duration, device=device)
        
        # Initialize phases if not provided
        if phases is None:
            phases = torch.zeros(batch_size, self.n_harmonics, device=device)
        
        # Sample indices and frame indices tensors
        sample_indices = torch.arange(0, duration, device=device)
        frame_positions = torch.arange(0, n_frames, device=device) * self.hop_length
        
        # For each frame, determine which samples it covers
        for i in range(n_frames):
            # Current frame's frequencies
            frame_freqs = frequencies[:, i, :]  # [B, n_harmonics]
            
            # Start and end sample for this frame
            start_sample = frame_positions[i]
            end_sample = frame_positions[i] + self.hop_length if i < n_frames - 1 else duration
            
            if start_sample >= duration:
                break
                
            # Time slice for this frame (samples since start of audio)
            t_frame = sample_indices[start_sample:end_sample].float() / self.sample_rate_tensor
            
            # Generate sine wave using the JIT-optimized kernel
            t_frame_expanded = t_frame.view(1, 1, -1)  # [1, 1, samples]
            freq_expanded = frame_freqs.unsqueeze(-1)  # [B, n_harmonics, 1]
            phase_expanded = phases.unsqueeze(-1)  # [B, n_harmonics, 1]
            
            sine_frame = self._fast_sine(freq_expanded, phase_expanded, t_frame_expanded, self.TWO_PI)
            
            # Add to output (in-place operation)
            output[:, :, start_sample:end_sample] = sine_frame
            
            # Update phases for the next frame to ensure continuity
            if i < n_frames - 1:
                # Calculate phase at the end of current frame
                end_phase = self.TWO_PI * frame_freqs * (self.hop_length / self.sample_rate)
                phases = (phases + end_phase) % self.TWO_PI
        
        return output
    
    # Additional optimizations could be added here for forward, apply_adsr, etc.


class ChunkedHarmonicGenerator(HarmonicGenerator):
    """
    Memory-efficient version of the HarmonicGenerator that processes audio in chunks
    to handle very long sequences without excessive memory usage.
    """
    
    def __init__(self, n_harmonics=80, sample_rate=24000, hop_length=240, use_adsr=True, chunk_size=24000):
        """
        Initialize the chunked HarmonicGenerator.
        
        Args:
            n_harmonics: Number of harmonics to generate
            sample_rate: Audio sample rate in Hz
            hop_length: Number of samples between frames
            use_adsr: Whether to use ADSR envelopes for amplitude shaping
            chunk_size: Size of audio chunks to process at once (samples)
        """
        super(ChunkedHarmonicGenerator, self).__init__(
            n_harmonics=n_harmonics,
            sample_rate=sample_rate,
            hop_length=hop_length,
            use_adsr=use_adsr
        )
        self.chunk_size = chunk_size
    
    def forward(self, f0, params, duration=None, note_on_frames=None, note_off_frames=None, initial_phase=None):
        """
        Process audio in chunks to save memory for long sequences.
        """
        batch_size, n_frames = f0.shape
        device = f0.device
        
        # Calculate duration if not provided
        if duration is None:
            duration = (n_frames + 1) * self.hop_length
        
        # For short audio, use the regular forward method
        if duration <= self.chunk_size:
            return super().forward(f0, params, duration, note_on_frames, note_off_frames, initial_phase)
        
        # For long audio, process in chunks
        num_chunks = (duration + self.chunk_size - 1) // self.chunk_size
        output = torch.zeros(batch_size, duration, device=device)
        
        # Process each chunk
        for i in range(num_chunks):
            chunk_start = i * self.chunk_size
            chunk_end = min((i + 1) * self.chunk_size, duration)
            chunk_duration = chunk_end - chunk_start
            
            # Find frames that correspond to this chunk
            frame_start = chunk_start // self.hop_length
            frame_end = min(n_frames, (chunk_end + self.hop_length - 1) // self.hop_length)
            
            # Extract chunk parameters
            chunk_f0 = f0[:, frame_start:frame_end]
            chunk_params = {
                'harmonic_amplitudes': params['harmonic_amplitudes'][:, frame_start:frame_end],
                'voiced_mix': params['voiced_mix'][:, frame_start:frame_end]
            }
            
            if 'adsr_params' in params:
                chunk_params['adsr_params'] = params['adsr_params'][:, frame_start:frame_end]
            
            # Adjust note frames to chunk-relative coordinates
            chunk_note_on = None
            chunk_note_off = None
            
            if note_on_frames is not None:
                # Only include notes that affect this chunk
                chunk_note_mask = (note_on_frames * self.hop_length < chunk_end) & \
                                 ((note_off_frames * self.hop_length if note_off_frames is not None 
                                   else torch.full_like(note_on_frames, duration)) > chunk_start)
                
                if chunk_note_mask.any():
                    chunk_note_on = note_on_frames.clone()
                    chunk_note_on = torch.max(chunk_note_on - frame_start, torch.zeros_like(chunk_note_on))
                    
                    if note_off_frames is not None:
                        chunk_note_off = note_off_frames.clone()
                        chunk_note_off = torch.max(chunk_note_off - frame_start, torch.zeros_like(chunk_note_off))
                        chunk_note_off = torch.min(chunk_note_off, 
                                                  torch.full_like(chunk_note_off, frame_end - frame_start))
            
            # Process chunk
            # Use inherited phase if not first chunk
            chunk_phase = initial_phase if i == 0 else current_phase
            
            chunk_output = super().forward(
                chunk_f0, chunk_params, chunk_duration, 
                chunk_note_on, chunk_note_off, chunk_phase
            )
            
            # Save current phase for next chunk
            frame_freqs = f0[:, frame_end-1].unsqueeze(1) * self.harmonic_indices.view(1, -1)
            phase_advance = self.TWO_PI * frame_freqs * (chunk_end - (frame_end-1) * self.hop_length) / self.sample_rate
            current_phase = (chunk_phase + phase_advance) % self.TWO_PI if chunk_phase is not None else phase_advance
            
            # Add chunk output to full output
            output[:, chunk_start:chunk_end] = chunk_output
        
        return output