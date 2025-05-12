import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class HarmonicGenerator(nn.Module):
    """
    Generates harmonic content for singing voice synthesis using oscillators and ADSR envelopes.
    
    This module takes fundamental frequency (f0) and parameters from the ParameterPredictor
    to generate harmonic signals with appropriate amplitude modulation over time.
    """
    def __init__(self, 
                 n_harmonics=80, 
                 sample_rate=24000, 
                 hop_length=240,
                 use_adsr=True):
        """
        Initialize the HarmonicGenerator.
        
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
        
        # Default ADSR parameters (used when not provided by parameter predictor)
        self.default_attack = 0.01   # 10ms
        self.default_decay = 0.05    # 50ms
        self.default_sustain = 0.7   # 70% of peak
        self.default_release = 0.1   # 100ms
        
    def generate_sine(self, frequencies, phases, duration):
        """
        Generate sine waves for each harmonic frequency.
        
        Args:
            frequencies: Frequencies for each harmonic [B, T, n_harmonics]
            phases: Initial phases [B, n_harmonics] or None
            duration: Number of samples to generate
            
        Returns:
            Sine waves for each harmonic [B, n_harmonics, duration]
        """
        batch_size, n_frames, _ = frequencies.shape
        device = frequencies.device
        
        # Time vector for the entire duration
        t = torch.arange(0, duration, device=device).float() / self.sample_rate
        
        # Initialize output tensor
        output = torch.zeros(batch_size, self.n_harmonics, duration, device=device)
        
        # Initialize phases if not provided
        if phases is None:
            phases = torch.zeros(batch_size, self.n_harmonics, device=device)
        
        # Frame positions in samples
        frame_positions = torch.arange(0, n_frames, device=device) * self.hop_length
        
        # For each frame, generate the corresponding part of the sine wave
        for i in range(n_frames):
            # Current frame's frequencies
            frame_freqs = frequencies[:, i, :]  # [B, n_harmonics]
            
            # Start and end sample for this frame
            start_sample = frame_positions[i]
            end_sample = frame_positions[i] + self.hop_length if i < n_frames - 1 else duration
            
            if start_sample >= duration:
                break
                
            # Time slice for this frame
            t_frame = t[start_sample:end_sample].unsqueeze(0).unsqueeze(0)  # [1, 1, hop_length]
            
            # Generate sine wave for this frame
            # 2πft + φ
            phase_term = 2 * math.pi * frame_freqs.unsqueeze(-1) * t_frame + phases.unsqueeze(-1)
            sine_frame = torch.sin(phase_term)
            
            # Add to output
            output[:, :, start_sample:end_sample] = sine_frame
            
            # Update phases for the next frame to ensure continuity
            if i < n_frames - 1:
                # Calculate phase at the end of current frame
                end_phase = 2 * math.pi * frame_freqs * (self.hop_length / self.sample_rate)
                phases = (phases + end_phase) % (2 * math.pi)
        
        return output
    
    def apply_adsr(self, signal, note_on_frames, note_off_frames=None, adsr_params=None):
        """
        Apply ADSR envelope to the signal.
        
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
                
                # Attack phase
                attack_end = min(note_start + attack_samples, signal_length)
                if attack_end > note_start:
                    t_attack = torch.arange(attack_end - note_start, device=device) / (attack_samples or 1)
                    envelope[b, 0, note_start:attack_end] = t_attack
                
                # Decay phase
                decay_end = min(attack_end + decay_samples, signal_length)
                if decay_end > attack_end:
                    t_decay = torch.arange(decay_end - attack_end, device=device) / (decay_samples or 1)
                    decay_values = 1.0 - (1.0 - sustain) * t_decay
                    envelope[b, 0, attack_end:decay_end] = decay_values
                
                # Sustain phase
                sustain_end = min(note_end, signal_length)
                if sustain_end > decay_end:
                    envelope[b, 0, decay_end:sustain_end] = sustain
                
                # Release phase
                release_end = min(note_end + release_samples, signal_length)
                if release_end > note_end:
                    t_release = torch.arange(release_end - note_end, device=device) / (release_samples or 1)
                    release_values = sustain * (1.0 - t_release)
                    envelope[b, 0, note_end:release_end] = release_values
        
        # Apply envelope to signal
        return signal * envelope
    
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
        
        # Expand f0 to get frequencies for all harmonics
        # [B, T, 1] * [1, 1, n_harmonics] = [B, T, n_harmonics]
        harmonic_frequencies = f0.unsqueeze(-1) * self.harmonic_indices.view(1, 1, -1)
        
        # Generate sine waves for all harmonics
        harmonic_signals = self.generate_sine(harmonic_frequencies, initial_phase, duration)  # [B, n_harmonics, N]
        
        # Reshape harmonic amplitudes for broadcasting
        # From [B, T, n_harmonics] to [B, n_harmonics, T]
        harmonic_amps_transposed = harmonic_amplitudes.transpose(1, 2)
        
        # Interpolate amplitudes to match signal length
        # From [B, n_harmonics, T] to [B, n_harmonics, N]
        hop_indices = torch.arange(0, n_frames, device=device) * self.hop_length
        sample_indices = torch.arange(0, duration, device=device)
        
        interpolated_amps = F.interpolate(
            harmonic_amps_transposed,
            size=duration,
            mode='linear',
            align_corners=False
        )
        
        # Similarly interpolate voiced_mix factor
        # From [B, T, 1] to [B, 1, N]
        voiced_mix_transposed = voiced_mix.transpose(1, 2)
        interpolated_voice_mix = F.interpolate(
            voiced_mix_transposed,
            size=duration,
            mode='linear',
            align_corners=False
        )
        
        # Apply harmonic amplitudes
        weighted_harmonics = harmonic_signals * interpolated_amps
        
        # Apply ADSR envelope if specified
        if self.use_adsr and note_on_frames is not None:
            weighted_harmonics = self.apply_adsr(weighted_harmonics, note_on_frames, note_off_frames, adsr_params)
        
        # Sum all harmonics
        summed_harmonics = weighted_harmonics.sum(dim=1)  # [B, N]
        
        # Apply voiced mix factor
        final_output = summed_harmonics * interpolated_voice_mix.squeeze(1)
        
        return final_output