import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HarmonicWaveGenerator(nn.Module):
    def __init__(self, n_mels=80, hop_length=240, num_harmonics=32, sample_rate=22050):
        super(HarmonicWaveGenerator, self).__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        
        # F0 processing - minimal, just normalization
        self.f0_norm = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Harmonic amplitude predictor from mel spectrogram
        self.amplitude_predictor = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_harmonics, kernel_size=3, padding=1),
        )
        
        # Initialize last layer to small values
        torch.nn.init.xavier_uniform_(self.amplitude_predictor[-1].weight, gain=0.1)
        torch.nn.init.zeros_(self.amplitude_predictor[-1].bias)
        
        # Add sigmoid to ensure amplitudes are in [0, 1] range
        self.amplitude_activation = nn.Sigmoid()
        
        # Phase predictor (optional, can be disabled)
        self.phase_predictor = nn.Sequential(
            nn.Conv1d(n_mels + 16, 64, kernel_size=3, padding=1),  # mel + f0 features
            nn.ReLU(),
            nn.Conv1d(64, num_harmonics, kernel_size=3, padding=1),
            nn.Tanh()  # Phase offsets between -π and π
        )
        
        # Initialize phase accumulator
        self.register_buffer('last_phase', torch.zeros(1, num_harmonics))
        
    def generate_harmonics(self, f0_hz, timestamps):
        """Generate harmonic frequencies from fundamental frequency.
        
        Args:
            f0_hz: Fundamental frequency in Hz (B, T)
            timestamps: Time values for each sample (B, T_samples)
        
        Returns:
            harmonic_freqs: Frequencies for all harmonics (B, num_harmonics, T)
        """
        B, T = f0_hz.shape
        
        # Generate harmonic numbers [1, 2, 3, ..., num_harmonics]
        harmonic_numbers = torch.arange(1, self.num_harmonics + 1, device=f0_hz.device)
        harmonic_numbers = harmonic_numbers.unsqueeze(0).unsqueeze(-1)  # (1, num_harmonics, 1)
        
        # Calculate harmonic frequencies
        harmonic_freqs = f0_hz.unsqueeze(1) * harmonic_numbers  # (B, num_harmonics, T)
        
        return harmonic_freqs
    
    def synthesize_harmonics(self, harmonic_freqs, amplitudes, phases, timestamps):
        """Synthesize waveform from harmonic components.
        
        Args:
            harmonic_freqs: Frequencies for all harmonics (B, num_harmonics, T)
            amplitudes: Amplitudes for all harmonics (B, num_harmonics, T)
            phases: Phase offsets for all harmonics (B, num_harmonics, T)
            timestamps: Time values for each sample (B, T_samples)
        
        Returns:
            waveform: Synthesized waveform (B, T_samples)
        """
        B, _, T = harmonic_freqs.shape
        
        # Interpolate parameters to sample rate
        # Use nearest neighbor for stability at edges
        harmonic_freqs_interp = F.interpolate(harmonic_freqs, size=timestamps.shape[1], mode='nearest')
        amplitudes_interp = F.interpolate(amplitudes, size=timestamps.shape[1], mode='nearest')
        phases_interp = F.interpolate(phases, size=timestamps.shape[1], mode='nearest')
        
        # Generate phase values
        # Use cumulative sum with proper scaling to prevent numerical issues
        dt = 1.0 / self.sample_rate
        phase_increments = harmonic_freqs_interp * 2 * math.pi * dt
        
        # Clamp phase increments to prevent instability
        phase_increments = torch.clamp(phase_increments, min=-math.pi, max=math.pi)
        
        # Calculate cumulative phase
        phase_integral = torch.cumsum(phase_increments, dim=-1)
        
        # Add phase offsets (scaled to [-π, π])
        total_phase = phase_integral + phases_interp * math.pi
        
        # Apply modulo to prevent phase overflow
        total_phase = torch.fmod(total_phase, 2 * math.pi)
        
        # Generate sinusoids
        harmonics = amplitudes_interp * torch.sin(total_phase)
        
        # Check for NaN
        if torch.isnan(harmonics).any():
            print("NaN in harmonics!")
            harmonics = torch.nan_to_num(harmonics, nan=0.0)
        
        # Sum all harmonics
        waveform = torch.sum(harmonics, dim=1)
        
        # Normalize to prevent clipping
        waveform_max = torch.max(torch.abs(waveform), dim=-1, keepdim=True)[0]
        waveform_max = torch.clamp(waveform_max, min=1e-8)  # Prevent division by zero
        waveform = waveform / waveform_max
        
        return waveform
        
    def forward(self, mel, f0, voicing):
        """
        Args:
            mel: Mel spectrogram (B, n_mels, T)
            f0: Fundamental frequency (B, 1, T)
            voicing: Voicing prediction (B, 1, T)
        
        Returns:
            waveform: Audio waveform (B, T * hop_length)
        """
        B, n_mels, T = mel.shape
        
        # Debug prints to check inputs
        if torch.isnan(mel).any():
            print("NaN in mel input!")
        if torch.isnan(f0).any():
            print("NaN in f0 input!")
        if torch.isnan(voicing).any():
            print("NaN in voicing input!")
        
        # Convert f0 to Hz - check if it's already in Hz or needs conversion
        f0_hz = f0.squeeze(1)  # (B, T)
        
        # If f0 is in MIDI or log scale, convert it
        # Common ranges:
        # - MIDI: typically 0-127
        # - Log scale: typically negative values
        # - Hz: typically 50-2000 for speech/singing
        if f0_hz.mean() < 50:  # Likely not in Hz
            if f0_hz.mean() < 0:  # Likely log scale
                f0_hz = torch.exp(f0_hz)
            elif f0_hz.mean() < 128:  # Likely MIDI
                f0_hz = 440.0 * (2.0 ** ((f0_hz - 69) / 12.0))
        
        # Clip f0 to reasonable range to prevent numerical issues
        f0_hz = torch.clamp(f0_hz, min=50.0, max=4000.0)
        
        # Use the provided voicing instead of simple thresholding
        voiced = voicing.squeeze(1)  # (B, T)
        f0_hz = f0_hz * voiced + 100.0 * (1 - voiced)  # Replace unvoiced f0 with safe value
        
        # Process f0
        f0_features = self.f0_norm(f0)
        
        # Predict harmonic amplitudes from mel
        amplitude_logits = self.amplitude_predictor(mel)  # (B, num_harmonics, T)
        amplitudes = self.amplitude_activation(amplitude_logits)
        amplitudes = amplitudes * voiced.unsqueeze(1)  # Zero out unvoiced regions
        
        # Add a small value to prevent division by zero
        amplitudes = amplitudes + 1e-8
        
        # Predict phase offsets
        combined_features = torch.cat([mel, f0_features], dim=1)
        phase_offsets = self.phase_predictor(combined_features)  # (B, num_harmonics, T)
        
        # Generate harmonic frequencies
        harmonic_freqs = self.generate_harmonics(f0_hz, None)  # (B, num_harmonics, T)
        
        # Create timestamps for the output waveform
        timestamps = torch.arange(T * self.hop_length, device=mel.device).float() / self.sample_rate
        timestamps = timestamps.unsqueeze(0).expand(B, -1)  # (B, T_samples)
        
        # Synthesize waveform
        waveform = self.synthesize_harmonics(harmonic_freqs, amplitudes, phase_offsets, timestamps)
        
        # Check for NaN in output
        if torch.isnan(waveform).any():
            print("NaN in waveform output!")
            waveform = torch.nan_to_num(waveform, nan=0.0)
        
        # Apply tanh to bound the output
        waveform = torch.tanh(waveform)
        
        return waveform