import torch
import torch.nn as nn
import numpy as np

from decoder.core import remove_above_nyquist

class NoiseGenerator(nn.Module):
    """Generate noise with spectral control using linear network approach"""
    def __init__(self, fs, noise_dim=64, is_remove_above_nyquist=True):
        super().__init__()
        self.fs = fs
        self.noise_dim = noise_dim
        self.is_remove_above_nyquist = is_remove_above_nyquist
        
        # Linear network layers for noise generation and shaping
        self.freq_mapper = nn.Linear(1, noise_dim)
        self.amp_mapper = nn.Linear(1, noise_dim)
        self.noise_basis = nn.Linear(noise_dim, noise_dim)
        self.spectral_shaping = nn.Linear(noise_dim, noise_dim)
        self.output_mixer = nn.Linear(noise_dim, 1, bias=False)
        self.phase_tracker = nn.Linear(1, 1, bias=False)
        
        # Initialize with reasonable defaults
        with torch.no_grad():
            # Initialize frequency mapper to spread across spectrum
            freq_init = torch.linspace(0, 1, noise_dim).reshape(noise_dim, 1)
            self.freq_mapper.weight.copy_(freq_init)
            self.freq_mapper.bias.zero_()
            
            # Initialize noise basis as identity
            self.noise_basis.weight.copy_(torch.eye(noise_dim))
            self.noise_basis.bias.zero_()
            
            # Initialize spectral shaping for pink-ish noise
            spectral_weights = torch.linspace(1, 0.1, noise_dim).reshape(noise_dim, 1)
            self.spectral_shaping.weight.copy_(spectral_weights * torch.eye(noise_dim))
            self.spectral_shaping.bias.zero_()
            
            # Equal mix for output
            self.output_mixer.weight.fill_(1.0 / noise_dim)
            
            # Phase tracking as identity
            self.phase_tracker.weight.fill_(1.0)
    
    def forward(self, f0, amplitudes, initial_phase=None):
        '''
                    f0: B x T x 1 (Hz)
            amplitudes: B x T x n_harmonic
         initial_phase: B x 1 x 1
          ---
              signal: B x T
         final_phase: B x 1 x 1
        '''
        batch_size, time_steps, n_harmonic = amplitudes.shape
        device = f0.device
        
        if initial_phase is None:
            initial_phase = torch.zeros(batch_size, 1, 1).to(device)
        
        # Use f0 to control noise characteristics
        mask = (f0 > 0).detach()
        f0 = f0.detach()
        
        # Generate random noise seeds that are consistent across time for each batch
        noise_seeds = torch.randn(batch_size, 1, self.noise_dim).to(device)
        noise_seeds = noise_seeds.expand(-1, time_steps, -1)
        
        # Use frequency to control spectral characteristics
        f0_norm = f0 / (self.fs/2)  # Normalize by Nyquist
        freq_weights = self.freq_mapper(f0_norm)
        
        # Apply frequency-dependent weighting to noise
        shaped_noise = noise_seeds * freq_weights
        
        # Transform through noise basis
        noise_components = self.noise_basis(shaped_noise)
        
        # Anti-aliasing (similar to harmonic oscillator)
        if self.is_remove_above_nyquist:
            # Average harmonics from amplitudes for spectral control
            harmonic_weights = amplitudes.mean(dim=-1, keepdim=True)
            
            # Apply anti-aliasing based on f0 similar to remove_above_nyquist
            nyquist = self.fs / 2
            harmonic_freqs = torch.linspace(1, n_harmonic, n_harmonic).to(device)
            harmonic_freqs = harmonic_freqs.reshape(1, 1, n_harmonic)
            harmonic_mask = (f0 * harmonic_freqs < nyquist).float()
            
            # Calculate spectral weights with anti-aliasing
            spectral_weights = (harmonic_weights * harmonic_mask).sum(dim=-1, keepdim=True) / n_harmonic
            spectral_weights = spectral_weights.expand(-1, -1, self.noise_dim)
        else:
            # Use raw amplitudes for spectral weights without anti-aliasing
            spectral_weights = amplitudes.mean(dim=-1, keepdim=True).expand(-1, -1, self.noise_dim)
        
        # Apply spectral shaping
        noise_output = self.spectral_shaping(noise_components * spectral_weights)
        
        # Mix down to mono signal
        signal = self.output_mixer(noise_output).squeeze(-1)
        
        # Apply mask (similar to harmonic oscillator)
        signal = signal * mask.squeeze(-1)
        
        # Track phase for API compatibility
        # For noise, phase doesn't have traditional meaning, but we track it anyway
        phase_change = torch.sum(f0 / self.fs * 2 * np.pi, dim=1, keepdim=True)
        final_phase = (initial_phase + self.phase_tracker(phase_change)) % (2 * np.pi)
        
        return signal, final_phase.detach()