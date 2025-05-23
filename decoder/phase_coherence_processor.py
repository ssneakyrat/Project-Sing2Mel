import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PhaseCoherenceProcessor(nn.Module):
    """
    Handles phase relationships between harmonics for more natural vocal synthesis.
    Models phonation-dependent phase offsets and formant-related phase dispersal.
    The transient/reset mechanism has been removed to prevent broken phoneme transitions.
    """
    def __init__(self, num_harmonics=100, input_channels=128):
        super(PhaseCoherenceProcessor, self).__init__()
        self.num_harmonics = num_harmonics
        
        # Create harmonic indices tensor (registered as buffer)
        harmonic_indices = torch.arange(1, num_harmonics + 1).float().view(1, -1, 1)
        self.register_buffer('harmonic_indices', harmonic_indices)
        
        # Phonation-dependent phase offset network
        self.phonation_phase_net = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, num_harmonics, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range, will be scaled to [-π, π]
        )
        
        # Formant-related phase dispersal network
        self.formant_phase_processor = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, num_harmonics, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range, will be scaled to [-π/2, π/2]
        )
        
        # Voice quality phase coefficient network
        self.voice_quality_net = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1], controls coherence amount
        )
        
    def forward(self, condition, f0, formant_centers=None, formant_bandwidths=None):
        """
        Process phase relationships for enhanced coherence.
        Reset mechanism has been removed to prevent broken phoneme transitions.
        
        Args:
            condition: Conditioning features [B, input_channels, T]
            f0: Fundamental frequency [B, 1, T]
            formant_centers: Optional formant center frequencies [B, T, num_formants]
            formant_bandwidths: Optional formant bandwidths [B, T, num_formants]
            
        Returns:
            phase_offsets: Phase offset tensor to add to base phases [B, num_harmonics, T]
        """
        batch_size, _, time_steps = condition.shape
        
        # 1. Generate phonation-dependent phase offsets
        phonation_offsets = self.phonation_phase_net(condition)  # [B, num_harmonics, T]
        # Scale from [-1, 1] to [-π, π]
        phonation_offsets = phonation_offsets * math.pi
        
        # 2. Generate formant-related phase dispersal
        formant_phase_shifts = self.formant_phase_processor(condition)  # [B, num_harmonics, T]
        # Scale from [-1, 1] to [-π/2, π/2] for less extreme shifts
        formant_phase_shifts = formant_phase_shifts * (math.pi / 2)
        
        # If formant information is provided, enhance the formant-related phase shifts
        if formant_centers is not None and formant_bandwidths is not None:
            enhanced_shifts = self._enhance_formant_phase_shifts(
                formant_phase_shifts, formant_centers, formant_bandwidths, f0
            )
            formant_phase_shifts = enhanced_shifts
        
        # Generate voice quality coefficient (controls phase coherence amount)
        voice_quality = self.voice_quality_net(condition)  # [B, 1, T]
        
        # Combine phase modifications
        phase_offsets = (voice_quality * phonation_offsets) + formant_phase_shifts
        
        return phase_offsets
    
    def _enhance_formant_phase_shifts(self, base_shifts, formant_centers, formant_bandwidths, f0):
        """
        Enhance formant-related phase shifts based on specific formant information.
        
        Args:
            base_shifts: Base phase shifts from the network [B, num_harmonics, T]
            formant_centers: Formant center frequencies [B, T, num_formants]
            formant_bandwidths: Formant bandwidths [B, T, num_formants]
            f0: Fundamental frequency [B, 1, T]
            
        Returns:
            enhanced_shifts: Enhanced phase shifts [B, num_harmonics, T]
        """
        batch_size, num_harmonics, time_steps = base_shifts.shape
        num_formants = formant_centers.shape[-1]
        
        # Initialize enhanced shifts with base shifts
        enhanced_shifts = base_shifts.clone()
        
        # Compute harmonic frequencies (f0 * harmonic_index)
        # Expand f0 for broadcasting if needed
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        harmonic_freqs = f0 * self.harmonic_indices  # [B, num_harmonics, T]
        
        # For each formant, enhance phase shifts around formant regions
        for i in range(num_formants):
            # Extract formant parameters for the current formant
            f_center = formant_centers[:, :, i].unsqueeze(1)  # [B, 1, T]
            f_bw = formant_bandwidths[:, :, i].unsqueeze(1)  # [B, 1, T]
            
            # Compute normalized frequency distance from formant center
            freq_diff = (harmonic_freqs - f_center) / f_bw
            
            # Generate Gaussian-like weight for proximity to formant
            formant_proximity = torch.exp(-0.5 * freq_diff * freq_diff)  # [B, num_harmonics, T]
            
            # Generate progressive phase shift based on position relative to formant center
            # Harmonics below formant get negative shift, above get positive shift
            direction = torch.sign(harmonic_freqs - f_center)
            
            # Apply weighted phase shift based on proximity to formant
            # The closer to the formant, the more shift applied
            formant_specific_shift = direction * formant_proximity * (math.pi / 4)  # max ±π/4
            
            # Add to enhanced shifts
            enhanced_shifts = enhanced_shifts + formant_specific_shift
        
        return enhanced_shifts