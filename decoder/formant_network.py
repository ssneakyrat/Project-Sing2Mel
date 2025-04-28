import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FormantNetwork(nn.Module):
    """
    Network to model vocal formants that define vowel sounds and voice quality.
    Predicts formant frequencies and bandwidths, then applies formant filtering
    to harmonic amplitudes.
    
    ENHANCED:
    - Extended formant frequency range to 8000 Hz
    - Additive formant model instead of multiplicative
    - High-frequency floor to prevent excessive attenuation
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
        
        # MODIFIED: Extended range from 3500 Hz to 8000 Hz for better high-frequency content
        # Apply sigmoid and scale to reasonable frequency range (200-8000 Hz)
        formant_freqs = 200 + 7800 * torch.sigmoid(formant_freqs)  # Range: 200-8000 Hz
        
        formant_bandwidths = formant_params[:, self.num_formants:, :]  # [B, num_formants, T]
        # Apply softplus to ensure positive bandwidths (50-300 Hz range)
        # MODIFIED: Increased maximum bandwidth for higher formants
        formant_bandwidths = 50 + 250 * F.softplus(formant_bandwidths)  # Range: 50-300 Hz
        
        # Create harmonic frequency tensor
        # First, repeat f0 for each harmonic
        f0_expanded = f0.unsqueeze(1)  # [B, 1, T]
        harmonic_indices = torch.arange(1, num_harmonics + 1, device=device).reshape(1, -1, 1)  # [1, num_harmonics, 1]
        harmonic_freqs = harmonic_indices * f0_expanded  # [B, num_harmonics, T]
        
        # MODIFIED: Apply formant filtering using additive model instead of multiplicative
        shaped_amplitudes = self._apply_formant_filter_additive(
            harmonic_amplitudes, harmonic_freqs, formant_freqs, formant_bandwidths)
        
        return shaped_amplitudes
        
    def _apply_formant_filter_additive(self, harmonic_amplitudes, harmonic_freqs, formant_freqs, formant_bandwidths):
        """
        Apply formant filtering to harmonic amplitudes using an ADDITIVE model
        instead of multiplicative to prevent severe attenuation of high frequencies.
        
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
        
        # Initialize formant gains using ADDITIVE model (not multiplicative)
        formant_gain = torch.zeros_like(harmonic_amplitudes)
        
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
            
            # ADD the formant contribution instead of multiplying
            # This prevents cascading attenuation of frequencies outside any formant
            formant_gain = formant_gain + resonance
        
        # Normalize gain to a reasonable range (0.4 to 1.0)
        max_gain = torch.max(formant_gain, dim=1, keepdim=True)[0]
        normalized_gain = 0.4 + 0.6 * (formant_gain / torch.clamp(max_gain, min=1e-5))
        
        # NEW: Apply high-frequency floor to prevent complete attenuation
        # Create frequency-dependent floor that increases with harmonic number
        harmonic_indices = torch.arange(1, num_harmonics + 1, device=harmonic_amplitudes.device).reshape(1, -1, 1)
        normalized_index = (harmonic_indices - 1) / (num_harmonics - 1)  # Range: 0-1
        
        # Floor starts at 0 for lowest harmonic and increases to 0.15 for highest
        high_freq_floor = 0.15 * normalized_index
        
        # Apply formant gains to harmonic amplitudes, ensuring minimum values via high_freq_floor
        with_formants = harmonic_amplitudes * normalized_gain
        with_floor = harmonic_amplitudes * high_freq_floor
        
        # Take maximum of formant-shaped and floor-shaped amplitudes
        shaped_amplitudes = torch.maximum(with_formants, with_floor)
        
        return shaped_amplitudes
        
    # Keep old method for backward compatibility, but don't use it
    def _apply_formant_filter(self, harmonic_amplitudes, harmonic_freqs, formant_freqs, formant_bandwidths):
        """
        Original multiplicative formant filter (kept for compatibility)
        This function is kept only for compatibility - the additive version should be used instead.
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
            numerator = f_bw ** 2
            denominator = (harmonic_freqs - f_freq) ** 2 + numerator
            resonance = numerator / torch.clamp(denominator, min=1e-5)
            
            # Accumulate the effect of this formant
            formant_gain = formant_gain * (0.8 + 0.2 * resonance)
        
        # Apply formant gains to harmonic amplitudes
        shaped_amplitudes = harmonic_amplitudes * formant_gain
        
        return shaped_amplitudes