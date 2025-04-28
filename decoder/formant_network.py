import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FormantNetwork(nn.Module):
    """
    Optimized network to model vocal formants that define vowel sounds and voice quality.
    Predicts formant frequencies and bandwidths, then applies formant filtering
    to harmonic amplitudes using vectorized operations.
    
    ENHANCED:
    - Extended formant frequency range to 8000 Hz
    - Additive formant model instead of multiplicative
    - High-frequency floor to prevent excessive attenuation
    - Vectorized operations for improved efficiency
    - Precomputed tensors for common operations
    """
    def __init__(self, num_formants=5, hidden_dim=128, input_channels=None):
        super(FormantNetwork, self).__init__()
        
        # If input_channels is not provided, use hidden_dim
        if input_channels is None:
            input_channels = hidden_dim
        
        # Network to predict formant frequencies and bandwidths using depthwise separable convolutions
        self.formant_predictor = nn.Sequential(
            # First depthwise separable convolution
            nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),  # Depthwise
            nn.Conv1d(input_channels, 256, kernel_size=1),  # Pointwise
            nn.LeakyReLU(0.1),
            
            # Second depthwise separable convolution
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=256),  # Depthwise
            nn.Conv1d(256, 128, kernel_size=1),  # Pointwise
            nn.LeakyReLU(0.1),
            
            # Final pointwise convolution
            nn.Conv1d(128, num_formants * 2, kernel_size=1)  # Predict frequency and bandwidth
        )
        
        self.num_formants = num_formants
        
        # Precompute common tensors and constants
        # These will be registered as buffers to avoid recreating them each time
        self.register_buffer('freq_range_min', torch.tensor(200.0))  # Minimum formant frequency
        self.register_buffer('freq_range_max', torch.tensor(8000.0))  # Maximum formant frequency
        self.register_buffer('bw_range_min', torch.tensor(50.0))  # Minimum bandwidth
        self.register_buffer('bw_range_max', torch.tensor(300.0))  # Maximum bandwidth
        
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
        formant_params_split = torch.chunk(formant_params, 2, dim=1)
        formant_freqs_raw = formant_params_split[0]  # [B, num_formants, T]
        formant_bandwidths_raw = formant_params_split[1]  # [B, num_formants, T]
        
        # Apply sigmoid and scale to reasonable frequency range (200-8000 Hz)
        # Using precomputed constants from buffers for better efficiency
        freq_range = self.freq_range_max - self.freq_range_min
        formant_freqs = self.freq_range_min + freq_range * torch.sigmoid(formant_freqs_raw)
        
        # Apply softplus to ensure positive bandwidths (50-300 Hz range)
        bw_range = self.bw_range_max - self.bw_range_min
        formant_bandwidths = self.bw_range_min + bw_range * F.softplus(formant_bandwidths_raw)
        
        # Create harmonic frequency tensor with vectorized operations
        # First, repeat f0 for each harmonic
        f0_expanded = f0.unsqueeze(1)  # [B, 1, T]
        
        # Create harmonic indices tensor once
        harmonic_indices = torch.arange(1, num_harmonics + 1, device=device).reshape(1, -1, 1)  # [1, num_harmonics, 1]
        
        # Vectorized multiplication for all harmonics at once
        harmonic_freqs = harmonic_indices * f0_expanded  # [B, num_harmonics, T]
        
        # Apply formant filtering using additive model (vectorized implementation)
        shaped_amplitudes = self._apply_formant_filter_additive_vectorized(
            harmonic_amplitudes, harmonic_freqs, formant_freqs, formant_bandwidths)
        
        return shaped_amplitudes
        
    def _apply_formant_filter_additive_vectorized(self, harmonic_amplitudes, harmonic_freqs, formant_freqs, formant_bandwidths):
        """
        Apply formant filtering to harmonic amplitudes using an ADDITIVE model
        with fully vectorized operations for better efficiency.
        
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
        device = harmonic_amplitudes.device
        
        # Initialize formant gains using ADDITIVE model (not multiplicative)
        formant_gain = torch.zeros_like(harmonic_amplitudes)
        
        # Reshape tensors for vectorized operations
        # harmonic_freqs: [B, num_harmonics, T]
        # formant_freqs: [B, num_formants, T] -> [B, 1, num_formants, T]
        # formant_bandwidths: [B, num_formants, T] -> [B, 1, num_formants, T]
        
        formant_freqs_expanded = formant_freqs.unsqueeze(1)  # [B, 1, num_formants, T]
        formant_bw_expanded = formant_bandwidths.unsqueeze(1)  # [B, 1, num_formants, T]
        harmonic_freqs_expanded = harmonic_freqs.unsqueeze(2)  # [B, num_harmonics, 1, T]
        
        # Calculate resonance response for all harmonics and all formants at once
        # This replaces the loop over formants with a single vectorized operation
        
        # Calculate the numerator (bandwidth squared)
        numerator = formant_bw_expanded ** 2  # [B, 1, num_formants, T]
        
        # Calculate the denominator ((harmonic_freq - formant_freq)^2 + bandwidth^2)
        freq_diff = harmonic_freqs_expanded - formant_freqs_expanded  # [B, num_harmonics, num_formants, T]
        denominator = (freq_diff ** 2) + numerator  # [B, num_harmonics, num_formants, T]
        
        # Calculate the resonance factors (bandwidth^2 / denominator)
        resonance = numerator / torch.clamp(denominator, min=1e-5)  # [B, num_harmonics, num_formants, T]
        
        # Sum resonance factors across all formants
        formant_gain = torch.sum(resonance, dim=2)  # [B, num_harmonics, T]
        
        # Normalize gain to a reasonable range (0.4 to 1.0)
        # Use dim=1 to normalize across harmonics
        max_gain = torch.max(formant_gain, dim=1, keepdim=True)[0]  # [B, 1, T]
        normalized_gain = 0.4 + 0.6 * (formant_gain / torch.clamp(max_gain, min=1e-5))
        
        # Apply high-frequency floor to prevent complete attenuation
        # Create frequency-dependent floor that increases with harmonic number
        # Create harmonic indices tensor
        harmonic_indices = torch.arange(1, num_harmonics + 1, device=device).reshape(1, -1, 1)  # [1, num_harmonics, 1]
        normalized_index = (harmonic_indices - 1) / (num_harmonics - 1)  # [1, num_harmonics, 1]
        
        # Floor starts at 0 for lowest harmonic and increases to 0.15 for highest
        high_freq_floor = 0.15 * normalized_index
        
        # Apply formant gains to harmonic amplitudes in a vectorized manner
        with_formants = harmonic_amplitudes * normalized_gain
        with_floor = harmonic_amplitudes * high_freq_floor
        
        # Take elementwise maximum of formant-shaped and floor-shaped amplitudes
        shaped_amplitudes = torch.maximum(with_formants, with_floor)
        
        return shaped_amplitudes