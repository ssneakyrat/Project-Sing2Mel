import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VocalFilter(nn.Module):
    """
    Implements a formant filter bank to shape the spectrum of synthesized speech/singing.
    
    This module applies spectral modifications to STFTs based on parameters predicted
    by the ParameterPredictor to simulate resonant properties of the human vocal tract.
    """
    def __init__(
        self,
        n_fft=1024,
        sample_rate=24000,
        use_parallel_filters=True,
        filter_mode='resonator'  # 'resonator' or 'gaussian'
    ):
        """
        Initialize the VocalFilter.
        
        Args:
            n_fft: Size of the FFT
            sample_rate: Audio sample rate
            use_parallel_filters: Whether to use parallel formant filters (True)
                                or a single spectral envelope (False)
            filter_mode: Type of filter to use ('resonator' or 'gaussian')
        """
        super(VocalFilter, self).__init__()
        
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.n_freqs = n_fft // 2 + 1
        self.use_parallel_filters = use_parallel_filters
        self.filter_mode = filter_mode
        
        # Precompute frequency bins for efficiency
        self.register_buffer(
            'freq_bins',
            torch.linspace(0, sample_rate / 2, self.n_freqs)
        )
        
    def _create_formant_filters_batch(self, formant_freqs, bandwidths, amplitudes):
        """
        Create formant filter responses for a batch of inputs using vectorization.
        
        Args:
            formant_freqs: Center frequencies of formants [B, T, num_formants]
            bandwidths: Bandwidths of formants [B, T, num_formants]
            amplitudes: Amplitude gains of formants [B, T, num_formants]
            
        Returns:
            Filter responses [B, T, n_freqs]
        """
        batch_size, seq_len, num_formants = formant_freqs.shape
        device = formant_freqs.device
        
        # Reshape frequency bins for broadcasting
        # Shape: [1, 1, 1, n_freqs]
        f = self.freq_bins.to(device).view(1, 1, 1, self.n_freqs)
        
        # Reshape formant parameters for broadcasting
        # Shapes: [B, T, num_formants, 1]
        f_centers = formant_freqs.unsqueeze(-1)
        bws = bandwidths.unsqueeze(-1)
        gains = amplitudes.unsqueeze(-1)
        
        if self.filter_mode == 'resonator':
            # Vectorized computation of resonator response for all formants at once
            # Using the resonance formula: gain * (bw^2) / ((f - f_center)^2 + bw^2)
            # Shape: [B, T, num_formants, n_freqs]
            numerator = gains * (bws ** 2)
            denominator = ((f - f_centers) ** 2 + bws ** 2)
            filter_responses = numerator / denominator
            
        elif self.filter_mode == 'gaussian':
            # Gaussian formant shape: gain * exp(-(f - f_center)^2 / (2 * bw^2))
            # Shape: [B, T, num_formants, n_freqs]
            exponent = -((f - f_centers) ** 2) / (2 * bws ** 2)
            filter_responses = gains * torch.exp(exponent)
        
        # Sum all formant responses along formant dimension
        # Shape: [B, T, n_freqs]
        combined_response = torch.sum(filter_responses, dim=2)
        
        return combined_response
    
    def forward(self, stft, filter_params):
        """
        Apply formant filtering to the STFT using vectorized operations.
        
        Args:
            stft: Complex STFT [B, F, T]
            filter_params: Dictionary with filter parameters from ParameterPredictor
                'frequencies': Formant frequencies [B, T, num_formants]
                'bandwidths': Formant bandwidths [B, T, num_formants]
                'amplitudes': Formant amplitudes [B, T, num_formants]
                
        Returns:
            Filtered complex STFT [B, F, T]
        """
        # Extract parameters
        formant_freqs = filter_params['frequencies']   # [B, T, num_formants]
        bandwidths = filter_params['bandwidths']       # [B, T, num_formants]
        amplitudes = filter_params['amplitudes']       # [B, T, num_formants]
        
        batch_size, n_freqs, time_steps = stft.shape
        device = stft.device
        
        # Transpose STFT to [B, T, F] for time-varying filter application
        stft_time_first = stft.transpose(1, 2)  # [B, T, F]
        
        # Get magnitude and phase
        magnitude = torch.abs(stft_time_first)  # [B, T, F]
        phase = torch.angle(stft_time_first)    # [B, T, F]
        
        if self.use_parallel_filters:
            # Generate combined filter response using vectorized operations
            filter_response = self._create_formant_filters_batch(
                formant_freqs, bandwidths, amplitudes
            )  # [B, T, F]
            
            # Apply filter response to magnitude
            filtered_magnitude = magnitude * filter_response
            
        else:
            # Multiplicative spectral envelope approach
            # Create base spectral template (small constant to avoid nullifying entire spectrum)
            spectral_envelope = torch.ones_like(magnitude) * 0.01  # [B, T, F]
            
            # Generate filter response
            filter_response = self._create_formant_filters_batch(
                formant_freqs, bandwidths, amplitudes
            )  # [B, T, F]
            
            # Add filter response to envelope
            spectral_envelope = spectral_envelope + filter_response
            
            # Apply envelope to magnitude
            filtered_magnitude = magnitude * spectral_envelope
        
        # Reconstruct complex STFT using original phase
        # Use Euler's formula: e^(iθ) = cos(θ) + i*sin(θ)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        
        # Build complex tensor (PyTorch 1.7+ supports complex tensors directly)
        real_part = filtered_magnitude * cos_phase
        imag_part = filtered_magnitude * sin_phase
        filtered_stft_time_first = torch.complex(real_part, imag_part)
        
        # Transpose back to [B, F, T]
        filtered_stft = filtered_stft_time_first.transpose(1, 2)
        
        return filtered_stft
