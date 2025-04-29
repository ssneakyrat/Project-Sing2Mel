import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FormantProcessor(nn.Module):
    """
    Handles formant-based spectral shaping for realistic vocal synthesis.
    """
    def __init__(self, num_formants=5, input_channels=128, sample_rate=24000):
        super(FormantProcessor, self).__init__()
        self.num_formants = num_formants
        self.sample_rate = sample_rate
        
        # Formant parameter network - predicts center frequency, bandwidth, and amplitude
        # for each formant region (3 parameters per formant)
        self.formant_param_net = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, self.num_formants * 3, kernel_size=3, padding=1)
        )
        
        # Frequency bins for formant computation (pre-compute)
        nyquist = sample_rate / 2
        freq_bins = torch.linspace(0, nyquist, 101)[:-1]  # Using 100 as placeholder for num_harmonics
        self.register_buffer('freq_bins', freq_bins.view(1, -1, 1))
        
    def forward(self, condition, harmonic_amplitudes, num_harmonics):
        """
        Apply formant processing to harmonic amplitudes.
        
        Args:
            condition: Conditioning features [B, input_channels, T]
            harmonic_amplitudes: Base harmonic amplitudes [B, num_harmonics, T]
            f0: Fundamental frequency [B, 1, T] or [B, T]
            num_harmonics: Number of harmonics
            
        Returns:
            Enhanced harmonic amplitudes with formant resonances applied [B, num_harmonics, T]
        """
        batch_size, _, time_steps = harmonic_amplitudes.shape
        
        # Generate formant parameters: [B, num_formants*3, T]
        formant_params = self.formant_param_net(condition)
        
        # Split formant parameters into center frequencies, bandwidths, and amplitudes
        # Reshape to [B, num_formants, 3, T] then split on dim 2
        formant_params = formant_params.view(batch_size, self.num_formants, 3, time_steps)
        
        # Get center frequencies (Hz) - constrain to reasonable vocal range (80Hz - 11000Hz)
        formant_centers = 80 + 11000 * torch.sigmoid(formant_params[:, :, 0, :])  # [B, num_formants, T]

        # Get bandwidths (Hz) - constrain to reasonable range (50Hz - 1000Hz)
        formant_bandwidths = 50 + 950 * torch.sigmoid(formant_params[:, :, 1, :])  # [B, num_formants, T]

        # Get formant amplitudes - using softplus for positive values with smooth gradient
        formant_amplitudes = F.softplus(formant_params[:, :, 2, :])  # [B, num_formants, T]
        
        # Reshape to make it easier to work with
        formant_centers = formant_centers.transpose(1, 2)      # [B, T, num_formants]
        formant_bandwidths = formant_bandwidths.transpose(1, 2)  # [B, T, num_formants]
        formant_amplitudes = formant_amplitudes.transpose(1, 2)  # [B, T, num_formants]
            
        # Apply formants at frame rate
        enhanced_amplitudes = self._apply_formants_frame_level(
            harmonic_amplitudes,
            formant_centers,
            formant_bandwidths,
            formant_amplitudes,
            num_harmonics
        )
        
        return enhanced_amplitudes
        
    def _apply_formants_frame_level(self, harmonic_amplitudes, formant_centers, 
                                    formant_bandwidths, formant_amplitudes, num_harmonics):
        """
        Apply formant resonances to harmonic amplitudes at frame rate (before upsampling).
        
        Args:
            harmonic_amplitudes: Harmonic amplitudes [B, num_harmonics, T]
            f0: Fundamental frequency [B, 1, T]
            formant_centers: Formant center frequencies [B, T, num_formants]
            formant_bandwidths: Formant bandwidths [B, T, num_formants]
            formant_amplitudes: Formant amplitudes [B, T, num_formants]
            num_harmonics: Number of harmonics
            
        Returns:
            Enhanced harmonic amplitudes with formant resonances applied [B, num_harmonics, T]
        """
        batch_size, _, time_steps = harmonic_amplitudes.shape
        num_formants = formant_centers.shape[2]
        
        # Compute harmonic frequencies based on f0
        # First, create indices for each harmonic: [1, 2, 3, ..., num_harmonics]
        harmonic_idx = torch.arange(1, num_harmonics + 1, device=harmonic_amplitudes.device)
        harmonic_idx = harmonic_idx.view(1, num_harmonics, 1)  # [1, num_harmonics, 1]
        
        # Then multiply by f0 to get frequency of each harmonic
        # maybe add learn able f0 like processor for each formant
        harmonic_freqs = harmonic_idx

        # Initialize formant scaling tensor
        formant_scaling = torch.ones_like(harmonic_amplitudes)  # [B, num_harmonics, T]
        
        # For each formant, apply resonance effect
        for i in range(num_formants):
            # Reshape formant parameters for broadcasting with harmonic frequencies
            # Extract the current formant parameters
            f_center = formant_centers[:, :, i].unsqueeze(1)  # [B, 1, T]
            f_bw = formant_bandwidths[:, :, i].unsqueeze(1)  # [B, 1, T]
            f_amp = formant_amplitudes[:, :, i].unsqueeze(1)  # [B, 1, T]
            
            # Compute formant response using resonance formula
            # Normalize frequency distance by bandwidth - creates a Gaussian-like resonance
            freq_diff = (harmonic_freqs - f_center) / f_bw
            formant_response = f_amp * torch.exp(-0.5 * freq_diff * freq_diff)
            
            # Accumulate the formant effect
            formant_scaling = formant_scaling + formant_response
        
        # Apply the combined formant scaling to harmonic amplitudes
        return harmonic_amplitudes * formant_scaling
    
    def get_formant_params(self, condition):
        """
        Extract formant parameters for external use by other processors.
        
        Args:
            condition: Conditioning features [B, input_channels, T]
            
        Returns:
            formant_centers: Formant center frequencies [B, T, num_formants]
            formant_bandwidths: Formant bandwidths [B, T, num_formants]
            formant_amplitudes: Formant amplitudes [B, T, num_formants]
        """
        batch_size, _, time_steps = condition.shape
        
        # Generate formant parameters: [B, num_formants*3, T]
        formant_params = self.formant_param_net(condition)
        
        # Split formant parameters into center frequencies, bandwidths, and amplitudes
        # Reshape to [B, num_formants, 3, T] then split on dim 2
        formant_params = formant_params.view(batch_size, self.num_formants, 3, time_steps)
        
        # Get center frequencies (Hz) - constrain to reasonable vocal range (80Hz - 11000Hz)
        formant_centers = 80 + 11000 * torch.sigmoid(formant_params[:, :, 0, :])  # [B, num_formants, T]

        # Get bandwidths (Hz) - constrain to reasonable range (50Hz - 1000Hz)
        formant_bandwidths = 50 + 950 * torch.sigmoid(formant_params[:, :, 1, :])  # [B, num_formants, T]

        # Get formant amplitudes - using softplus for positive values with smooth gradient
        formant_amplitudes = F.softplus(formant_params[:, :, 2, :])  # [B, num_formants, T]
        
        # Reshape to make it easier to work with
        formant_centers = formant_centers.transpose(1, 2)      # [B, T, num_formants]
        formant_bandwidths = formant_bandwidths.transpose(1, 2)  # [B, T, num_formants]
        formant_amplitudes = formant_amplitudes.transpose(1, 2)  # [B, T, num_formants]
        
        return formant_centers, formant_bandwidths, formant_amplitudes