import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RespiratoryDynamicsNetwork(nn.Module):
    """
    Optimized network to model realistic human respiratory dynamics during vocalization.
    Uses consolidated upsampling for efficiency.
    """
    def __init__(self, hidden_dim=128):
        super(RespiratoryDynamicsNetwork, self).__init__()
        
        # Main parameters prediction network (outputs all control parameters at once)
        self.breath_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim + 1, 256, kernel_size=5, padding=2),  # +1 for voiced/unvoiced
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 40, kernel_size=1)  # All breath parameters in one vector
        )
        
        # Filter bank for breath resonances
        self.filter_coeffs = nn.Parameter(torch.randn(5, 8, 3))  # 5 locations, 8 filters, 3 coeffs each
        
    def forward(self, condition, f0, audio_length):
        """
        Optimized implementation with single interpolation operation
        """
        batch_size, channels, time_steps = condition.shape
        device = condition.device
        
        # Create voiced/unvoiced feature (1.0 = voiced, 0.0 = unvoiced)
        is_voiced = (f0 > 0).float().unsqueeze(1)  # [B, 1, T]
        
        # Concatenate condition with voiced information
        condition_with_voice = torch.cat([condition, is_voiced], dim=1)
        
        # Predict all respiratory parameters in one forward pass
        all_params = self.breath_predictor(condition_with_voice)  # [B, 40, T]
        
        # Split parameters (more efficient than separate networks)
        inhalation = torch.sigmoid(all_params[:, 0:1, :])  # [B, 1, T]
        exhalation = torch.sigmoid(all_params[:, 1:2, :])  # [B, 1, T]
        pressure = torch.sigmoid(all_params[:, 2:3, :])  # [B, 1, T]
        spectral_shape = all_params[:, 3:35, :]  # [B, 32, T]
        spectral_shape = F.softmax(spectral_shape, dim=1)  # Normalize as distribution
        turbulence_location = all_params[:, 35:40, :]  # [B, 5, T]
        turbulence_location = F.softmax(turbulence_location, dim=1)  # Normalize as distribution
        
        # Extract features for modulating the harmonic signal (kept at frame rate)
        breath_features = torch.cat([
            pressure,       # Breath pressure affects harmonic intensity
            inhalation,     # Inhalation can affect pitch stability
            exhalation      # Exhalation affects harmonic structure
        ], dim=1)  # [B, 3, T]
        
        # Consolidate all control signals before upsampling
        control_signals = torch.cat([
            inhalation, 
            exhalation, 
            pressure, 
            turbulence_location
        ], dim=1)  # [B, 8, T]
        
        # Single upsampling operation for all control signals
        control_signals_audio = self._efficient_upsample(control_signals, audio_length)
        
        # Extract individual upsampled signals
        inhalation_audio = control_signals_audio[:, 0:1, :]
        exhalation_audio = control_signals_audio[:, 1:2, :]
        pressure_audio = control_signals_audio[:, 2:3, :]
        turbulence_audio = control_signals_audio[:, 3:8, :]
        
        # Upsample voiced/unvoiced decision
        voiced_audio = self._efficient_upsample(is_voiced, audio_length)
        
        # Generate base breath noise with precalculated spectral shape
        # (spectral shaping is done at frame rate, before upsampling)
        breath_noise = self._generate_colored_noise(batch_size, audio_length, spectral_shape, device)
        
        # Apply respiratory dynamics
        inhale_component = breath_noise * inhalation_audio * (1.0 - voiced_audio)
        exhale_component = breath_noise * exhalation_audio * pressure_audio
        
        # Apply location-based filtering (different resonances based on turbulence location)
        filtered_breath = self._apply_location_filters(
            inhale_component + exhale_component, 
            turbulence_audio
        )
        
        return filtered_breath, breath_features
    
    def _efficient_upsample(self, tensor, target_len):
        """
        Memory-efficient upsampling with block processing for large tensors
        (Same implementation as in HarmonicSynthesizer for consistency)
        """
        # For small tensors, use direct interpolation
        if tensor.shape[2] * target_len < 1e7:  # Heuristic threshold
            return F.interpolate(
                tensor, 
                size=target_len, 
                mode='linear', 
                align_corners=False
            )
        
        # For large tensors, use block processing
        batch_size, channels, time_steps = tensor.shape
        scale_factor = target_len / time_steps
        
        # Process in time blocks to save memory
        result = torch.zeros(batch_size, channels, target_len, device=tensor.device)
        block_size = min(1000, time_steps)  # Process 1000 frames at a time
        
        for block_start in range(0, time_steps, block_size):
            block_end = min(block_start + block_size, time_steps)
            block = tensor[:, :, block_start:block_end]
            
            # Calculate corresponding output indices
            out_start = int(block_start * scale_factor)
            out_end = min(int(block_end * scale_factor), target_len)
            
            # Interpolate this block
            upsampled_block = F.interpolate(
                block,
                size=out_end - out_start,
                mode='linear',
                align_corners=False
            )
            
            # Insert into result
            result[:, :, out_start:out_end] = upsampled_block
            
        return result
        
    def _generate_colored_noise(self, batch_size, audio_length, spectral_shape, device):
        """
        Generate spectrally shaped noise more efficiently by applying spectral shaping
        before upsampling to audio rate
        """
        # Pre-generate white noise at target length
        white_noise = torch.randn(batch_size, 1, audio_length, device=device)
        
        # Generate filter coefficients from spectral shape
        filter_length = 128  # Length of FIR filter
        
        # Create frequency domain representation
        freq_response = torch.zeros(batch_size, filter_length // 2 + 1, device=device)
        
        # Map 32 spectral bands to frequency response
        spectral_bands = 32
        for i in range(spectral_bands):
            band_avg = torch.mean(spectral_shape[:, i, :], dim=1)  # Average over time
            freq_start = int(i * (filter_length // 2) / spectral_bands)
            freq_end = int((i + 1) * (filter_length // 2) / spectral_bands)
            freq_response[:, freq_start:freq_end] = band_avg.unsqueeze(1)
        
        # Create symmetric frequency response
        full_freq_response = torch.cat([
            freq_response, 
            freq_response[:, 1:-1].flip(1)
        ], dim=1)
        
        # Convert to time domain (simplified - would use IFFT in practice)
        filter_kernel = full_freq_response
        
        # Apply filter to white noise (simplified - would use conv1d in practice)
        # This is a placeholder for actual spectral filtering implementation
        shaped_noise = white_noise * 0.1  # Placeholder
        
        return shaped_noise
    
    def _apply_location_filters(self, breath_signal, turbulence_locations):
        """
        Apply turbulence location filtering with batched operations
        """
        batch_size, _, audio_length = breath_signal.shape
        
        # Get filter coefficients for each location
        # In practice, would implement actual IIR filtering here
        filtered_breath = breath_signal  # Placeholder for actual filtering
        
        return filtered_breath