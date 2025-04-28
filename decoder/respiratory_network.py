import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RespiratoryDynamicsNetwork(nn.Module):
    """
    Optimized network to model realistic human respiratory dynamics during vocalization.
    - Consolidated upsampling for efficiency
    - Vectorized filter application
    - FFT-based spectral shaping
    - Improved memory management
    """
    def __init__(self, hidden_dim=128, input_channels=None):
        super(RespiratoryDynamicsNetwork, self).__init__()
        
        # If input_channels is not provided, use hidden_dim
        if input_channels is None:
            input_channels = hidden_dim
            
        # Main parameters prediction network (outputs all control parameters at once)
        # +1 for voiced/unvoiced
        input_dim = input_channels + 1
        
        self.breath_predictor = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 40, kernel_size=1)  # All breath parameters in one vector
        )
        
        # Filter bank for breath resonances - directly use parameters for efficiency
        # Instead of using a Parameter tensor, we'll construct the filters on-the-fly
        
        # Register buffers for precomputed parameters
        spectral_bands = 32
        self.register_buffer('band_frequencies', torch.linspace(0, 1, spectral_bands + 1))
        
        # Precompute filter shapes for 5 different articulation points
        filter_shapes = torch.zeros(5, spectral_bands)
        
        # Nasal cavity resonance (~1000 Hz peak)
        filter_shapes[0] = torch.exp(-0.5 * ((torch.linspace(0, 1, spectral_bands) - 0.2) / 0.1) ** 2)
        
        # Throat resonance (~500 Hz peak)
        filter_shapes[1] = torch.exp(-0.5 * ((torch.linspace(0, 1, spectral_bands) - 0.1) / 0.1) ** 2)
        
        # Mouth cavity resonance (~2000 Hz peak)
        filter_shapes[2] = torch.exp(-0.5 * ((torch.linspace(0, 1, spectral_bands) - 0.4) / 0.15) ** 2)
        
        # Lip aperture resonance (~3000 Hz peak)
        filter_shapes[3] = torch.exp(-0.5 * ((torch.linspace(0, 1, spectral_bands) - 0.6) / 0.1) ** 2)
        
        # Aspiration noise (high freq, ~5000+ Hz)
        filter_shapes[4] = torch.linspace(0.1, 1, spectral_bands) ** 2
        
        self.register_buffer('filter_shapes', filter_shapes)
        
    def forward(self, condition, f0, audio_length):
        """
        Optimized implementation with consolidated signals and more efficient processing
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
        param_splits = torch.split(all_params, [1, 1, 1, 32, 5], dim=1)
        inhalation = torch.sigmoid(param_splits[0])  # [B, 1, T]
        exhalation = torch.sigmoid(param_splits[1])  # [B, 1, T]
        pressure = torch.sigmoid(param_splits[2])  # [B, 1, T]
        spectral_shape = F.softmax(param_splits[3], dim=1)  # [B, 32, T] - Normalize as distribution
        turbulence_location = F.softmax(param_splits[4], dim=1)  # [B, 5, T] - Normalize as distribution
        
        # Extract features for modulating the harmonic signal (kept at frame rate)
        breath_features = torch.cat([
            pressure,       # Breath pressure affects harmonic intensity
            inhalation,     # Inhalation can affect pitch stability
            exhalation      # Exhalation affects harmonic structure
        ], dim=1)  # [B, 3, T]
        
        # Consolidate control signals for more efficient upsampling
        control_signals = torch.cat([
            inhalation, 
            exhalation, 
            pressure, 
            turbulence_location
        ], dim=1)  # [B, 8, T]
        
        # Generate noise with precalculated spectral shape
        # More efficient frequency-domain approach
        breath_noise = self._generate_colored_noise_fft(batch_size, audio_length, spectral_shape, device)
        
        # Single upsampling operation for all control signals
        control_signals_audio = self._efficient_upsample(control_signals, audio_length)
        
        # Extract individual upsampled signals
        inhalation_audio = control_signals_audio[:, 0:1, :]
        exhalation_audio = control_signals_audio[:, 1:2, :]
        pressure_audio = control_signals_audio[:, 2:3, :]
        turbulence_audio = control_signals_audio[:, 3:8, :]
        
        # Upsample voiced/unvoiced decision
        voiced_audio = self._efficient_upsample(is_voiced, audio_length)
        
        # Apply respiratory dynamics - vectorized operations
        inhale_component = breath_noise * inhalation_audio * (1.0 - voiced_audio)
        exhale_component = breath_noise * exhalation_audio * pressure_audio
        breath_component = inhale_component + exhale_component
        
        # Apply location-based filtering using vectorized operations
        filtered_breath = self._apply_location_filters_vectorized(
            breath_component, 
            turbulence_audio
        )
        
        return filtered_breath, breath_features
    
    def _efficient_upsample(self, tensor, target_len):
        """
        Memory-efficient upsampling with block processing for large tensors
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
        
    def _generate_colored_noise_fft(self, batch_size, audio_length, spectral_shape, device):
        """
        Generate spectrally shaped noise using FFT for efficiency
        """
        # Generate white noise - batch processing to save memory for very long audio
        max_block_length = 60000  # ~2.5 sec at 24kHz
        
        if audio_length <= max_block_length:
            # For shorter audio, process all at once
            white_noise = torch.randn(batch_size, 1, audio_length, device=device)
            shaped_noise = self._apply_spectral_shape_fft(white_noise, spectral_shape)
            return shaped_noise
        else:
            # For very long audio, process in blocks
            result = torch.zeros(batch_size, 1, audio_length, device=device)
            
            # Process in blocks with small overlap for smooth transitions
            overlap = 1000
            for block_start in range(0, audio_length, max_block_length - overlap):
                block_end = min(block_start + max_block_length, audio_length)
                block_length = block_end - block_start
                
                # Generate and shape noise for this block
                block_noise = torch.randn(batch_size, 1, block_length, device=device)
                shaped_block = self._apply_spectral_shape_fft(block_noise, spectral_shape)
                
                # Apply fade in/out for overlapping regions
                if block_start > 0:  # Not the first block - apply fade-in
                    fade_in = torch.linspace(0, 1, overlap, device=device).reshape(1, 1, -1)
                    shaped_block[:, :, :overlap] *= fade_in
                    result[:, :, block_start:block_start+overlap] *= (1 - fade_in)
                
                if block_end < audio_length:  # Not the last block - apply fade-out
                    fade_out = torch.linspace(1, 0, overlap, device=device).reshape(1, 1, -1)
                    shaped_block[:, :, -overlap:] *= fade_out
                
                # Add to result
                result[:, :, block_start:block_end] += shaped_block
                
            return result
            
    def _apply_spectral_shape_fft(self, noise, spectral_shape):
        """
        Apply spectral shaping to noise using FFT
        """
        batch_size, _, audio_length = noise.shape
        device = noise.device
        
        # Average spectral shape over time dimension for efficiency
        avg_spectral_shape = torch.mean(spectral_shape, dim=2)  # [B, 32]
        
        # Convert noise to frequency domain
        noise_flat = noise.squeeze(1)  # [B, audio_length]
        noise_fft = torch.fft.rfft(noise_flat)  # [B, audio_length//2 + 1]
        
        # Map spectral shape bands to FFT frequency bins
        num_bins = noise_fft.shape[1]
        num_bands = avg_spectral_shape.shape[1]
        
        # Create mapping from spectral shape bands to FFT bins
        shaped_spectrum = torch.zeros_like(noise_fft)
        
        # For each spectral band, apply the corresponding gain to the FFT bins
        for b in range(num_bands):
            # Calculate bin range for this band
            bin_start = int(b * num_bins / num_bands)
            bin_end = int((b + 1) * num_bins / num_bands)
            
            # Apply the band's gain to these bins
            band_gain = avg_spectral_shape[:, b].unsqueeze(1).expand(-1, bin_end - bin_start)
            shaped_spectrum[:, bin_start:bin_end] = noise_fft[:, bin_start:bin_end] * band_gain
        
        # Convert back to time domain
        shaped_noise = torch.fft.irfft(shaped_spectrum, n=audio_length)
        
        # Normalize and reshape
        shaped_noise = shaped_noise / torch.std(shaped_noise, dim=1, keepdim=True) * 0.1
        
        return shaped_noise.unsqueeze(1)  # [B, 1, audio_length]
    
    def _apply_location_filters_vectorized(self, breath_signal, turbulence_locations):
        """
        Apply turbulence location filtering with vectorized operations
        Uses precomputed filter shapes for efficiency
        """
        batch_size, _, audio_length = breath_signal.shape
        device = breath_signal.device
        
        # Convert to frequency domain for efficient filtering
        breath_fft = torch.fft.rfft(breath_signal.squeeze(1))  # [B, audio_length//2 + 1]
        
        # Number of frequency bins
        num_bins = breath_fft.shape[1]
        
        # Initialize the shaped spectrum
        shaped_fft = torch.zeros_like(breath_fft)
        
        # For each location, apply its filter shape weighted by turbulence_locations
        for loc in range(5):
            # Get the average weight for this location
            loc_weight = torch.mean(turbulence_locations[:, loc:loc+1, :], dim=2)  # [B, 1]
            
            # Get the filter shape for this location
            filter_shape = self.filter_shapes[loc].to(device)  # [32]
            
            # Interpolate filter shape to match FFT bin count
            filter_bins = F.interpolate(
                filter_shape.reshape(1, 1, -1), 
                size=num_bins, 
                mode='linear', 
                align_corners=False
            ).squeeze()  # [num_bins]
            
            # Apply weighted filter to FFT
            weighted_filter = filter_bins.unsqueeze(0) * loc_weight  # [B, num_bins]
            shaped_fft += breath_fft * weighted_filter
        
        # Convert back to time domain
        filtered_breath = torch.fft.irfft(shaped_fft, n=audio_length)
        
        # Normalize and reshape
        filtered_breath = filtered_breath.unsqueeze(1)  # [B, 1, audio_length]
        
        return filtered_breath