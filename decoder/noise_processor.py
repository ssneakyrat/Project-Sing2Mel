import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseProcessor(nn.Module):
    """
    Generates and shapes noise based on conditioning information.
    This creates noise components that can be mixed with harmonic signals
    to better model breathy or noisy vocal characteristics.
    """
    def __init__(self, input_channels=128):
        super(NoiseProcessor, self).__init__()
        
        # Network to predict noise parameters
        self.noise_net = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Control the noise level (0-1)
        )
        
        # Spectral shaping network for colored noise
        self.spectral_shape_net = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.Softplus()  # Ensure positive filter coefficients
        )
        
    def forward(self, condition, audio_length):
        """
        Generate and shape noise based on conditioning information.
        
        Args:
            condition: Conditioning features [B, input_channels, T]
            audio_length: Length of the target audio
            
        Returns:
            shaped_noise: Conditioned noise signal [B, 1, audio_length]
        """
        # Generate noise amplitude envelope
        noise_level = self.noise_net(condition)  # [B, 1, T]
        
        # Generate spectral shaping coefficients
        spectral_shape = self.spectral_shape_net(condition)  # [B, 16, T]
        
        # Upsample the conditioning signals to audio length
        noise_level_upsampled = F.interpolate(
            noise_level, 
            size=audio_length, 
            mode='linear', 
            align_corners=False
        )  # [B, 1, audio_length]
        
        spectral_shape_upsampled = F.interpolate(
            spectral_shape, 
            size=audio_length, 
            mode='linear', 
            align_corners=False
        )  # [B, 16, audio_length]
        
        # Generate white noise
        batch_size = condition.shape[0]
        white_noise = torch.randn(batch_size, 1, audio_length, device=condition.device)
        
        # Apply spectral shaping using frequency-domain filtering
        # 1. Convert to frequency domain
        noise_fft = torch.fft.rfft(white_noise, dim=2)  # [B, 1, fft_size//2 + 1]
        fft_size = noise_fft.shape[2]
        
        # 2. Interpolate spectral shape to match FFT size
        filter_shape = F.interpolate(
            spectral_shape_upsampled, 
            size=fft_size, 
            mode='linear', 
            align_corners=False
        )  # [B, 16, fft_size]
        
        # 3. Combine filter coefficients into single filter response
        # Sum weighted basis filters (this is a simplified approach)
        filter_response = torch.sum(filter_shape, dim=1, keepdim=True)  # [B, 1, fft_size]
        
        # 4. Apply filter in frequency domain
        filtered_fft = noise_fft * filter_response
        
        # 5. Convert back to time domain
        colored_noise = torch.fft.irfft(filtered_fft, dim=2, n=audio_length)
        
        # 6. Apply the learned noise level amplitude envelope
        shaped_noise = colored_noise * noise_level_upsampled
        
        # Normalize to prevent excessive amplitude
        shaped_noise = shaped_noise / (torch.std(shaped_noise, dim=2, keepdim=True) + 1e-8)
        
        return shaped_noise