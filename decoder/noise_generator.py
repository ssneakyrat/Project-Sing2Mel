import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FilteredNoiseGenerator(nn.Module):
    """Generates filtered noise components"""
    def __init__(self, n_fft=1024, hop_length=240, sample_rate=24000):
        super(FilteredNoiseGenerator, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # Filter coefficient predictor network (as frequency-domain filter)
        self.filter_net = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, n_fft // 2 + 1, kernel_size=3, padding=1),
            nn.Softplus()
        )
        
        # Noise amplitude envelope predictor
        self.envelope_net = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 1, kernel_size=3, padding=1),
            nn.Softplus()
        )
        
    def forward(self, condition, audio_length):
        """
        Args:
            condition: Conditioning information [B, C, T]
            audio_length: Target audio length in samples
        Returns:
            filtered_noise: Generated filtered noise signal [B, audio_length]
        """
        batch_size, _, time_steps = condition.shape
        
        # Predict filter coefficients - spectral shape for each frame
        filter_coeffs = self.filter_net(condition)  # [B, n_fft//2+1, T]
        
        # Predict noise amplitude envelope
        envelope = self.envelope_net(condition)  # [B, 1, T]
        
        # Generate white noise
        noise = torch.randn(batch_size, audio_length, device=condition.device)
        
        # Compute STFT of the noise
        noise_stft = torch.stft(
            noise, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=torch.hann_window(self.n_fft, device=condition.device),
            return_complex=True,
            normalized=True
        )  # [B, n_fft//2+1, num_frames]
        
        # Number of frames in the STFT
        num_frames = noise_stft.shape[2]
        
        # Upsample filter coefficients to match STFT frames if needed
        if num_frames != filter_coeffs.shape[2]:
            filter_coeffs = F.interpolate(
                filter_coeffs, 
                size=num_frames, 
                mode='linear', 
                align_corners=False
            )  # [B, n_fft//2+1, num_frames]
        
        # Apply filter in the frequency domain
        filtered_stft = noise_stft * filter_coeffs.to(torch.complex64)
        
        # Convert back to time domain
        filtered_noise = torch.istft(
            filtered_stft, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=torch.hann_window(self.n_fft, device=condition.device),
            length=audio_length,
            normalized=True
        )  # [B, audio_length]
        
        # Upsample and apply envelope
        envelope_upsampled = F.interpolate(
            envelope, 
            size=audio_length, 
            mode='linear', 
            align_corners=False
        ).squeeze(1)  # [B, audio_length]
        
        filtered_noise = filtered_noise * envelope_upsampled
        
        return filtered_noise