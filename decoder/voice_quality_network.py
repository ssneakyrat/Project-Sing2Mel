import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VoiceQualityNetwork(nn.Module):
    """
    Network to model voice quality characteristics like jitter, shimmer and breathiness.
    These micro-variations add naturalness to synthesized vocal sounds.
    """
    def __init__(self, hidden_dim=128):
        super(VoiceQualityNetwork, self).__init__()
        
        # Network to predict jitter, shimmer, and breathiness
        self.quality_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 3, kernel_size=1),  # Jitter, shimmer, breathiness
            nn.Sigmoid()  # Normalized parameters
        )
        
    def forward(self, condition, audio_length):
        """
        Args:
            condition: Conditioning information [B, C, T]
            audio_length: Length of the audio to generate
        Returns:
            quality_params: Voice quality parameters [B, 3, audio_length]
                - jitter: Phase variation [B, 1, audio_length]
                - shimmer: Amplitude variation [B, 1, audio_length]
                - breathiness: Noise level [B, 1, audio_length]
        """
        batch_size, channels, time_steps = condition.shape
        device = condition.device
        
        # Predict quality parameters at frame rate
        quality_params_frames = self.quality_predictor(condition)  # [B, 3, T]
        
        # Upsample to audio rate
        quality_params = F.interpolate(
            quality_params_frames,
            size=audio_length,
            mode='linear',
            align_corners=False
        )  # [B, 3, audio_length]
        
        # Scale parameters to appropriate ranges
        jitter = quality_params[:, 0:1, :] * 0.05  # Small phase variations (0-0.05)
        shimmer = quality_params[:, 1:2, :] * 0.15  # Amplitude variations (0-0.15)
        breathiness = quality_params[:, 2:3, :] * 0.3  # Breathiness level (0-0.3)
        
        return jitter, shimmer, breathiness
        
    def apply_voice_qualities(self, harmonic_signal, phase, jitter, shimmer, breathiness):
        """
        Apply voice quality effects to the harmonic signal
        Args:
            harmonic_signal: Base harmonic signal [B, audio_length]
            phase: Phase information [B, audio_length]
            jitter: Phase variation [B, 1, audio_length]
            shimmer: Amplitude variation [B, 1, audio_length]
            breathiness: Noise level [B, 1, audio_length]
        Returns:
            enhanced_signal: Signal with added voice qualities [B, audio_length]
        """
        batch_size, audio_length = harmonic_signal.shape
        device = harmonic_signal.device
        
        # Generate random phase variations for jitter
        jitter_noise = torch.randn(batch_size, audio_length, device=device) * jitter.squeeze(1)
        
        # Generate random amplitude variations for shimmer
        shimmer_envelope = 1.0 + (torch.randn(batch_size, audio_length, device=device) * shimmer.squeeze(1))
        
        # Generate breath noise
        breath_noise = torch.randn(batch_size, audio_length, device=device) * breathiness.squeeze(1)
        
        # Apply shimmer to the harmonic signal
        shimmer_signal = harmonic_signal * shimmer_envelope
        
        # Add breath noise
        enhanced_signal = shimmer_signal + breath_noise
        
        return enhanced_signal