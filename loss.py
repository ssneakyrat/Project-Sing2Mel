import torch
import torch.nn as nn
import torch.nn.functional as F

class DurationAwareLoss(nn.Module):
    """Combined loss for mel spectrogram and duration prediction"""
    def __init__(self, duration_weight=0.1, l1_weight=1.0):
        super().__init__()
        self.duration_weight = duration_weight
        self.l1_weight = l1_weight
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predicted_mels, target_mels, 
                predicted_durations, target_durations):
        """
        Args:
            predicted_mels: [batch, time, n_mels]
            target_mels: [batch, time, n_mels]
            predicted_durations: [batch, phoneme_seq_len]
            target_durations: [batch, phoneme_seq_len]
        """
        # Mel reconstruction loss
        mel_loss = self.l1_loss(predicted_mels, target_mels)
        
        # Fix shape mismatch - squeeze the extra dimension if present
        if predicted_durations.dim() == 3 and predicted_durations.shape[-1] == 1:
            predicted_durations = predicted_durations.squeeze(-1)

        # Duration prediction loss
        duration_loss = F.mse_loss(predicted_durations, target_durations)
        
        # Log-scale duration loss for better gradient flow
        log_duration_loss = F.mse_loss(
            torch.log(predicted_durations + 1e-8),
            torch.log(target_durations + 1e-8)
        )
        
        total_loss = self.l1_weight * mel_loss + \
                     self.duration_weight * (duration_loss + log_duration_loss)
        
        return total_loss, {
            'mel_loss': mel_loss.item(),
            'duration_loss': duration_loss.item(),
            'log_duration_loss': log_duration_loss.item(),
            'total_loss': total_loss.item()
        }