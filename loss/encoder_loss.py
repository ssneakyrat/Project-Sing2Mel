import torch
import torch.nn as nn

class EncoderLoss(nn.Module):
    """Combined loss function for mel decoder including STFT loss"""
    def __init__(self, mel_loss_weight=0.5):
        super().__init__()
        self.mel_loss_weight = mel_loss_weight
        
        # Mel loss
        self.mel_criterion = nn.L1Loss()
    
    def forward(self, predicted_mel, target_mel):
        """Compute combined loss"""
        
        if predicted_mel.shape[2] > target_mel.shape[2]:
            predicted_mel_aligned = predicted_mel[:, :, :target_mel.shape[2]]
            target_mel_aligned = target_mel
        elif target_mel.shape[2] > predicted_mel.shape[2]:
            target_mel_aligned = target_mel[:, :, :predicted_mel.shape[2]]
            predicted_mel_aligned = predicted_mel
        else:
            # Shapes are already the same
            predicted_mel_aligned = predicted_mel
            target_mel_aligned = target_mel

        # Mel spectrogram loss
        mel_loss = self.mel_criterion(predicted_mel_aligned, target_mel_aligned)
        
        # Combined loss
        total_loss = (self.mel_loss_weight * mel_loss)
        
        return total_loss, mel_loss