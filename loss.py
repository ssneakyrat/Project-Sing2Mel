import torch
import torch.nn as nn

class DecoderLoss(nn.Module):
    """Combined loss function for mel decoder including STFT loss and formant regularization"""
    def __init__(self, stft_loss_weight=0.5, mel_loss_weight=0.5, 
                 formant_reg_weight=0.1,
                 stft_n_ffts=[512, 1024, 2048], stft_hop_lengths=[50, 120, 240], 
                 stft_win_lengths=[240, 600, 1200]):
        super().__init__()
        self.stft_loss_weight = stft_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.formant_reg_weight = formant_reg_weight
        
        # Mel loss
        self.mel_criterion = nn.L1Loss()
        
        # Multi-resolution STFT loss components
        self.stft_n_ffts = stft_n_ffts
        self.stft_hop_lengths = stft_hop_lengths
        self.stft_win_lengths = stft_win_lengths
        
        self.stft_mag_criterion = nn.L1Loss()
        
    def compute_stft_loss(self, predicted_audio, target_audio):
        """Compute multi-resolution STFT loss"""
        stft_loss = 0.0
        
        for n_fft, hop_length, win_length in zip(self.stft_n_ffts, self.stft_hop_lengths, self.stft_win_lengths):
            # Compute STFT for predicted audio
            pred_stft = torch.stft(
                predicted_audio, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length,
                window=torch.hann_window(win_length).to(predicted_audio.device),
                return_complex=True
            )
            
            # Compute STFT for target audio
            target_stft = torch.stft(
                target_audio, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length,
                window=torch.hann_window(win_length).to(target_audio.device),
                return_complex=True
            )
            
            # Compute magnitude spectrogram
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            if pred_mag.shape[2] > target_mag.shape[2]:
                predicted_mel_aligned = pred_mag[:, :, :target_mag.shape[2]]
                target_mel_aligned = target_mag
            elif target_mag.shape[2] > pred_mag.shape[2]:
                target_mel_aligned = target_mag[:, :, :pred_mag.shape[2]]
                predicted_mel_aligned = pred_mag
            else:
                # Shapes are already the same
                predicted_mel_aligned = pred_mag
                target_mel_aligned = target_mag

            # Magnitude loss
            mag_loss = self.stft_mag_criterion(predicted_mel_aligned, target_mel_aligned)
            
            # Spectral convergence loss
            sc_loss = torch.norm(target_mel_aligned - predicted_mel_aligned, p='fro') / torch.norm(target_mel_aligned, p='fro')
            
            stft_loss += mag_loss + sc_loss
        
        # Average over all resolutions
        stft_loss /= len(self.stft_n_ffts)
        return stft_loss
    
    def forward(self, predicted_audio, target_audio, mel_transform, model_outputs=None):
        """
        Compute combined loss
        
        Args:
            predicted_audio: Synthesized audio from model
            target_audio: Ground truth audio
            mel_transform: Function to convert audio to mel spectrogram
            model_outputs: Dictionary containing model outputs, including regularization loss
        """
        # Compute mel spectrogram from predicted and target audio
        # Add small epsilon to avoid log(0)
        predicted_mel = torch.log(mel_transform(predicted_audio) + 1e-7)
        target_mel = torch.log(mel_transform(target_audio) + 1e-7)
        
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
        
        # Multi-resolution STFT loss
        stft_loss = self.compute_stft_loss(predicted_audio, target_audio)
        
        # Formant regularization loss (if available)
        formant_reg_loss = 0.0
        if model_outputs is not None and 'regularization_loss' in model_outputs:
            formant_reg_loss = model_outputs['regularization_loss']
        
        # Combined loss
        total_loss = (self.mel_loss_weight * mel_loss + 
                      self.stft_loss_weight * stft_loss +
                      self.formant_reg_weight * formant_reg_loss)
        
        # Create loss dictionary for detailed monitoring
        loss_dict = {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'stft_loss': stft_loss,
            'formant_reg_loss': formant_reg_loss
        }
        
        return total_loss, loss_dict, predicted_mel
        
        
class RegisterAwareLoss(nn.Module):
    """
    Loss function that encourages proper separation between registers
    and smooth transitions between them.
    """
    def __init__(self, sharpness_weight=0.1, smoothness_weight=0.5):
        super().__init__()
        self.sharpness_weight = sharpness_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(self, register_weights, f0):
        """
        Compute register loss.
        
        Args:
            register_weights: Predicted register weights [B, T, 3]
            f0: Fundamental frequency [B, T]
            
        Returns:
            loss: Register loss
        """
        batch_size, seq_len = f0.shape
        
        # Loss to encourage sharpness in classification
        # When we're clearly in chest or head voice, we want clear classification
        entropy = -torch.sum(register_weights * torch.log(register_weights + 1e-7), dim=-1)  # [B, T]
        sharpness_loss = entropy.mean()
        
        # Loss to encourage smooth transitions
        # Register weights should change smoothly with f0
        register_weights_shifted = register_weights[:, 1:, :]  # [B, T-1, 3]
        register_weights_prev = register_weights[:, :-1, :]    # [B, T-1, 3]
        
        # L1 difference between consecutive register weights
        diff = torch.abs(register_weights_shifted - register_weights_prev)  # [B, T-1, 3]
        
        # But we want to allow changes when f0 is changing
        f0_shifted = f0[:, 1:]  # [B, T-1]
        f0_prev = f0[:, :-1]    # [B, T-1]
        f0_diff = torch.abs(f0_shifted - f0_prev).unsqueeze(-1)  # [B, T-1, 1]
        
        # Scale diff by f0_diff (allow bigger changes when f0 changes more)
        normalized_diff = diff / (f0_diff + 1.0)  # Add 1.0 to avoid division by zero
        
        smoothness_loss = normalized_diff.mean()
        
        # Combined loss
        loss = self.sharpness_weight * sharpness_loss + self.smoothness_weight * smoothness_loss
        
        return loss