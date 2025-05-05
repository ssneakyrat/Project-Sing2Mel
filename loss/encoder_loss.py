import torch
import torch.nn as nn
import torch.nn.functional as F

class MelSSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for Mel spectrograms.
    Adapted from image-based SSIM to work with spectrograms.
    """
    def __init__(self, window_size=11, sigma=1.5, channels=1):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.register_buffer('window', self._create_window())
        
    def _create_window(self):
        # Create a Gaussian window
        coords = torch.arange(self.window_size, dtype=torch.float)
        coords -= self.window_size // 2
        
        # 1D Gaussian window
        gaussian = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        gaussian /= gaussian.sum()
        
        # Expand to 2D and reshape for conv2d
        window_1d = gaussian.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t())
        window_2d = window_2d.unsqueeze(0).unsqueeze(0)
        
        # Repeat for each channel
        window_2d = window_2d.expand(self.channels, 1, self.window_size, self.window_size)
        
        return window_2d
    
    def forward(self, x, y):
        """
        Calculate SSIM loss between two mel spectrograms.
        Args:
            x, y: Batch of mel spectrograms [batch_size, n_mels, time_steps]
        Returns:
            SSIM loss (1 - SSIM to make it a loss function)
        """
        # Add channel dimension if not already present
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
        
        c1 = 0.01 ** 2  # Constants for stability
        c2 = 0.03 ** 2
        
        # Adjust the window to match the number of channels
        window = self.window.expand(x.size(1), 1, self.window_size, self.window_size)
        
        # Calculate means
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=x.size(1))
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=y.size(1))
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Calculate variances and covariance
        sigma_x_sq = F.conv2d(x ** 2, window, padding=self.window_size // 2, groups=x.size(1)) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, window, padding=self.window_size // 2, groups=y.size(1)) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=x.size(1)) - mu_xy
        
        # SSIM calculation
        ssim_numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        ssim_denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = ssim_numerator / ssim_denominator
        
        # Return 1 - SSIM to convert to a loss (0 = identical, 1 = completely different)
        return 1 - ssim_map.mean()


class EncoderLoss(nn.Module):
    """Combined loss function for mel decoder including L1 loss and SSIM perceptual loss"""
    def __init__(self, mel_loss_weight=0.5, ssim_loss_weight=0.5):
        super().__init__()
        self.mel_loss_weight = mel_loss_weight
        self.ssim_loss_weight = ssim_loss_weight
        
        # Mel loss
        self.mel_criterion = nn.L1Loss()
        
        # SSIM loss
        self.ssim_criterion = MelSSIMLoss()
    
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

        # Mel spectrogram loss (L1)
        mel_loss = self.mel_criterion(predicted_mel_aligned, target_mel_aligned)
        
        # SSIM loss
        ssim_loss = self.ssim_criterion(predicted_mel_aligned, target_mel_aligned)
        
        # Combined loss
        total_loss = (self.mel_loss_weight * mel_loss) + (self.ssim_loss_weight * ssim_loss)
        
        return total_loss, mel_loss, ssim_loss