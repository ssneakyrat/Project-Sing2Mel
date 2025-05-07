import torch
import torch.nn as nn
import torch.nn.functional as F


class MelLoss(nn.Module):
    """
    Loss function for mel spectrogram prediction
    Combines L1, L2, and optional perceptual losses
    """
    
    def __init__(self, 
                 l1_weight=0.5,
                 l2_weight=0.5,
                 duration_weight=0.1,
                 use_ssim=False,
                 ssim_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.duration_weight = duration_weight
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        
        # MSE Loss
        self.mse_loss = nn.MSELoss()
        
        # L1 Loss
        self.l1_loss = nn.L1Loss()
    
    def forward(self, 
                predicted_mel, 
                target_mel, 
                predicted_durations=None, 
                target_durations=None):
        """
        Args:
            predicted_mel: [B, T, M] Predicted mel spectrogram
            target_mel: [B, T, M] Target mel spectrogram
            predicted_durations: [B, T_ph] Predicted phoneme durations (optional)
            target_durations: [B, T_ph] Target phoneme durations (optional)
            
        Returns:
            total_loss: Combined loss value
            losses_dict: Dictionary with individual loss components
        """
        # Compute L1 loss (absolute error)
        l1 = self.l1_loss(predicted_mel, target_mel)
        
        # Compute L2 loss (MSE)
        l2 = self.mse_loss(predicted_mel, target_mel)
        
        # Initialize losses dictionary
        losses = {
            'l1_loss': l1.item(),
            'l2_loss': l2.item()
        }
        
        # Combine L1 and L2 losses
        mel_loss = self.l1_weight * l1 + self.l2_weight * l2
        losses['mel_loss'] = mel_loss.item()
        
        # Add SSIM (Structural Similarity Index) loss if enabled
        if self.use_ssim:
            ssim_loss = self.compute_ssim_loss(predicted_mel, target_mel)
            mel_loss = mel_loss + self.ssim_weight * ssim_loss
            losses['ssim_loss'] = ssim_loss.item()
        
        # Add duration loss if durations are provided
        dur_loss = 0
        if predicted_durations is not None and target_durations is not None:
            dur_loss = self.compute_duration_loss(predicted_durations, target_durations)
            losses['duration_loss'] = dur_loss.item()
        
        # Compute final loss
        total_loss = mel_loss + self.duration_weight * dur_loss
        losses['total_loss'] = total_loss.item()
        
        return total_loss, losses
    
    def compute_duration_loss(self, predicted_durations, target_durations):
        """
        Compute duration prediction loss
        Args:
            predicted_durations: [B, T] Predicted durations
            target_durations: [B, T] Target durations
            
        Returns:
            duration_loss: Loss value for duration prediction
        """
        # Use L1 loss for durations (mean absolute error)
        return self.l1_loss(predicted_durations, target_durations)
    
    def compute_ssim_loss(self, predicted_mel, target_mel, window_size=11):
        """
        Compute SSIM loss between mel spectrograms
        Args:
            predicted_mel: [B, T, M] Predicted mel spectrogram
            target_mel: [B, T, M] Target mel spectrogram
            window_size: Size of the SSIM window
            
        Returns:
            ssim_loss: 1 - SSIM value (lower is better)
        """
        # Ensure inputs are the right shape for SSIM computation
        # SSIM expects [B, C, H, W] format
        predicted = predicted_mel.permute(0, 2, 1).unsqueeze(1)  # [B, 1, M, T]
        target = target_mel.permute(0, 2, 1).unsqueeze(1)  # [B, 1, M, T]
        
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create a 1D Gaussian kernel for the window
        kernel_size = window_size
        sigma = 1.5
        
        # Create a 1D Gaussian kernel
        kernel_x = torch.arange(kernel_size, device=predicted.device) - (kernel_size - 1) / 2
        kernel_x = torch.exp(-0.5 * kernel_x ** 2 / sigma ** 2)
        kernel_x = kernel_x / kernel_x.sum()
        
        # Convert to 2D kernel
        kernel = kernel_x.view(1, 1, kernel_size, 1) * kernel_x.view(1, 1, 1, kernel_size)
        
        # Expand to match input channels
        kernel = kernel.expand(1, 1, kernel_size, kernel_size).contiguous()
        
        # Compute means using convolution with the Gaussian kernel
        mu_x = F.conv2d(predicted, kernel, padding=kernel_size//2, groups=1)
        mu_y = F.conv2d(target, kernel, padding=kernel_size//2, groups=1)
        
        # Compute squared means
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Compute variances and covariance
        sigma_x_sq = F.conv2d(predicted ** 2, kernel, padding=kernel_size//2, groups=1) - mu_x_sq
        sigma_y_sq = F.conv2d(target ** 2, kernel, padding=kernel_size//2, groups=1) - mu_y_sq
        sigma_xy = F.conv2d(predicted * target, kernel, padding=kernel_size//2, groups=1) - mu_xy
        
        # SSIM formula
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        # 1 - SSIM for loss (SSIM is 1 when images are identical)
        return 1 - ssim_map.mean()


class AdversarialLoss(nn.Module):
    """
    Optional adversarial loss for improving spectrogram quality
    Based on GAN training approach
    """
    
    def __init__(self, discriminator_weight=0.1):
        super().__init__()
        self.discriminator_weight = discriminator_weight
        
        # Binary cross entropy loss for adversarial training
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def generator_loss(self, discriminator_outputs):
        """
        Generator (spectrogram predictor) loss
        Args:
            discriminator_outputs: Discriminator outputs for generated spectrograms
            
        Returns:
            generator_loss: Loss value to train generator
        """
        # Create targets (all ones since generator wants to fool discriminator)
        target = torch.ones_like(discriminator_outputs)
        
        # Compute generator loss
        return self.bce_loss(discriminator_outputs, target)
    
    def discriminator_loss(self, real_outputs, fake_outputs):
        """
        Discriminator loss
        Args:
            real_outputs: Discriminator outputs for real spectrograms
            fake_outputs: Discriminator outputs for generated spectrograms
            
        Returns:
            discriminator_loss: Loss value to train discriminator
        """
        # Create targets (ones for real, zeros for fake)
        real_target = torch.ones_like(real_outputs)
        fake_target = torch.zeros_like(fake_outputs)
        
        # Compute losses
        real_loss = self.bce_loss(real_outputs, real_target)
        fake_loss = self.bce_loss(fake_outputs, fake_target)
        
        # Combine
        return real_loss + fake_loss


class MelSynthLoss(nn.Module):
    """
    Singing Voice Synthesis loss combining mel reconstruction and optional
    adversarial losses
    """
    
    def __init__(self, 
                 l1_weight=0.5,
                 l2_weight=0.5, 
                 duration_weight=0.1,
                 use_adversarial=False,
                 adversarial_weight=0.1,
                 use_ssim=False,
                 ssim_weight=0.1):
        super().__init__()
        
        # Main mel spectrogram reconstruction loss
        self.mel_loss = MelLoss(
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            duration_weight=duration_weight,
            use_ssim=use_ssim,
            ssim_weight=ssim_weight
        )
        
        # Optional adversarial loss
        self.use_adversarial = use_adversarial
        if use_adversarial:
            self.adversarial_loss = AdversarialLoss()
            self.adversarial_weight = adversarial_weight
    
    def forward(self, 
                predicted_mel, 
                target_mel, 
                predicted_durations=None, 
                target_durations=None,
                discriminator_outputs=None):
        """
        Args:
            predicted_mel: [B, T, M] Predicted mel spectrogram
            target_mel: [B, T, M] Target mel spectrogram
            predicted_durations: [B, T_ph] Predicted phoneme durations (optional)
            target_durations: [B, T_ph] Target phoneme durations (optional)
            discriminator_outputs: Discriminator outputs for adversarial training (optional)
            
        Returns:
            total_loss: Combined loss value
            losses_dict: Dictionary with individual loss components
        """
        # Get mel reconstruction loss
        mel_loss, losses_dict = self.mel_loss(
            predicted_mel, target_mel, predicted_durations, target_durations
        )
        
        # Initialize final loss with mel loss
        total_loss = mel_loss
        
        # Add adversarial loss if enabled
        if self.use_adversarial and discriminator_outputs is not None:
            gen_loss = self.adversarial_loss.generator_loss(discriminator_outputs)
            total_loss = total_loss + self.adversarial_weight * gen_loss
            losses_dict['adversarial_loss'] = gen_loss.item()
        
        # Update total loss in dictionary
        losses_dict['total_loss'] = total_loss.item()
        
        return total_loss, losses_dict