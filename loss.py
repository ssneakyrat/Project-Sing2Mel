import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

class DecoderLoss(nn.Module):
    """Combined loss function for mel decoder including STFT loss"""
    def __init__(self, stft_loss_weight=0.5, mel_loss_weight=0.5, 
                 stft_n_ffts=[512, 1024, 2048], stft_hop_lengths=[50, 120, 240], 
                 stft_win_lengths=[240, 600, 1200]):
        super().__init__()
        self.stft_loss_weight = stft_loss_weight
        self.mel_loss_weight = mel_loss_weight
        
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
    
    def forward(self, predicted_audio, target_audio, mel_transform):
        """Compute combined loss"""
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
        
        # Combined loss
        total_loss = (self.mel_loss_weight * mel_loss + 
                     self.stft_loss_weight * stft_loss)
        
        return total_loss, mel_loss, stft_loss, predicted_mel

class AdversarialDecoderLoss(DecoderLoss):
    """Extended loss function with adversarial components using WGAN approach"""
    def __init__(self, 
                 stft_loss_weight=0.5,
                 mel_loss_weight=0.3, 
                 adv_loss_weight=0.2,
                 feature_matching_weight=0.1,
                 stft_n_ffts=[512, 1024, 2048], 
                 stft_hop_lengths=[50, 120, 240], 
                 stft_win_lengths=[240, 600, 1200],
                 progressive_steps=50000,
                 detect_anomaly_threshold=5.0):
        
        # Initialize parent class
        super().__init__(stft_loss_weight, mel_loss_weight, stft_n_ffts, stft_hop_lengths, stft_win_lengths)
        
        # Adversarial loss weights
        self.adv_loss_weight = adv_loss_weight
        self.feature_matching_weight = feature_matching_weight
        
        # Progressive training parameters
        self.progressive_steps = progressive_steps
        self.current_step = 0
        
        # Loss spike detection
        self.detect_anomaly_threshold = detect_anomaly_threshold
        self.prev_gen_loss = None
        
        # Feature matching loss
        self.feature_matching_criterion = nn.L1Loss()
        
        # Initialize loss tracking
        self.loss_history = {
            'gen_loss': [],
            'disc_loss': [],
            'adv_loss': [],
            'mel_loss': [],
            'stft_loss': [],
            'feature_matching_loss': [],
            'total_loss': [],
            'adv_weight': []
        }
        
    def get_adv_weight(self):
        """Get current weight for adversarial loss based on progressive training"""
        if self.progressive_steps <= 0:
            return self.adv_loss_weight
            
        # Linear ramp up for adversarial weight
        progress = min(1.0, self.current_step / self.progressive_steps)
        return progress * self.adv_loss_weight
        
    def compute_feature_matching_loss(self, disc_features_real, disc_features_fake):
        """Compute feature matching loss between real and fake features"""
        feature_matching_loss = 0.0
        
        # Iterate through scales
        for scale_real, scale_fake in zip(disc_features_real, disc_features_fake):
            # Iterate through layers
            for feat_real, feat_fake in zip(scale_real, scale_fake):
                ft = feat_real.detach()
                #print(ft.shape)
                #print(feat_fake.shape)
                feat_fake = feat_fake[:, :, :ft.shape[2]]
                fmc = self.feature_matching_criterion(feat_fake, ft)
                
                feature_matching_loss += fmc
                
        # Normalize by number of feature maps
        feature_matching_loss /= (len(disc_features_real) * len(disc_features_real[0]))
        return feature_matching_loss
        
    def discriminator_loss(self, disc_outputs_real, disc_outputs_fake):
        """WGAN discriminator loss"""
        loss_real = 0.0
        loss_fake = 0.0
        
        for real, fake in zip(disc_outputs_real, disc_outputs_fake):
            # WGAN formulation (critic loss)
            loss_real += -torch.mean(real)
            loss_fake += torch.mean(fake)
            
        total_loss = (loss_real + loss_fake) / len(disc_outputs_real)
        return total_loss
        
    def generator_adversarial_loss(self, disc_outputs_fake):
        """WGAN generator adversarial loss"""
        loss = 0.0
        for fake in disc_outputs_fake:
            # WGAN formulation (generator wants critic to give high scores)
            loss += -torch.mean(fake)
            
        return loss / len(disc_outputs_fake)
        
    def detect_training_instability(self, gen_loss):
        """Detect if generator loss is unstable"""
        if self.prev_gen_loss is not None:
            # Check for significant spikes in loss
            relative_change = abs(gen_loss - self.prev_gen_loss) / (abs(self.prev_gen_loss) + 1e-8)
            if relative_change > self.detect_anomaly_threshold:
                return True
                
        self.prev_gen_loss = gen_loss
        return False
        
    def forward_generator(self, predicted_audio, target_audio, mel_transform, 
                         disc_outputs_fake, disc_features_real, disc_features_fake):
        """Full generator loss including adversarial components"""
        # Get reconstruction losses from parent class
        base_loss, mel_loss, stft_loss, predicted_mel = super().forward(
            predicted_audio, target_audio, mel_transform
        )
        
        # Compute adversarial loss
        adv_loss = self.generator_adversarial_loss(disc_outputs_fake)
        
        # Compute feature matching loss
        feature_matching_loss = self.compute_feature_matching_loss(
            disc_features_real, disc_features_fake
        )
        
        # Get current adversarial weight
        adv_weight = self.get_adv_weight()
        
        # Combined loss with weighted components
        total_loss = (
            base_loss + 
            adv_weight * adv_loss + 
            self.feature_matching_weight * feature_matching_loss
        )
        
        # Increment step counter for progressive training
        self.current_step += 1
        
        # Check for training instability
        is_unstable = self.detect_training_instability(adv_loss.item())
        
        # Track losses for plotting
        self.loss_history['gen_loss'].append(adv_loss.item())
        self.loss_history['mel_loss'].append(mel_loss.item())
        self.loss_history['stft_loss'].append(stft_loss.item())
        self.loss_history['feature_matching_loss'].append(feature_matching_loss.item())
        self.loss_history['adv_loss'].append(adv_loss.item())
        self.loss_history['total_loss'].append(total_loss.item())
        self.loss_history['adv_weight'].append(adv_weight)
        
        return total_loss, mel_loss, stft_loss, adv_loss, feature_matching_loss, predicted_mel, is_unstable
    
    def plot_losses(self, save_dir='visuals/decoder/loss', filename='loss_plot'):
        """Plot and save loss history"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subplots for different loss components
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Plot reconstruction losses
        axes[0].plot(self.loss_history['mel_loss'], label='Mel Loss')
        axes[0].plot(self.loss_history['stft_loss'], label='STFT Loss')
        axes[0].plot(self.loss_history['total_loss'], label='Total Loss')
        axes[0].set_title('Reconstruction Losses')
        axes[0].set_ylabel('Loss Value')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot adversarial losses
        axes[1].plot(self.loss_history['gen_loss'], label='Generator Loss')
        if 'disc_loss' in self.loss_history and len(self.loss_history['disc_loss']) > 0:
            axes[1].plot(self.loss_history['disc_loss'], label='Discriminator Loss')
        axes[1].plot(self.loss_history['adv_loss'], label='Adversarial Loss')
        axes[1].plot(self.loss_history['feature_matching_loss'], label='Feature Matching Loss')
        axes[1].set_title('Adversarial Losses')
        axes[1].set_ylabel('Loss Value')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot adversarial weight
        axes[2].plot(self.loss_history['adv_weight'], label='Adversarial Weight')
        axes[2].set_title('Adversarial Weight Progression')
        axes[2].set_xlabel('Training Steps')
        axes[2].set_ylabel('Weight Value')
        axes[2].legend()
        axes[2].grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{filename}.png")
        plt.close()