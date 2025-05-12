import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvGANLoss(nn.Module):
    """Adversarial loss with hinge loss formulation"""
    def __init__(self):
        super().__init__()
        
    def generator_loss(self, disc_outputs):
        """Generator loss tries to make discriminator predict 1 (real)"""
        loss = 0
        
        # Process waveform discriminator outputs (multiple scales)
        if 'wave' in disc_outputs:
            for output in disc_outputs['wave']:
                loss += -torch.mean(output)
                
        # Process spectrogram discriminator output
        if 'spec' in disc_outputs:
            loss += -torch.mean(disc_outputs['spec'])
            
        return loss
    
    def discriminator_loss(self, real_outputs, fake_outputs):
        """Discriminator loss with hinge formulation"""
        d_loss = 0
        
        # Process waveform discriminator outputs (multiple scales)
        if 'wave' in real_outputs and 'wave' in fake_outputs:
            for real_output, fake_output in zip(real_outputs['wave'], fake_outputs['wave']):
                d_loss_real = torch.mean(F.relu(1.0 - real_output))
                d_loss_fake = torch.mean(F.relu(1.0 + fake_output))
                d_loss += d_loss_real + d_loss_fake
                
        # Process spectrogram discriminator output
        if 'spec' in real_outputs and 'spec' in fake_outputs:
            d_loss_real = torch.mean(F.relu(1.0 - real_outputs['spec']))
            d_loss_fake = torch.mean(F.relu(1.0 + fake_outputs['spec']))
            d_loss += d_loss_real + d_loss_fake
            
        return d_loss
    
    def feature_matching_loss(self, real_features, fake_features):
        """Feature matching loss for improved generator training"""
        loss = 0
        
        # Process waveform discriminator features
        if 'wave' in real_features and 'wave' in fake_features:
            for real_feat, fake_feat in zip(real_features['wave'], fake_features['wave']):
                loss += F.l1_loss(fake_feat, real_feat.detach())
                
        # Process spectrogram discriminator features
        if 'spec' in real_features and 'spec' in fake_features:
            for real_feat, fake_feat in zip(real_features['spec'], fake_features['spec']):
                loss += F.l1_loss(fake_feat, real_feat.detach())
                
        return loss

class DecoderLoss(nn.Module):
    """Combined loss function for mel decoder including STFT loss and adversarial loss"""
    def __init__(self, stft_loss_weight=0.5, mel_loss_weight=0.5, 
                 adv_loss_weight=0.1, feature_matching_weight=10.0,
                 stft_n_ffts=[512, 1024, 2048], stft_hop_lengths=[50, 120, 240], 
                 stft_win_lengths=[240, 600, 1200]):
        super().__init__()
        self.stft_loss_weight = stft_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.feature_matching_weight = feature_matching_weight
        
        # Mel loss
        self.mel_criterion = nn.L1Loss()
        
        # Multi-resolution STFT loss components
        self.stft_n_ffts = stft_n_ffts
        self.stft_hop_lengths = stft_hop_lengths
        self.stft_win_lengths = stft_win_lengths
        
        self.stft_mag_criterion = nn.L1Loss()
        
        # Adversarial loss
        self.adv_criterion = AdvGANLoss()
        
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
    
    def forward(self, predicted_audio, target_audio, mel_transform, 
                disc_outputs=None, real_disc_features=None, fake_disc_features=None,
                train_generator=True):
        """
        Compute combined loss
        
        Args:
            predicted_audio: Generated audio from model
            target_audio: Ground truth audio
            mel_transform: Mel spectrogram transform
            disc_outputs: Discriminator outputs for adversarial loss (optional)
            real_disc_features: Discriminator features from real data (optional)
            fake_disc_features: Discriminator features from fake data (optional)
            train_generator: Whether this is a generator update (True) or discriminator update (False)
            
        Returns:
            Loss values depending on training mode
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
        
        # For generator training
        if train_generator:
            adv_loss = 0
            feature_matching_loss = 0
            
            # Add adversarial loss if discriminator outputs are provided
            if disc_outputs is not None:
                adv_loss = self.adv_criterion.generator_loss(disc_outputs)
                
            # Add feature matching loss if discriminator features are provided
            if real_disc_features is not None and fake_disc_features is not None:
                feature_matching_loss = self.adv_criterion.feature_matching_loss(
                    real_disc_features, fake_disc_features)
                
            # Combined loss for generator
            total_loss = (self.mel_loss_weight * mel_loss + 
                         self.stft_loss_weight * stft_loss + 
                         self.adv_loss_weight * adv_loss + 
                         self.feature_matching_weight * feature_matching_loss)
            
            return total_loss, mel_loss, stft_loss, adv_loss, feature_matching_loss, predicted_mel
        
        # For discriminator training, just return the reconstruction losses
        else:
            return mel_loss, stft_loss, predicted_mel