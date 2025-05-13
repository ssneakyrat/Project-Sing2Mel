import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    """Lightweight CNN for extracting hierarchical features from mel spectrograms"""
    def __init__(self, n_mels, channels=[16, 32, 64, 128]):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        ))
        
        # Additional layers
        for i in range(len(channels) - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2)
            ))
            
    def forward(self, x):
        """
        Forward pass returning features from all layers
        
        Args:
            x: Input mel spectrogram [B, 1, F, T] where F is n_mels
            
        Returns:
            List of feature maps from each layer
        """
        features = []
        
        # Pass through each layer and collect features
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            
        return features

class FeatureMatchingLoss(nn.Module):
    """Loss comparing features between predicted and target spectrograms"""
    def __init__(self, n_mels, layer_weights=None):
        super().__init__()
        
        # Default layer weights if not provided (emphasize earlier layers)
        if layer_weights is None:
            self.layer_weights = [1.0, 0.75, 0.5, 0.25]
        else:
            self.layer_weights = layer_weights
            
        # Create feature extractor
        self.feature_extractor = FeatureExtractor(n_mels)
        
    def forward(self, predicted_mel, target_mel):
        """
        Compute feature matching loss between predicted and target mels
        
        Args:
            predicted_mel: Predicted log mel spectrogram [B, F, T]
            target_mel: Target log mel spectrogram [B, F, T]
            
        Returns:
            Feature matching loss (weighted sum of L1 distances between feature maps)
        """
        # Add channel dimension [B, F, T] -> [B, 1, F, T]
        pred_mel = predicted_mel.unsqueeze(1)
        target_mel = target_mel.unsqueeze(1)
        
        # Extract features
        pred_features = self.feature_extractor(pred_mel)
        target_features = self.feature_extractor(target_mel)
        
        # Compute loss at each layer
        fm_loss = 0.0
        for i, (p_feat, t_feat) in enumerate(zip(pred_features, target_features)):
            # Calculate L1 distance between feature maps (normalized by size)
            layer_loss = F.l1_loss(p_feat, t_feat) * self.layer_weights[i]
            fm_loss += layer_loss
            
        return fm_loss

class DecoderLoss(nn.Module):
    """Combined loss function for mel decoder including STFT loss and feature matching"""
    def __init__(self, stft_loss_weight=0.5, mel_loss_weight=0.3, feature_match_weight=0.2,
                 stft_n_ffts=[512, 1024, 2048], stft_hop_lengths=[50, 120, 240], 
                 stft_win_lengths=[240, 600, 1200], n_mels=128):
        super().__init__()
        self.stft_loss_weight = stft_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.feature_match_weight = feature_match_weight
        
        # Mel loss
        self.mel_criterion = nn.L1Loss()
        
        # Multi-resolution STFT loss components
        self.stft_n_ffts = stft_n_ffts
        self.stft_hop_lengths = stft_hop_lengths
        self.stft_win_lengths = stft_win_lengths
        
        self.stft_mag_criterion = nn.L1Loss()
        
        # Feature matching loss
        self.feature_matching = FeatureMatchingLoss(n_mels)
        
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
        
        # Feature matching loss
        feature_match_loss = self.feature_matching(predicted_mel_aligned, target_mel_aligned)
        
        # Combined loss
        total_loss = (self.mel_loss_weight * mel_loss + 
                      self.stft_loss_weight * stft_loss +
                      self.feature_match_weight * feature_match_loss)
        
        return total_loss, mel_loss, stft_loss, predicted_mel, feature_match_loss