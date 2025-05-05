import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Noise conditioner network with F0 conditioning
class NoiseGenerator(nn.Module):
    def __init__(self, input_dim):
        super(NoiseGenerator, self).__init__()
        
        # Feature processing network
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # F0 processing network - takes mean, min, max, std of F0
        self.f0_network = nn.Sequential(
            nn.Linear(4, 32),  # 4 features: mean, min, max, std
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Combined network for final output
        self.combined_network = nn.Sequential(
            nn.Linear(128, 64),  # 64 from feature_network + 64 from f0_network
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1 to control noise magnitude
        )
    
    def forward(self, conditioning_features, f0):
        """
        Args:
            conditioning_features: [batch_size, feature_dim]
            f0: [batch_size, seq_length] - Fundamental frequency trajectory
        """
        # Process conditioning features
        features = self.feature_network(conditioning_features)
        
        # Extract f0 statistics
        # Handle mask for unvoiced frames (f0 <= 0)
        voiced_mask = (f0 > 0).float()
        
        # Extract statistics only from voiced frames
        # Add small epsilon to avoid division by zero
        voiced_frames = f0 * voiced_mask
        epsilon = 1e-7
        
        # Mean of voiced frames
        mean_f0 = torch.sum(voiced_frames, dim=1) / (torch.sum(voiced_mask, dim=1) + epsilon)
        
        # Min of voiced frames (replace 0s with large value first)
        voiced_frames_for_min = voiced_frames + (1 - voiced_mask) * 1000.0
        min_f0 = torch.min(voiced_frames_for_min, dim=1)[0]
        
        # Max 
        max_f0 = torch.max(voiced_frames, dim=1)[0]
        
        # Standard deviation
        mean_f0_expanded = mean_f0.unsqueeze(1).expand_as(voiced_frames)
        squared_diff = (voiced_frames - mean_f0_expanded) ** 2
        variance = torch.sum(squared_diff * voiced_mask, dim=1) / (torch.sum(voiced_mask, dim=1) + epsilon)
        std_f0 = torch.sqrt(variance + epsilon)
        
        # Combine F0 statistics
        f0_features_combined = torch.stack([mean_f0, min_f0, max_f0, std_f0], dim=1)
        
        # Process F0 features
        f0_features = self.f0_network(f0_features_combined)
        
        # Combine all features
        combined = torch.cat([features, f0_features], dim=1)
        
        # Generate final noise conditioning factor
        return self.combined_network(combined)