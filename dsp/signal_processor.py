import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalProcessor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, condition_dim=256):
        """
        Signal processor with convolutional layers and conditional input.
        
        Args:
            input_dim: Dimension of input features (default: 1 for raw audio)
            hidden_dim: Dimension of hidden layer
            condition_dim: Dimension of conditioning features
        """
        super(SignalProcessor, self).__init__()
        
        # Conditioning projection
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Simplified conditioning - directly to signal modulation
        self.condition_modulation = nn.Sequential(
            nn.Linear(hidden_dim, 1),  # Project to 1 channel to match input_dim
            nn.Tanh()  # Constrain to [-1, 1]
        )
        
        # Conditioning strength - gradually increase during training
        self.condition_strength = nn.Parameter(torch.tensor(0.01))  # Start very small
        
        # Convolutional network for processing (replacing the linear layers)
        self.process_network = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, signal, condition=None):
        """
        Process an input signal using convolutional layers with optional conditioning.
        
        Args:
            signal: Input audio signal [B, T]
            condition: Conditioning features [B, T, condition_dim]
            
        Returns:
            processed_signal: Processed audio signal with same shape [B, T]
        """
        # Reshape for convolutional processing [B, C, T]
        reshaped_signal = signal.unsqueeze(1)  # [B, 1, T]
        
        # Process through convolutional network
        processed = self.process_network(reshaped_signal)  # [B, 1, T]
        
        # Apply conditioning if provided
        if condition is not None:
            # Project conditioning features
            condition_features = self.condition_projection(condition)  # [B, T, hidden_dim]
            
            # Generate modulation with constrained range
            modulation = self.condition_modulation(condition_features)  # [B, T, 1]
            
            # Reshape modulation to match signal shape [B, 1, T_cond]
            modulation = modulation.transpose(1, 2)  # [B, 1, T_cond]
            
            # Interpolate modulation to match signal length
            target_length = processed.shape[2]
            modulation = F.interpolate(
                modulation, 
                size=target_length, 
                mode='linear', 
                align_corners=False
            )  # [B, 1, T_audio]
            
            # Apply very conservative additive conditioning with learnable strength
            # Use clipped strength parameter to ensure it stays positive but small
            strength = torch.clamp(self.condition_strength, 0.0, 0.3)
            
            # Apply as a residual connection - original processed + small modulation
            processed = processed + modulation * strength
        
        # Squeeze back to original dimensions [B, T]
        processed_signal = processed.squeeze(1)
        
        return processed_signal