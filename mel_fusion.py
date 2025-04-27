import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise feature recalibration."""
    
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, channels, seq_len]
        """
        batch, channels, seq_len = x.size()
        
        # Global average pooling
        y = self.avg_pool(x).view(batch, channels)
        
        # Channel attention
        y = self.fc(y).view(batch, channels, 1)
        
        # Scale the input
        return x * y.expand_as(x)


class CrossAttention(nn.Module):
    """Lightweight cross-attention mechanism."""
    
    def __init__(self, channels, heads=1):
        super(CrossAttention, self).__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        
        assert channels % heads == 0, "channels must be divisible by heads"
        
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        self.scale = (self.head_dim) ** -0.5
        
    def forward(self, query, key, value):
        """
        Args:
            query, key, value: Input tensors [batch, seq_len, channels]
        """
        batch, seq_len, _ = query.size()
        
        # Linear projections
        q = self.query(query).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.channels)
        output = self.out_proj(attn_output)
        
        return output


class MelFusion(nn.Module):
    """Hybrid fusion layer combining SE blocks with cross-attention."""
    
    def __init__(self, n_mels=80, reduction=4, attention_heads=1):
        super(MelFusion, self).__init__()
        
        self.n_mels = n_mels
        
        # SE blocks for each component
        self.se_harmonic = SEBlock(n_mels, reduction)
        self.se_melodic = SEBlock(n_mels, reduction)
        self.se_noise = SEBlock(n_mels, reduction)
        
        # Cross-attention layers
        self.harmonic_to_melodic = CrossAttention(n_mels, attention_heads)
        self.harmonic_to_noise = CrossAttention(n_mels, attention_heads)
        self.melodic_to_harmonic = CrossAttention(n_mels, attention_heads)
        self.melodic_to_noise = CrossAttention(n_mels, attention_heads)
        self.noise_to_harmonic = CrossAttention(n_mels, attention_heads)
        self.noise_to_melodic = CrossAttention(n_mels, attention_heads)
        
        # Final fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(n_mels * 3, n_mels, kernel_size=1),
            nn.GroupNorm(8, n_mels),  # GroupNorm for better stability
            nn.ReLU(inplace=True),
            nn.Conv1d(n_mels, n_mels, kernel_size=1)
        )
        
        # Learnable weights for skip connection
        self.harmonic_weight = nn.Parameter(torch.ones(1) * 0.33)
        self.melodic_weight = nn.Parameter(torch.ones(1) * 0.33)
        self.noise_weight = nn.Parameter(torch.ones(1) * 0.33)
        
    def forward(self, harmonic_mel, melodic_mel, noise_mel):
        """
        Args:
            harmonic_mel: Harmonic component [batch, seq_len, n_mels]
            melodic_mel: Melodic component [batch, seq_len, n_mels]
            noise_mel: Noise component [batch, seq_len, n_mels]
        Returns:
            fused_mel: Fused mel spectrogram [batch, seq_len, n_mels]
        """
        # Transpose for SE blocks (they expect [batch, channels, seq_len])
        harmonic_t = harmonic_mel.transpose(1, 2)
        melodic_t = melodic_mel.transpose(1, 2)
        noise_t = noise_mel.transpose(1, 2)
        
        # Apply SE blocks
        harmonic_se = self.se_harmonic(harmonic_t).transpose(1, 2)
        melodic_se = self.se_melodic(melodic_t).transpose(1, 2)
        noise_se = self.se_noise(noise_t).transpose(1, 2)
        
        # Cross-attention
        # Harmonic attends to melodic and noise
        harmonic_from_melodic = self.harmonic_to_melodic(harmonic_se, melodic_se, melodic_se)
        harmonic_from_noise = self.harmonic_to_noise(harmonic_se, noise_se, noise_se)
        harmonic_attended = harmonic_se + 0.5 * (harmonic_from_melodic + harmonic_from_noise)
        
        # Melodic attends to harmonic and noise
        melodic_from_harmonic = self.melodic_to_harmonic(melodic_se, harmonic_se, harmonic_se)
        melodic_from_noise = self.melodic_to_noise(melodic_se, noise_se, noise_se)
        melodic_attended = melodic_se + 0.5 * (melodic_from_harmonic + melodic_from_noise)
        
        # Noise attends to harmonic and melodic
        noise_from_harmonic = self.noise_to_harmonic(noise_se, harmonic_se, harmonic_se)
        noise_from_melodic = self.noise_to_melodic(noise_se, melodic_se, melodic_se)
        noise_attended = noise_se + 0.5 * (noise_from_harmonic + noise_from_melodic)
        
        # Concatenate for final fusion
        concat_features = torch.cat([
            harmonic_attended.transpose(1, 2),
            melodic_attended.transpose(1, 2),
            noise_attended.transpose(1, 2)
        ], dim=1)  # [batch, n_mels*3, seq_len]
        
        # Apply fusion convolution
        fused_features = self.fusion_conv(concat_features)  # [batch, n_mels, seq_len]
        fused_features = fused_features.transpose(1, 2)  # [batch, seq_len, n_mels]
        
        # Skip connection with weighted sum of inputs
        skip_connection = (
            self.harmonic_weight * harmonic_mel +
            self.melodic_weight * melodic_mel +
            self.noise_weight * noise_mel
        )
        
        # Final output with skip connection
        fused_mel = fused_features + skip_connection
        
        return fused_mel