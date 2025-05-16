import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embeddings to introduce a notion of word order.
    """
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter but should be saved and loaded with the model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feed-forward networks
    """
    def __init__(self, dim, heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 4
            
        # Layer normalization and multi-head attention
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention block with residual connection
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.attention(x, x, x, attn_mask=mask)
        x = residual + self.dropout1(x_attn)
        
        # Feed-forward block with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ff(x))
        return x


class FeatureProcessor(nn.Module):
    """
    Enhanced FeatureProcessor using Transformer architecture for better 
    sequence modeling capabilities.
    """
    def __init__(self, 
                 phoneme_embed_dim=128,
                 singer_embed_dim=16, 
                 language_embed_dim=8,
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=2,
                 ff_dim=None,
                 max_seq_length=1000,
                 num_harmonics=100,
                 num_mag_harmonic=80,
                 num_mag_noise=80,
                 dropout=0.1):
        super().__init__()
        
        # Calculate total input dimension
        total_input_dim = phoneme_embed_dim + 1 + singer_embed_dim + language_embed_dim
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)
        
        # Optional positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_length, dropout)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projections for control parameters
        self.harmonic_projector = nn.Linear(hidden_dim, num_mag_harmonic)
        self.noise_projector = nn.Linear(hidden_dim, num_mag_noise)
        
        # Simple sigmoid activation for scaling outputs to [0,1]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, f0, phoneme_emb, singer_emb, language_emb):
        """
        Process linguistic features through transformer architecture to
        generate control parameters.
        
        Args:
            f0: Fundamental frequency trajectory [B, T, 1]
            phoneme_emb: Phoneme embeddings [B, T, phoneme_embed_dim]
            singer_emb: Singer embeddings [B, singer_embed_dim]
            language_emb: Language embeddings [B, language_embed_dim]
            
        Returns:
            Dictionary of control parameters:
            - harmonic_magnitude [B, T, num_mag_harmonic]
            - noise_magnitude [B, T, num_mag_noise]
        """
        batch_size, seq_len = f0.shape[0], f0.shape[1]
        
        # Expand style features to match sequence length
        singer_expanded = singer_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, singer_dim]
        language_expanded = language_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, language_dim]
        
        # Concatenate all features
        combined_features = torch.cat([
            phoneme_emb,       # [B, T, phoneme_embed_dim]
            f0,                # [B, T, 1]
            singer_expanded,   # [B, T, singer_dim]
            language_expanded  # [B, T, language_dim]
        ], dim=-1)
        
        # Project to hidden dimension
        x = self.input_proj(combined_features)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # Final normalization
        x = self.norm(x)
        
        # Generate control parameters
        harmonic_magnitude = self.sigmoid(self.harmonic_projector(x))
        noise_magnitude = self.sigmoid(self.noise_projector(x))
        
        # Create output dictionary with the same structure as the original
        output = {
            'harmonic_magnitude': harmonic_magnitude,
            'noise_magnitude': noise_magnitude
        }
        
        return output