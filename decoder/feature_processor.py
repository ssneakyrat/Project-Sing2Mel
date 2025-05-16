import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-based models.
    """
    def __init__(self, d_model, max_seq_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.d_model = d_model
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, D]
        Returns:
            Output tensor with positional encoding added [B, T, D]
        """
        return x + self.pe[:, :x.size(1), :]

class FeatureProcessor(nn.Module):
    """
    Direct linguistic feature to control parameter processor.
    Replaces both LatentEncoder and FeatureExtractor for more efficient processing.
    """
    def __init__(self, 
                 phoneme_embed_dim=128,
                 singer_embed_dim=16, 
                 language_embed_dim=8,
                 hidden_dim=256,
                 num_harmonics=100,
                 num_mag_harmonic=80,
                 num_mag_noise=80,
                 dropout=0.1):
        super().__init__()
        
        # F0 processing with enhanced representation
        self.f0_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )
        
        # Style processing (singer + language)
        self.style_processor = nn.Sequential(
            nn.Linear(singer_embed_dim + language_embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for sequence data
        self.positional_encoding = PositionalEncoding(phoneme_embed_dim)
        
        # Calculate fusion input dimension
        fusion_input_dim = phoneme_embed_dim + 64 + 64  # phoneme + f0 + style
        
        # Feature fusion transformer block
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_input_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Feature processing network
        self.feature_net = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Temporal context modeling (BiGRU + attention)
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Multi-head self-attention for global context
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output projections for control parameters
        self.harmonic_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_mag_harmonic),
            nn.Sigmoid()  # Scale to [0,1]
        )
        
        self.noise_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_mag_noise),
            nn.Sigmoid()  # Scale to [0,1]
        )
        
        # Style-specific adaptors for expressive control
        self.style_harmonic_adaptor = nn.Sequential(
            nn.Linear(64, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_mag_harmonic)
        )
        
        self.style_noise_adaptor = nn.Sequential(
            nn.Linear(64, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_mag_noise)
        )
        
    def forward(self, f0, phoneme_emb, singer_emb, language_emb):
        """
        Direct conversion from linguistic features to control parameters
        
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
        
        # 1. Process F0
        f0_features = self.f0_encoder(f0)  # [B, T, 64]
        
        # 2. Process style information (singer and language)
        style_features = torch.cat([singer_emb, language_emb], dim=-1)  # [B, singer_dim + language_dim]
        style_features = self.style_processor(style_features)  # [B, 64]
        
        # Expand style features to match sequence length
        style_features = style_features.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, 64]
        
        # 3. Apply positional encoding to phoneme embeddings
        phoneme_features = self.positional_encoding(phoneme_emb)  # [B, T, phoneme_embed_dim]
        
        # 4. Concatenate all features
        combined_features = torch.cat([
            phoneme_features,  # [B, T, phoneme_embed_dim]
            f0_features,       # [B, T, 64]
            style_features     # [B, T, 64]
        ], dim=-1)  # [B, T, fusion_input_dim]
        
        # 5. Apply fusion transformer for integrated feature processing
        fused_features = self.fusion_transformer(combined_features)  # [B, T, fusion_input_dim]
        
        # 6. Process through feature network
        processed_features = self.feature_net(fused_features)  # [B, T, hidden_dim]
        
        # 7. Apply temporal context modeling (parallel processing)
        # a. GRU for sequential processing
        gru_out, _ = self.temporal_gru(processed_features)  # [B, T, hidden_dim]
        
        # b. Self-attention for global context
        attn_out, _ = self.self_attention(
            processed_features, 
            processed_features, 
            processed_features
        )  # [B, T, hidden_dim]
        
        # c. Combine GRU and attention outputs
        context_combined = torch.cat([gru_out, attn_out], dim=-1)  # [B, T, hidden_dim*2]
        context_features = self.context_fusion(context_combined)  # [B, T, hidden_dim]
        
        # 8. Generate control parameters
        harmonic_magnitude = self.harmonic_projector(context_features)  # [B, T, num_mag_harmonic]
        noise_magnitude = self.noise_projector(context_features)  # [B, T, num_mag_noise]
        
        # 9. Apply style-specific adaptations
        style_harmonic = self.style_harmonic_adaptor(style_features)  # [B, T, num_mag_harmonic]
        style_noise = self.style_noise_adaptor(style_features)  # [B, T, num_mag_noise]
        
        # 10. Add style adaptations (residual connection)
        harmonic_magnitude = harmonic_magnitude + 0.1 * style_harmonic
        noise_magnitude = noise_magnitude + 0.1 * style_noise
        
        # 11. Create output dictionary
        output = {
            'harmonic_magnitude': harmonic_magnitude,
            'noise_magnitude': noise_magnitude
        }
        
        return output

def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))