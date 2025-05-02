import torch
import torch.nn as nn

class MelEncoder(nn.Module):
    """
    Encodes fundamental frequency and linguistic features into mel spectrograms.
    Uses simple linear layers to predict mel spectrograms from f0, phoneme, singer, and language embeddings.
    """
    def __init__(self, 
                 n_mels=80, 
                 phoneme_embed_dim=128, 
                 singer_embed_dim=16, 
                 language_embed_dim=8,
                 hidden_dim=128):
        super(MelEncoder, self).__init__()
        
        # Input dimensions
        self.n_mels = n_mels
        self.phoneme_embed_dim = phoneme_embed_dim
        self.singer_embed_dim = singer_embed_dim
        self.language_embed_dim = language_embed_dim
        
        # Calculate total input dimension
        f0_dim = 8  # Projected dimension for f0
        total_input_dim = phoneme_embed_dim + singer_embed_dim + language_embed_dim + f0_dim
        
        # F0 projection layer
        self.f0_projection = nn.Linear(1, f0_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
        ])
        
        # Residual connections
        self.residual_projections = nn.ModuleList([
            nn.Linear(total_input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, n_mels)
        
    def forward(self, f0, phoneme_emb, singer_emb, language_emb):
        """
        Forward pass to generate mel spectrogram from linguistic features.
        
        Args:
            f0: Fundamental frequency trajectory [B, T, 1]
            phoneme_emb: Phoneme embeddings [B, T, phoneme_embed_dim]
            singer_emb: Singer embeddings [B, singer_embed_dim]
            language_emb: Language embeddings [B, language_embed_dim]
            
        Returns:
            Predicted mel spectrogram [B, T, n_mels]
        """
        batch_size, seq_len = f0.shape[0], f0.shape[1]
        
        # Project f0
        f0_proj = self.f0_projection(f0)  # [B, T, f0_dim]
        
        # Expand singer and language embeddings to match sequence length
        singer_emb_expanded = singer_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, singer_embed_dim]
        language_emb_expanded = language_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, language_embed_dim]
        
        # Concatenate all inputs
        encoder_input = torch.cat([phoneme_emb, f0_proj, singer_emb_expanded, language_emb_expanded], dim=-1)
        
        # Process through encoder layers with residual connections
        x = encoder_input
        residual = self.residual_projections[0](x)
        x = self.encoder_layers[0](x) + residual
        
        residual = self.residual_projections[1](x)
        x = self.encoder_layers[1](x) + residual
        
        x = self.encoder_layers[2](x) + x  # Self-residual for the last layer
        
        # Project to mel spectrogram
        mel_pred = torch.sigmoid(self.output_projection(x))  # Apply sigmoid to constrain values
        
        return mel_pred