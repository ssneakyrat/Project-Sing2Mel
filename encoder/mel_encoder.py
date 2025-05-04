import torch
import torch.nn as nn
import torch.nn.functional as F
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

class DilatedConvBlock(nn.Module):
    """
    Dilated convolutional block with gated activations.
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super(DilatedConvBlock, self).__init__()
        self.conv_layer = nn.Conv1d(
            channels, 
            channels * 2,  # Double for gated activation
            kernel_size=kernel_size,
            padding=(kernel_size-1)//2 * dilation,
            dilation=dilation
        )
        self.layer_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C]
        Returns:
            Output tensor [B, T, C]
        """
        # Save residual
        residual = x
        
        # Switch to channels-first for convolution
        x = x.transpose(1, 2)  # [B, C, T]
        
        # Apply convolution
        x = self.conv_layer(x)
        
        # Gated activation
        filter_x, gate_x = torch.chunk(x, 2, dim=1)
        x = torch.sigmoid(gate_x) * torch.tanh(filter_x)
        
        # Back to original shape
        x = x.transpose(1, 2)  # [B, T, C]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Skip connection
        x = residual + x
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [B, T, D]
            mask: Optional mask tensor [B, T, T]
        Returns:
            Output tensor [B, T, D]
        """
        batch_size = x.size(0)
        
        # Linear projections
        q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        k = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        v = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, T, T]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, d_k]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.w_o(attn_output)
        
        return output

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, D]
        Returns:
            Output tensor [B, T, D]
        """
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention and feed-forward network.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [B, T, D]
            mask: Optional mask tensor [B, T, T]
        Returns:
            Output tensor [B, T, D]
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class MelEncoder(nn.Module):
    """
    Encodes fundamental frequency and linguistic features into mel spectrograms.
    Uses a hybrid architecture with convolutional layers and self-attention to predict 
    mel spectrograms from f0, phoneme, singer, and language embeddings.
    """
    def __init__(self, 
                 n_mels=80, 
                 phoneme_embed_dim=128, 
                 singer_embed_dim=16, 
                 language_embed_dim=8,
                 hidden_dim=128,
                 num_heads=4,
                 num_encoder_layers=2,
                 num_conv_blocks=2,
                 dropout=0.1):
        super(MelEncoder, self).__init__()
        
        # Input dimensions
        self.n_mels = n_mels
        self.phoneme_embed_dim = phoneme_embed_dim
        self.singer_embed_dim = singer_embed_dim
        self.language_embed_dim = language_embed_dim
        
        # Enhanced f0 processing with MLP
        f0_dim = 32  # Larger projected dimension for f0
        self.f0_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, f0_dim),
            nn.LayerNorm(f0_dim)
        )
        
        # Calculate total input dimension
        total_input_dim = phoneme_embed_dim + singer_embed_dim + language_embed_dim + f0_dim
        
        # Input projection layer
        self.input_projection = nn.Linear(total_input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Dilated convolutional blocks for local feature extraction
        self.conv_blocks = nn.ModuleList()
        for i in range(num_conv_blocks):
            dilation = 2 ** i  # Exponentially increasing dilation
            self.conv_blocks.append(DilatedConvBlock(hidden_dim, dilation=dilation, dropout=dropout))
        
        # Transformer encoder layers for global feature extraction
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.encoder_layers.append(EncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout))
        
        # Conditional output projection based on singer and language
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim + singer_embed_dim + language_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mels),
            nn.Sigmoid()  # Constrain values
        )
        
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
        
        # Process f0 with MLP
        f0_processed = self.f0_mlp(f0)  # [B, T, f0_dim]
        
        # Expand singer and language embeddings to match sequence length
        singer_emb_expanded = singer_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, singer_embed_dim]
        language_emb_expanded = language_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, language_embed_dim]
        
        # Concatenate all inputs
        combined_input = torch.cat([phoneme_emb, f0_processed, singer_emb_expanded, language_emb_expanded], dim=-1)
        
        # Project to hidden dimension
        x = self.input_projection(combined_input)  # [B, T, hidden_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply dilated convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Apply transformer encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # Concatenate with singer and language for conditional output
        x = torch.cat([x, singer_emb_expanded, language_emb_expanded], dim=-1)
        
        # Project to mel spectrogram
        mel_pred = self.output_projection(x)  # [B, T, n_mels]
        
        return mel_pred