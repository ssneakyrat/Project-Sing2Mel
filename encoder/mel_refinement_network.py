import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer-based architecture
    """
    def __init__(self, d_model, max_seq_length=2000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input of shape [batch_size, seq_length, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PerformerAttention(nn.Module):
    """
    Performer attention with random feature approximation for efficient processing
    """
    def __init__(self, dim, num_heads=8, kernel_ratio=0.5, causal=False, dropout=0.1, dim_head=64):
        super(PerformerAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.causal = causal
        
        inner_dim = num_heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Performer-specific: random projection for feature map approximation
        self.feature_dim = int(kernel_ratio * dim_head)
        self.ortho_scaling = 0  # orthogonal random features
        self.epsilon = 1e-8
        
        # Create random projection matrix
        self.create_projection()
    
    def create_projection(self):
        # Create random orthogonal matrix
        random_matrix = torch.randn(self.dim_head, self.feature_dim)
        q, r = torch.linalg.qr(random_matrix)
        self.register_buffer('proj_matrix', q * math.sqrt(self.dim_head))
        
    def kernel_fn(self, x):
        # ReLU feature map: φ(x) = max(0, x)
        return F.relu(x) / math.sqrt(self.feature_dim)
    
    def performer_attention(self, q, k, v, mask=None):
        # q, k, v shape: [batch_size, num_heads, seq_length, dim_head]
        batch_size, num_heads, seq_length, dim_head = q.shape
        
        # Project q and k to the feature space
        q_prime = self.kernel_fn(torch.einsum('bhnd,df->bhnf', q, self.proj_matrix))  # [b, h, n, f]
        k_prime = self.kernel_fn(torch.einsum('bhnd,df->bhnf', k, self.proj_matrix))  # [b, h, n, f]
        
        # Causal masking for autoregressive attention
        if self.causal:
            # Create causal mask
            i = torch.arange(seq_length)
            mask = i[:, None] <= i[None, :]  # Upper triangular matrix
            mask = mask.to(q.device)
            k_prime = k_prime * mask.unsqueeze(0).unsqueeze(-1)
        
        # Compute K^TV
        kv = torch.einsum('bhnf,bhnd->bhfd', k_prime, v)
        
        # Compute attention approximation
        z = 1.0 / (torch.einsum('bhnf,bhf->bhn', q_prime, k_prime.sum(dim=2)) + self.epsilon)
        attn_output = torch.einsum('bhnf,bhfd,bhn->bhnd', q_prime, kv, z)
        
        return attn_output
    
    def forward(self, x, context=None, mask=None):
        """
        x: Input of shape [batch_size, seq_length, dim]
        context: Optional context tensor for cross-attention
        mask: Optional attention mask
        """
        batch_size, seq_length, _ = x.shape
        
        # If context is not provided (self-attention)
        if context is None:
            context = x
        
        # Project to queries, keys, and values
        q = self.to_q(x).reshape(batch_size, seq_length, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(batch_size, context.size(1), self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(batch_size, context.size(1), self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        
        # Apply performer attention
        attn_output = self.performer_attention(q, k, v, mask)
        
        # Reshape and project back to original dimension
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, -1)
        out = self.to_out(attn_output)
        
        return self.dropout(out)

class FeedForward(nn.Module):
    """
    Standard feed-forward network with expansion and projection
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class PerformerBlock(nn.Module):
    """
    Basic Performer block with self-attention and feed-forward network
    """
    def __init__(self, dim, num_heads=8, hidden_dim_factor=4, dropout=0.1, dim_head=64):
        super(PerformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = PerformerAttention(dim, num_heads, dropout=dropout, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * hidden_dim_factor, dropout)
    
    def forward(self, x):
        # Self-attention block with residual connection
        x = x + self.self_attn(self.norm1(x))
        # Feed-forward block with residual connection
        x = x + self.ff(self.norm2(x))
        return x

class ConditionalPerformerBlock(nn.Module):
    """
    Performer block with additional cross-attention for conditioning
    """
    def __init__(self, dim, conditioning_dim, num_heads=8, hidden_dim_factor=4, dropout=0.1, dim_head=64):
        super(ConditionalPerformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = PerformerAttention(dim, num_heads, dropout=dropout, dim_head=dim_head)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = PerformerAttention(dim, num_heads, dropout=dropout, dim_head=dim_head)
        
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * hidden_dim_factor, dropout)
        
        # Conditioning projection
        self.cond_proj = nn.Linear(conditioning_dim, dim)
    
    def forward(self, x, conditioning):
        # Project conditioning to model dimension
        conditioning = self.cond_proj(conditioning)
        
        # Self-attention block with residual connection
        x = x + self.self_attn(self.norm1(x))
        
        # Cross-attention to conditioning with residual connection
        x = x + self.cross_attn(self.norm2(x), context=conditioning)
        
        # Feed-forward block with residual connection
        x = x + self.ff(self.norm3(x))
        
        return x

class DownsampleLayer(nn.Module):
    """
    Downsampling layer for hierarchical processing
    """
    def __init__(self, in_dim, out_dim):
        super(DownsampleLayer, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, in_dim]
        x = x.transpose(1, 2)  # [batch_size, in_dim, seq_len]
        x = self.pool(x)       # [batch_size, in_dim, seq_len/2]
        x = x.transpose(1, 2)  # [batch_size, seq_len/2, in_dim]
        x = self.proj(x)       # [batch_size, seq_len/2, out_dim]
        return x

class UpsampleLayer(nn.Module):
    """
    Upsampling layer for hierarchical processing
    """
    def __init__(self, in_dim, out_dim):
        super(UpsampleLayer, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, skip=None):
        # x shape: [batch_size, seq_len, in_dim]
        batch_size, seq_len, in_dim = x.shape
        
        if skip is not None:
            # Ensure upsampled size exactly matches skip connection size
            target_size = skip.size(1)
            
            # Upsample sequence length with exact output size
            x = F.interpolate(
                x.transpose(1, 2),  # [batch_size, in_dim, seq_len]
                size=target_size,
                mode='nearest'
            ).transpose(1, 2)  # [batch_size, target_size, in_dim]
        else:
            # If no skip connection, just double the size
            x = F.interpolate(
                x.transpose(1, 2),  # [batch_size, in_dim, seq_len]
                scale_factor=2,
                mode='nearest'
            ).transpose(1, 2)  # [batch_size, seq_len*2, in_dim]
        
        # Project to output dimension
        x = self.proj(x)  # [batch_size, upsampled_size, out_dim]
        
        # Add skip connection if provided
        if skip is not None:
            x = torch.cat([x, skip], dim=-1)
        
        return x

class PerformerMelRefinementNetwork(nn.Module):
    """
    Performer-based architecture for mel spectrogram refinement
    with hierarchical processing and conditioning
    """
    def __init__(self, n_mels, d_model=256, num_heads=8, num_layers=4, 
                 hidden_dim_factor=4, dropout=0.1, phoneme_embed_dim=128, 
                 singer_embed_dim=16, language_embed_dim=8):
        super(PerformerMelRefinementNetwork, self).__init__()
        
        # Store dimensions
        self.n_mels = n_mels
        self.d_model = d_model
        self.phoneme_embed_dim = phoneme_embed_dim
        self.singer_embed_dim = singer_embed_dim
        self.language_embed_dim = language_embed_dim
        
        # Calculate total conditioning dimension
        self.conditioning_dim = phoneme_embed_dim + singer_embed_dim + language_embed_dim + 1  # +1 for f0
        
        # Define the dimension after processing
        self.processed_conditioning_dim = d_model
        
        # Input processing - project mel spectrogram to model dimension
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Conditioning network
        self.conditioning_network = nn.Sequential(
            nn.Linear(self.conditioning_dim, self.processed_conditioning_dim),
            nn.ReLU(),
            nn.Linear(self.processed_conditioning_dim, self.processed_conditioning_dim)
        )
        
        # Hierarchical layers
        # Encoder (downsampling)
        self.down1 = DownsampleLayer(d_model, d_model)
        self.down2 = DownsampleLayer(d_model, d_model)
        
        # Performer layers for each level
        self.level1_blocks = nn.ModuleList([
            ConditionalPerformerBlock(
                d_model, self.processed_conditioning_dim, num_heads, 
                hidden_dim_factor, dropout
            ) for _ in range(num_layers)
        ])
        
        self.level2_blocks = nn.ModuleList([
            ConditionalPerformerBlock(
                d_model, self.processed_conditioning_dim, num_heads, 
                hidden_dim_factor, dropout
            ) for _ in range(num_layers)
        ])
        
        self.level3_blocks = nn.ModuleList([
            ConditionalPerformerBlock(
                d_model, self.processed_conditioning_dim, num_heads, 
                hidden_dim_factor, dropout
            ) for _ in range(num_layers)
        ])
        
        # Decoder (upsampling)
        self.up1 = UpsampleLayer(d_model, d_model)
        self.up2 = UpsampleLayer(d_model * 2, d_model)  # *2 because of skip connection
        
        # Output projection
        self.output_proj = nn.Linear(d_model * 2, n_mels)  # *2 because of skip connection
    
    def forward(self, mel, f0, phoneme_emb, singer_emb, language_emb):
        """
        Refine the predicted mel spectrogram with conditioning.
        
        Args:
            mel: Predicted mel spectrogram [B, T, n_mels]
            f0: Fundamental frequency trajectory [B, T]
            phoneme_emb: Phoneme embeddings [B, T, phoneme_dim]
            singer_emb: Singer embeddings [B, singer_dim]
            language_emb: Language embeddings [B, language_dim]
        
        Returns:
            Refined mel spectrogram [B, T, n_mels]
        """
        batch_size, n_frames, _ = mel.shape
        
        # Process f0, expand global conditioning
        f0 = f0.unsqueeze(-1)  # [B, T, 1]
        singer_emb_expanded = singer_emb.unsqueeze(1).expand(-1, n_frames, -1)  # [B, T, singer_dim]
        language_emb_expanded = language_emb.unsqueeze(1).expand(-1, n_frames, -1)  # [B, T, language_dim]
        
        # Create conditioning tensor [B, T, conditioning_dim]
        conditioning = torch.cat([f0, phoneme_emb, singer_emb_expanded, language_emb_expanded], dim=-1)
        
        # Process the conditioning through the conditioning network
        # Average over time for global conditioning
        global_conditioning = conditioning.mean(dim=1)  # [B, conditioning_dim]
        conditioning_bottleneck = self.conditioning_network(global_conditioning)  # [B, processed_conditioning_dim]
        conditioning_bottleneck = conditioning_bottleneck.unsqueeze(1).expand(-1, n_frames, -1)  # [B, T, processed_conditioning_dim]
        
        # Initial projection and positional encoding
        x = self.input_proj(mel)  # [B, T, d_model]
        x = self.pos_encoding(x)  # Add positional encoding
        
        # Save the original input for residual connection
        input_x = x
        
        # Hierarchical Encoder with skip connections
        skip1 = x
        
        # Level 1 processing
        for block in self.level1_blocks:
            x = block(x, conditioning_bottleneck)
        
        # Downsample to Level 2
        x = self.down1(x)
        level2_frames = x.size(1)  # Store size for conditioning
        
        # Level 2 processing
        for block in self.level2_blocks:
            # Ensure conditioning has correct length by using interpolate instead of slicing
            level2_conditioning = F.interpolate(
                conditioning_bottleneck.transpose(1, 2),  # [B, C, T]
                size=level2_frames,
                mode='nearest'
            ).transpose(1, 2)  # [B, level2_frames, C]
            
            x = block(x, level2_conditioning)
        
        skip2 = x
        
        # Downsample to Level 3
        x = self.down2(x)
        level3_frames = x.size(1)  # Store size for conditioning
        
        # Level 3 processing (bottleneck)
        for block in self.level3_blocks:
            # Ensure conditioning has correct length by using interpolate instead of slicing
            level3_conditioning = F.interpolate(
                conditioning_bottleneck.transpose(1, 2),  # [B, C, T]
                size=level3_frames,
                mode='nearest'
            ).transpose(1, 2)  # [B, level3_frames, C]
            
            x = block(x, level3_conditioning)
        
        # Hierarchical Decoder with skip connections
        # Upsample from Level 3 to Level 2
        x = self.up1(x)
        
        # Concatenate with skip connection from Level 2
        x = torch.cat([x, skip2], dim=-1)
        
        # Upsample from Level 2 to Level 1
        x = self.up2(x, skip1)
        
        # Final output projection
        x = self.output_proj(x)
        
        # Residual connection - learn refinements rather than entire spectrogram
        return mel + x

# Example usage (for reference)
"""
n_mels = 80
phoneme_embed_dim = 128
singer_embed_dim = 16
language_embed_dim = 8

# Create the model
model = PerformerMelRefinementNetwork(
    n_mels=n_mels,
    phoneme_embed_dim=phoneme_embed_dim,
    singer_embed_dim=singer_embed_dim,
    language_embed_dim=language_embed_dim
)

# Dummy inputs
batch_size = 2
seq_length = 128
mel = torch.randn(batch_size, seq_length, n_mels)
f0 = torch.randn(batch_size, seq_length)
phoneme_emb = torch.randn(batch_size, seq_length, phoneme_embed_dim)
singer_emb = torch.randn(batch_size, singer_embed_dim)
language_emb = torch.randn(batch_size, language_embed_dim)

# Forward pass
refined_mel = model(mel, f0, phoneme_emb, singer_emb, language_emb)
print(refined_mel.shape)  # Should be [batch_size, seq_length, n_mels]
"""