import torch
import torch.nn as nn
import torch.nn.functional as F

class MelEncoder(nn.Module):
    def __init__(self, n_mels=80, phoneme_embed_dim=128, singer_embed_dim=16, language_embed_dim=8):
        super(MelEncoder, self).__init__()
        
        # Store dimensions
        self.n_mels = n_mels
        self.phoneme_embed_dim = phoneme_embed_dim
        self.singer_embed_dim = singer_embed_dim
        self.language_embed_dim = language_embed_dim
        
        # Calculate combined input dimension
        self.combined_dim = phoneme_embed_dim + singer_embed_dim + language_embed_dim + 1  # +1 for f0
        
        # Hidden dimensions
        self.hidden_dim = 256
        
        # 1. Feature Integration Module
        self.feature_projection = nn.Sequential(
            nn.Linear(self.combined_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )
        
        # 2. Temporal Processing Module
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Combine bidirectional outputs
        self.temporal_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        # 3. Spectral Representation Module
        self.spectral_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # 4. Output Refinement Module (Postnet)
        self.output_projection = nn.Linear(self.hidden_dim, n_mels)
        
        # Postnet: 1D convolutions for refinement
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
        )
    
    def forward(self, f0, phoneme_emb, singer_emb, language_emb):
        """
        Forward pass of the MelEncoder.
        
        Args:
            f0: Fundamental frequency trajectory [B, T, 1]
            phoneme_emb: Phoneme embeddings [B, T, phoneme_embed_dim]
            singer_emb: Singer embeddings [B, singer_embed_dim]
            language_emb: Language embeddings [B, language_embed_dim]
            
        Returns:
            mel_spectrogram: Predicted mel spectrogram [B, T, n_mels]
        """
        batch_size, seq_len = phoneme_emb.shape[0], phoneme_emb.shape[1]
        
        # Expand singer and language embeddings to match sequence length
        singer_emb_expanded = singer_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, singer_embed_dim]
        language_emb_expanded = language_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, language_embed_dim]
        
        # Concatenate all features
        combined_features = torch.cat(
            [f0, phoneme_emb, singer_emb_expanded, language_emb_expanded], 
            dim=-1
        )  # [B, T, combined_dim]
        
        # 1. Feature Integration
        integrated_features = self.feature_projection(combined_features)  # [B, T, hidden_dim]
        
        # 2. Temporal Processing
        lstm_out, _ = self.lstm(integrated_features)  # [B, T, hidden_dim*2]
        temporal_features = self.temporal_projection(lstm_out)  # [B, T, hidden_dim]
        
        # Add residual connection
        temporal_features = temporal_features + integrated_features
        
        # 3. Spectral Representation
        spectral_features = self.spectral_layers(temporal_features)  # [B, T, hidden_dim]
        
        # Add another residual connection
        spectral_features = spectral_features + temporal_features
        
        # 4. Output Projection
        mel_output = self.output_projection(spectral_features)  # [B, T, n_mels]
        
        # Apply postnet for refinement
        mel_output_transposed = mel_output.transpose(1, 2)  # [B, n_mels, T]
        postnet_output = self.postnet(mel_output_transposed)  # [B, n_mels, T]
        refined_mel = (mel_output_transposed + postnet_output).transpose(1, 2)  # [B, T, n_mels]
        
        return refined_mel