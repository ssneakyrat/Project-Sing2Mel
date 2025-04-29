import torch
import torch.nn as nn

class ConditioningNetwork(nn.Module):
    def __init__(self, n_mels=80, num_phonemes=100, hidden_dim=128, output_dim=None):
        super(ConditioningNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        # If output_dim is not provided, default to hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        
        # Process mel spectrogram - lightweight 1D convolution approach
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU()
        )
        
        # Process F0 - simple convolutional encoder
        self.f0_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU()
        )
        
        # Process phoneme features
        self.phoneme_encoder = nn.Sequential(
            nn.Conv1d(64, hidden_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU()
        )
        
        # Process singer embedding
        self.singer_encoder = nn.Sequential(
            nn.Linear(32, hidden_dim//8),
            nn.ReLU()
        )
        
        # Process language embedding
        self.language_encoder = nn.Sequential(
            nn.Linear(32, hidden_dim//8),
            nn.ReLU()
        )
        
        # Calculate total channels after concatenation
        total_channels = (hidden_dim // 2) + (hidden_dim // 4) + (hidden_dim // 4) + (hidden_dim // 8) + (hidden_dim // 8)
        
        # Combine all features efficiently with correct input channel count
        self.combiner = nn.Sequential(
            nn.Conv1d(total_channels, self.output_dim, kernel_size=1),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU()
        )
        
    def forward(self, mel, f0, phoneme_embedded, singer_embedded, language_embedded):
        # Get sequence length
        seq_len = mel.size(2)
        batch_size = mel.size(0)
        
        # Process mel spectrogram
        mel_features = self.mel_encoder(mel)
        
        # Process F0
        f0_features = self.f0_encoder(f0)
        
        # Process phoneme embeddings (now already in the right format)
        phoneme_features = self.phoneme_encoder(phoneme_embedded)
        
        # Process singer embedding - expand to match time dimension
        singer_features = self.singer_encoder(singer_embedded)
        singer_features = singer_features.unsqueeze(2).expand(-1, -1, seq_len)
        
        # Process language embedding - expand to match time dimension
        language_features = self.language_encoder(language_embedded)
        language_features = language_features.unsqueeze(2).expand(-1, -1, seq_len)
        
        # Combine all features
        combined = torch.cat([
            mel_features,              # [B, hidden_dim//2, T]
            f0_features,               # [B, hidden_dim//4, T]
            phoneme_features,          # [B, hidden_dim//4, T]
            singer_features,           # [B, hidden_dim//8, T]
            language_features          # [B, hidden_dim//8, T]
        ], dim=1)
        
        # Final conditioning
        condition = self.combiner(combined)
        
        return condition