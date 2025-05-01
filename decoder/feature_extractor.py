import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            input_channel,
            output_splits,
            phoneme_dim=128,
            singer_dim=16,
            language_dim=8,
            hidden_dim=256):
        super().__init__()
        self.output_splits = output_splits
        self.hidden_dim = hidden_dim

        # 1. Mel-spectrogram processing
        self.mel_conv = nn.Sequential(
            weight_norm(nn.Conv1d(input_channel, 128, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Dropout(0.1),
            weight_norm(nn.Conv1d(128, 128, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. F0 processing
        self.f0_conv = nn.Sequential(
            weight_norm(nn.Conv1d(1, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 3. Feature fusion
        # Calculate fusion input dimension
        fusion_input_dim = 128 + 32 + phoneme_dim + singer_dim + language_dim
        
        self.fusion_layers = nn.Sequential(
            weight_norm(nn.Linear(fusion_input_dim, 256)),
            nn.ReLU(),
            nn.Dropout(0.1),
            weight_norm(nn.Linear(256, 256)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 4. Temporal processing (Bidirectional GRU)
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 5. Pre-output layers with residual connection
        self.pre_out = nn.Sequential(
            weight_norm(nn.Linear(hidden_dim, 128)),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final transformation from pre-out to hidden state
        self.pre_out_to_hidden = weight_norm(nn.Linear(128, 64))
        
        # 6. Output projection
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(nn.Linear(64, self.n_out))

    def forward(self, mel, f0, phoneme_seq, singer_id, language_id):
        '''
        input: 
            mel: Mel-spectrogram [B, T, n_mels]
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T, phoneme_dim]
            singer_id: Singer IDs [B, singer_dim]
            language_id: Language IDs [B, language_dim]
        return: 
            dict of B x n_frames x feat
        '''
        batch_size, seq_length = mel.shape[0], mel.shape[1]
        
        # Process mel-spectrogram [B, T, n_mels] -> [B, 128, T]
        mel_trans = mel.transpose(1, 2)  # [B, n_mels, T]
        mel_features = self.mel_conv(mel_trans)  # [B, 128, T]
        mel_features = mel_features.transpose(1, 2)  # [B, T, 128]
        
        # Process f0 [B, T] -> [B, T, 32]
        f0 = f0.unsqueeze(-1)  # [B, T, 1]
        f0_trans = f0.transpose(1, 2)  # [B, 1, T]
        f0_features = self.f0_conv(f0_trans)  # [B, 32, T]
        f0_features = f0_features.transpose(1, 2)  # [B, T, 32]
        
        # Expand singer and language embeddings to match sequence length
        singer_expanded = singer_id.unsqueeze(1).expand(-1, seq_length, -1)  # [B, T, singer_dim]
        language_expanded = language_id.unsqueeze(1).expand(-1, seq_length, -1)  # [B, T, language_dim]
        
        # Concatenate all features
        concat_features = torch.cat([
            mel_features,             # [B, T, 128]
            f0_features,              # [B, T, 32]
            phoneme_seq,              # [B, T, phoneme_dim]
            singer_expanded,          # [B, T, singer_dim]
            language_expanded         # [B, T, language_dim]
        ], dim=-1)  # [B, T, fusion_input_dim]
        
        # Apply fusion layers
        fused = self.fusion_layers(concat_features)  # [B, T, 256]
        
        # Apply GRU for temporal modeling
        gru_out, _ = self.gru(fused)  # [B, T, hidden_dim]
        
        # Apply pre-output layers
        pre_out = self.pre_out(gru_out)  # [B, T, 128]
        
        # Apply residual connection and final pre-output transformation
        x = self.pre_out_to_hidden(pre_out)  # [B, T, 64]
        
        # Output projection
        e = self.dense_out(x)  # [B, T, n_out]
        controls = split_to_dict(e, self.output_splits)
        
        return controls