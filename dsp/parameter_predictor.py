import torch
import torch.nn as nn

class ParameterPredictor(nn.Module):
    """
    Predicts formant filter parameters, harmonic amplitudes, and noise parameters from input features.
    
    This module uses phoneme embeddings, singer embeddings, language embeddings,
    and fundamental frequency (f0) to predict parameters for a formant filter bank,
    harmonic amplitudes for the source signal, and noise parameters for the noise component.
    """
    def __init__(
        self,
        phoneme_dim=128,
        singer_dim=16,
        language_dim=8,
        hidden_dim=256,
        num_formants=5,
        num_harmonics=8,
        n_noise_bands=8,  # Added parameter for noise bands
        use_lstm=True
    ):
        """
        Initialize the ParameterPredictor.
        
        Args:
            phoneme_dim: Dimension of phoneme embeddings
            singer_dim: Dimension of singer embeddings
            language_dim: Dimension of language embeddings
            hidden_dim: Dimension of hidden layers
            num_formants: Number of formants to model
            num_harmonics: Number of harmonics to model
            n_noise_bands: Number of frequency bands for noise spectral shaping
            use_lstm: Whether to use LSTM for temporal modeling
        """
        super(ParameterPredictor, self).__init__()
        
        self.num_formants = num_formants
        self.num_harmonics = num_harmonics
        self.n_noise_bands = n_noise_bands
        self.use_lstm = use_lstm
        
        # Input feature dimension
        input_dim = phoneme_dim + singer_dim + language_dim + 1  # +1 for f0
        
        # Fully connected layers for initial feature processing
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM for temporal modeling
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.1
            )
        
        # Output layer for harmonic amplitudes
        self.harmonic_amplitude_fc = nn.Linear(hidden_dim, num_harmonics)
        
        # Output layers for noise parameters (new)
        self.noise_gain_fc = nn.Linear(hidden_dim, 1)
        self.spectral_shape_fc = nn.Linear(hidden_dim, n_noise_bands)
        self.voiced_mix_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, f0, phoneme_emb, singer_emb, language_emb):
        """
        Forward pass to predict formant filter parameters, harmonic amplitudes, and noise parameters.
        
        Args:
            f0: Fundamental frequency [B, T]
            phoneme_emb: Phoneme embeddings [B, T, phoneme_dim]
            singer_emb: Singer embeddings [B, singer_dim]
            language_emb: Language embeddings [B, language_dim]
            
        Returns:
            Dictionary with parameters:
                'frequencies': Formant frequencies [B, T, num_formants]
                'bandwidths': Formant bandwidths [B, T, num_formants]
                'amplitudes': Formant amplitudes [B, T, num_formants]
                'harmonic_amplitudes': Harmonic amplitudes [B, T, num_harmonics]
                'noise_gain': Overall noise level [B, T, 1]
                'spectral_shape': Noise spectral shape [B, T, n_noise_bands]
                'voiced_mix': Mix ratio between voiced/unvoiced [B, T, 1]
        """
        batch_size, seq_len = f0.shape
        device = f0.device
        
        # Expand singer and language embeddings to match sequence length
        singer_emb_expanded = singer_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, singer_dim]
        language_emb_expanded = language_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, language_dim]
        
        # Reshape f0 to match embeddings
        f0_expanded = f0.unsqueeze(-1)  # [B, T, 1]
        
        # Concatenate all inputs - vectorized operation
        x = torch.cat([phoneme_emb, singer_emb_expanded, language_emb_expanded, f0_expanded], dim=-1)
        
        # Process through fully connected layers
        x = self.input_fc(x)  # [B, T, hidden_dim]
        
        # Process through LSTM if enabled
        if self.use_lstm:
            x, _ = self.lstm(x)  # [B, T, hidden_dim]
        
        # Harmonic amplitudes - values in [0, 1] range
        # This controls the relative strength of each harmonic in the source signal
        harmonic_amplitudes = torch.sigmoid(self.harmonic_amplitude_fc(x))
        
        # Noise parameters - all new
        # Overall noise gain (how much noise to add)
        noise_gain = torch.sigmoid(self.noise_gain_fc(x))  # [0, 1] range
        
        # Spectral shape of the noise (using softmax to ensure values sum to 1)
        # This represents the energy distribution across frequency bands
        spectral_shape = torch.softmax(self.spectral_shape_fc(x), dim=-1)
        
        # Voiced/unvoiced mix factor
        # 1 = fully voiced (harmonic), 0 = fully unvoiced (noise)
        voiced_mix = torch.sigmoid(self.voiced_mix_fc(x))
        
        return {
            'harmonic_amplitudes': harmonic_amplitudes, # [B, T, num_harmonics]
            'noise_gain': noise_gain,                   # [B, T, 1]
            'spectral_shape': spectral_shape,           # [B, T, n_noise_bands]
            'voiced_mix': voiced_mix,                   # [B, T, 1]
            'hidden_features': x                        # [B, T, hidden_dim]
        }