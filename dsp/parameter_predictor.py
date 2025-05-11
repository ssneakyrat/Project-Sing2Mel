import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELSTM(nn.Module):
    """
    Mixture of Experts LSTM implementation that distributes learning across multiple
    specialized LSTM 'experts', with a gating network to determine expert weighting.
    """
    def __init__(self, input_size, hidden_size, num_experts=4, num_layers=2, dropout=0.1):
        super(MoELSTM, self).__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
        # Create multiple expert LSTMs
        self.experts = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
            ) for _ in range(num_experts)
        ])
        
        # Gating network to determine which expert to use
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Calculate expert weights for each sequence
        # We use the average of the input sequence to determine gate values
        # This ensures consistent expert assignment throughout the sequence
        x_avg = torch.mean(x, dim=1)  # [batch_size, input_size]
        gates = self.gate(x_avg).unsqueeze(1)  # [batch_size, 1, num_experts]
        
        # Get outputs from each expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            output, _ = expert(x)  # [batch_size, seq_len, hidden_size]
            expert_outputs.append(output)
        
        # Stack expert outputs along a new dimension
        stacked_outputs = torch.stack(expert_outputs, dim=-1)  # [batch_size, seq_len, hidden_size, num_experts]
        
        # Expand gates to match stacked_outputs dimensions for broadcasting
        expanded_gates = gates.unsqueeze(2)  # [batch_size, 1, 1, num_experts]
        
        # Weighted sum of expert outputs
        weighted_outputs = stacked_outputs * expanded_gates
        combined_output = torch.sum(weighted_outputs, dim=-1)  # [batch_size, seq_len, hidden_size]
        
        return combined_output


class ParameterPredictor(nn.Module):
    """
    Predicts formant filter parameters, harmonic amplitudes, and noise parameters from input features.
    Uses a Mixture of Experts LSTM architecture for better handling of large datasets.
    
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
        n_noise_bands=8,
        use_lstm=True,
        num_experts=4,  # Number of LSTM experts in MoE
        lstm_layers=2   # Number of layers in each LSTM expert
    ):
        """
        Initialize the ParameterPredictor with MoE LSTM.
        
        Args:
            phoneme_dim: Dimension of phoneme embeddings
            singer_dim: Dimension of singer embeddings
            language_dim: Dimension of language embeddings
            hidden_dim: Dimension of hidden layers
            num_formants: Number of formants to model
            num_harmonics: Number of harmonics to model
            n_noise_bands: Number of frequency bands for noise spectral shaping
            use_lstm: Whether to use LSTM for temporal modeling
            num_experts: Number of expert LSTMs in the MoE
            lstm_layers: Number of layers in each LSTM expert
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
        
        # MoE LSTM for temporal modeling
        if use_lstm:
            self.lstm = MoELSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_experts=num_experts,
                num_layers=lstm_layers,
                dropout=0.1
            )
            
            # Layer normalization after LSTM for more stable training
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Define scaling factors for more controlled parameter ranges
        self.register_buffer('formant_base_freqs', 
                            torch.tensor([500.0, 1500.0, 2500.0, 3500.0, 4500.0][:num_formants]))
        self.register_buffer('formant_max_deviation', 
                            torch.tensor([300.0, 800.0, 1000.0, 1000.0, 1000.0][:num_formants]))
        self.register_buffer('bandwidth_base', 
                            torch.tensor([80.0, 100.0, 120.0, 150.0, 200.0][:num_formants]))
        self.register_buffer('bandwidth_scaling', 
                            torch.tensor([100.0, 150.0, 200.0, 250.0, 300.0][:num_formants]))
        
        # Output layers for formant parameters
        # For each formant: frequency deviation, bandwidth, and amplitude
        self.freq_fc = nn.Linear(hidden_dim, num_formants)
        self.bandwidth_fc = nn.Linear(hidden_dim, num_formants)
        self.amplitude_fc = nn.Linear(hidden_dim, num_formants)
        
        # Output layer for harmonic amplitudes
        self.harmonic_amplitude_fc = nn.Linear(hidden_dim, num_harmonics)
        
        # Output layers for noise parameters
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
        x_fc = self.input_fc(x)  # [B, T, hidden_dim]
        
        # Process through MoE LSTM if enabled
        if self.use_lstm:
            x_lstm = self.lstm(x_fc)  # [B, T, hidden_dim]
            x = self.layer_norm(x_lstm)
            
            # Add residual connection from the input_fc output
            x = x + x_fc  # Residual connection helps training
        else:
            x = x_fc
        
        # Get formant parameters using separate projection layers
        # This allows more specialized tuning for each parameter type
        
        # Frequency deviations (to be added to base frequencies)
        freq_deviations = torch.tanh(self.freq_fc(x))  # Range: [-1, 1]
        
        # Scale deviations and add to base frequencies
        # [B, T, num_formants] * [num_formants] -> [B, T, num_formants]
        formant_base = self.formant_base_freqs.to(device).expand(batch_size, seq_len, -1)
        formant_dev_scaled = freq_deviations * self.formant_max_deviation.to(device).expand(batch_size, seq_len, -1)
        frequencies = formant_base + formant_dev_scaled
        
        # Make sure adjacent formants maintain proper ordering (F1 < F2 < F3 etc.)
        # Sort formants along the last dimension
        frequencies, _ = torch.sort(frequencies, dim=-1)
        
        # Bandwidths - positive values with base offset
        # sigmoid -> [0, 1] range
        bandwidth_factors = torch.sigmoid(self.bandwidth_fc(x))
        
        # Scale bandwidth factors and add base values
        # [B, T, num_formants] * [num_formants] -> [B, T, num_formants]
        bandwidth_base = self.bandwidth_base.to(device).expand(batch_size, seq_len, -1)
        bandwidth_scaling = self.bandwidth_scaling.to(device).expand(batch_size, seq_len, -1)
        bandwidths = bandwidth_base + bandwidth_factors * bandwidth_scaling
        
        # Formant amplitudes - values in [0, 1] range
        formant_amplitudes = torch.sigmoid(self.amplitude_fc(x))
        
        # Harmonic amplitudes - values in [0, 1] range
        # This controls the relative strength of each harmonic in the source signal
        harmonic_amplitudes = torch.sigmoid(self.harmonic_amplitude_fc(x))
        
        # Noise parameters
        # Overall noise gain (how much noise to add)
        noise_gain = torch.sigmoid(self.noise_gain_fc(x))  # [0, 1] range
        
        # Spectral shape of the noise (using softmax to ensure values sum to 1)
        # This represents the energy distribution across frequency bands
        spectral_shape = torch.softmax(self.spectral_shape_fc(x), dim=-1)
        
        # Voiced/unvoiced mix factor
        # 1 = fully voiced (harmonic), 0 = fully unvoiced (noise)
        voiced_mix = torch.sigmoid(self.voiced_mix_fc(x))
        
        return {
            'frequencies': frequencies,                 # [B, T, num_formants]
            'bandwidths': bandwidths,                   # [B, T, num_formants]
            'amplitudes': formant_amplitudes,           # [B, T, num_formants]
            'harmonic_amplitudes': harmonic_amplitudes, # [B, T, num_harmonics]
            'noise_gain': noise_gain,                   # [B, T, 1]
            'spectral_shape': spectral_shape,           # [B, T, n_noise_bands]
            'voiced_mix': voiced_mix                    # [B, T, 1]
        }


# Usage example:
def example_usage():
    # Create an instance of the ParameterPredictor
    model = ParameterPredictor(
        phoneme_dim=128,
        singer_dim=16,
        language_dim=8,
        hidden_dim=256,
        num_formants=5,
        num_harmonics=8,
        n_noise_bands=8,
        use_lstm=True,
        num_experts=6,  # Adjust based on dataset size and complexity
        lstm_layers=2
    )
    
    # Example input dimensions
    batch_size = 16
    seq_len = 100
    
    # Create dummy inputs
    f0 = torch.rand(batch_size, seq_len)  # Fundamental frequencies
    phoneme_emb = torch.rand(batch_size, seq_len, 128)  # Phoneme embeddings
    singer_emb = torch.rand(batch_size, 16)  # Singer embeddings
    language_emb = torch.rand(batch_size, 8)  # Language embeddings
    
    # Forward pass
    output = model(f0, phoneme_emb, singer_emb, language_emb)
    
    # Access the predicted parameters
    formant_freqs = output['frequencies']
    formant_bandwidths = output['bandwidths']
    formant_amplitudes = output['amplitudes']
    harmonic_amplitudes = output['harmonic_amplitudes']
    noise_gain = output['noise_gain']
    spectral_shape = output['spectral_shape']
    voiced_mix = output['voiced_mix']
    
    return output


# Training function example (for reference)
def train_model(model, train_loader, optimizer, criterion, epochs=10, device='cuda', grad_clip=1.0):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            # Unpack your batch data here
            f0, phoneme_emb, singer_emb, language_emb, targets = batch
            
            # Move to device
            f0 = f0.to(device)
            phoneme_emb = phoneme_emb.to(device)
            singer_emb = singer_emb.to(device)
            language_emb = language_emb.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(f0, phoneme_emb, singer_emb, language_emb)
            
            # Calculate loss (example - adapt to your specific loss function)
            loss = 0
            for param_name in outputs:
                if param_name in targets:
                    loss += criterion(outputs[param_name], targets[param_name])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")