import torch
import torch.nn as nn

class RegisterClassifier(nn.Module):
    """
    Predicts the vocal register (chest, mixed, head) based on f0 and singer characteristics.
    """
    def __init__(self, singer_dim=16, hidden_dim=32):
        super(RegisterClassifier, self).__init__()
        
        # Input: f0 and singer embedding
        input_dim = 1 + singer_dim  # f0 + singer embedding
        
        # Simple MLP for register classification
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),  # 3 registers: chest, mixed, head
        )
    
    def forward(self, f0, singer_emb):
        """
        Predict register weights.
        
        Args:
            f0: Fundamental frequency [B, T, 1]
            singer_emb: Singer embeddings [B, T, singer_dim]
            
        Returns:
            register_weights: Weights for each register [B, T, 3]
        """
        # Concatenate inputs
        x = torch.cat([f0, singer_emb], dim=-1)  # [B, T, 1+singer_dim]
        
        # Get register logits
        register_logits = self.net(x)  # [B, T, 3]
        
        # Get register weights with softmax for smooth interpolation
        register_weights = torch.softmax(register_logits, dim=-1)  # [B, T, 3]
        
        return register_weights


class ParameterPredictor(nn.Module):
    """
    Predicts formant filter parameters, harmonic amplitudes, and noise parameters from input features.
    
    This module uses phoneme embeddings, singer embeddings, language embeddings,
    and fundamental frequency (f0) to predict parameters for a formant filter bank,
    harmonic amplitudes for the source signal, and noise parameters for the noise component.
    
    Now includes register-adaptive formant parameters with physics-based constraints.
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
        
        # Add register classifier
        self.register_classifier = RegisterClassifier(singer_dim=singer_dim)
        
        # Define learnable register-dependent formant parameters
        # For each register (chest, mixed, head), define base parameters
        
        # Initial formant base frequencies - now learnable Parameters
        chest_formants_init = torch.tensor([500.0, 1500.0, 2500.0, 3500.0, 4500.0][:num_formants])
        mixed_formants_init = torch.tensor([550.0, 1650.0, 2550.0, 3550.0, 4550.0][:num_formants])
        head_formants_init = torch.tensor([600.0, 1800.0, 2600.0, 3600.0, 4600.0][:num_formants])
        
        # Learnable parameters for each register's formant frequencies
        self.chest_formant_freqs = nn.Parameter(chest_formants_init, requires_grad=True)
        self.mixed_formant_freqs = nn.Parameter(mixed_formants_init, requires_grad=True)
        self.head_formant_freqs = nn.Parameter(head_formants_init, requires_grad=True)
        
        # Initial bandwidth bases - now learnable Parameters
        chest_bw_init = torch.tensor([80.0, 100.0, 120.0, 150.0, 200.0][:num_formants])
        mixed_bw_init = torch.tensor([100.0, 120.0, 140.0, 170.0, 220.0][:num_formants])
        head_bw_init = torch.tensor([120.0, 150.0, 180.0, 200.0, 250.0][:num_formants])
        
        # Learnable parameters for each register's bandwidth bases
        self.chest_bandwidth_base = nn.Parameter(chest_bw_init, requires_grad=True)
        self.mixed_bandwidth_base = nn.Parameter(mixed_bw_init, requires_grad=True)
        self.head_bandwidth_base = nn.Parameter(head_bw_init, requires_grad=True)
        
        # Maximum deviations are still fixed buffers since they represent physical constraints
        self.register_buffer('formant_max_deviation', 
                          torch.tensor([300.0, 800.0, 1000.0, 1000.0, 1000.0][:num_formants]))
        
        # Bandwidth scaling is still a fixed buffer
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
        
        # Phoneme-specific formant adjustments
        self.phoneme_formant_adjust = nn.Linear(phoneme_dim, num_formants)
        
    def get_formant_regularization_loss(self):
        """
        Compute loss to enforce physically plausible formant relationships.
        """
        loss = 0.0
        
        # Function to check formant spacing and ranges for one register's formants
        def register_formant_constraints(formants):
            # Get device from formants tensor
            device = formants.device
            
            # Minimum spacing between adjacent formants (Hz) - move to correct device
            min_spacing = torch.tensor([500.0, 500.0, 500.0, 500.0][:self.num_formants-1], device=device)
            
            # Check spacing between adjacent formants
            formant_diffs = formants[1:] - formants[:-1]
            spacing_loss = torch.relu(min_spacing - formant_diffs).sum()
            
            # Check absolute formant ranges - move to correct device
            min_formant_values = torch.tensor([300.0, 800.0, 1800.0, 2800.0, 3800.0][:self.num_formants], device=device)
            max_formant_values = torch.tensor([1000.0, 2500.0, 3500.0, 4500.0, 5500.0][:self.num_formants], device=device)
            
            range_loss_min = torch.relu(min_formant_values - formants).sum()
            range_loss_max = torch.relu(formants - max_formant_values).sum()
            
            return spacing_loss + range_loss_min + range_loss_max
        
        # Apply constraints to each register's formants
        loss += register_formant_constraints(self.chest_formant_freqs)
        loss += register_formant_constraints(self.mixed_formant_freqs)
        loss += register_formant_constraints(self.head_formant_freqs)
        
        # Get device
        device = self.chest_formant_freqs.device
        
        # Check register progression (head > mixed > chest)
        # First formant should generally increase with register
        f1_progression_loss = torch.relu(self.chest_formant_freqs[0] - self.mixed_formant_freqs[0]) + \
                             torch.relu(self.mixed_formant_freqs[0] - self.head_formant_freqs[0])
        
        # Second formant should generally increase with register
        f2_progression_loss = torch.relu(self.chest_formant_freqs[1] - self.mixed_formant_freqs[1]) + \
                             torch.relu(self.mixed_formant_freqs[1] - self.head_formant_freqs[1])
        
        loss += f1_progression_loss + f2_progression_loss
        
        # Bandwidth constraints - bandwidths should increase with register
        # Move all tensors to the same device
        chest_bw = self.chest_bandwidth_base.to(device)
        mixed_bw = self.mixed_bandwidth_base.to(device)
        head_bw = self.head_bandwidth_base.to(device)
        
        bw_progression_loss = torch.relu(chest_bw - mixed_bw).sum() + \
                             torch.relu(mixed_bw - head_bw).sum()
        
        loss += bw_progression_loss
        
        return loss
        
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
                'register_weights': Register classification weights [B, T, 3]
                'regularization_loss': Loss term for formant constraints
        """
        batch_size, seq_len = f0.shape
        device = f0.device
        
        # Expand singer and language embeddings to match sequence length
        singer_emb_expanded = singer_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, singer_dim]
        language_emb_expanded = language_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, language_dim]
        
        # Reshape f0 to match embeddings
        f0_expanded = f0.unsqueeze(-1)  # [B, T, 1]
        
        # Classify vocal register
        register_weights = self.register_classifier(f0_expanded, singer_emb_expanded)  # [B, T, 3]
        
        # Concatenate all inputs - vectorized operation
        x = torch.cat([phoneme_emb, singer_emb_expanded, language_emb_expanded, f0_expanded], dim=-1)
        
        # Process through fully connected layers
        x = self.input_fc(x)  # [B, T, hidden_dim]
        
        # Process through LSTM if enabled
        if self.use_lstm:
            x, _ = self.lstm(x)  # [B, T, hidden_dim]
        
        # Calculate phoneme-specific formant adjustments
        phoneme_adjust = torch.tanh(self.phoneme_formant_adjust(phoneme_emb)) * 200.0  # Scale to reasonable range
        
        # Get formant parameters using separate projection layers
        # Frequency deviations (to be added to base frequencies)
        freq_deviations = torch.tanh(self.freq_fc(x))  # Range: [-1, 1]
        
        # Blend formant base frequencies using register weights
        # Move learnable parameters to device
        chest_formants = self.chest_formant_freqs.to(device)
        mixed_formants = self.mixed_formant_freqs.to(device)
        head_formants = self.head_formant_freqs.to(device)
        
        # Expand to batch and time dimensions [B, T, num_formants]
        chest_formants_exp = chest_formants.expand(batch_size, seq_len, -1)
        mixed_formants_exp = mixed_formants.expand(batch_size, seq_len, -1)
        head_formants_exp = head_formants.expand(batch_size, seq_len, -1)
        
        # Weight each register's formants by register weights
        # [B, T, 1, 3] * [B, T, 3, num_formants] -> [B, T, 1, num_formants]
        formant_bases = torch.stack([
            chest_formants_exp, 
            mixed_formants_exp, 
            head_formants_exp
        ], dim=2)  # [B, T, 3, num_formants]
        
        register_weights_reshaped = register_weights.unsqueeze(3)  # [B, T, 3, 1]
        
        # Blend formant bases using register weights
        blended_formant_base = torch.sum(register_weights_reshaped * formant_bases, dim=2)  # [B, T, num_formants]
        
        # Add phoneme-specific adjustments
        blended_formant_base = blended_formant_base + phoneme_adjust
        
        # Scale deviations and add to base frequencies
        formant_dev_scaled = freq_deviations * self.formant_max_deviation.to(device).expand(batch_size, seq_len, -1)
        frequencies = blended_formant_base + formant_dev_scaled
        
        # Make sure adjacent formants maintain proper ordering (F1 < F2 < F3 etc.)
        frequencies, _ = torch.sort(frequencies, dim=-1)
        
        # Blend bandwidth bases by register
        chest_bw = self.chest_bandwidth_base.to(device)
        mixed_bw = self.mixed_bandwidth_base.to(device)
        head_bw = self.head_bandwidth_base.to(device)
        
        # Expand to batch and time dimensions
        chest_bw_exp = chest_bw.expand(batch_size, seq_len, -1)
        mixed_bw_exp = mixed_bw.expand(batch_size, seq_len, -1)
        head_bw_exp = head_bw.expand(batch_size, seq_len, -1)
        
        # Stack for register-weighted blending
        bandwidth_bases = torch.stack([
            chest_bw_exp,
            mixed_bw_exp,
            head_bw_exp
        ], dim=2)  # [B, T, 3, num_formants]
        
        # Blend bandwidth bases using register weights
        blended_bandwidth_base = torch.sum(register_weights_reshaped * bandwidth_bases, dim=2)  # [B, T, num_formants]
        
        # Bandwidths - positive values with base offset
        # sigmoid -> [0, 1] range
        bandwidth_factors = torch.sigmoid(self.bandwidth_fc(x))
        
        # Scale bandwidth factors
        bandwidth_scaling = self.bandwidth_scaling.to(device).expand(batch_size, seq_len, -1)
        bandwidths = blended_bandwidth_base + bandwidth_factors * bandwidth_scaling
        
        # Formant amplitudes - values in [0, 1] range
        formant_amplitudes = torch.sigmoid(self.amplitude_fc(x))
        
        # Harmonic amplitudes - values in [0, 1] range
        harmonic_amplitudes = torch.sigmoid(self.harmonic_amplitude_fc(x))
        
        # Noise parameters
        noise_gain = torch.sigmoid(self.noise_gain_fc(x))
        spectral_shape = torch.softmax(self.spectral_shape_fc(x), dim=-1)
        voiced_mix = torch.sigmoid(self.voiced_mix_fc(x))
        
        # Compute regularization loss
        regularization_loss = self.get_formant_regularization_loss()
        
        return {
            'frequencies': frequencies,                 # [B, T, num_formants]
            'bandwidths': bandwidths,                   # [B, T, num_formants]
            'amplitudes': formant_amplitudes,           # [B, T, num_formants]
            'harmonic_amplitudes': harmonic_amplitudes, # [B, T, num_harmonics]
            'noise_gain': noise_gain,                   # [B, T, 1]
            'spectral_shape': spectral_shape,           # [B, T, n_noise_bands]
            'voiced_mix': voiced_mix,                   # [B, T, 1]
            'register_weights': register_weights,       # [B, T, 3]
            'regularization_loss': regularization_loss  # scalar
        }