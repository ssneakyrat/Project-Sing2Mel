import torch
import torch.nn as nn

class NoiseGenerator(nn.Module):
    def __init__(self, n_mels=80, n_bands=4):
        super(NoiseGenerator, self).__init__()
        
        self.n_mels = n_mels
        self.n_bands = n_bands
        self.band_size = n_mels // n_bands
        
        # Process each frequency band separately
        self.band_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2 * self.band_size, self.band_size, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),  # LeakyReLU preserves negative values for noise
                nn.Conv1d(self.band_size, self.band_size, kernel_size=1),
                nn.LeakyReLU(0.2),
            ) for _ in range(n_bands)
        ])
        
        # Noise injection parameters
        self.noise_scales = nn.Parameter(torch.ones(n_bands) * 0.1)
        
        # Amplitude modulation for creating breath patterns
        self.amplitude_modulator = nn.Sequential(
            nn.Conv1d(n_mels * 2, n_mels, kernel_size=5, padding=2),
            nn.Sigmoid()  # Creates smooth envelope for noise
        )
        
    def forward(self, harmonic_mel, melodic_mel):
        batch_size, seq_len, _ = harmonic_mel.shape
        
        # Generate amplitude envelope for breath patterns
        combined_features = torch.cat([harmonic_mel, melodic_mel], dim=-1)
        amplitude_env = self.amplitude_modulator(combined_features.transpose(1, 2)).transpose(1, 2)
        
        # Split into frequency bands
        harmonic_bands = harmonic_mel.reshape(batch_size, seq_len, self.n_bands, -1)
        melodic_bands = melodic_mel.reshape(batch_size, seq_len, self.n_bands, -1)
        
        noise_bands = []
        for i in range(self.n_bands):
            # Process each band
            band_input = torch.cat([harmonic_bands[:, :, i], melodic_bands[:, :, i]], dim=-1)
            band_input = band_input.transpose(1, 2)  # [batch, channels, seq_len]
            
            # Generate structured noise
            structured_noise = self.band_processors[i](band_input)
            
            # Add random noise with learnable scaling
            random_noise = torch.randn_like(structured_noise) * self.noise_scales[i]
            
            # Combine structured and random noise
            band_noise = structured_noise + random_noise
            noise_bands.append(band_noise)
        
        # Reconstruct full mel spectrogram
        noise = torch.cat(noise_bands, dim=1).transpose(1, 2)
        
        # Apply amplitude envelope to create realistic breath patterns
        noise = noise * amplitude_env
        
        # Add global residual noise to maintain graininess
        global_noise = torch.randn_like(noise) * 0.05
        noise = noise + global_noise
        
        return noise