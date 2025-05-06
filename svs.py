import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoder.mel_encoder import MelEncoder
from decoder.feature_extractor import FeatureExtractor
from decoder.wave_generator_oscillator import WaveGeneratorOscillator
from decoder.core import scale_function, frequency_filter, upsample
from decoder.human_vocal_filter import vocal_frequency_filter
from decoder.noise_generator import NoiseGenerator
from decoder.enhancement_network import PhaseAwareEnhancer

# Modified SVS class with MelEncoder integration
class SVS(nn.Module):
    """
    Lightweight DDSP-based singing voice synthesis model with separated
    expressive control prediction and signal processing components.
    """
    def __init__(self, 
                 num_phonemes, 
                 num_singers, 
                 num_languages,
                 n_mels=80, 
                 hop_length=240, 
                 sample_rate=24000,
                 num_harmonics=80, 
                 num_mag_harmonic=256,
                 num_mag_noise=80,
                 ):
        super(SVS, self).__init__()
        
        # Basic parameters
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        self.num_mag_harmonic = num_mag_harmonic
        self.num_mag_noise = num_mag_noise
        
        # Define embedding dimensions
        self.phoneme_embed_dim = 128
        self.singer_embed_dim = 16
        self.language_embed_dim = 8
        
        # Embedding layers
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, self.phoneme_embed_dim)
        self.singer_embed = nn.Embedding(num_singers, self.singer_embed_dim)
        self.language_embed = nn.Embedding(num_languages, self.language_embed_dim)   

        # Register buffers for use in forward pass
        self.register_buffer("sampling_rate", torch.tensor(sample_rate))
        self.register_buffer("block_size", torch.tensor(hop_length))
        
        # Define feature extractor output splits with added vocal parameters
        split_map = {
            'harmonic_magnitude': num_mag_harmonic,
            'noise_magnitude': num_mag_noise,
            # Harmonic filter parameters
            'harmonic_articulation': 1,
            'harmonic_presence_amount': 1,
            'harmonic_exciter_amount': 1,
            'harmonic_breathiness': 1,
            # Noise filter parameters
            'noise_articulation': 1,
            'noise_presence_amount': 1,
            'noise_exciter_amount': 1,
            'noise_breathiness': 1,
            # NEW: Formant parameter offsets
            'formant1_offset': 1,  # F1 frequency offset
            'formant2_offset': 1,  # F2 frequency offset
            'formant3_offset': 1,  # F3 frequency offset
            'formant4_offset': 1,  # F4 frequency offset
            # NEW: Vocal range parameters
            'vocal_range_min': 1,  # Lower boundary of vocal range
            'vocal_range_max': 1,  # Upper boundary of vocal range
            # NEW: Formant parameter offsets
            'noise_formant1_offset': 1,  # F1 frequency offset
            'noise_formant2_offset': 1,  # F2 frequency offset
            'noise_formant3_offset': 1,  # F3 frequency offset
            'noise_formant4_offset': 1,  # F4 frequency offset
            # NEW: Vocal range parameters
            'noise_vocal_range_min': 1,  # Lower boundary of vocal range
            'noise_vocal_range_max': 1,  # Upper boundary of vocal range
        }

        # Initialize feature extractor with proper dimensions
        self.feature_extractor = FeatureExtractor(
            input_channel=n_mels,
            output_splits=split_map,
            phoneme_dim=self.phoneme_embed_dim,
            singer_dim=self.singer_embed_dim,
            language_dim=self.language_embed_dim
        )

        # Harmonic Synthesizer parameters
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, num_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        # Initialize harmonic synthesizer
        self.harmonic_synthesizer = WaveGeneratorOscillator(
            sample_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)
        
        self.noise_generator = NoiseGenerator()

        # Initialize mel encoder
        self.mel_encoder = MelEncoder(
            n_mels=n_mels,
            phoneme_embed_dim=self.phoneme_embed_dim,
            singer_embed_dim=self.singer_embed_dim,
            language_embed_dim=self.language_embed_dim
        )
        
        self.refinement = PhaseAwareEnhancer()

    def forward(self, f0, phoneme_seq, singer_id, language_id, initial_phase=None):
        """
        Forward pass with separated expressive control and signal processing.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices)
            language_id: Language IDs [B] (indices)
            mel: Optional mel-spectrogram [B, T, n_mels] (if None, it will be predicted)
            initial_phase: Optional initial phase for the harmonic oscillator
            
        Returns:
            Audio signal [B, T*hop_length], predicted_mel, and vocal parameters dict
        """
        batch_size, n_frames = f0.shape[0], f0.shape[1]
        
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]

        # Prepare f0 for mel encoder
        f0_unsqueeze = f0.unsqueeze(2)  # [B, T, 1]
         # Process F0 - make sure it's in Hz and properly shaped
        f0_unsqueeze = torch.clamp(f0_unsqueeze, min=0.0, max=1000.0)
        f0_unsqueeze[f0_unsqueeze < 80] = 0 + 1e-7  # Set unvoiced regions to 0

        # Generate mel spectrogram if not provided
        predicted_mel = self.mel_encoder(f0_unsqueeze, phoneme_emb, singer_emb, language_emb)

        # Get control parameters from feature extractor
        ctrls = self.feature_extractor(predicted_mel, f0, phoneme_emb, singer_emb, language_emb)

        # Process harmonic and noise parameters
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])
        
        # Apply sigmoid to constrain vocal filter parameters between 0 and 1
        # Convert tensors to scalar values by taking the mean
        harmonic_articulation = torch.sigmoid(ctrls['harmonic_articulation']).mean().item()
        harmonic_presence_amount = torch.sigmoid(ctrls['harmonic_presence_amount']).mean().item()
        harmonic_exciter_amount = torch.sigmoid(ctrls['harmonic_exciter_amount']).mean().item() 
        harmonic_breathiness = torch.sigmoid(ctrls['harmonic_breathiness']).mean().item()
        
        noise_articulation = torch.sigmoid(ctrls['noise_articulation']).mean().item()
        noise_presence_amount = torch.sigmoid(ctrls['noise_presence_amount']).mean().item()
        noise_exciter_amount = torch.sigmoid(ctrls['noise_exciter_amount']).mean().item()
        noise_breathiness = torch.sigmoid(ctrls['noise_breathiness']).mean().item()
        
        # NEW: Process formant offsets
        # Map sigmoid output to appropriate Hz range offsets for each formant
        formant1_offset = (torch.sigmoid(ctrls['formant1_offset']) * 2 - 1) * 100  # ±100Hz range for F1
        formant2_offset = (torch.sigmoid(ctrls['formant2_offset']) * 2 - 1) * 200  # ±200Hz range for F2
        formant3_offset = (torch.sigmoid(ctrls['formant3_offset']) * 2 - 1) * 300  # ±300Hz range for F3
        formant4_offset = (torch.sigmoid(ctrls['formant4_offset']) * 2 - 1) * 400  # ±400Hz range for F4
        
        # Convert to scalar values
        formant1_offset = formant1_offset.mean().item()
        formant2_offset = formant2_offset.mean().item()
        formant3_offset = formant3_offset.mean().item()
        formant4_offset = formant4_offset.mean().item()
        
        # NEW: Process vocal range parameters
        # Map sigmoid output to frequency ranges
        vocal_range_min = torch.sigmoid(ctrls['vocal_range_min']) * 300  # 0-300Hz range
        vocal_range_max = 300 + torch.sigmoid(ctrls['vocal_range_max']) * 1200  # 300-1500Hz range
        
        # Convert to scalar values
        vocal_range_min = vocal_range_min.mean().item()
        vocal_range_max = vocal_range_max.mean().item()

        noise_formant1_offset = (torch.sigmoid(ctrls['noise_formant1_offset']) * 2 - 1) * 100  # ±100Hz range for F1
        noise_formant2_offset = (torch.sigmoid(ctrls['noise_formant2_offset']) * 2 - 1) * 200  # ±200Hz range for F2
        noise_formant3_offset = (torch.sigmoid(ctrls['noise_formant3_offset']) * 2 - 1) * 300  # ±300Hz range for F3
        noise_formant4_offset = (torch.sigmoid(ctrls['noise_formant4_offset']) * 2 - 1) * 400  # ±400Hz range for F4
        
        # Convert to scalar values
        noise_formant1_offset = noise_formant1_offset.mean().item()
        noise_formant2_offset = noise_formant2_offset.mean().item()
        noise_formant3_offset = noise_formant3_offset.mean().item()
        noise_formant4_offset = noise_formant4_offset.mean().item()
        
        # NEW: Process vocal range parameters
        # Map sigmoid output to frequency ranges
        noise_vocal_range_min = torch.sigmoid(ctrls['noise_vocal_range_min']) * 300  # 0-300Hz range
        noise_vocal_range_max = 300 + torch.sigmoid(ctrls['noise_vocal_range_max']) * 1200  # 300-1500Hz range
        
        # Convert to scalar values
        noise_vocal_range_min = noise_vocal_range_min.mean().item()
        noise_vocal_range_max = noise_vocal_range_max.mean().item()

        # upsample
        pitch = upsample(f0_unsqueeze, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthesizer(pitch, initial_phase)
        harmonic = vocal_frequency_filter(
            harmonic, 
            src_param, 
            gender="neutral",
            formant_emphasis=True,
            vocal_range_boost=True,
            articulation=harmonic_articulation,
            presence_amount=harmonic_presence_amount,
            exciter_amount=harmonic_exciter_amount,
            breathiness=harmonic_breathiness,
            # NEW: Pass formant offsets
            formant1_offset=formant1_offset,
            formant2_offset=formant2_offset,
            formant3_offset=formant3_offset,
            formant4_offset=formant4_offset,
            # NEW: Pass vocal range parameters
            vocal_range_min=vocal_range_min,
            vocal_range_max=vocal_range_max
        )

        # noise part
        noise = self.noise_generator(harmonic, noise_param, pitch)
        noise = vocal_frequency_filter(
            noise, 
            noise_param, 
            gender="neutral",  # Or dynamically set based on singer
            formant_emphasis=True,
            vocal_range_boost=True,
            articulation=noise_articulation,
            presence_amount=noise_presence_amount,
            exciter_amount=noise_exciter_amount,
            breathiness=noise_breathiness,
            # NEW: Pass formant offsets
            formant1_offset=noise_formant1_offset,
            formant2_offset=noise_formant2_offset,
            formant3_offset=noise_formant3_offset,
            formant4_offset=noise_formant4_offset,
            # NEW: Pass vocal range parameters
            vocal_range_min=noise_vocal_range_min,
            vocal_range_max=noise_vocal_range_max
        )
        
        signal = harmonic + noise
        
        # Create dictionary of vocal parameters
        vocal_params = {
            # Harmonic filter parameters
            'harmonic_articulation': ctrls['harmonic_articulation'],
            'harmonic_presence_amount': ctrls['harmonic_presence_amount'],
            'harmonic_exciter_amount': ctrls['harmonic_exciter_amount'],
            'harmonic_breathiness': ctrls['harmonic_breathiness'],
            # Noise filter parameters
            'noise_articulation': ctrls['noise_articulation'],
            'noise_presence_amount': ctrls['noise_presence_amount'],
            'noise_exciter_amount': ctrls['noise_exciter_amount'],
            'noise_breathiness': ctrls['noise_breathiness'],
            # NEW: Formant offsets
            'formant1_offset': ctrls['formant1_offset'],
            'formant2_offset': ctrls['formant2_offset'],
            'formant3_offset': ctrls['formant3_offset'],
            'formant4_offset': ctrls['formant4_offset'],
            # NEW: Vocal range parameters
            'vocal_range_min': ctrls['vocal_range_min'],
            'vocal_range_max': ctrls['vocal_range_max']
        }
        
        signal = self.refinement(signal)

        # Return audio signal, predicted mel, and vocal parameters
        return signal, predicted_mel, vocal_params