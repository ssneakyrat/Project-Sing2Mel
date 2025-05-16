import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from decoder.feature_processor import FeatureProcessor
from decoder.harmonic_oscillator import HarmonicOscillator
from decoder.core import scale_function, frequency_filter, upsample
from decoder.vocal_filter import vocal_frequency_filter
from decoder.phaser_network import PhaseAwareEnhancer

class SVS(nn.Module):
    """
    Lightweight DDSP-based singing voice synthesis model with streamlined
    direct linguistic feature to control parameter processing.
    Enhanced with singer and language mixture capabilities.
    """
    def __init__(self, 
                 num_phonemes, 
                 num_singers, 
                 num_languages,
                 n_mels=80, 
                 hop_length=256, 
                 sample_rate=44100,
                 num_harmonics=100, 
                 num_mag_harmonic=80,
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
        self.num_singers = num_singers
        self.num_languages = num_languages
        
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
        
        # Initialize the feature processor
        self.feature_processor = FeatureProcessor(
            phoneme_embed_dim=self.phoneme_embed_dim,
            singer_embed_dim=self.singer_embed_dim,
            language_embed_dim=self.language_embed_dim,
            hidden_dim=256,
            num_harmonics=num_harmonics,
            num_mag_harmonic=num_mag_harmonic,
            num_mag_noise=num_mag_noise,
            dropout=0.1
        )

        # Harmonic Synthesizer parameters
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, num_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        # Initialize harmonic synthesizer
        self.harmonic_synthesizer = HarmonicOscillator(
            sample_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)

        # Phase-aware enhancers for harmonic and noise components
        self.harmonic_phaser = PhaseAwareEnhancer(hidden_dim=512)
        self.noise_phaser = PhaseAwareEnhancer(hidden_dim=256)
    
    def get_weighted_embedding(self, weights, embedding_layer):
        """
        Calculate weighted embeddings from a weight vector
        
        Args:
            weights: Weights for each embedding [B, num_embeddings]
            embedding_layer: The embedding layer to use
            
        Returns:
            Weighted embeddings [B, embedding_dim]
        """
        # Get all embeddings
        all_embeddings = embedding_layer.weight  # [num_embeddings, embedding_dim]
        
        # Apply weights and sum
        # [B, num_embeddings] @ [num_embeddings, embedding_dim] -> [B, embedding_dim]
        return torch.matmul(weights, all_embeddings)
        
    def convert_to_weights(self, ids, num_classes, device):
        """
        Convert IDs to one-hot weight vectors
        
        Args:
            ids: IDs [B]
            num_classes: Number of possible classes
            device: Device to place tensor on
            
        Returns:
            One-hot weights [B, num_classes]
        """
        batch_size = ids.size(0)
        weights = torch.zeros(batch_size, num_classes, device=device)
        weights.scatter_(1, ids.unsqueeze(1), 1.0)
        return weights

    def forward(self, f0, phoneme_seq, singer_id=None, language_id=None, 
                singer_weights=None, language_weights=None, initial_phase=None):
        """
        Forward pass with support for both single IDs and mixture weights.
        
        Args:
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices) - optional if singer_weights provided
            language_id: Language IDs [B] (indices) - optional if language_weights provided
            singer_weights: Weights for each singer [B, num_singers] - optional
            language_weights: Weights for each language [B, num_languages] - optional
            initial_phase: Optional initial phase for the harmonic oscillator
            
        Returns:
            Audio signal [B, T*hop_length]
        """
        batch_size, n_frames = f0.shape[0], f0.shape[1]
        device = f0.device
        
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        
        # Handle singer embedding - either from ID or weights
        if singer_weights is not None:
            # Use provided weights
            singer_emb = self.get_weighted_embedding(singer_weights, self.singer_embed)
        elif singer_id is not None:
            # Convert single ID to one-hot weights if mixture required
            if hasattr(singer_id, 'shape') and len(singer_id.shape) > 1 and singer_id.shape[1] > 1:
                # Already in weight form
                singer_emb = self.get_weighted_embedding(singer_id, self.singer_embed)
            else:
                # Traditional single ID lookup
                singer_emb = self.singer_embed(singer_id)
        else:
            raise ValueError("Either singer_id or singer_weights must be provided")
        
        # Handle language embedding - either from ID or weights
        if language_weights is not None:
            # Use provided weights
            language_emb = self.get_weighted_embedding(language_weights, self.language_embed)
        elif language_id is not None:
            # Convert single ID to one-hot weights if mixture required
            if hasattr(language_id, 'shape') and len(language_id.shape) > 1 and language_id.shape[1] > 1:
                # Already in weight form
                language_emb = self.get_weighted_embedding(language_id, self.language_embed)
            else:
                # Traditional single ID lookup
                language_emb = self.language_embed(language_id)
        else:
            raise ValueError("Either language_id or language_weights must be provided")

        # Prepare f0 for feature processor
        f0_unsqueeze = f0.unsqueeze(2)  # [B, T, 1]
        
        # Get control parameters directly from feature processor
        ctrls = self.feature_processor(f0_unsqueeze, phoneme_emb, singer_emb, language_emb)

        # Process harmonic and noise parameters
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # Process F0 - make sure it's in Hz and properly shaped
        f0_unsqueeze = torch.clamp(f0_unsqueeze, min=0.0, max=1000.0)
        f0_unsqueeze[f0_unsqueeze < 80] = 0 + 1e-7  # Set unvoiced regions to 0

        # upsample
        pitch = upsample(f0_unsqueeze, self.block_size)
        
        # harmonic
        harmonic, final_phase = self.harmonic_synthesizer(pitch, initial_phase)
        harmonic = vocal_frequency_filter(
            harmonic, 
            src_param, 
            gender="neutral",  # Or dynamically set based on singer
            formant_emphasis=False,
            vocal_range_boost=False,
            breathiness=0,
            multi_resolution=False
        )

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = vocal_frequency_filter(
            noise, 
            noise_param, 
            gender="neutral",  # Or dynamically set based on singer
            formant_emphasis=False,
            vocal_range_boost=False,
            breathiness=0,
            multi_resolution=False
        )
        
        # Apply phase-aware enhancement
        harmonic = self.harmonic_phaser(harmonic)
        noise = self.noise_phaser(noise)

        # Final audio signal is the sum of harmonic and noise components
        signal = harmonic + noise

        return signal
        
    def get_singer_mixture(self, mixture_dict):
        """
        Helper method to create singer mixture weights from a dictionary
        
        Args:
            mixture_dict: Dictionary mapping singer IDs to weights
                         (e.g., {0: 0.7, 2: 0.3})
                         
        Returns:
            Tensor of normalized weights [num_singers]
        """
        weights = torch.zeros(self.num_singers)
        for id, weight in mixture_dict.items():
            weights[id] = weight
            
        # Normalize to sum to 1.0
        if weights.sum() > 0:
            weights = weights / weights.sum()
            
        return weights
        
    def get_language_mixture(self, mixture_dict):
        """
        Helper method to create language mixture weights from a dictionary
        
        Args:
            mixture_dict: Dictionary mapping language IDs to weights
                         (e.g., {0: 0.7, 1: 0.3})
                         
        Returns:
            Tensor of normalized weights [num_languages]
        """
        weights = torch.zeros(self.num_languages)
        for id, weight in mixture_dict.items():
            weights[id] = weight
            
        # Normalize to sum to 1.0
        if weights.sum() > 0:
            weights = weights / weights.sum()
            
        return weights