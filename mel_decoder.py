import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from decoder.conditioning_network import ConditioningNetwork
from decoder.harmonic_synthesizer import HarmonicSynthesizer

class MelDecoder(nn.Module):
    """
    DDSP-based singing vocal synthesis model.
    """
    def __init__(self, num_phonemes, num_singers, num_languages, 
                 n_mels=80, hop_length=240, num_harmonics=100, sample_rate=24000):
        super(MelDecoder, self).__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # Define embedding dimensions
        phoneme_embed_dim = 64
        singer_embed_dim = 32
        language_embed_dim = 32
        conditioning_hidden_dim = 128
        
        # Pre-calculate desired output dimension for conditioning network
        conditioning_output_dim = conditioning_hidden_dim
        
        # Embedding layers (efficient implementation)
        self.phoneme_embed = nn.Embedding(num_phonemes + 1, phoneme_embed_dim)
        self.singer_embed = nn.Embedding(num_singers, singer_embed_dim)
        self.language_embed = nn.Embedding(num_languages, language_embed_dim)
        
        # Conditioning network with explicit output dimension
        self.conditioning_network = ConditioningNetwork(
            n_mels=n_mels,
            num_phonemes=num_phonemes,
            hidden_dim=conditioning_hidden_dim,
            output_dim=conditioning_output_dim
        )
        
        # DDSP Components with correct input channels
        self.harmonic_synth = HarmonicSynthesizer(
            sample_rate=sample_rate,
            hop_length=hop_length,
            num_harmonics=num_harmonics,
            input_channels=conditioning_output_dim
        )
        
    def forward(self, mel, f0, phoneme_seq, singer_id, language_id):
        # preprocess dimension conditioning
        mel = mel.transpose(1, 2)  # now [B, n_mels, T]
        f0 = f0.unsqueeze(1)  # Make it [B, 1, T]
        
        # Apply embedding layers to convert integer indices to embeddings
        # For phonemes: convert from [B, T] to [B, 64, T]
        phoneme_embedded = self.phoneme_embed(phoneme_seq)  # [B, T, 64]
        phoneme_embedded = phoneme_embedded.transpose(1, 2)  # [B, 64, T]
        
        # Convert singer_id and language_id to embeddings
        singer_embedded = self.singer_embed(singer_id)  # [B, 32]
        language_embedded = self.language_embed(language_id)  # [B, 32]
        
        # Generate conditioning information from inputs with embeddings
        condition = self.conditioning_network(mel, f0, phoneme_embedded, 
                                            singer_embedded, language_embedded)
        
        # Generate harmonic component using optimized synthesizer
        harmonic_signal = self.harmonic_synth(f0, mel, condition)
        
        return harmonic_signal