import torch
import torch.nn as nn
from decoder.feature_extractor import FeatureExtractor
from decoder.harmonic_oscillator import HarmonicOscillator
from decoder.wave_generator_oscillator import WaveGeneratorOscillator
from decoder.core import unit_to_hz2, scale_function, upsample, frequency_filter

class MelDecoder(nn.Module):
    """
    Lightweight DDSP-based singing voice synthesis model.
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
                 n_fft=1024):
        super(MelDecoder, self).__init__()
        
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
        
        # Feature extractor output projection
        '''
        split_map = {
            'A': 1,
            'amplitudes': num_harmonics,
            'harmonic_magnitude': num_mag_harmonic,
            'noise_magnitude': num_mag_noise
        }
        '''

        split_map = {
            'harmonic_magnitude': num_mag_harmonic,
            'noise_magnitude': num_mag_noise
        }

        # Initialize feature extractor with proper dimensions
        self.feature_extractor = FeatureExtractor(
            input_channel=n_mels,
            output_splits=split_map,
            phoneme_dim=self.phoneme_embed_dim,
            singer_dim=self.singer_embed_dim,
            language_dim=self.language_embed_dim
        )

        # Harmonic Synthesizer
        #self.harmonic_synthesizer = HarmonicOscillator(sample_rate)
        
        # Harmonic Synthsizer
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, num_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        self.harmonic_synthsizer = WaveGeneratorOscillator(
            sample_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)

    def forward(self, mel, f0, phoneme_seq, singer_id, language_id, initial_phase=None):
        """
        Optimized forward pass with dynamic formant modeling.
        
        Args:
            mel: Mel-spectrogram [B, T, n_mels]
            f0: Fundamental frequency trajectory [B, T]
            phoneme_seq: Phoneme sequence [B, T] (indices)
            singer_id: Singer IDs [B] (indices)
            language_id: Language IDs [B] (indices)
            initial_phase: Optional initial phase for the harmonic oscillator
            
        Returns:
            Audio signal [B, T*hop_length]
        """
        # Apply embeddings
        phoneme_emb = self.phoneme_embed(phoneme_seq)  # [B, T, phoneme_dim]
        singer_emb = self.singer_embed(singer_id)      # [B, singer_dim]
        language_emb = self.language_embed(language_id) # [B, language_dim]

        # Get control parameters from feature extractor
        ctrls = self.feature_extractor(mel, f0, phoneme_emb, singer_emb, language_emb)

        # Process F0
        '''
        f0_unsqueeze = f0.unsqueeze(2)  
        f0_unit = torch.sigmoid(f0_unsqueeze)
        f0 = unit_to_hz2(f0_unit, hz_min=80.0, hz_max=1000.0)
        f0[f0 < 80] = 0
        pitch = f0
        '''

        # Assuming f0 is already in Hz from Parselmouth
        f0_unsqueeze = f0.unsqueeze(2)  
        # Optional: Clip to desired range if needed
        f0_unsqueeze = torch.clamp(f0_unsqueeze, min=0.0, max=1000.0)
        # Set frequencies below threshold to 0 (for unvoiced)
        f0_unsqueeze[f0_unsqueeze < 80] = 0
        pitch = f0_unsqueeze

        '''
        # Process control parameters
        A = scale_function(ctrls['A'])
        amplitudes = scale_function(ctrls['amplitudes'])
        harmonic_mag = scale_function(ctrls['harmonic_magnitude'])
        noise_mag = scale_function(ctrls['noise_magnitude'])

        # Normalize amplitudes to distribution and apply amplitude scaling
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= A * 2 - 1
        
        # Upsample control signals to audio rate
        pitch = upsample(pitch, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)

        # Generate harmonic component
        harmonic, final_phase = self.harmonic_synthesizer(
            pitch, amplitudes, initial_phase)
        harmonic = frequency_filter(harmonic, harmonic_mag)
        
        # Generate noise component
        noise = torch.rand_like(harmonic).to(noise_mag) * 2 - 1
        noise = frequency_filter(noise, noise_mag)
        
        # Combine harmonic and noise components
        signal = harmonic + noise
        '''
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # exciter signal
        #B, n_frames, _ = pitch.shape

        # upsample
        pitch = upsample(pitch, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(pitch, initial_phase)
        harmonic = frequency_filter(
                        harmonic,
                        src_param)

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal