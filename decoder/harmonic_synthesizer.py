import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from decoder.formant_processor import FormantProcessor
from decoder.spectral_processor import SpectralProcessor
from decoder.harmonic_processor import HarmonicProcessor
from decoder.noise_processor import NoiseProcessor
from decoder.phase_coherence_processor import PhaseCoherenceProcessor
from decoder.refiner_network import RefinementNetwork    

class HarmonicSynthesizer(nn.Module):
    """
    Enhanced DDSP-based Harmonic Synthesizer with improved spectral shaping
    for human vocal clarity, optimized to minimize upsampling operations.
    Includes noise component for modeling breathy and noisy vocal characteristics.
    Now with enhanced phase coherence for more natural vocal synthesis.
    """
    def __init__(self, sample_rate=24000, hop_length=240, num_harmonics=100, input_channels=128, noise_mix_ratio=0.2):
        super(HarmonicSynthesizer, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.num_harmonics = num_harmonics
        self.input_channels = input_channels
        self.noise_mix_ratio = noise_mix_ratio
        
        # Create harmonic indices tensor (registered as buffer)
        harmonic_indices = torch.arange(1, num_harmonics + 1).float().view(1, -1, 1)
        self.register_buffer('harmonic_indices', harmonic_indices)
        
        # Learnable harmonic amplitude network
        self.harmonic_processor = HarmonicProcessor(num_harmonics=num_harmonics, input_channels=input_channels)
        
        # Initialize the formant and spectral processors
        self.formant_processor = FormantProcessor(num_formants=5, input_channels=input_channels, sample_rate=sample_rate)
        self.spectral_processor = SpectralProcessor(num_harmonics=num_harmonics, input_channels=input_channels)
        
        # Initialize the noise processor
        self.noise_processor = NoiseProcessor(input_channels=input_channels)
        
        # Initialize the phase coherence processor (NEW)
        self.phase_processor = PhaseCoherenceProcessor(num_harmonics=num_harmonics, input_channels=input_channels)

        # Adaptive noise mixer network - learns when to apply more/less noise
        self.noise_mixer = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output between 0-1 for noise mixing ratio
        )

        self.refinement_network = RefinementNetwork(
            input_channels=self.input_channels,
            fft_size=1024,  # Single FFT size for stability
            hop_factor=4
        )
        
        # Register additional buffers for efficient computation
        self.register_buffer('hop_length_tensor', torch.tensor(hop_length, dtype=torch.float))
        self.register_buffer('default_noise_ratio', torch.tensor(noise_mix_ratio, dtype=torch.float))
        
    def forward(self, f0, mel, condition):
        # Get batch size and sequence length
        batch_size = mel.shape[0]
        time_steps = mel.size(2)
        
        audio_length = (time_steps * self.hop_length_tensor).long()

        # Generate base harmonic amplitudes from conditioning
        harmonic_amplitudes = self.harmonic_processor(condition)  # [B, num_harmonics, T]
        
        # Apply formant processing
        formant_amplitudes = self.formant_processor(condition, harmonic_amplitudes, self.num_harmonics)
        
        # Apply spectral processing
        enhanced_amplitudes = self.spectral_processor(condition, formant_amplitudes)
        
        # Get formant information for enhanced phase processing (NEW)
        # This assumes formant_processor exposes formant parameters
        formant_centers = None
        formant_bandwidths = None
        if hasattr(self.formant_processor, 'get_formant_params'):
            formant_centers, formant_bandwidths, _ = self.formant_processor.get_formant_params(condition)
        
        # ===== ONLY UPSAMPLE ESSENTIAL SIGNALS =====
        
        # 1. Prepare f0 for upsampling
        if f0.dim() == 2:  # If f0 is [B, T]
            f0_expanded = f0.unsqueeze(1)  # Make it [B, 1, T]
        else:
            f0_expanded = f0  # Already [B, 1, T]

        # 2. Perform the only two necessary upsamplings
        
        f0_upsampled = self._efficient_upsample(f0_expanded, audio_length)  # [B, 1, audio_length]
        enhanced_amplitudes_upsampled = self._efficient_upsample(enhanced_amplitudes, audio_length)  # [B, num_harmonics, audio_length]
        
        # ===== AUDIO GENERATION AT UPSAMPLED RATE =====
        
        # 1. Compute phase increments (radians per sample)
        phase_increments = 2 * math.pi * f0_upsampled / self.sample_rate  # [B, 1, audio_length]
        
        # 2. Compute cumulative phase (integrate frequency)
        phase = torch.cumsum(phase_increments, dim=2)  # [B, 1, audio_length]
        
        # 3. Apply phase coherence processing (NEW)
        phase_offsets, reset_points, reset_patterns = self.phase_processor(
            condition, f0_expanded, formant_centers, formant_bandwidths
        )
        
        # Upsample phase modifications
        phase_offsets_upsampled = self._efficient_upsample(phase_offsets, audio_length)
        reset_points_upsampled = self._efficient_upsample(reset_points, audio_length)
        reset_patterns_upsampled = self._efficient_upsample(reset_patterns, audio_length)
        
        # 4. Generate all harmonics at once with enhanced phase coherence
        base_harmonic_phases = phase * self.harmonic_indices  # [B, num_harmonics, audio_length]
        
        # Apply phase offsets to base phases
        modified_harmonic_phases = base_harmonic_phases + phase_offsets_upsampled
        
        # Apply phase resets at appropriate points
        final_harmonic_phases = self.phase_processor.apply_phase_resets(
            modified_harmonic_phases, 
            reset_points_upsampled, 
            reset_patterns_upsampled
        )
        
        # Generate harmonic signals with enhanced phase coherence
        harmonic_signals = torch.sin(final_harmonic_phases)  # [B, num_harmonics, audio_length]
        
        # 5. Apply enhanced amplitudes to harmonic signals
        weighted_harmonics = harmonic_signals * enhanced_amplitudes_upsampled  # [B, num_harmonics, audio_length]
        
        # 6. Sum all harmonics
        harmonic_signal = torch.sum(weighted_harmonics, dim=1, keepdim=True)  # [B, 1, audio_length]
        
        # ===== NOISE GENERATION AND MIXING =====
        
        # 1. Generate conditioned noise
        noise_signal = self.noise_processor(condition, audio_length)  # [B, 1, audio_length]
        
        # 2. Determine adaptive mixing ratio based on conditioning
        adaptive_mix_ratio = self.noise_mixer(condition)  # [B, 1, T]
        adaptive_mix_ratio = self._efficient_upsample(adaptive_mix_ratio, audio_length)  # [B, 1, audio_length]
        
        # Combine default ratio with adaptive ratio
        final_mix_ratio = self.default_noise_ratio * adaptive_mix_ratio
        
        # 3. Mix harmonic and noise signals
        # Ensure mix preserves overall energy
        harmonic_weight = 1.0 - final_mix_ratio
        output_signal = harmonic_weight * harmonic_signal + final_mix_ratio * noise_signal

        refined_signal = self.refinement_network(output_signal.squeeze(1), condition)
            
        return refined_signal  # [B, audio_length]
        #return output_signal.squeeze(1)  # [B, audio_length]
        
    def _efficient_upsample(self, tensor, target_len):
        """More efficient upsampling with reduced memory footprint"""
        # Ensure tensor is 3D for F.interpolate with mode='linear'
        orig_shape = tensor.shape
        squeezed = False
        
        if len(orig_shape) == 4:
            tensor = tensor.squeeze(1)  # Remove the second dimension
            squeezed = True
        
        # For small tensors, interpolate is fine
        if tensor.shape[2] * target_len < 1e7:  # Heuristic threshold
            upsampled = F.interpolate(
                tensor, 
                size=target_len, 
                mode='linear', 
                align_corners=False
            )
            
            # Restore original dimensionality if needed
            if squeezed:
                upsampled = upsampled.unsqueeze(1)
            return upsampled
        
        # For large tensors, use block processing
        batch_size, channels, time_steps = tensor.shape
        scale_factor = target_len / time_steps
        
        # Process in time blocks to save memory
        result = torch.zeros(batch_size, channels, target_len, device=tensor.device)
        block_size = min(1000, time_steps)  # Process 1000 frames at a time
        
        for block_start in range(0, time_steps, block_size):
            block_end = min(block_start + block_size, time_steps)
            block = tensor[:, :, block_start:block_end]
            
            # Calculate corresponding output indices
            out_start = int(block_start * scale_factor)
            out_end = min(int(block_end * scale_factor), target_len)
            
            # Interpolate this block
            upsampled_block = F.interpolate(
                block,
                size=out_end - out_start,
                mode='linear',
                align_corners=False
            )
            
            # Insert into result
            result[:, :, out_start:out_end] = upsampled_block
        
        # Restore original dimensionality if needed
        if squeezed:
            result = result.unsqueeze(1)
            
        return result