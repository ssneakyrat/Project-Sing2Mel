import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SignalProcessor(nn.Module):
    """
    Handles all DSP signal processing for expressive control effects.
    Applies vibrato, tension, vocal fry, and breathiness to audio signals.
    """
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Tension-related spectral shaping parameters
        self.tension_spectral_tilt = nn.Parameter(
            torch.tensor([0.5]), requires_grad=True
        )
        
        self.tension_formant_emphasis = nn.Parameter(
            torch.tensor([0.3]), requires_grad=True
        )
        
        self.tension_harmonic_distribution = nn.Parameter(
            torch.ones(32), requires_grad=True
        )
        
        self.tension_attack_sharpness = nn.Parameter(
            torch.tensor([0.4]), requires_grad=True
        )
    
    def apply_vibrato(self, f0, time_idx, expressive_params):
        """
        Apply vibrato to F0
        
        Args:
            f0: Input F0 trajectory [B, T, 1]
            time_idx: Time indices for each frame [B, T]
            expressive_params: Dictionary of expressive parameters
            
        Returns:
            F0 with vibrato applied [B, T, 1]
        """
        # Extract vibrato parameters
        rate = expressive_params['vibrato_rate']    # [B, T_params]
        depth = expressive_params['vibrato_depth']  # [B, T_params]
        phase = expressive_params['vibrato_phase']  # [B, T_params]
        
        # Get shapes
        batch_size, n_frames = f0.shape[0], f0.shape[1]
        params_frames = rate.shape[1]
        
        # Handle shape mismatches - ensure all tensors have compatible shapes
        if n_frames != params_frames:
            # Interpolate parameters to match f0 timeline
            rate = F.interpolate(
                rate.unsqueeze(1), 
                size=n_frames, 
                mode='linear',
                align_corners=False
            ).squeeze(1)
            
            depth = F.interpolate(
                depth.unsqueeze(1), 
                size=n_frames, 
                mode='linear',
                align_corners=False
            ).squeeze(1)
            
            phase = F.interpolate(
                phase.unsqueeze(1), 
                size=n_frames, 
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        # Make sure time_idx has the right shape
        if time_idx.shape[1] != n_frames:
            time_idx = torch.arange(n_frames, device=f0.device).float().unsqueeze(0)
            time_idx = time_idx.expand(batch_size, -1) / 100.0
        
        # Combined calculation with proper broadcasting
        vibrato_modulation = torch.sin(2 * np.pi * rate.unsqueeze(-1) * time_idx.unsqueeze(-1) + phase.unsqueeze(-1))
        f0_with_vibrato = f0 * (1.0 + (depth.unsqueeze(-1) / 1200.0) * vibrato_modulation)
        
        return f0_with_vibrato
    
    def calculate_tension_derived_params(self, tension):
        """
        Calculate derived parameters from tension
        
        Args:
            tension: Tension parameter [B, T, 1]
            
        Returns:
            Dictionary of tension-derived parameters
        """
        batch_size, seq_len, _ = tension.shape
        device = tension.device
        
        # Spectral tilt
        spectral_tilt = -0.3 + tension * self.tension_spectral_tilt * 1.2
        
        # Formant emphasis/bandwidth
        formant_emphasis = 0.5 + tension * self.tension_formant_emphasis * 1.5
        
        # Harmonic distribution
        harmonic_dist = self.tension_harmonic_distribution.unsqueeze(0).unsqueeze(0)
        harmonic_dist = harmonic_dist.expand(batch_size, seq_len, -1)
        harmonic_indices = torch.arange(32, device=device).float() / 8.0
        harmonic_scaling = 1.0 + (tension * 2 - 1) * torch.sin(harmonic_indices * math.pi)
        harmonic_dist = harmonic_dist * harmonic_scaling.unsqueeze(0).unsqueeze(0)
        
        # Attack characteristics
        attack_sharpness = tension * self.tension_attack_sharpness
        
        return {
            'spectral_tilt': spectral_tilt,
            'formant_emphasis': formant_emphasis,
            'harmonic_distribution': harmonic_dist,
            'attack_sharpness': attack_sharpness
        }
    
    def interpolate_to_audio_rate(self, param, audio_length):
        """
        Interpolate a parameter to audio rate
        
        Args:
            param: Parameter tensor [B, T, C]
            audio_length: Target length in samples
            
        Returns:
            Interpolated parameter at audio rate
        """
        if param.shape[1] != audio_length:
            param_audio = F.interpolate(
                param.transpose(1, 2),
                size=audio_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            return param_audio
        return param
    
    def interpolate_parameters(self, expressive_params, audio_length):
        """
        Interpolate all parameters to audio rate only, deferring STFT rate interpolation
        
        Args:
            expressive_params: Dictionary of expressive parameters
            audio_length: Length of audio in samples
            
        Returns:
            Dictionary with parameters interpolated to audio rate
        """
        # Create new dictionary to hold interpolated parameters
        synced_params = {}
        
        # Interpolate parameters to audio rate only
        for param_name in ['tension', 'breathiness', 'vocal_fry']:
            param = expressive_params[param_name]
            param_audio = self.interpolate_to_audio_rate(param, audio_length)
            synced_params[param_name] = param_audio
        
        # Copy vibrato parameters directly
        synced_params['vibrato_rate'] = expressive_params['vibrato_rate']
        synced_params['vibrato_depth'] = expressive_params['vibrato_depth']
        synced_params['vibrato_phase'] = expressive_params['vibrato_phase']
        
        # Calculate tension-derived parameters at audio rate
        synced_params['tension_derived'] = self.calculate_tension_derived_params(synced_params['tension'])
        
        return synced_params
    
    def process_audio(self, harmonic, noise, f0, time_idx, expressive_params):
        """
        Combined processing pipeline for all audio effects
        
        Args:
            harmonic: Harmonic component [B, T_audio]
            noise: Noise component [B, T_audio]
            f0: Fundamental frequency [B, T_f0, 1]
            time_idx: Time indices [B, T_f0]
            expressive_params: Dictionary of expressive parameters [B, T_params]
            
        Returns:
            Processed audio signal [B, T_audio], F0 with vibrato [B, T_f0, 1]
        """
        batch_size, audio_length = harmonic.shape
        
        # 1. Apply vibrato to F0 (will handle shape mismatches internally)
        f0_with_vibrato = self.apply_vibrato(f0, time_idx, expressive_params)
        
        # 2. Interpolate all parameters at once to both audio and STFT rates
        # This is the key optimization - do all interpolations in one place
        synced_params = self.interpolate_parameters(expressive_params, audio_length)
        
        # 3. Apply combined spectral and time-domain effects with synced parameters
        harmonic_processed = self.apply_combined_effects(harmonic, f0_with_vibrato, synced_params)
        
        # 4. Apply breathiness with synced parameters
        output = self.apply_breathiness(harmonic_processed, noise, synced_params)
        
        return output, f0_with_vibrato
    
    def apply_combined_effects(self, harmonic, f0, synced_params):
        """
        Combined application of tension and vocal fry effects
        
        Args:
            harmonic: Harmonic component [B, T]
            f0: Fundamental frequency with vibrato [B, T, 1]
            synced_params: Dictionary of pre-interpolated parameters
            
        Returns:
            Modified harmonic signal [B, T]
        """
        # Extract parameters (already at audio rate from interpolate_parameters)
        tension = synced_params['tension']  # [B, T_audio, 1]
        tension_derived = synced_params['tension_derived']
        vocal_fry = synced_params['vocal_fry']  # [B, T_audio, 1]
        
        batch_size, audio_length = harmonic.shape
        device = harmonic.device
        
        # Get tension at audio rate, handling different tensor formats
        if tension.dim() == 3:
            tension_expanded = tension.squeeze(-1)
        else:  # dim == 2
            tension_expanded = tension
            
        # Apply STFT for frequency-domain processing
        n_fft = 1024
        hop_length = 256
        
        harmonic_stft = torch.stft(
            harmonic, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=torch.hann_window(n_fft).to(device),
            return_complex=True
        )
        
        # Get magnitude and phase
        magnitude = torch.abs(harmonic_stft)
        phase = torch.angle(harmonic_stft)
        
        # Calculate frequency bin indices
        num_frames = harmonic_stft.shape[1]
        num_bins = harmonic_stft.shape[2]
        
        # Interpolate tension-derived parameters to STFT frame rate
        # Do this interpolation now that we know the actual STFT dimensions
        spectral_tilt = F.interpolate(
            tension_derived['spectral_tilt'].transpose(1, 2),
            size=num_frames,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        formant_emphasis = F.interpolate(
            tension_derived['formant_emphasis'].transpose(1, 2),
            size=num_frames,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        attack_sharpness = F.interpolate(
            tension_derived['attack_sharpness'].transpose(1, 2),
            size=num_frames,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        # Also interpolate tension to STFT frame rate for harmonic emphasis
        tension_stft = F.interpolate(
            tension.transpose(1, 2) if tension.dim() == 3 else tension.unsqueeze(1),
            size=num_frames,
            mode='linear',
            align_corners=False
        ).transpose(1, 2) if tension.dim() == 3 else F.interpolate(
            tension.unsqueeze(1),
            size=num_frames,
            mode='linear',
            align_corners=False
        ).squeeze(1)
        
        # Create spectral tilt modifier 
        freq_bins = torch.arange(num_bins, device=device).float() / num_bins
        
        # Apply spectral tilt: Higher tension = boosted high frequencies
        tilt_factor = torch.exp(spectral_tilt * freq_bins.unsqueeze(0).unsqueeze(0) * 2)
        magnitude = magnitude * tilt_factor
        
        # Apply formant modifications
        is_peak = torch.zeros_like(magnitude, dtype=torch.bool)
        is_peak[:, :, 1:-1] = (magnitude[:, :, 1:-1] > magnitude[:, :, :-2]) & \
                              (magnitude[:, :, 1:-1] > magnitude[:, :, 2:])
        
        # Apply emphasis to peaks (sharpen formants with higher tension)
        formant_gain = torch.ones_like(magnitude)
        formant_emphasis_expanded = formant_emphasis.squeeze(-1).unsqueeze(-1).expand(-1, -1, magnitude.shape[2])
        formant_gain[is_peak] = 1.0 + formant_emphasis_expanded[is_peak] * 0.5
        magnitude = magnitude * formant_gain
        
        # Apply harmonic distribution modifications
        harmonic_dist_expanded = torch.ones((batch_size, num_frames, num_bins), device=device)
        freq_bins = torch.linspace(0, 1, num_bins, device=device)
        
        # Ensure tension_factor has the right shape for broadcasting
        if isinstance(tension_stft, torch.Tensor) and tension_stft.dim() == 3:
            tension_factor = tension_stft.mean(dim=2, keepdim=True)
        else:
            tension_factor = tension_stft.unsqueeze(-1) if tension_stft.dim() == 2 else tension_stft
            
        harmonic_emphasis = 1.0 + tension_factor * (torch.sin(freq_bins.unsqueeze(0).unsqueeze(0) * 3 * math.pi) * 0.5)
        harmonic_dist_expanded = harmonic_dist_expanded * harmonic_emphasis
        magnitude = magnitude * harmonic_dist_expanded
        
        # Vectorize attack processing
        env = torch.sum(magnitude, dim=2)
        env_diff = torch.zeros_like(env)
        env_diff[:, 1:] = torch.clamp(env[:, 1:] - env[:, :-1], min=0)
        is_attack = (env_diff > 0.1 * torch.max(env_diff, dim=1, keepdim=True)[0])
        
        # Create attack gain profiles for all frames at once
        attack_gain = torch.ones_like(magnitude)
        
        # Vectorized attack processing
        for frame_offset in range(min(5, num_frames)):
            if frame_offset > 0:
                # Calculate which frames have attacks with this offset
                attack_frames = is_attack[:, :-frame_offset] if frame_offset > 0 else is_attack
                
                # Only process if there are any attacks
                if torch.any(attack_frames):
                    # Create a mask for frames with attacks
                    attack_mask = torch.zeros_like(is_attack, dtype=torch.float)
                    attack_mask[:, frame_offset:] = attack_frames.float()
                    
                    # Calculate gain based on attack sharpness and offset
                    attack_width_tensor = torch.clamp(5 * (1 - attack_sharpness.squeeze(-1)), min=1)
                    gain_factor = 1.0 + (1.0 - frame_offset / attack_width_tensor) * attack_sharpness.squeeze(-1) * 0.5
                    
                    # Apply gain where attacks are present
                    for b in range(batch_size):
                        frames_with_attacks = torch.where(attack_mask[b] > 0)[0]
                        if frames_with_attacks.shape[0] > 0:
                            attack_gain[b, frames_with_attacks] = gain_factor[b, frames_with_attacks-frame_offset].unsqueeze(-1)
        
        magnitude = magnitude * attack_gain
        
        # Convert back to complex STFT
        modified_stft = torch.polar(magnitude, phase)
        
        # Convert back to time domain
        modified_harmonic = torch.istft(
            modified_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft).to(device),
            length=audio_length
        )
        
        # Apply nonlinear waveshaping based on tension
        nonlinear_amount = tension_expanded * 0.2
        nonlinear_harmonic = torch.tanh(modified_harmonic * (1.0 + nonlinear_amount))
        output = modified_harmonic * (1.0 - nonlinear_amount) + nonlinear_harmonic * nonlinear_amount
        
        # Get vocal_fry at audio rate
        if vocal_fry.dim() == 3:
            vocal_fry_expanded = vocal_fry.squeeze(-1)
        else:  # dim == 2
            vocal_fry_expanded = vocal_fry
        
        # Generate pulse rates (20-60 Hz)
        pulse_rate = (20 + 40 * vocal_fry.mean(dim=1 if vocal_fry.dim() == 2 else (1, 2))).view(batch_size, 1)  # Hz
        samples_per_pulse = (self.sample_rate / pulse_rate).long()
        
        # Create modulation signal for fry effect - vectorized
        fry_mod = torch.ones_like(output)
        
        for b in range(batch_size):
            # Create pulse pattern more efficiently
            steps = max(1, samples_per_pulse[b].item())
            num_pulses = audio_length // steps + 1
            
            # Generate random amplitudes for the entire sequence
            random_amps = torch.rand(num_pulses, device=device) * 0.5 + 0.5
            
            # Create indices directly
            indices = torch.arange(0, audio_length, steps, device=device)
            indices = indices[indices < audio_length]
            
            # Create the pattern once and fill it
            pattern = torch.ones(audio_length, device=device)
            pattern[indices] = random_amps[:indices.shape[0]]
            
            # Apply the pattern
            fry_mod[b] = pattern
        
        # Smooth the modulation signal
        fry_mod = torch.nn.functional.pad(fry_mod.unsqueeze(1), (2, 2), 'replicate')
        fry_mod = torch.nn.functional.avg_pool1d(fry_mod, 5, stride=1).squeeze(1)
        
        # Apply fry effect
        fry_effect = output * fry_mod
        output_with_fry = (1.0 - vocal_fry_expanded) * output + vocal_fry_expanded * fry_effect
        
        return output_with_fry
    
    def apply_breathiness(self, harmonic, noise, synced_params):
        """
        Apply breathiness control
        
        Args:
            harmonic: Processed harmonic component [B, T]
            noise: Noise component [B, T]
            synced_params: Dictionary of pre-interpolated parameters
            
        Returns:
            Mixed signal with breathiness control [B, T]
        """
        # Use pre-interpolated breathiness parameter directly - no need to interpolate again!
        breathiness = synced_params['breathiness']
        
        batch_size, audio_length = harmonic.shape
        
        # Get breathiness at audio rate, handling different tensor formats
        if breathiness.dim() == 3:
            breathiness_expanded = breathiness.squeeze(-1)
        else:  # dim == 2
            breathiness_expanded = breathiness

        # Mix harmonic and noise with breathiness control
        output = (1 - breathiness_expanded) * harmonic + breathiness_expanded * noise
        
        return output