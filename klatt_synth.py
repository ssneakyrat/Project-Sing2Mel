"""
Main module for the Klatt speech synthesizer.

This module provides a simple interface to the Klatt synthesizer.
"""

import numpy as np
from typing import List, Optional, Dict, Union, Tuple

from klatt.parameters import (
    GlottalSourceType, MAX_ORAL_FORMANTS,
    MainParams, FrameParams
)
from klatt.generator import generate_sound
from klatt.transfer_function import get_vocal_tract_transfer_function_coefficients
from utils.save_audio import save_wav

class KlattSynthesizer:
    """Main class for the Klatt speech synthesizer."""
    
    def __init__(self, sample_rate: float = 44100, glottal_source_type: GlottalSourceType = GlottalSourceType.NATURAL):
        """
        Initialize the Klatt synthesizer.
        
        Args:
            sample_rate: Sample rate in Hz.
            glottal_source_type: Type of glottal source to use.
        """
        self.main_params = MainParams(
            sample_rate=sample_rate,
            glottal_source_type=glottal_source_type
        )
    
    def create_frame_params(self, **kwargs) -> FrameParams:
        """
        Create a FrameParams object with default values that can be overridden.
        
        Args:
            **kwargs: Parameter values to override defaults.
            
        Returns:
            FrameParams object with specified parameters.
        """
        # Default values for a neutral vowel
        defaults = {
            'duration': 0.5,  # seconds
            'f0': 120.0,  # Hz
            'flutter_level': 0.25,
            'open_phase_ratio': 0.7,
            'breathiness_db': -25.0,
            'tilt_db': 0.0,
            'gain_db': float('nan'),  # Use AGC
            'agc_rms_level': 0.1,
            
            'nasal_formant_freq': None,
            'nasal_formant_bw': None,
            'oral_formant_freq': [500.0, 1500.0, 2500.0, 3500.0, 4500.0, 5500.0],
            'oral_formant_bw': [60.0, 90.0, 150.0, 200.0, 250.0, 300.0],
            
            'cascade_enabled': True,
            'cascade_voicing_db': 0.0,
            'cascade_aspiration_db': -25.0,
            'cascade_aspiration_mod': 0.5,
            'nasal_antiformant_freq': None,
            'nasal_antiformant_bw': None,
            
            'parallel_enabled': True,
            'parallel_voicing_db': -6.0,
            'parallel_aspiration_db': -25.0,
            'parallel_aspiration_mod': 0.5,
            'frication_db': -60.0,
            'frication_mod': 0.5,
            'parallel_bypass_db': -6.0,
            'nasal_formant_db': 0.0,
            'oral_formant_db': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        
        # Override defaults with provided values
        params = defaults.copy()
        params.update(kwargs)
        
        # Convert None to float('nan') for numeric fields that can be NaN
        for key in ['gain_db', 'nasal_formant_freq', 'nasal_formant_bw', 
                    'nasal_antiformant_freq', 'nasal_antiformant_bw']:
            if params[key] is None:
                params[key] = float('nan')
        
        # Convert any None values in lists to float('nan')
        for key in ['oral_formant_freq', 'oral_formant_bw', 'oral_formant_db']:
            params[key] = [float('nan') if x is None else x for x in params[key]]
            
        return FrameParams(**params)
    
    def synthesize(self, frame_params_list: List[FrameParams]) -> np.ndarray:
        """
        Synthesize speech using the provided frame parameters.
        
        Args:
            frame_params_list: List of frame parameters.
            
        Returns:
            Generated audio samples as numpy array.
        """
        return generate_sound(self.main_params, frame_params_list)
    
    def get_transfer_function(self, frame_params: FrameParams) -> List[List[float]]:
        """
        Get the transfer function coefficients for the specified frame parameters.
        
        Args:
            frame_params: Frame parameters.
            
        Returns:
            Transfer function coefficients as [numerator, denominator].
        """
        return get_vocal_tract_transfer_function_coefficients(self.main_params, frame_params)


# Example usage
if __name__ == "__main__":
    # Create a synthesizer
    klatt = KlattSynthesizer(sample_rate=44100)
    
    # Create frame parameters for an 'a' vowel
    a_vowel = klatt.create_frame_params(
        duration=0.5,
        f0=120.0,
        oral_formant_freq=[800, 1200, 2500, 3500, 4500, 5500],
        oral_formant_bw=[80, 90, 150, 200, 250, 300]
    )
    
    # Create frame parameters for an 'i' vowel
    i_vowel = klatt.create_frame_params(
        duration=0.5,
        f0=130.0,
        oral_formant_freq=[280, 2250, 2800, 3500, 4500, 5500],
        oral_formant_bw=[60, 90, 150, 200, 250, 300]
    )
    
    # Synthesize the vowel sequence
    audio = klatt.synthesize([a_vowel, i_vowel])
    
    save_wav( 'test.wav', audio)
        
    print("Audio shape:", audio.shape)
    print("Audio min/max:", audio.min(), audio.max())
