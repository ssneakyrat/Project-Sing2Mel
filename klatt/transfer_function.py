"""
Functions for calculating transfer function coefficients for the Klatt speech synthesizer.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Union

from klatt.parameters import MainParams, FrameParams, MAX_ORAL_FORMANTS, GlottalSourceType
from klatt.filters import LpFilter1, Resonator, AntiResonator, DifferencingFilter
from klatt.utils import db_to_lin
from klatt.generator import (
    set_tilt_filter, set_nasal_formant_casc, set_nasal_antiformant_casc,
    set_oral_formant_casc, set_nasal_formant_par, set_oral_formant_par
)


# Small epsilon value for numerical stability in polynomial operations
EPS = 1e-10


def multiply_fractions(a: List[float], b: List[float], eps: float = EPS) -> List[List[float]]:
    """
    Multiply two rational fractions.
    
    Args:
        a: First fraction as [numerator, denominator].
        b: Second fraction as [numerator, denominator].
        eps: Small value for numerical stability.
        
    Returns:
        Product of fractions as [numerator, denominator].
    """
    # Extract the numerator and denominator of each fraction
    num_a, den_a = a
    num_b, den_b = b
    
    # Multiply the numerators and denominators
    num_result = np.convolve(num_a, num_b)
    den_result = np.convolve(den_a, den_b)
    
    # Remove small coefficients for numerical stability
    num_result[np.abs(num_result) < eps] = 0
    den_result[np.abs(den_result) < eps] = 0
    
    return [num_result.tolist(), den_result.tolist()]


def add_fractions(a: List[List[float]], b: List[List[float]], eps: float = EPS) -> List[List[float]]:
    """
    Add two rational fractions.
    
    Args:
        a: First fraction as [numerator, denominator].
        b: Second fraction as [numerator, denominator].
        eps: Small value for numerical stability.
        
    Returns:
        Sum of fractions as [numerator, denominator].
    """
    # Extract the numerator and denominator of each fraction
    num_a, den_a = a
    num_b, den_b = b
    
    # Calculate the common denominator
    den_result = np.convolve(den_a, den_b)
    
    # Calculate the numerators over the common denominator
    term_a = np.convolve(num_a, den_b)
    term_b = np.convolve(num_b, den_a)
    
    # Add the numerators
    num_result = term_a + term_b
    
    # Remove small coefficients for numerical stability
    num_result[np.abs(num_result) < eps] = 0
    den_result[np.abs(den_result) < eps] = 0
    
    return [num_result.tolist(), den_result.tolist()]


def get_vocal_tract_transfer_function_coefficients(
    main_params: MainParams, 
    frame_params: FrameParams
) -> List[List[float]]:
    """
    Returns the polynomial coefficients of the overall filter transfer function in the z-plane.
    
    The returned array contains the top and bottom coefficients of the rational fraction, 
    ordered in ascending powers.
    
    Args:
        main_params: Main parameters for the synthesizer.
        frame_params: Frame parameters.
        
    Returns:
        Transfer function coefficients as [numerator, denominator].
    """
    # Start with voice source
    voice = [[1], [1]]
    
    # Apply tilt filter
    tilt_filter = LpFilter1(main_params.sample_rate)
    set_tilt_filter(tilt_filter, frame_params.tilt_db)
    tilt_trans = tilt_filter.get_transfer_function_coefficients()
    voice = multiply_fractions(voice, tilt_trans)
    
    # Calculate cascade and parallel branch transfer functions
    cascade_trans = (
        get_cascade_branch_transfer_function_coefficients(main_params, frame_params)
        if frame_params.cascade_enabled else [[0], [1]]
    )
    
    parallel_trans = (
        get_parallel_branch_transfer_function_coefficients(main_params, frame_params)
        if frame_params.parallel_enabled else [[0], [1]]
    )
    
    # Combine branches
    branches_trans = add_fractions(cascade_trans, parallel_trans)
    out = multiply_fractions(voice, branches_trans)
    
    # Apply output low-pass filter
    output_lp_filter = Resonator(main_params.sample_rate)
    output_lp_filter.set(0, main_params.sample_rate / 2)
    output_lp_trans = output_lp_filter.get_transfer_function_coefficients()
    out = multiply_fractions(out, output_lp_trans)
    
    # Apply overall gain
    gain_lin = db_to_lin(frame_params.gain_db if not math.isnan(frame_params.gain_db) else 0)
    out = multiply_fractions(out, [[gain_lin], [1]])
    
    return out


def get_cascade_branch_transfer_function_coefficients(
    main_params: MainParams, 
    frame_params: FrameParams
) -> List[List[float]]:
    """
    Returns the polynomial coefficients of the cascade branch transfer function in the z-plane.
    
    Args:
        main_params: Main parameters for the synthesizer.
        frame_params: Frame parameters.
        
    Returns:
        Transfer function coefficients as [numerator, denominator].
    """
    # Start with cascade voicing gain
    cascade_voicing_lin = db_to_lin(frame_params.cascade_voicing_db)
    v = [[cascade_voicing_lin], [1]]
    
    # Apply nasal antiformant
    nasal_antiformant_casc = AntiResonator(main_params.sample_rate)
    set_nasal_antiformant_casc(nasal_antiformant_casc, frame_params)
    nasal_antiformant_trans = nasal_antiformant_casc.get_transfer_function_coefficients()
    v = multiply_fractions(v, nasal_antiformant_trans)
    
    # Apply nasal formant
    nasal_formant_casc = Resonator(main_params.sample_rate)
    set_nasal_formant_casc(nasal_formant_casc, frame_params)
    nasal_formant_trans = nasal_formant_casc.get_transfer_function_coefficients()
    v = multiply_fractions(v, nasal_formant_trans)
    
    # Apply oral formants
    for i in range(MAX_ORAL_FORMANTS):
        oral_formant_casc = Resonator(main_params.sample_rate)
        set_oral_formant_casc(oral_formant_casc, frame_params, i)
        oral_formant_casc_trans = oral_formant_casc.get_transfer_function_coefficients()
        v = multiply_fractions(v, oral_formant_casc_trans)
        
    return v


def get_parallel_branch_transfer_function_coefficients(
    main_params: MainParams, 
    frame_params: FrameParams
) -> List[List[float]]:
    """
    Returns the polynomial coefficients of the parallel branch transfer function in the z-plane.
    
    Args:
        main_params: Main parameters for the synthesizer.
        frame_params: Frame parameters.
        
    Returns:
        Transfer function coefficients as [numerator, denominator].
    """
    # Start with parallel voicing gain
    parallel_voicing_lin = db_to_lin(frame_params.parallel_voicing_db)
    source = [[parallel_voicing_lin], [1]]
    
    # Apply differencing filter
    differencing_filter_par = DifferencingFilter()
    differencing_filter_trans = differencing_filter_par.get_transfer_function_coefficients()
    source2 = multiply_fractions(source, differencing_filter_trans)
    
    # Start with zero
    v = [[0], [1]]
    
    # Add nasal formant (applied to source)
    nasal_formant_par = Resonator(main_params.sample_rate)
    set_nasal_formant_par(nasal_formant_par, frame_params)
    nasal_formant_trans = nasal_formant_par.get_transfer_function_coefficients()
    v = add_fractions(v, multiply_fractions(source, nasal_formant_trans))
    
    # Add oral formants
    for i in range(MAX_ORAL_FORMANTS):
        oral_formant_par = Resonator(main_params.sample_rate)
        set_oral_formant_par(oral_formant_par, main_params, frame_params, i)
        oral_formant_trans = oral_formant_par.get_transfer_function_coefficients()
        
        # F1 is applied to source, F2 to F6 are applied to source2
        formant_in = source if i == 0 else source2
        formant_out = multiply_fractions(formant_in, oral_formant_trans)
        
        # Apply alternating sign
        alternating_sign = 1 if i % 2 == 0 else -1
        v2 = multiply_fractions(formant_out, [[alternating_sign], [1]])
        
        v = add_fractions(v, v2)
    
    # Add parallel bypass (applied to source2)
    parallel_bypass_lin = db_to_lin(frame_params.parallel_bypass_db)
    parallel_bypass = multiply_fractions(source2, [[parallel_bypass_lin], [1]])
    v = add_fractions(v, parallel_bypass)
    
    return v