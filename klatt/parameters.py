"""
Parameter definitions for the Klatt speech synthesizer.
"""

from enum import Enum, auto
from typing import List, Dict, Optional, Union
from dataclasses import dataclass


class GlottalSourceType(Enum):
    """Types of glottal source signals."""
    IMPULSIVE = 0
    NATURAL = 1
    NOISE = 2


# Maximum number of oral formants
MAX_ORAL_FORMANTS = 6


@dataclass
class MainParams:
    """Parameters for the whole sound."""
    sample_rate: float
    glottal_source_type: GlottalSourceType


@dataclass
class FrameParams:
    """Parameters for a sound frame."""
    # Basic parameters
    duration: float                    # frame duration in seconds
    f0: float                          # fundamental frequency in Hz
    flutter_level: float               # F0 flutter level, 0 .. 1, typically 0.25
    open_phase_ratio: float            # relative length of the open phase of the glottis, 0 .. 1, typically 0.7
    breathiness_db: float              # breathiness in voicing (turbulence) in dB, positive to amplify or negative to attenuate
    tilt_db: float                     # spectral tilt for glottal source in dB. Attenuation at 3 kHz in dB. 0 = no tilt.
    gain_db: float                     # overall gain (output gain) in dB, positive to amplify, negative to attenuate, NaN for automatic gain control (AGC)
    agc_rms_level: float               # RMS level for automatic gain control (AGC), only relevant when gain_db is NaN
    
    # Formant parameters
    nasal_formant_freq: Optional[float]  # nasal formant frequency in Hz, or None
    nasal_formant_bw: Optional[float]    # nasal formant bandwidth in Hz, or None
    oral_formant_freq: List[Optional[float]]  # oral format frequencies in Hz, or None
    oral_formant_bw: List[Optional[float]]    # oral format bandwidths in Hz, or None

    # Cascade branch parameters
    cascade_enabled: bool              # true = cascade branch enabled
    cascade_voicing_db: float          # voicing amplitude for cascade branch in dB, positive to amplify or negative to attenuate
    cascade_aspiration_db: float       # aspiration (glottis noise) amplitude for cascade branch in dB, positive to amplify or negative to attenuate
    cascade_aspiration_mod: float      # amplitude modulation factor for aspiration in cascade branch, 0 = no modulation, 1 = maximum modulation
    nasal_antiformant_freq: Optional[float]  # nasal antiformant frequency in Hz, or None
    nasal_antiformant_bw: Optional[float]    # nasal antiformant bandwidth in Hz, or None

    # Parallel branch parameters
    parallel_enabled: bool             # true = parallel branch enabled
    parallel_voicing_db: float         # voicing amplitude for parallel branch in dB, positive to amplify or negative to attenuate
    parallel_aspiration_db: float      # aspiration (glottis noise) amplitude for parallel branch in dB, positive to amplify or negative to attenuate
    parallel_aspiration_mod: float     # amplitude modulation factor for aspiration in parallel branch, 0 = no modulation, 1 = maximum modulation
    frication_db: float                # frication noise level in dB
    frication_mod: float               # amplitude modulation factor for frication noise in parallel branch, 0 = no modulation, 1 = maximum modulation
    parallel_bypass_db: float          # parallel bypass level in dB, used to bypass differentiated glottal and frication signals around resonators F2 to F6
    nasal_formant_db: float            # nasal formant level in dB
    oral_formant_db: List[Optional[float]]  # oral format levels in dB, or None


@dataclass
class FrameState:
    """Variables of the currently active frame."""
    breathiness_lin: float            # linear breathiness level
    gain_lin: float                   # linear overall gain

    # Cascade branch
    cascade_voicing_lin: float        # linear voicing amplitude for cascade branch
    cascade_aspiration_lin: float     # linear aspiration amplitude for cascade branch

    # Parallel branch
    parallel_voicing_lin: float       # linear voicing amplitude for parallel branch
    parallel_aspiration_lin: float    # linear aspiration amplitude for parallel branch
    frication_lin: float              # linear frication noise level
    parallel_bypass_lin: float        # linear parallel bypass level


@dataclass
class PeriodState:
    """Variables of the currently active F0 period (aka glottal period)."""
    f0: float                         # modulated fundamental frequency for this period, in Hz, or 0
    period_length: int                # period length in samples
    open_phase_length: int            # open glottis phase length in samples
    
    # Per sample values
    position_in_period: int           # current sample position within F0 period
    lp_noise: float                   # LP filtered noise
