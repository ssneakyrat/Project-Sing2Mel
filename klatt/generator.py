"""
Main generator class for the Klatt speech synthesizer.
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Union

from klatt.parameters import (
    GlottalSourceType, MAX_ORAL_FORMANTS,
    MainParams, FrameParams, FrameState, PeriodState
)
from klatt.filters import LpFilter1, Resonator, AntiResonator, DifferencingFilter
from klatt.glottal_sources import ImpulsiveGlottalSource, NaturalGlottalSource
from klatt.noise_sources import get_white_noise, LpNoiseSource
from klatt.utils import perform_frequency_modulation, db_to_lin, adjust_signal_gain


class Generator:
    """Sound generator controller."""

    def __init__(self, main_params: MainParams):
        """
        Initialize the generator.

        Args:
            main_params: Main parameters for the synthesizer.
        """
        self.main_params = main_params
        self.frame_params = None  # Currently active frame parameters
        self.new_frame_params = None  # New frame parameters for start of next F0 period
        self.frame_state = FrameState(
            breathiness_lin=0,
            gain_lin=0,
            cascade_voicing_lin=0,
            cascade_aspiration_lin=0,
            parallel_voicing_lin=0,
            parallel_aspiration_lin=0,
            frication_lin=0,
            parallel_bypass_lin=0
        )
        self.period_state = None  # F0 period state variables
        self.abs_position = 0  # Current absolute sample position
        self.tilt_filter = LpFilter1(main_params.sample_rate)
        self.output_lp_filter = Resonator(main_params.sample_rate)
        self.output_lp_filter.set(0, main_params.sample_rate / 2)
        self.flutter_time_offset = random.random() * 1000

        # Initialize glottal source
        self.init_glottal_source()

        # Create noise sources
        self.aspiration_source_casc = LpNoiseSource(main_params.sample_rate)
        self.aspiration_source_par = LpNoiseSource(main_params.sample_rate)
        self.frication_source_par = LpNoiseSource(main_params.sample_rate)

        # Initialize cascade branch variables
        self.nasal_formant_casc = Resonator(main_params.sample_rate)
        self.nasal_antiformant_casc = AntiResonator(main_params.sample_rate)
        self.oral_formant_casc = [Resonator(main_params.sample_rate) for _ in range(MAX_ORAL_FORMANTS)]

        # Initialize parallel branch variables
        self.nasal_formant_par = Resonator(main_params.sample_rate)
        self.oral_formant_par = [Resonator(main_params.sample_rate) for _ in range(MAX_ORAL_FORMANTS)]
        self.differencing_filter_par = DifferencingFilter()

    def generate_frame(self, frame_params: FrameParams, out_buf: np.ndarray) -> None:
        """
        Generates a frame of the sound.

        Args:
            frame_params: Frame parameters.
            out_buf: Output buffer to fill with generated samples.
        """
        if frame_params is self.frame_params:
            raise ValueError("FrameParams structure must not be re-used.")
            
        self.new_frame_params = frame_params
        
        for out_pos in range(len(out_buf)):
            if not self.period_state or self.period_state.position_in_period >= self.period_state.period_length:
                self.start_new_period()
                
            out_buf[out_pos] = self.compute_next_output_signal_sample()
            self.period_state.position_in_period += 1
            self.abs_position += 1

        # Automatic gain control (AGC)
        if math.isnan(frame_params.gain_db):
            adjust_signal_gain(out_buf, frame_params.agc_rms_level)

    def compute_next_output_signal_sample(self) -> float:
        """
        Compute the next output signal sample.

        Returns:
            The next output sample value.
        """
        frame_params = self.frame_params
        frame_state = self.frame_state
        period_state = self.period_state
        
        # Get base voice signal
        voice = self.glottal_source()
        
        # Apply spectral tilt
        voice = self.tilt_filter.step(voice)
        
        # Add breathiness (turbulence) if within glottal open phase
        if period_state.position_in_period < period_state.open_phase_length:
            voice += get_white_noise() * frame_state.breathiness_lin
            
        # Calculate outputs from both branches
        cascade_out = self.compute_cascade_branch(voice) if frame_params.cascade_enabled else 0
        parallel_out = self.compute_parallel_branch(voice) if frame_params.parallel_enabled else 0
        
        # Combine and apply output filter and gain
        out = cascade_out + parallel_out
        out = self.output_lp_filter.step(out)
        out *= frame_state.gain_lin
        
        return out

    def compute_cascade_branch(self, voice: float) -> float:
        """
        Compute the cascade branch output.

        Args:
            voice: Input voice signal.

        Returns:
            Cascade branch output value.
        """
        frame_params = self.frame_params
        frame_state = self.frame_state
        period_state = self.period_state
        
        # Apply voicing gain
        cascade_voice = voice * frame_state.cascade_voicing_lin
        
        # Calculate aspiration modulation
        current_aspiration_mod = frame_params.cascade_aspiration_mod if period_state.position_in_period >= period_state.period_length / 2 else 0
        
        # Get aspiration noise
        aspiration = self.aspiration_source_casc.get_next() * frame_state.cascade_aspiration_lin * (1 - current_aspiration_mod)
        
        # Combine voice and aspiration
        v = cascade_voice + aspiration
        
        # Apply filters in sequence
        v = self.nasal_antiformant_casc.step(v)
        v = self.nasal_formant_casc.step(v)
        
        for i in range(MAX_ORAL_FORMANTS):
            v = self.oral_formant_casc[i].step(v)
            
        return v

    def compute_parallel_branch(self, voice: float) -> float:
        """
        Compute the parallel branch output.

        Args:
            voice: Input voice signal.

        Returns:
            Parallel branch output value.
        """
        frame_params = self.frame_params
        frame_state = self.frame_state
        period_state = self.period_state
        
        # Apply voicing gain
        parallel_voice = voice * frame_state.parallel_voicing_lin
        
        # Calculate aspiration modulation
        current_aspiration_mod = frame_params.parallel_aspiration_mod if period_state.position_in_period >= period_state.period_length / 2 else 0
        
        # Get aspiration noise
        aspiration = self.aspiration_source_par.get_next() * frame_state.parallel_aspiration_lin * (1 - current_aspiration_mod)
        
        # Combine voice and aspiration
        source = parallel_voice + aspiration
        
        # Apply differencing filter
        source_difference = self.differencing_filter_par.step(source)
        
        # Calculate frication modulation
        current_frication_mod = frame_params.frication_mod if period_state.position_in_period >= period_state.period_length / 2 else 0
        
        # Get frication noise
        frication_noise = self.frication_source_par.get_next() * frame_state.frication_lin * (1 - current_frication_mod)
        
        # Combine source difference and frication
        source2 = source_difference + frication_noise
        
        # Initialize output
        v = 0
        
        # Add nasal formant (applied directly to source)
        v += self.nasal_formant_par.step(source)
        
        # Add F1 (applied directly to source)
        v += self.oral_formant_par[0].step(source)
        
        # Add F2 to F6 (applied to source difference + frication)
        for i in range(1, MAX_ORAL_FORMANTS):
            # Alternating sign according to Klatt (1980) Fig. 13
            alternating_sign = 1 if i % 2 == 0 else -1
            v += alternating_sign * self.oral_formant_par[i].step(source2)
            
        # Add bypass (applied to source difference + frication)
        v += frame_state.parallel_bypass_lin * source2
        
        return v

    def start_new_period(self) -> None:
        """Start a new F0 period."""
        # If we have new frame parameters, activate them at the start of a new F0 period
        if self.new_frame_params:
            self.frame_params = self.new_frame_params
            self.new_frame_params = None
            self.start_using_new_frame_parameters()

        # Initialize period state if needed
        if not self.period_state:
            self.period_state = PeriodState(
                f0=0,
                period_length=0,
                open_phase_length=0,
                position_in_period=0,
                lp_noise=0
            )

        # Get local references
        period_state = self.period_state
        main_params = self.main_params
        frame_params = self.frame_params
        
        # Calculate modulated f0
        flutter_time = self.abs_position / main_params.sample_rate + self.flutter_time_offset
        period_state.f0 = perform_frequency_modulation(frame_params.f0, frame_params.flutter_level, flutter_time)
        
        # Calculate period and open phase lengths
        period_state.period_length = round(main_params.sample_rate / period_state.f0) if period_state.f0 > 0 else 1
        period_state.open_phase_length = round(period_state.period_length * frame_params.open_phase_ratio) if period_state.period_length > 1 else 0
        period_state.position_in_period = 0
        
        # Start new period for the glottal source
        self.start_glottal_source_period()

    def start_using_new_frame_parameters(self) -> None:
        """Apply new frame parameters."""
        main_params = self.main_params
        frame_params = self.frame_params
        frame_state = self.frame_state
        
        # Set basic frame state values
        frame_state.breathiness_lin = db_to_lin(frame_params.breathiness_db)
        frame_state.gain_lin = db_to_lin(frame_params.gain_db if not math.isnan(frame_params.gain_db) else 0)
        
        # Set tilt filter
        set_tilt_filter(self.tilt_filter, frame_params.tilt_db)

        # Adjust cascade branch
        frame_state.cascade_voicing_lin = db_to_lin(frame_params.cascade_voicing_db)
        frame_state.cascade_aspiration_lin = db_to_lin(frame_params.cascade_aspiration_db)
        set_nasal_formant_casc(self.nasal_formant_casc, frame_params)
        set_nasal_antiformant_casc(self.nasal_antiformant_casc, frame_params)
        
        for i in range(MAX_ORAL_FORMANTS):
            set_oral_formant_casc(self.oral_formant_casc[i], frame_params, i)

        # Adjust parallel branch
        frame_state.parallel_voicing_lin = db_to_lin(frame_params.parallel_voicing_db)
        frame_state.parallel_aspiration_lin = db_to_lin(frame_params.parallel_aspiration_db)
        frame_state.frication_lin = db_to_lin(frame_params.frication_db)
        frame_state.parallel_bypass_lin = db_to_lin(frame_params.parallel_bypass_db)
        set_nasal_formant_par(self.nasal_formant_par, frame_params)
        
        for i in range(MAX_ORAL_FORMANTS):
            set_oral_formant_par(self.oral_formant_par[i], main_params, frame_params, i)

    def init_glottal_source(self) -> None:
        """Initialize the glottal source based on the glottal source type."""
        if self.main_params.glottal_source_type == GlottalSourceType.IMPULSIVE:
            self.impulsive_g_source = ImpulsiveGlottalSource(self.main_params.sample_rate)
            self.glottal_source = self.impulsive_g_source.get_next
        elif self.main_params.glottal_source_type == GlottalSourceType.NATURAL:
            self.natural_g_source = NaturalGlottalSource()
            self.glottal_source = self.natural_g_source.get_next
        elif self.main_params.glottal_source_type == GlottalSourceType.NOISE:
            self.glottal_source = get_white_noise
        else:
            raise ValueError("Undefined glottal source type.")

    def start_glottal_source_period(self) -> None:
        """Start a new period for the glottal source."""
        if self.main_params.glottal_source_type == GlottalSourceType.IMPULSIVE:
            self.impulsive_g_source.start_period(self.period_state.open_phase_length)
        elif self.main_params.glottal_source_type == GlottalSourceType.NATURAL:
            self.natural_g_source.start_period(self.period_state.open_phase_length)


def set_tilt_filter(tilt_filter: LpFilter1, tilt_db: float) -> None:
    """
    Set the tilt filter parameters.

    Args:
        tilt_filter: The tilt filter to configure.
        tilt_db: Tilt in dB.
    """
    if not tilt_db:
        tilt_filter.set_passthrough()
    else:
        tilt_filter.set(3000, db_to_lin(-tilt_db))


def set_nasal_formant_casc(nasal_formant_casc: Resonator, frame_params: FrameParams) -> None:
    """
    Set the nasal formant parameters for the cascade branch.

    Args:
        nasal_formant_casc: The nasal formant resonator to configure.
        frame_params: Frame parameters.
    """
    if (frame_params.nasal_formant_freq is not None and 
            not math.isnan(frame_params.nasal_formant_freq) and 
            frame_params.nasal_formant_bw is not None and 
            not math.isnan(frame_params.nasal_formant_bw)):
        nasal_formant_casc.set(frame_params.nasal_formant_freq, frame_params.nasal_formant_bw)
    else:
        nasal_formant_casc.set_passthrough()


def set_nasal_antiformant_casc(nasal_antiformant_casc: AntiResonator, frame_params: FrameParams) -> None:
    """
    Set the nasal antiformant parameters for the cascade branch.

    Args:
        nasal_antiformant_casc: The nasal antiformant resonator to configure.
        frame_params: Frame parameters.
    """
    if (frame_params.nasal_antiformant_freq is not None and 
            not math.isnan(frame_params.nasal_antiformant_freq) and 
            frame_params.nasal_antiformant_bw is not None and 
            not math.isnan(frame_params.nasal_antiformant_bw)):
        nasal_antiformant_casc.set(frame_params.nasal_antiformant_freq, frame_params.nasal_antiformant_bw)
    else:
        nasal_antiformant_casc.set_passthrough()


def set_oral_formant_casc(oral_formant_casc: Resonator, frame_params: FrameParams, i: int) -> None:
    """
    Set the oral formant parameters for the cascade branch.

    Args:
        oral_formant_casc: The oral formant resonator to configure.
        frame_params: Frame parameters.
        i: Formant index.
    """
    f = frame_params.oral_formant_freq[i] if i < len(frame_params.oral_formant_freq) else None
    bw = frame_params.oral_formant_bw[i] if i < len(frame_params.oral_formant_bw) else None
    
    if (f is not None and not math.isnan(f) and 
            bw is not None and not math.isnan(bw)):
        oral_formant_casc.set(f, bw)
    else:
        oral_formant_casc.set_passthrough()


def set_nasal_formant_par(nasal_formant_par: Resonator, frame_params: FrameParams) -> None:
    """
    Set the nasal formant parameters for the parallel branch.

    Args:
        nasal_formant_par: The nasal formant resonator to configure.
        frame_params: Frame parameters.
    """
    if (frame_params.nasal_formant_freq is not None and 
            not math.isnan(frame_params.nasal_formant_freq) and 
            frame_params.nasal_formant_bw is not None and 
            not math.isnan(frame_params.nasal_formant_bw) and
            frame_params.nasal_formant_db is not None and
            not math.isnan(frame_params.nasal_formant_db) and
            db_to_lin(frame_params.nasal_formant_db) > 0):
        nasal_formant_par.set(frame_params.nasal_formant_freq, frame_params.nasal_formant_bw)
        nasal_formant_par.adjust_peak_gain(db_to_lin(frame_params.nasal_formant_db))
    else:
        nasal_formant_par.set_mute()


def set_oral_formant_par(
    oral_formant_par: Resonator, 
    main_params: MainParams, 
    frame_params: FrameParams, 
    i: int
) -> None:
    """
    Set the oral formant parameters for the parallel branch.

    Args:
        oral_formant_par: The oral formant resonator to configure.
        main_params: Main parameters.
        frame_params: Frame parameters.
        i: Formant index.
    """
    formant = i + 1
    f = frame_params.oral_formant_freq[i] if i < len(frame_params.oral_formant_freq) else None
    bw = frame_params.oral_formant_bw[i] if i < len(frame_params.oral_formant_bw) else None
    db = frame_params.oral_formant_db[i] if i < len(frame_params.oral_formant_db) else None
    
    peak_gain = db_to_lin(db) if db is not None and not math.isnan(db) else 0
    
    if (f is not None and not math.isnan(f) and 
            bw is not None and not math.isnan(bw) and 
            peak_gain > 0):
        oral_formant_par.set(f, bw)
        
        # Compensate differencing filter for F2 to F6
        w = 2 * math.pi * f / main_params.sample_rate
        diff_gain = math.sqrt(2 - 2 * math.cos(w))  # gain of differencing filter
        filter_gain = peak_gain / diff_gain if formant >= 2 else peak_gain
        
        oral_formant_par.adjust_peak_gain(filter_gain)
    else:
        oral_formant_par.set_mute()


def generate_sound(main_params: MainParams, frame_params_array: List[FrameParams]) -> np.ndarray:
    """
    Generates a sound that consists of multiple frames.

    Args:
        main_params: Main parameters.
        frame_params_array: Array of frame parameters.

    Returns:
        Generated sound samples.
    """
    generator = Generator(main_params)
    
    # Calculate total buffer length
    out_buf_len = 0
    for frame_params in frame_params_array:
        out_buf_len += round(frame_params.duration * main_params.sample_rate)
        
    # Create output buffer
    out_buf = np.zeros(out_buf_len, dtype=np.float64)
    
    # Generate each frame
    out_buf_pos = 0
    for frame_params in frame_params_array:
        frame_len = round(frame_params.duration * main_params.sample_rate)
        frame_buf = out_buf[out_buf_pos:out_buf_pos + frame_len]
        generator.generate_frame(frame_params, frame_buf)
        out_buf_pos += frame_len
        
    return out_buf