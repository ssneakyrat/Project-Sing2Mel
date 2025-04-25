"""
Utility functions for the Klatt speech synthesizer.
"""

import math
import numpy as np
from typing import List, Tuple, Union, Optional


def perform_frequency_modulation(f0: float, flutter_level: float, time: float) -> float:
    """
    Modulates the fundamental frequency (F0).

    Sine-wave frequencies of 12.7, 7.1 and 4.7 Hz were chosen so as to ensure
    a long period before repetition of the perturbation that is introduced.
    A value of flutter_level = 0.25 results in synthetic vowels with a quite
    realistic deviation from constant pitch.

    Args:
        f0: Fundamental frequency.
        flutter_level: Flutter level between 0 and 1.
        time: Relative signal position in seconds.

    Returns:
        Modulated fundamental frequency.
    """
    if flutter_level <= 0:
        return f0
        
    w = 2 * math.pi * time
    a = math.sin(12.7 * w) + math.sin(7.1 * w) + math.sin(4.7 * w)
    return f0 * (1 + a * flutter_level / 50)


def db_to_lin(db: float) -> float:
    """
    Convert a dB value into a linear value.
    dB values of -99 and below or NaN are converted to 0.

    Args:
        db: Input value in decibels.

    Returns:
        Linear value.
    """
    if db <= -99 or math.isnan(db):
        return 0
    else:
        return 10 ** (db / 20)


def adjust_signal_gain(buf: np.ndarray, target_rms: float) -> None:
    """
    Adjusts the gain of a signal to achieve a target RMS value.

    Args:
        buf: Signal buffer to adjust.
        target_rms: Target RMS level.
    """
    n = len(buf)
    if n == 0:
        return
        
    rms = compute_rms(buf)
    if rms == 0:
        return
        
    r = target_rms / rms
    buf *= r


def compute_rms(buf: np.ndarray) -> float:
    """
    Compute the root mean square (RMS) of a signal.

    Args:
        buf: Signal buffer.

    Returns:
        RMS value.
    """
    n = len(buf)
    if n == 0:
        return 0
        
    return np.sqrt(np.mean(buf ** 2))
