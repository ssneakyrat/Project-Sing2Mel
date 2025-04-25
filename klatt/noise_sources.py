"""
Noise source implementations for the Klatt speech synthesizer.
"""

import random
import math
from typing import List, Tuple, Optional
from klatt.filters import LpFilter1


def get_white_noise() -> float:
    """
    Returns a random number within the range -1 .. 1.

    Returns:
        Random float between -1 and 1.
    """
    return random.random() * 2 - 1


class LpNoiseSource:
    """
    A low-pass filtered noise source.
    """

    def __init__(self, sample_rate: float):
        """
        Initialize the low-pass filtered noise source.

        Args:
            sample_rate: Sample rate in Hz.
        """
        # The original program logic used a first order LP filter with a filter coefficient
        # of b=0.75 and a sample rate of 10 kHz.
        old_b = 0.75
        old_sample_rate = 10000
        
        # Compute the gain at 1000 Hz with a sample rate of 10 kHz and a DC gain of 1.
        f = 1000
        w = 2 * math.pi * f / old_sample_rate
        g = (1 - old_b) / math.sqrt(1 - 2 * old_b * math.cos(w) + old_b ** 2)
        
        # Compensate amplitude for output range -1 .. +1
        extra_gain = 2.5 * (sample_rate / 10000) ** 0.33
        
        # Create an LP filter with the same characteristics but with our sampling rate.
        self.lp_filter = LpFilter1(sample_rate)
        self.lp_filter.set(f, g, extra_gain)

    def get_next(self) -> float:
        """
        Returns an LP-filtered random number.

        Returns:
            LP-filtered random float.
        """
        x = get_white_noise()
        return self.lp_filter.step(x)
