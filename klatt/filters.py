"""
Digital filters for the Klatt speech synthesizer.
"""

import math
from typing import List, Tuple, Optional


class LpFilter1:
    """
    A first-order IIR LP filter.

    Formulas:
     Variables:
       x = input samples
       y = output samples
       a = first filter coefficient
       b = second filter coefficient, >0 for LP filter, <0 for HP filter
       f = frequency in Hz
       w = 2 * PI * f / sampleRate
       g = gain at frequency f
     Filter function:
       y[n] = a * x[n] + b * y[n-1]
     Transfer function:
       H(w) = a / ( 1 - b * e^(-jw) )
     Frequency response:
       |H(w)| = a / sqrt(1 - 2b * cos(w) + b^2)
     Gain at DC:
       |H(0)| = a / sqrt(1 - 2b * cos(0) + b^2)
              = a / sqrt(1 - 2b + b^2)
              = a / (1 - b)                                 for b < 1
    """

    def __init__(self, sample_rate: float):
        """
        Initialize the filter.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self.sample_rate = sample_rate
        self.a = 0  # filter coefficient a
        self.b = 0  # filter coefficient b
        self.y1 = 0  # y[n-1], last output value
        self.passthrough = True
        self.muted = False

    def set(self, f: float, g: float, extra_gain: float = 1.0):
        """
        Adjusts the filter parameters without resetting the inner state.

        Args:
            f: Frequency at which the gain is specified.
            g: Gain at frequency f. Between 0 and 1 for LP filter. Greater than 1 for HP filter.
            extra_gain: Extra gain factor. This is the resulting DC gain.
                The resulting gain at `f` will be `g * extra_gain`.
        """
        if (f <= 0 or f >= self.sample_rate / 2 or g <= 0 or g >= 1 or 
                not math.isfinite(f) or not math.isfinite(g) or not math.isfinite(extra_gain)):
            raise ValueError("Invalid filter parameters.")
            
        w = 2 * math.pi * f / self.sample_rate
        q = (1 - g ** 2 * math.cos(w)) / (1 - g ** 2)
        self.b = q - math.sqrt(q ** 2 - 1)
        self.a = (1 - self.b) * extra_gain
        self.passthrough = False
        self.muted = False

    def set_passthrough(self):
        """Set filter to passthrough mode (no filtering)."""
        self.passthrough = True
        self.muted = False
        self.y1 = 0

    def set_mute(self):
        """Set filter to mute mode (no output)."""
        self.passthrough = False
        self.muted = True
        self.y1 = 0

    def get_transfer_function_coefficients(self) -> List[List[float]]:
        """
        Returns the polynomial coefficients of the filter transfer function in the z-plane.
        
        The returned array contains the top and bottom coefficients of the 
        rational fraction, ordered in ascending powers.

        Returns:
            A list containing two lists: numerator and denominator coefficients.
        """
        if self.passthrough:
            return [[1], [1]]
        if self.muted:
            return [[0], [1]]
        return [[self.a], [1, -self.b]]

    def step(self, x: float) -> float:
        """
        Performs a filter step.

        Args:
            x: Input signal value.

        Returns:
            Output signal value.
        """
        if self.passthrough:
            return x
        if self.muted:
            return 0
        y = self.a * x + self.b * self.y1
        self.y1 = y
        return y


class Resonator:
    """
    A Klatt resonator.
    This is a second order IIR filter.
    With f=0 it can also be used as a low-pass filter.

    Formulas:
     Variables:
       x = input samples
       y = output samples
       a/b/c = filter coefficients
       f = frequency in Hz
       w = 2 * PI * f / sampleRate
       f0 = resonator frequency in Hz
       w0 = 2 * PI * f0 / sampleRate
       bw = Bandwidth in Hz
       r = exp(- PI * bw / sampleRate)
     Filter function:
       y[n] = a * x[n] + b * y[n-1] + c * y[n-2]
     Transfer function:
       H(w) = a / ( 1 - b * e^(-jw) - c * e^(-2jw) )
    """

    def __init__(self, sample_rate: float):
        """
        Initialize the resonator.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self.sample_rate = sample_rate
        self.a = 0  # filter coefficient a
        self.b = 0  # filter coefficient b
        self.c = 0  # filter coefficient c
        self.y1 = 0  # y[n-1], last output value
        self.y2 = 0  # y[n-2], second-last output value
        self.r = 0
        self.passthrough = True
        self.muted = False

    def set(self, f: float, bw: float, dc_gain: float = 1.0):
        """
        Adjusts the filter parameters without resetting the inner state.

        Args:
            f: Frequency of resonator in Hz. May be 0 for LP filtering.
            bw: Bandwidth of resonator in Hz.
            dc_gain: DC gain level.
        """
        if (f < 0 or f >= self.sample_rate / 2 or bw <= 0 or dc_gain <= 0 or
                not math.isfinite(f) or not math.isfinite(bw) or not math.isfinite(dc_gain)):
            raise ValueError("Invalid resonator parameters.")
            
        self.r = math.exp(- math.pi * bw / self.sample_rate)
        w = 2 * math.pi * f / self.sample_rate
        self.c = - (self.r ** 2)
        self.b = 2 * self.r * math.cos(w)
        self.a = (1 - self.b - self.c) * dc_gain
        self.passthrough = False
        self.muted = False

    def set_passthrough(self):
        """Set resonator to passthrough mode (no filtering)."""
        self.passthrough = True
        self.muted = False
        self.y1 = 0
        self.y2 = 0

    def set_mute(self):
        """Set resonator to mute mode (no output)."""
        self.passthrough = False
        self.muted = True
        self.y1 = 0
        self.y2 = 0

    def adjust_impulse_gain(self, new_a: float):
        """
        Adjust the impulse gain.

        Args:
            new_a: New filter coefficient a.
        """
        self.a = new_a

    def adjust_peak_gain(self, peak_gain: float):
        """
        Adjust the peak gain.

        Args:
            peak_gain: New peak gain.
        """
        if peak_gain <= 0 or not math.isfinite(peak_gain):
            raise ValueError("Invalid resonator peak gain.")
        self.a = peak_gain * (1 - self.r)

    def get_transfer_function_coefficients(self) -> List[List[float]]:
        """
        Returns the polynomial coefficients of the filter transfer function in the z-plane.
        
        The returned array contains the top and bottom coefficients of the 
        rational fraction, ordered in ascending powers.

        Returns:
            A list containing two lists: numerator and denominator coefficients.
        """
        if self.passthrough:
            return [[1], [1]]
        if self.muted:
            return [[0], [1]]
        return [[self.a], [1, -self.b, -self.c]]

    def step(self, x: float) -> float:
        """
        Performs a filter step.

        Args:
            x: Input signal value.

        Returns:
            Output signal value.
        """
        if self.passthrough:
            return x
        if self.muted:
            return 0
        y = self.a * x + self.b * self.y1 + self.c * self.y2
        self.y2 = self.y1
        self.y1 = y
        return y


class AntiResonator:
    """
    A Klatt anti-resonator.
    This is a second order FIR filter.

    Formulas:
     Variables:
       x = input samples
       y = output samples
       a/b/c = filter coefficients
       f = frequency in Hz
       w = 2 * PI * f / sampleRate
     Filter function:
       y[n] = a * x[n] + b * x[n-1] + c * x[n-2]
     Transfer function:
       H(w) = a + b * e^(-jw) + c * e^(-2jw)
    """
    
    def __init__(self, sample_rate: float):
        """
        Initialize the anti-resonator.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self.sample_rate = sample_rate
        self.a = 0  # filter coefficient a
        self.b = 0  # filter coefficient b
        self.c = 0  # filter coefficient c
        self.x1 = 0  # x[n-1], last input value
        self.x2 = 0  # x[n-2], second-last input value
        self.passthrough = True
        self.muted = False

    def set(self, f: float, bw: float):
        """
        Adjusts the filter parameters without resetting the inner state.

        Args:
            f: Frequency of anti-resonator in Hz.
            bw: bandwidth of anti-resonator in Hz.
        """
        if (f <= 0 or f >= self.sample_rate / 2 or bw <= 0 or 
                not math.isfinite(f) or not math.isfinite(bw)):
            raise ValueError("Invalid anti-resonator parameters.")
            
        r = math.exp(- math.pi * bw / self.sample_rate)
        w = 2 * math.pi * f / self.sample_rate
        c0 = - (r * r)
        b0 = 2 * r * math.cos(w)
        a0 = 1 - b0 - c0
        
        if a0 == 0:
            self.a = 0
            self.b = 0
            self.c = 0
            return
            
        self.a = 1 / a0
        self.b = - b0 / a0
        self.c = - c0 / a0
        self.passthrough = False
        self.muted = False

    def set_passthrough(self):
        """Set anti-resonator to passthrough mode (no filtering)."""
        self.passthrough = True
        self.muted = False
        self.x1 = 0
        self.x2 = 0

    def set_mute(self):
        """Set anti-resonator to mute mode (no output)."""
        self.passthrough = False
        self.muted = True
        self.x1 = 0
        self.x2 = 0

    def get_transfer_function_coefficients(self) -> List[List[float]]:
        """
        Returns the polynomial coefficients of the filter transfer function in the z-plane.
        
        The returned array contains the top and bottom coefficients of the 
        rational fraction, ordered in ascending powers.

        Returns:
            A list containing two lists: numerator and denominator coefficients.
        """
        if self.passthrough:
            return [[1], [1]]
        if self.muted:
            return [[0], [1]]
        return [[self.a, self.b, self.c], [1]]

    def step(self, x: float) -> float:
        """
        Performs a filter step.

        Args:
            x: Input signal value.

        Returns:
            Output signal value.
        """
        if self.passthrough:
            return x
        if self.muted:
            return 0
        y = self.a * x + self.b * self.x1 + self.c * self.x2
        self.x2 = self.x1
        self.x1 = x
        return y


class DifferencingFilter:
    """
    A differencing filter.
    This is a first-order FIR HP filter.

    Formulas:
     Variables:
       x = input samples
       y = output samples
       f = frequency in Hz
       w = 2 * PI * f / sampleRate
     Filter function:
       y[n] = x[n] - x[n-1]
     Transfer function:
       H(w) = 1 - e^(-jw)
     Frequency response:
       |H(w)| = sqrt(2 - 2 * cos(w))
    """
    
    def __init__(self):
        """Initialize the differencing filter."""
        self.x1 = 0  # x[n-1], last input value

    def get_transfer_function_coefficients(self) -> List[List[float]]:
        """
        Returns the polynomial coefficients of the filter transfer function in the z-plane.
        
        The returned array contains the top and bottom coefficients of the 
        rational fraction, ordered in ascending powers.

        Returns:
            A list containing two lists: numerator and denominator coefficients.
        """
        return [[1, -1], [1]]

    def step(self, x: float) -> float:
        """
        Performs a filter step.

        Args:
            x: Input signal value.

        Returns:
            Output signal value.
        """
        y = x - self.x1
        self.x1 = x
        return y