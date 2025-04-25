"""
Glottal source implementations for the Klatt speech synthesizer.
"""

from klatt.filters import Resonator


class ImpulsiveGlottalSource:
    """
    Generates a glottal source signal by LP filtering a pulse train.
    """

    def __init__(self, sample_rate: float):
        """
        Initialize the impulsive glottal source.

        Args:
            sample_rate: Sample rate in Hz.
        """
        self.sample_rate = sample_rate
        self.resonator = None  # resonator used as an LP filter
        self.position_in_period = 0  # current sample position within F0 period

    def start_period(self, open_phase_length: float):
        """
        Start a new period.

        Args:
            open_phase_length: Duration of the open glottis phase of the F0 period, in samples.
        """
        if not open_phase_length:
            self.resonator = None
            return
            
        if not self.resonator:
            self.resonator = Resonator(self.sample_rate)
            
        bw = self.sample_rate / open_phase_length
        self.resonator.set(0, bw)
        self.resonator.adjust_impulse_gain(1)
        self.position_in_period = 0

    def get_next(self) -> float:
        """
        Get the next sample of the glottal source signal.

        Returns:
            The next sample value.
        """
        if not self.resonator:
            return 0
            
        # Generate pulse at specific positions in the period
        pulse = 0
        if self.position_in_period == 1:
            pulse = 1
        elif self.position_in_period == 2:
            pulse = -1
            
        self.position_in_period += 1
        return self.resonator.step(pulse)


class NaturalGlottalSource:
    """
    Generates a "natural" glottal source signal according to the KLGLOTT88 model.
    Formula of the glottal flow: t^2 - t^3
    Formula of the derivative: 2 * t - 3 * t^2
    The derivative is used as the glottal source.

    At the end of the open glottal phase there is an abrupt jump from the minimum value to zero.
    This jump is not smoothed in the classic Klatt model. In Praat this "collision phase" is smoothed.
    """

    def __init__(self):
        """Initialize the natural glottal source."""
        self.start_period(0)

    def start_period(self, open_phase_length: float):
        """
        Start a new period.

        Args:
            open_phase_length: Duration of the open glottis phase of the F0 period, in samples.
        """
        self.open_phase_length = open_phase_length
        self.x = 0  # current signal value
        
        # Set up coefficients for the glottal flow derivative
        amplification = 5
        self.b = -amplification / (open_phase_length ** 2) if open_phase_length > 0 else 0  # second derivative
        self.a = -self.b * open_phase_length / 3  # first derivative
        self.position_in_period = 0

    def get_next(self) -> float:
        """
        Get the next sample of the glottal source signal.

        Returns:
            The next sample value.
        """
        if self.position_in_period >= self.open_phase_length:
            self.x = 0
            self.position_in_period += 1
            return 0
            
        self.a += self.b
        self.x += self.a
        self.position_in_period += 1
        return self.x
