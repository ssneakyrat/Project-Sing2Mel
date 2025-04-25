"""
Test script for the Klatt synthesizer using the demo parameters from the original TypeScript implementation.
"""

import math
import numpy as np
from klatt_synth import KlattSynthesizer
from klatt.parameters import GlottalSourceType
from utils.save_audio import save_wav

def test_with_demo_params():
    """
    Test the Klatt synthesizer with the demo parameters from the TypeScript implementation.
    """
    # Create a synthesizer with natural glottal source
    klatt = KlattSynthesizer(
        sample_rate=44100,
        glottal_source_type=GlottalSourceType.NATURAL
    )
    
    # Create frame parameters based on the demo values from TypeScript
    demo_params = klatt.create_frame_params(
        # Basic parameters
        duration=1,
        f0=247,
        flutter_level=0.25,
        open_phase_ratio=0.7,
        breathiness_db=-25,
        tilt_db=0,
        gain_db=float('nan'),
        agc_rms_level=0.18,
        
        # Formant parameters
        nasal_formant_freq=None,
        nasal_formant_bw=None,
        oral_formant_freq=[520, 1006, 2831, 3168, 4135, 5020],
        oral_formant_bw=[76, 102, 72, 102, 816, 596],
        
        # Cascade branch parameters
        cascade_enabled=True,
        cascade_voicing_db=0,
        cascade_aspiration_db=-25,
        cascade_aspiration_mod=0.5,
        nasal_antiformant_freq=None,
        nasal_antiformant_bw=None,
        
        # Parallel branch parameters
        parallel_enabled=False,
        parallel_voicing_db=0,
        parallel_aspiration_db=-25,
        parallel_aspiration_mod=0.5,
        frication_db=-30,
        frication_mod=0.5,
        parallel_bypass_db=-99,
        nasal_formant_db=float('nan'),
        oral_formant_db=[0, -8, -15, -19, -30, -35]
    )
    
    print("Synthesizing audio with demo parameters...")
    
    # Synthesize audio with the demo parameters
    audio = klatt.synthesize([demo_params])
    
    print(f"Generated audio samples: {len(audio)}")
    print(f"Audio min/max: {audio.min():.4f}/{audio.max():.4f}")
    
    save_wav("demo_klatt_sequence.wav", audio, 44100)

    '''
    # Save to a WAV file
    try:
        from scipy.io import wavfile
        wavfile.write("demo_klatt.wav", 44100, audio)
        print("Saved audio to demo_klatt.wav")
    except ImportError:
        print("scipy not found, audio not saved. Install scipy to save WAV files.")
    '''
    return audio


if __name__ == "__main__":
    # Run the test
    audio = test_with_demo_params()
    
    # Try to visualize if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Plot waveform
        plt.figure(figsize=(12, 6))
        time = np.arange(len(audio)) / 44100
        plt.plot(time, audio)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Klatt Synthesizer Demo - Waveform")
        plt.ylim(-1, 1)
        plt.grid(True)
        plt.savefig("demo_waveform.png")
        plt.close()
        
        # Try to plot spectrum if signal processing libraries are available
        try:
            from scipy import signal
            
            # Calculate and plot spectrogram
            plt.figure(figsize=(12, 6))
            f, t, Sxx = signal.spectrogram(audio, 44100, nperseg=1024)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.ylim(0, 5000)  # Focus on 0-5kHz range
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.title('Klatt Synthesizer Demo - Spectrogram')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.savefig("demo_spectrogram.png")
            plt.close()
            
            print("Generated visualization plots: demo_waveform.png and demo_spectrogram.png")
        except ImportError:
            print("Could not generate spectrogram plot. Install scipy for signal processing.")
            
    except ImportError:
        print("matplotlib not found, visualizations not generated. Install matplotlib to visualize.")