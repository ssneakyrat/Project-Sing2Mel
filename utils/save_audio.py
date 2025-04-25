"""
Improved audio saving utilities for the Klatt synthesizer.
"""

import numpy as np
import os

def save_wav(filename, audio, sample_rate=44100):
    """
    Save audio data as a WAV file with proper PCM conversion.
    
    Args:
        filename: Output filename
        audio: Audio data as floating-point array between -1 and 1
        sample_rate: Sample rate in Hz
    """
    try:
        from scipy.io import wavfile
        
        # Ensure the audio is within the range -1 to 1
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        # Scale to the range of 16-bit integers and convert
        audio_pcm = (audio * 32767).astype(np.int16)
        
        # Save the file
        wavfile.write(filename, sample_rate, audio_pcm)
        
        print(f"Successfully saved audio to {filename}")
        print(f"File size: {os.path.getsize(filename)} bytes")
        
        return True
    except Exception as e:
        print(f"Error saving WAV file: {e}")
        return False

def try_multiple_formats(base_filename, audio, sample_rate=44100):
    """
    Try saving audio in multiple formats for compatibility.
    
    Args:
        base_filename: Base name for the output files (without extension)
        audio: Audio data as floating-point array between -1 and 1
        sample_rate: Sample rate in Hz
    """
    results = {}
    
    # Try 16-bit PCM WAV
    results["16bit_pcm"] = save_wav(f"{base_filename}_16bit.wav", audio, sample_rate)
    
    # Try alternative formats if available
    try:
        import soundfile as sf
        
        # Try saving as 32-bit float WAV
        try:
            audio_float = np.clip(audio, -1.0, 1.0)
            sf.write(f"{base_filename}_float.wav", audio_float, sample_rate, subtype='FLOAT')
            print(f"Successfully saved audio to {base_filename}_float.wav")
            results["float_wav"] = True
        except Exception as e:
            print(f"Error saving 32-bit float WAV: {e}")
            results["float_wav"] = False
            
        # Try saving as FLAC
        try:
            audio_pcm = np.clip(audio, -1.0, 1.0)
            sf.write(f"{base_filename}.flac", audio_pcm, sample_rate)
            print(f"Successfully saved audio to {base_filename}.flac")
            results["flac"] = True
        except Exception as e:
            print(f"Error saving FLAC: {e}")
            results["flac"] = False
            
    except ImportError:
        print("soundfile not installed, only saving as 16-bit PCM WAV")
        results["float_wav"] = False
        results["flac"] = False
        
    return results


if __name__ == "__main__":
    # Example usage - generate a simple test tone
    
    # Create a 1-second 440 Hz sine wave
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    test_tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Save in multiple formats
    try_multiple_formats("test_tone", test_tone, sample_rate)
    
    print("Test tone files created. Please check if you can play any of them.")
