import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from klatt.parameters import GlottalSourceType
from klatt.generator import generate_sound
from utils import save_wav

# Import your KlattSynthesizer class
from klatt_synth import KlattSynthesizer

# Define phoneme-to-formant mapping for English phonemes
# These values are approximate and can be adjusted for better quality
PHONEME_TO_FORMANTS = {
    # Vowels - format: [F1, F2, F3, F4, F5, F6]
    'aa': [730, 1090, 2440, 3500, 4500, 5500],  # father
    'ae': [660, 1720, 2410, 3500, 4500, 5500],  # cat
    'ah': [520, 1190, 2390, 3500, 4500, 5500],  # but
    'ao': [570, 840, 2410, 3500, 4500, 5500],   # dog
    'aw': [680, 1060, 2410, 3500, 4500, 5500],  # down
    'ay': [660, 1200, 2550, 3500, 4500, 5500],  # hide
    'eh': [530, 1840, 2480, 3500, 4500, 5500],  # red
    'er': [490, 1350, 1690, 3500, 4500, 5500],  # bird
    'ey': [480, 1870, 2540, 3500, 4500, 5500],  # say
    'ih': [390, 1990, 2550, 3500, 4500, 5500],  # sit
    'iy': [270, 2290, 3010, 3500, 4500, 5500],  # see
    'ow': [450, 1110, 2680, 3500, 4500, 5500],  # go
    'oy': [440, 1020, 2240, 3500, 4500, 5500],  # boy
    'uh': [370, 1500, 2500, 3500, 4500, 5500],  # book
    'uw': [300, 870, 2240, 3500, 4500, 5500],   # you
    
    # Semivowels/Glides
    'w':  [310, 610, 2200, 3500, 4500, 5500],
    'y':  [270, 2290, 3010, 3500, 4500, 5500],
    'r':  [420, 1300, 1600, 3500, 4500, 5500],
    'l':  [380, 880, 2575, 3500, 4500, 5500],
    
    # Nasals
    'm':  [480, 1000, 2200, 3500, 4500, 5500],
    'n':  [480, 1780, 2300, 3500, 4500, 5500],
    'ng': [480, 1900, 2300, 3500, 4500, 5500],
    
    # Fricatives
    'f':  [340, 1100, 2300, 3500, 4500, 5500],
    'v':  [300, 1500, 2300, 3500, 4500, 5500],
    'th': [300, 1400, 2300, 3500, 4500, 5500],
    'dh': [300, 1600, 2300, 3500, 4500, 5500],
    's':  [300, 1700, 2700, 3500, 4500, 5500],
    'z':  [300, 1700, 2700, 3500, 4500, 5500],
    'sh': [300, 1600, 2300, 3500, 4500, 5500],
    'zh': [300, 1600, 2300, 3500, 4500, 5500],
    'h':  [500, 1500, 2500, 3500, 4500, 5500],
    'hh': [500, 1500, 2500, 3500, 4500, 5500],
    
    # Affricates
    'jh': [300, 1600, 2300, 3500, 4500, 5500],
    'ch': [300, 1600, 2300, 3500, 4500, 5500],
    
    # Stops
    'p':  [300, 800, 2300, 3500, 4500, 5500],
    'b':  [300, 800, 2300, 3500, 4500, 5500],
    't':  [300, 1700, 2600, 3500, 4500, 5500],
    'd':  [300, 1700, 2600, 3500, 4500, 5500],
    'k':  [300, 1500, 2600, 3500, 4500, 5500],
    'g':  [300, 1500, 2600, 3500, 4500, 5500],
    
    # Other
    'dx': [300, 1700, 2600, 3500, 4500, 5500],  # flap
    'pau': [0, 0, 0, 0, 0, 0],  # pause (silent)
    'spn': [500, 1500, 2500, 3500, 4500, 5500],  # spoken noise
    'fry': [300, 1500, 2500, 3500, 4500, 5500],  # vocal fry
    'gs_t': [300, 1500, 2500, 3500, 4500, 5500], # glottal stop
    'enb': [480, 1780, 2300, 3500, 4500, 5500],  # nasal sound
}

# Default formants for unknown phonemes
DEFAULT_FORMANTS = [500, 1500, 2500, 3500, 4500, 5500]

# Define bandwidth scaling relative to formant frequencies
# Generally, higher formants have wider bandwidths
BW_SCALING = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

def load_dataset_segment(cache_path="./cache/singing_voice_train_data_24000hz_80mels.pkl", segment_idx=0):
    """
    Load a segment from the dataset.
    
    Args:
        cache_path: Path to the cached dataset
        segment_idx: Index of the segment to load
        
    Returns:
        A dictionary containing the segment data
    """
    print(f"Loading dataset from {cache_path}")
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Extract the data for the segment
        segments = cache_data.get('data', [])
        if segment_idx >= len(segments):
            print(f"Warning: Segment index {segment_idx} out of range. Using first segment.")
            segment_idx = 0
            
        segment = segments[segment_idx]
        
        # Get the phone mapping
        phone_map = cache_data.get('phone_map', {})
        inv_phone_map = cache_data.get('inv_phone_map', {})
        
        return segment, phone_map, inv_phone_map
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

def convert_to_klatt_frames(segment, inv_phone_map, sample_rate=24000, hop_length=240):
    """
    Convert dataset segment to Klatt synthesizer frame parameters.
    
    Args:
        segment: Dataset segment with phone sequence, F0, etc.
        inv_phone_map: Inverse phone map to convert indices to phoneme symbols
        sample_rate: Audio sample rate
        hop_length: Hop length used in the dataset
        
    Returns:
        List of Klatt frame parameters for synthesis
    """
    klatt = KlattSynthesizer(sample_rate=sample_rate)
    
    # Extract data from the segment
    phone_seq = segment['phone_seq_mel']
    f0_values = segment['f0']
    
    # Convert phone sequences to actual phoneme symbols
    phoneme_seq = []
    for p_idx in phone_seq:
        if p_idx == 0:  # This is usually a padding index
            phoneme = 'pau'
        else:
            phoneme = inv_phone_map.get(int(p_idx), 'pau')
        phoneme_seq.append(phoneme.lower())  # Convert to lowercase
    
    # Find phone transitions (when the phone changes)
    transitions = []
    current_phone = None
    current_start = 0
    
    for i, phone in enumerate(phoneme_seq):
        if phone != current_phone:
            if current_phone is not None:
                transitions.append((current_phone, current_start, i))
            current_phone = phone
            current_start = i
    
    # Add the last phone
    if current_phone is not None:
        transitions.append((current_phone, current_start, len(phoneme_seq)))
    
    # Create Klatt frames
    frames = []
    for phone, start_frame, end_frame in transitions:
        # Calculate duration in seconds
        duration = (end_frame - start_frame) * hop_length / sample_rate
        
        # Skip very short segments
        if duration < 0.01:
            continue
        
        # Get average F0 for this segment (ignore zeros)
        f0_segment = f0_values[start_frame:end_frame]
        f0_valid = f0_segment[f0_segment > 0]
        f0 = f0_valid.mean() if len(f0_valid) > 0 else 120.0  # Default F0 if all zeros
        
        # Get formants for this phoneme
        formants = PHONEME_TO_FORMANTS.get(phone, DEFAULT_FORMANTS)
        
        # Calculate bandwidths based on formant frequencies
        bandwidths = [max(50, freq * scale) for freq, scale in zip(formants, BW_SCALING)]
        
        # For unvoiced consonants, reduce voicing and increase frication/aspiration
        voicing_db = 0.0
        frication_db = -60.0
        aspiration_db = -25.0
        
        if phone in ['f', 'th', 's', 'sh', 'h', 'ch', 'p', 't', 'k']:
            voicing_db = -30.0
            frication_db = -20.0
            aspiration_db = -15.0
        elif phone in ['v', 'dh', 'z', 'zh', 'jh', 'b', 'd', 'g']:
            voicing_db = -5.0
            frication_db = -30.0
        elif phone == 'pau':
            voicing_db = -60.0
            formants = [0, 0, 0, 0, 0, 0]
            bandwidths = [100, 100, 100, 100, 100, 100]
        
        # Special handling for nasal sounds
        nasal_formant_freq = float('nan')
        nasal_formant_bw = float('nan')
        if phone in ['m', 'n', 'ng']:
            nasal_formant_freq = 250.0
            nasal_formant_bw = 100.0
        
        # Create a frame parameter with these settings
        frame_params = klatt.create_frame_params(
            duration=duration,
            f0=f0,
            oral_formant_freq=formants,
            oral_formant_bw=bandwidths,
            nasal_formant_freq=nasal_formant_freq,
            nasal_formant_bw=nasal_formant_bw,
            cascade_voicing_db=voicing_db,
            parallel_voicing_db=voicing_db - 5.0,
            frication_db=frication_db,
            cascade_aspiration_db=aspiration_db,
            flutter_level=0.15  # Add some flutter for singing
        )
        
        frames.append(frame_params)
    
    return frames

def synthesize_singing(segment, inv_phone_map, output_file="singing_output.wav", sample_rate=24000):
    """
    Synthesize singing from a dataset segment.
    
    Args:
        segment: Dataset segment
        inv_phone_map: Inverse phone map
        output_file: Output WAV file path
        sample_rate: Sample rate for synthesis
    """
    # Create Klatt synthesizer
    klatt = KlattSynthesizer(sample_rate=sample_rate, 
                             glottal_source_type=GlottalSourceType.NATURAL)
    
    # Convert segment to Klatt frames
    frames = convert_to_klatt_frames(segment, inv_phone_map, sample_rate)
    
    # Synthesize audio
    print(f"Synthesizing {len(frames)} phoneme frames...")
    audio = klatt.synthesize(frames)
    
    # Save audio to file
    save_wav(output_file, audio)
    print(f"Saved singing output to {output_file}")
    
    return audio

def plot_data(segment, audio, output_file="singing_analysis.png"):
    """Plot F0 and audio waveform for analysis"""
    plt.figure(figsize=(12, 8))
    
    # Plot F0
    plt.subplot(2, 1, 1)
    plt.plot(segment['f0'])
    plt.title('F0 Contour from Dataset')
    plt.ylabel('Frequency (Hz)')
    
    # Plot audio waveform
    plt.subplot(2, 1, 2)
    plt.plot(audio)
    plt.title('Synthesized Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved analysis plot to {output_file}")

def main():
    # Set paths
    cache_path = "./cache/singing_voice_train_data_24000hz_80mels.pkl"
    output_wav = "singing_output.wav"
    output_plot = "singing_analysis.png"
    
    # Load a segment from the dataset
    segment, phone_map, inv_phone_map = load_dataset_segment(cache_path, segment_idx=6)
    
    if segment is None:
        print("Failed to load dataset segment. Please check the cache path.")
        return
    
    # Print segment info
    print(f"Loaded segment: {segment['filename']}")
    print(f"Number of phones: {len(segment['phone_seq_mel'])}")
    print(f"F0 range: {segment['f0'].min():.1f} - {segment['f0'].max():.1f} Hz")
    
    # Synthesize singing
    audio = synthesize_singing(segment, inv_phone_map, output_wav, sample_rate=24000)
    
    # Plot data for analysis
    plot_data(segment, audio, output_plot)

if __name__ == "__main__":
    main()
