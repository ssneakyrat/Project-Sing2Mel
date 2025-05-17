import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
import logging

logger = logging.getLogger("AudioProcessor")

class AudioProcessor:
    """Handles audio loading, processing, and feature extraction"""
    
    def __init__(self, config):
        # Initialize with audio parameters from config
        self.sr = config['model']['sample_rate'] if config else 24000
        self.hop_length = config['model']['hop_length'] if config else 240
        self.win_length = config['model']['win_length'] if config else 1024
        self.n_mels = config['model']['n_mels'] if config else 80
    
    def load_audio(self, file_path):
        """Load and process audio file, resampling if necessary"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Load audio
        audio, sr = sf.read(file_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != self.sr:
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sr
            )
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()
            sr = self.sr
        
        return audio, sr
    
    def extract_features(self, audio, sr):
        """Extract mel spectrogram and F0 from audio"""
        # Convert audio to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            
        # Extract mel spectrogram using torchaudio
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=40,  # Common minimum frequency
            f_max=sr/2,  # Nyquist frequency
            n_mels=self.n_mels,
            power=2.0
        )
        
        mel_spec = mel_transform(audio_tensor)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        mel_norm = self.normalize_mel(mel_db)
        
        # Convert to numpy for plotting
        mel_np = mel_norm.squeeze(0).cpu().numpy()
        
        # Extract F0
        f0 = self.extract_f0(audio, sr, self.hop_length)
        
        return {
            'mel_spectrogram': mel_np,
            'f0': f0,
            'duration': len(audio) / sr
        }
    
    def normalize_mel(self, mel_db):
        """Normalize mel spectrogram"""
        # Simple normalization to 0-1 range
        mel_min = torch.min(mel_db)
        mel_max = torch.max(mel_db)
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-5)
        return mel_norm
    
    def extract_f0(self, audio, sr, hop_length):
        """Extract F0 (fundamental frequency) from audio"""
        try:
            # Try to import parselmouth for better F0 extraction
            import parselmouth
            from parselmouth.praat import call
            
            # Create a Praat Sound object
            sound = parselmouth.Sound(audio, sr)
            
            # Extract pitch using Praat
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            
            # Extract F0 values at regular intervals
            f0_values = []
            for i in range(0, len(audio), hop_length):
                time = i / sr
                f0 = call(pitch, "Get value at time", time, "Hertz", "Linear")
                if np.isnan(f0):
                    f0 = 0.0  # Replace NaN with 0
                f0_values.append(f0)
            
            return np.array(f0_values)
            
        except ImportError:
            # Fall back to a simple method if parselmouth is not available
            # This is just a placeholder and won't give accurate F0
            return np.zeros(len(audio) // hop_length + 1)
    
    def normalize_audio(self, file_path, target_db_fs=-18, suppress_outliers=True, outlier_percentile=80, outlier_threshold=3.0):
        """
        Normalize audio file to target dB FS level with protection against ice pick amplitudes
        
        Args:
            file_path (str): Path to the audio file
            target_db_fs (float): Target dB FS level, typically -18 dB FS for EBU R128 reference
            suppress_outliers (bool): Whether to detect and suppress outlier peaks
            outlier_percentile (float): Percentile to use for outlier detection (99-99.9 recommended)
            outlier_threshold (float): How many times above the percentile peak to consider as an outlier
                
        Returns:
            bool: Success status
            dict: Normalization details
        """
        if not os.path.exists(file_path):
            return False, {"error": f"File not found: {file_path}"}
            
        try:
            # Load the audio file
            audio, sr = sf.read(file_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            
            # Create a working copy of the audio
            processed_audio = np.copy(audio)
            
            # Calculate absolute amplitude values
            abs_audio = np.abs(audio)
            
            # Original peak before any processing
            original_peak = np.max(abs_audio)
            original_db_fs = 20 * np.log10(original_peak) if original_peak > 0 else -np.inf
            
            # Suppression stats
            suppression_applied = False
            num_samples_suppressed = 0
            
            # Check for and suppress outliers if requested
            if suppress_outliers:
                # Calculate percentile-based peak
                percentile_peak = np.percentile(abs_audio, outlier_percentile)
                
                # Set threshold for outlier detection
                outlier_threshold_value = percentile_peak * outlier_threshold
                
                # Find outlier samples
                outlier_mask = abs_audio > outlier_threshold_value
                num_outliers = np.sum(outlier_mask)
                
                if num_outliers > 0:
                    # Calculate suppression factor to bring outliers down to the percentile peak level
                    # Use original sign but scale magnitude
                    suppression_factors = np.ones_like(processed_audio)
                    suppression_factors[outlier_mask] = (outlier_threshold_value / abs_audio[outlier_mask])
                    
                    # Apply suppression only to outlier samples
                    processed_audio = processed_audio * suppression_factors
                    
                    # Update stats
                    suppression_applied = True
                    num_samples_suppressed = num_outliers
                    
                    logger.info(f"Suppressed {num_outliers} outlier samples ({num_outliers/len(audio)*100:.4f}% of audio)")
                    logger.info(f"Outlier threshold: {outlier_threshold_value:.6f} ({outlier_threshold}x the {outlier_percentile}th percentile)")
            
            # Recalculate peak amplitude after possible suppression
            peak_amplitude = np.max(np.abs(processed_audio))
            
            # Current level in dB FS
            current_db_fs = 20 * np.log10(peak_amplitude) if peak_amplitude > 0 else -np.inf
            
            # Calculate needed gain in dB
            gain_db = target_db_fs - current_db_fs
            
            # Convert dB gain to amplitude scalar
            gain_factor = 10 ** (gain_db / 20)
            
            # Apply gain
            normalized_audio = processed_audio * gain_factor
            
            # Final check to prevent clipping
            max_amplitude = np.max(np.abs(normalized_audio))
            if max_amplitude > 1.0:
                normalized_audio = normalized_audio / max_amplitude
                logger.warning(f"Audio was clipping after normalization. Applied additional scaling.")
            
            # Write back to the original file
            sf.write(file_path, normalized_audio, sr)
            
            # Return success and details
            return True, {
                "original_db_fs": original_db_fs,
                "processed_db_fs": current_db_fs,
                "target_db_fs": target_db_fs,
                "gain_db": gain_db,
                "outliers_suppressed": suppression_applied,
                "num_samples_suppressed": num_samples_suppressed,
                "suppression_percentage": (num_samples_suppressed/len(audio)*100) if num_samples_suppressed > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {str(e)}")
            return False, {"error": str(e)}
    
    def batch_normalize(self, file_list, target_db_fs=-18, progress_callback=None):
        """
        Normalize multiple audio files to target dB FS level
        
        Args:
            file_list (list): List of audio file paths
            target_db_fs (float): Target dB FS level
            progress_callback (callable): Function to call with progress updates
            
        Returns:
            dict: Counts of successful, skipped, and failed files
        """
        results = {
            "successful": [],
            "skipped": [],
            "failed": []
        }
        
        total_files = len(file_list)
        
        for i, file_path in enumerate(file_list):
            # Update progress if callback provided
            if progress_callback:
                progress_callback(i, total_files, os.path.basename(file_path))
            
            # Check if file exists
            if not os.path.exists(file_path):
                results["failed"].append((file_path, "File not found"))
                continue
            
            try:
                # Load the audio file
                audio, sr = sf.read(file_path, dtype='float32')
                
                # Convert to mono if stereo
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = audio.mean(axis=1)
                
                # Calculate peak amplitude
                peak_amplitude = np.max(np.abs(audio))
                
                # Current level in dB FS
                current_db_fs = 20 * np.log10(peak_amplitude) if peak_amplitude > 0 else -np.inf
                
                # Skip if already within 0.5 dB of target (to avoid unnecessary processing)
                if abs(current_db_fs - target_db_fs) < 0.5:
                    results["skipped"].append(file_path)
                    logger.info(f"Skipped {os.path.basename(file_path)} - already at {current_db_fs:.2f} dB FS (target: {target_db_fs:.2f} dB FS)")
                    continue
                
                # Calculate needed gain in dB
                gain_db = target_db_fs - current_db_fs
                
                # Convert dB gain to amplitude scalar
                gain_factor = 10 ** (gain_db / 20)
                
                # Apply gain
                normalized_audio = audio * gain_factor
                
                # Ensure we don't clip (shouldn't happen when normalizing to negative dB FS)
                if np.max(np.abs(normalized_audio)) > 1.0:
                    normalized_audio = normalized_audio / np.max(np.abs(normalized_audio))
                    logger.warning(f"Audio was clipping after normalization. Applied additional scaling.")
                
                # Write back to the original file
                sf.write(file_path, normalized_audio, sr)
                
                # Add to successful files
                results["successful"].append(file_path)
                logger.info(f"Normalized {os.path.basename(file_path)} from {current_db_fs:.2f} dB FS to {target_db_fs:.2f} dB FS (gain: {gain_db:.2f} dB)")
                
            except Exception as e:
                # Log the error and add to failed files
                error_message = f"Error normalizing {os.path.basename(file_path)}: {str(e)}"
                logger.error(error_message)
                results["failed"].append((file_path, str(e)))
        
        return results