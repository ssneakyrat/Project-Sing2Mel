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
    
    def normalize_audio(self, file_path, target_db_fs=-18, detect_end_spikes=True, outlier_percentile=99.5):
        """
        Normalize audio file to target dB FS level with protection against ice pick amplitudes
        
        Args:
            file_path (str): Path to the audio file
            target_db_fs (float): Target dB FS level, typically -18 dB FS for EBU R128 reference
            detect_end_spikes (bool): Whether to detect and trim ice pick artifacts at the end
            outlier_percentile (float): Percentile to use for outlier detection (99-99.9 recommended)
                
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
            
            original_length = len(audio)
            trimmed_samples = 0
            
            # Detect and trim ice pick at the end if requested
            if detect_end_spikes and len(audio) > sr * 0.1:  # At least 100ms of audio
                # Define analysis windows
                end_window_ms = 50  # Analyze last 50ms for potential ice picks
                end_window_samples = int((end_window_ms / 1000) * sr)
                
                # Get the main body of the audio (everything except the very end)
                main_body = audio[:-end_window_samples] if end_window_samples < len(audio) else audio
                
                # Get the end section 
                end_section = audio[-end_window_samples:] if end_window_samples < len(audio) else np.array([])
                
                if len(end_section) > 0:
                    # Calculate statistics for both sections
                    main_body_percentile = np.percentile(np.abs(main_body), outlier_percentile)
                    end_section_max = np.max(np.abs(end_section))
                    
                    # Define what qualifies as an ice pick (end max significantly higher than main body)
                    # 5.0 is a threshold meaning "5 times louder" - can be adjusted based on your specific audio
                    ice_pick_ratio_threshold = 5.0
                    
                    if end_section_max > main_body_percentile * ice_pick_ratio_threshold:
                        # Ice pick detected! Find where it starts
                        end_array = np.abs(end_section)
                        spike_threshold = main_body_percentile * 3.0  # Less strict for finding start point
                        
                        # Find where the spike begins by going backwards from the end
                        spike_start_idx = 0
                        for i in range(len(end_array)-1, 0, -1):
                            if end_array[i] > spike_threshold and end_array[i-1] < spike_threshold:
                                spike_start_idx = i
                                break
                        
                        # Calculate how many samples to trim
                        trimmed_samples = end_window_samples - spike_start_idx
                        if trimmed_samples > 0:
                            audio = audio[:-trimmed_samples]
                            logger.warning(f"Ice pick detected at end of audio: {trimmed_samples} samples ({trimmed_samples/sr*1000:.1f}ms) trimmed")
            
            # Calculate absolute amplitude values
            abs_audio = np.abs(audio)
            
            # Get max peak
            raw_peak_amplitude = np.max(abs_audio)
            
            # Get percentile-based peak (to handle any remaining outliers)
            percentile_peak = np.percentile(abs_audio, outlier_percentile)
            
            # Determine if outliers exist throughout the audio
            peak_ratio = raw_peak_amplitude / percentile_peak if percentile_peak > 0 else 1.0
            has_outliers = peak_ratio > 2.0
            
            # Choose peak amplitude for normalization
            if has_outliers:
                logger.warning(f"Detected outlier peaks in audio: max={raw_peak_amplitude:.4f}, {outlier_percentile}th percentile={percentile_peak:.4f}, ratio={peak_ratio:.2f}")
                peak_amplitude = percentile_peak
            else:
                peak_amplitude = raw_peak_amplitude
            
            # Current level in dB FS
            current_db_fs = 20 * np.log10(peak_amplitude) if peak_amplitude > 0 else -np.inf
            
            # Calculate needed gain in dB
            gain_db = target_db_fs - current_db_fs
            
            # Convert dB gain to amplitude scalar
            gain_factor = 10 ** (gain_db / 20)
            
            # Apply gain
            normalized_audio = audio * gain_factor
            
            # Final check to prevent clipping
            max_amplitude = np.max(np.abs(normalized_audio))
            if max_amplitude > 1.0:
                normalized_audio = normalized_audio / max_amplitude
                logger.warning(f"Audio was clipping after normalization. Applied additional scaling.")
            
            # Write back to the original file
            sf.write(file_path, normalized_audio, sr)
            
            # Return success and details
            return True, {
                "original_db_fs": current_db_fs,
                "target_db_fs": target_db_fs,
                "gain_db": gain_db,
                "has_outliers": has_outliers,
                "trimmed_samples": trimmed_samples,
                "trimmed_ms": (trimmed_samples / sr * 1000) if trimmed_samples > 0 else 0,
                "original_duration_sec": original_length / sr,
                "new_duration_sec": len(normalized_audio) / sr
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