import os
import logging

logger = logging.getLogger("LabFileHandler")

class LabFileHandler:
    """Handles reading and writing phoneme boundary files"""
    
    def __init__(self, phone_map=None):
        self.phone_map = phone_map or {}
    
    def read_lab_file(self, file_path):
        """
        Read phone labels from lab file
        
        Args:
            file_path (str): Path to the lab file
            
        Returns:
            tuple: (phones, start_times, end_times)
        """
        phones = []
        start_times = []
        end_times = []
        
        if not os.path.exists(file_path):
            logger.warning(f"Lab file not found: {file_path}")
            return phones, start_times, end_times
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        phone = parts[2]
                        
                        start_times.append(start_time)
                        end_times.append(end_time)
                        phones.append(phone)
            
            return phones, start_times, end_times
        except Exception as e:
            logger.error(f"Error reading lab file: {str(e)}")
            return [], [], []
    
    def write_lab_file(self, file_path, phones, start_times, end_times):
        """
        Write phoneme boundaries to lab file
        
        Args:
            file_path (str): Path to the lab file
            phones (list): List of phoneme labels
            start_times (list): List of start times
            end_times (list): List of end times
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure all lists have the same length
            if len(phones) != len(start_times) or len(phones) != len(end_times):
                logger.error("Mismatched lengths: phones, start_times, and end_times must have the same length")
                return False
                
            with open(file_path, 'w') as f:
                for phone, start, end in zip(phones, start_times, end_times):
                    f.write(f"{start:.6f} {end:.6f} {phone}\n")
            
            logger.info(f"Successfully wrote {len(phones)} phoneme boundaries to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing lab file: {str(e)}")
            return False
    
    def validate_boundaries(self, phones, start_times, end_times, audio_duration=None):
        """
        Validate phoneme boundaries for consistency and correctness
        
        Args:
            phones (list): List of phoneme labels
            start_times (list): List of start times
            end_times (list): List of end times
            audio_duration (float, optional): Duration of the audio file in seconds
            
        Returns:
            tuple: (is_valid, validation_messages)
        """
        validation_messages = []
        
        # Check if all lists are the same length
        if len(phones) != len(start_times) or len(phones) != len(end_times):
            validation_messages.append("Mismatched lengths: phones, start_times, and end_times must have the same length")
            return False, validation_messages
        
        # Check for empty lists
        if len(phones) == 0:
            validation_messages.append("No phoneme data provided")
            return False, validation_messages
        
        # Check for overlapping boundaries
        for i in range(len(phones) - 1):
            if end_times[i] > start_times[i + 1]:
                validation_messages.append(f"Overlapping boundaries at phoneme {i} ({phones[i]}) and {i+1} ({phones[i+1]})")
        
        # Check for negative durations
        for i in range(len(phones)):
            if end_times[i] < start_times[i]:
                validation_messages.append(f"Negative duration for phoneme {i} ({phones[i]})")
        
        # Check alignment with audio duration
        if audio_duration is not None:
            if start_times[0] < 0:
                validation_messages.append(f"First phoneme starts before 0 ({start_times[0]})")
                
            if end_times[-1] > audio_duration + 0.5:  # Allow 0.5s slack
                validation_messages.append(f"Last phoneme ends after audio duration (phoneme end: {end_times[-1]}, audio: {audio_duration})")
        
        # Check for valid phoneme labels
        if self.phone_map:
            for i, phone in enumerate(phones):
                if phone not in self.phone_map:
                    validation_messages.append(f"Unknown phoneme at position {i}: {phone}")
        
        # Return validation result
        is_valid = len(validation_messages) == 0
        return is_valid, validation_messages