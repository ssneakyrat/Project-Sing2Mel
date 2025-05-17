import os
import glob
import yaml
import logging
from collections import namedtuple

logger = logging.getLogger("DatasetManager")

# Create a namedtuple for file metadata
FileMetadata = namedtuple('FileMetadata', [
    'wav_file', 'lab_file', 'singer_id', 'language_id', 'singer_idx', 
    'language_idx', 'base_name'
])

class DatasetManager:
    """Manages dataset structure and provides access to audio/lab files"""
    
    def __init__(self, config):
        self.config = config
        
        # Set dataset directory and map file from config
        self.dataset_dir = config['data']['dataset_dir'] if config else './datasets/'
        self.map_file = config['data']['map_file'] if config else 'mappings.yaml'
        
        # Initialize mappings
        self.singer_map = {}
        self.language_map = {}
        self.phone_map = {}
        
        # Load mappings
        self.load_mappings()
        
        # File structure and statistics
        self.file_tasks = []
        self.dataset_structure = {}
        self.statistics = {
            "singers": 0,
            "languages": 0,
            "files": 0
        }
    
    def load_mappings(self):
        """Load singer and language mappings from the map file"""
        try:
            if os.path.exists(self.map_file):
                with open(self.map_file, 'r') as f:
                    mappings = yaml.safe_load(f)
                self.singer_map = mappings.get('singer_map', {})
                self.language_map = mappings.get('lang_map', {})
                self.phone_map = mappings.get('phone_map', {})
                logger.info(f"Loaded mappings: {len(self.singer_map)} singers, {len(self.language_map)} languages, {len(self.phone_map)} phones")
            else:
                logger.warning(f"Map file not found: {self.map_file}")
        except Exception as e:
            logger.error(f"Error loading mappings: {str(e)}")
    
    def scan_dataset(self):
        """
        Scan dataset directory and find WAV and LAB file pairs
        
        Returns:
            tuple: (file_tasks, dataset_structure)
        """
        self.file_tasks = []
        self.dataset_structure = {}
        
        if not os.path.exists(self.dataset_dir):
            logger.error(f"Dataset directory not found: {self.dataset_dir}")
            return self.file_tasks, self.dataset_structure
            
        try:
            # First find the singer directories
            singer_dirs = glob.glob(os.path.join(self.dataset_dir, '*'))
            singer_dirs = [d for d in singer_dirs if os.path.isdir(d)]
            
            for singer_dir in singer_dirs:
                singer_id = os.path.basename(singer_dir)
                singer_idx = self.singer_map.get(singer_id, -1)
                
                if singer_idx == -1:
                    logger.warning(f"Skipping unmapped singer: {singer_id}")
                    continue  # Skip unmapped singers
                
                # Initialize singer in dataset structure
                if singer_id not in self.dataset_structure:
                    self.dataset_structure[singer_id] = {}
                
                # Find language directories within this singer
                language_dirs = glob.glob(os.path.join(singer_dir, '*'))
                language_dirs = [d for d in language_dirs if os.path.isdir(d)]
                
                for language_dir in language_dirs:
                    language_id = os.path.basename(language_dir)
                    language_idx = self.language_map.get(language_id, -1)
                   
                    if language_idx == -1:
                        logger.warning(f"Skipping unmapped language: {language_id}")
                        continue  # Skip unmapped languages
                    
                    # Initialize language in dataset structure
                    if language_id not in self.dataset_structure[singer_id]:
                        self.dataset_structure[singer_id][language_id] = []
                    
                    # Find all lab files
                    lab_dir = os.path.join(language_dir, 'lab')
                    if not os.path.exists(lab_dir):
                        logger.warning(f"Lab directory not found: {lab_dir}")
                        continue
                        
                    lab_files = glob.glob(os.path.join(lab_dir, '*.lab'))
                    
                    for lab_file in lab_files:
                        base_name = os.path.splitext(os.path.basename(lab_file))[0]
                        wav_dir = os.path.join(language_dir, 'wav')
                        wav_file = os.path.join(wav_dir, f"{base_name}.wav")
                        
                        # Check if WAV file exists
                        if not os.path.exists(wav_file):
                            logger.warning(f"WAV file not found for {lab_file}: {wav_file}")
                            continue
                        
                        # Create a file task
                        task = FileMetadata(
                            wav_file=wav_file,
                            lab_file=lab_file,
                            singer_id=singer_id,
                            language_id=language_id,
                            singer_idx=singer_idx,
                            language_idx=language_idx,
                            base_name=base_name
                        )
                        
                        # Add to list and structure
                        self.file_tasks.append(task)
                        self.dataset_structure[singer_id][language_id].append(task)

            # Update statistics
            self.statistics = self.calculate_statistics()
            
            logger.info(f"Scanned dataset: {self.statistics['singers']} singers, {self.statistics['languages']} languages, {self.statistics['files']} files")
            return self.file_tasks, self.dataset_structure
            
        except Exception as e:
            logger.error(f"Error scanning directory: {str(e)}")
            return self.file_tasks, self.dataset_structure
    
    def calculate_statistics(self):
        """Calculate dataset statistics"""
        total_singers = len(self.dataset_structure)
        
        total_languages = 0
        for singer_id in self.dataset_structure:
            total_languages += len(self.dataset_structure[singer_id])
        
        total_files = len(self.file_tasks)
        
        return {
            "singers": total_singers,
            "languages": total_languages,
            "files": total_files
        }
    
    def get_statistics(self):
        """Get dataset statistics"""
        return self.statistics
    
    def get_file_tasks(self):
        """Get all file tasks"""
        return self.file_tasks
    
    def get_dataset_structure(self):
        """Get dataset structure"""
        return self.dataset_structure
    
    def get_singer_languages(self, singer_id):
        """Get languages for a specific singer"""
        if singer_id in self.dataset_structure:
            return list(self.dataset_structure[singer_id].keys())
        return []
    
    def get_language_files(self, singer_id, language_id):
        """Get files for a specific singer and language"""
        if singer_id in self.dataset_structure and language_id in self.dataset_structure[singer_id]:
            return self.dataset_structure[singer_id][language_id]
        return []