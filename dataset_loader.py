#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset loader module for interfacing with the SingingVoiceDataset class.
"""

import os
import pickle
import torch
import numpy as np
import logging
from threading import Thread
from PyQt5.QtCore import QObject, pyqtSignal, QThread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DatasetLoader")

class DatasetWorker(QThread):
    """
    Worker thread for loading dataset files without blocking the UI.
    """
    # Define signals
    progressChanged = pyqtSignal(int)
    statusChanged = pyqtSignal(str)
    error = pyqtSignal(str)
    completed = pyqtSignal(object)
    
    def __init__(self, dataset_path=None, cache_path=None, max_files=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.cache_path = cache_path
        self.max_files = max_files
        self.stop_requested = False
    
    def run(self):
        """Run the worker thread."""
        try:
            self.statusChanged.emit("Starting dataset loading...")
            
            if self.cache_path and os.path.exists(self.cache_path):
                # Load dataset from cache
                self.statusChanged.emit(f"Loading cached dataset from {self.cache_path}")
                self.progressChanged.emit(10)
                
                # Placeholder for actual loading code
                # In Phase 2, this would import and use the SingingVoiceDataset class
                # dataset = SingingVoiceDataset(cache_path=self.cache_path)
                
                # Simulate loading time
                import time
                for i in range(11, 101, 10):
                    if self.stop_requested:
                        return
                    time.sleep(0.2)
                    self.progressChanged.emit(i)
                
                # Create a mock dataset for UI testing
                dataset = self._create_mock_dataset()
                
                self.statusChanged.emit("Dataset loaded from cache")
                self.completed.emit(dataset)
                
            elif self.dataset_path and os.path.isdir(self.dataset_path):
                # Create dataset from directory
                self.statusChanged.emit(f"Creating dataset from {self.dataset_path}")
                self.progressChanged.emit(5)
                
                # Placeholder for actual creation code
                # In Phase 2, this would import and use the SingingVoiceDataset class
                # dataset = SingingVoiceDataset(dataset_dir=self.dataset_path, max_files=self.max_files)
                
                # Simulate processing time
                import time
                for i in range(6, 101, 5):
                    if self.stop_requested:
                        return
                    time.sleep(0.3)
                    self.progressChanged.emit(i)
                
                # Create a mock dataset for UI testing
                dataset = self._create_mock_dataset()
                
                self.statusChanged.emit("Dataset created from directory")
                self.completed.emit(dataset)
                
            else:
                self.error.emit("Invalid dataset path or cache file")
        
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            self.error.emit(f"Error loading dataset: {str(e)}")
    
    def _create_mock_dataset(self):
        """
        Create a mock dataset for UI testing in Phase 1.
        In Phase 2, this will be replaced with actual dataset loading.
        """
        # Create a simple dictionary to simulate dataset
        mock_data = {
            'singers': ['Singer 1', 'Singer 2', 'Singer 3'],
            'languages': ['English', 'Mandarin', 'Spanish'],
            'phones': ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH'],
            'files': [
                {'filename': 'song1', 'singer': 'Singer 1', 'language': 'English', 'duration': 45.2},
                {'filename': 'song2', 'singer': 'Singer 1', 'language': 'Mandarin', 'duration': 32.8},
                {'filename': 'song3', 'singer': 'Singer 2', 'language': 'English', 'duration': 28.5},
                {'filename': 'song4', 'singer': 'Singer 2', 'language': 'Spanish', 'duration': 37.1},
                {'filename': 'song5', 'singer': 'Singer 3', 'language': 'Mandarin', 'duration': 42.9}
            ],
            'sample_rate': 24000,
            'hop_length': 240,
            'n_mels': 80
        }
        
        logger.info("Created mock dataset")
        return mock_data
    
    def stop(self):
        """Request the worker to stop."""
        self.stop_requested = True

class DatasetLoader(QObject):
    """
    Interface for loading and accessing the singing voice dataset.
    """
    # Define signals
    datasetLoaded = pyqtSignal(object)
    loadProgress = pyqtSignal(int)
    loadStatus = pyqtSignal(str)
    loadError = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset = None
        self.worker = None
    
    def load_dataset(self, dataset_path=None, cache_path=None, max_files=None):
        """
        Load the dataset from a directory or cache file.
        
        Args:
            dataset_path: Path to the dataset directory
            cache_path: Path to a cached dataset file
            max_files: Maximum number of files to load
        """
        # Stop any existing worker
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        # Create a new worker thread
        self.worker = DatasetWorker(dataset_path, cache_path, max_files)
        
        # Connect signals
        self.worker.progressChanged.connect(self.loadProgress)
        self.worker.statusChanged.connect(self.loadStatus)
        self.worker.error.connect(self.loadError)
        self.worker.completed.connect(self._on_dataset_loaded)
        
        # Start the worker
        self.worker.start()
    
    def _on_dataset_loaded(self, dataset):
        """Handle dataset loaded event."""
        self.dataset = dataset
        self.datasetLoaded.emit(dataset)
    
    def get_item(self, item_id):
        """
        Get a specific item from the dataset.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            Dictionary with item data or None if not found
        """
        if not self.dataset or not self.dataset.get('files'):
            return None
        
        # In Phase 1, simply return a mock item
        # In Phase 2, this would retrieve actual data
        
        # Create mock data for visualization testing
        mock_item = {
            'audio': np.sin(np.linspace(0, 100, 24000 * 3)) * 0.5,  # 3 seconds of sine wave
            'mel': np.random.rand(80, 300) * 0.5 + 0.2,  # Random mel spectrogram
            'f0': 220 + 30 * np.sin(np.linspace(0, 10, 300)),  # Sine wave f0
            'phones': ['SIL', 'HH', 'AH', 'L', 'OW', 'W', 'ER', 'L', 'D', 'SIL'],  # Example phones
            'phone_timestamps': [0, 20, 40, 60, 100, 140, 180, 220, 240, 280],  # Example timestamps
            'sample_rate': 24000,
            'hop_length': 240
        }
        
        return mock_item
    
    def get_singer_files(self, singer_id):
        """Get all files for a specific singer."""
        if not self.dataset or not self.dataset.get('files'):
            return []
        
        # Filter files by singer
        return [f for f in self.dataset['files'] if f['singer'] == singer_id]
    
    def get_language_files(self, language_id):
        """Get all files for a specific language."""
        if not self.dataset or not self.dataset.get('files'):
            return []
        
        # Filter files by language
        return [f for f in self.dataset['files'] if f['language'] == language_id]
    
    def get_statistics(self):
        """Get dataset statistics."""
        if not self.dataset:
            return None
        
        # In Phase 1, return mock statistics
        # In Phase 2, compute actual statistics from the dataset
        
        stats = {
            'total_files': len(self.dataset.get('files', [])),
            'total_duration': sum(f.get('duration', 0) for f in self.dataset.get('files', [])),
            'singers': self.dataset.get('singers', []),
            'languages': self.dataset.get('languages', []),
            'phones': self.dataset.get('phones', [])
        }
        
        return stats
