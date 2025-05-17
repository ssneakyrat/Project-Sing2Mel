import sys
import os
import yaml
import numpy as np
import traceback
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel,
                            QSplitter, QMessageBox, QPushButton, QScrollArea,
                            QSlider, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import glob
from collections import namedtuple

from ui.dataset.spectrogram_canvas import SpectrogramCanvas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DatasetViewer")

import soundfile as sf
import torchaudio
import torch

# Create a namedtuple for file metadata
FileMetadata = namedtuple('FileMetadata', [
    'wav_file', 'lab_file', 'singer_id', 'language_id', 'singer_idx', 
    'language_idx', 'base_name'
])

class ScrollableSpectrogramWidget(QScrollArea):
    """A scrollable container for the spectrogram canvas"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create the spectrogram canvas
        self.spectrogram_canvas = SpectrogramCanvas(self, width=8, height=6)
        
        # Set the canvas as the widget for the scroll area
        self.setWidget(self.spectrogram_canvas)
        
        # Keep this False but properly size the canvas
        self.setWidgetResizable(False)  
        
        # Always show scrollbars for debugging purposes
        # Change back to AsNeeded after confirming they work
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        # Set frame style
        self.setFrameStyle(QScrollArea.StyledPanel)

class DatasetViewer(QMainWindow):
    """Main application window for the dataset viewer"""
    def __init__(self):
        super().__init__()
        
        # Load config
        try:
            with open('config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
                
            self.dataset_dir = self.config['data']['dataset_dir']
            self.map_file = self.config['data']['map_file']
        except Exception as e:
            self.show_error(f"Error loading config: {str(e)}")
            self.config = None
            self.dataset_dir = './datasets/'  # Default value
            self.map_file = 'mappings.yaml'   # Default value
        
        # Load mappings
        try:
            with open(self.map_file, 'r') as f:
                self.mappings = yaml.safe_load(f)
            self.singer_map = self.mappings.get('singer_map', {})
            self.language_map = self.mappings.get('lang_map', {})
            self.phone_map = self.mappings.get('phone_map', {})  # Load phone map for reference
        except Exception as e:
            self.show_error(f"Error loading mappings: {str(e)}")
            self.singer_map = {}
            self.language_map = {}
            self.phone_map = {}
            
        # Initialize audio processing params from config
        self.sr = self.config['model']['sample_rate'] if self.config else 24000
        self.hop_length = self.config['model']['hop_length'] if self.config else 240
        self.win_length = self.config['model']['win_length'] if self.config else 1024
        self.n_mels = self.config['model']['n_mels'] if self.config else 80
        
        # Width scale factor for spectrograms
        self.width_scale_factor = 2.0
        
        # Audio playback
        self.media_player = QMediaPlayer()
        self.current_audio_file = None
        self.update_timer = QTimer()
        self.update_timer.setInterval(50)  # Update every 50ms
        self.update_timer.timeout.connect(self.update_playback_position)
        
        # Set up the UI
        self.init_ui()
        
        # Load the dataset
        self.load_dataset()
            
    def show_error(self, message):
        """Show an error message box"""
        QMessageBox.critical(self, "Error", message)
        
    def init_ui(self):
        """Initialize the user interface with scrollable spectrogram and waveform display"""
        self.setWindowTitle('Singing Voice Dataset Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Dataset tree and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Moved spectrogram controls to the top of the left panel
        # Add spectrogram controls
        controls_group = QGroupBox("Spectrogram Controls")
        controls_layout = QVBoxLayout()
        
        # Add width scale slider with label
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Width Scale:")
        self.width_scale_slider = QSlider(Qt.Horizontal)
        self.width_scale_slider.setRange(10, 500)  # 0.1x to 5.0x
        self.width_scale_slider.setValue(200)  # Default 2.0x
        self.width_scale_slider.setTickPosition(QSlider.TicksBelow)
        self.width_scale_slider.setTickInterval(50)
        self.width_scale_value_label = QLabel("2.0x")
        
        # Connect slider value change signal
        self.width_scale_slider.valueChanged.connect(self.on_scale_slider_changed)
        
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.width_scale_slider)
        scale_layout.addWidget(self.width_scale_value_label)
        controls_layout.addLayout(scale_layout)
        
        # Add dB FS scale controls
        db_scale_layout = QHBoxLayout()
        db_min_label = QLabel("Min dB FS:")
        self.db_min_slider = QSlider(Qt.Horizontal)
        self.db_min_slider.setRange(-100, -10)  # -100 dB to -10 dB
        self.db_min_slider.setValue(-60)  # Default -60 dB
        self.db_min_slider.setTickPosition(QSlider.TicksBelow)
        self.db_min_slider.setTickInterval(10)
        self.db_min_value_label = QLabel("-60 dB")
        
        # Connect db min slider value change signal
        self.db_min_slider.valueChanged.connect(self.on_db_min_slider_changed)
        
        db_scale_layout.addWidget(db_min_label)
        db_scale_layout.addWidget(self.db_min_slider)
        db_scale_layout.addWidget(self.db_min_value_label)
        controls_layout.addLayout(db_scale_layout)
        
        # Add playback controls
        playback_layout = QHBoxLayout()
        
        # Play button
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)  # Disabled until audio is loaded
        self.play_button.clicked.connect(self.on_play_clicked)
        playback_layout.addWidget(self.play_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)  # Disabled until audio is playing
        self.stop_button.clicked.connect(self.on_stop_clicked)
        playback_layout.addWidget(self.stop_button)
        
        # Current position and duration label
        self.time_label = QLabel("Position: 0:00 / 0:00")
        playback_layout.addWidget(self.time_label)
        
        controls_layout.addLayout(playback_layout)
        
        # Add save phoneme boundaries button
        phoneme_layout = QHBoxLayout()
        
        # Save button for phoneme boundaries
        self.save_boundaries_button = QPushButton("Save Phoneme Boundaries")
        self.save_boundaries_button.setEnabled(False)  # Disabled until phonemes are modified
        self.save_boundaries_button.setToolTip("Save modified phoneme boundaries to the .lab file")
        self.save_boundaries_button.clicked.connect(self.on_save_boundaries_clicked)
        self.save_boundaries_button.setStyleSheet("background-color: #f8d8a8; font-weight: bold;")  # Light orange background
        phoneme_layout.addWidget(self.save_boundaries_button)
        
        controls_layout.addLayout(phoneme_layout)
        
        # Add audio processing controls
        processing_layout = QHBoxLayout()
        
        # Normalize to -18 dB FS button
        self.normalize_button = QPushButton("Normalize to -18 dB FS")
        self.normalize_button.setEnabled(False)  # Disabled until audio is loaded
        self.normalize_button.setToolTip("Process audio to -18 dB FS loudness standard and overwrite the file")
        self.normalize_button.clicked.connect(self.on_normalize_clicked)
        self.normalize_button.setStyleSheet("background-color: #a8d8a8; font-weight: bold;")  # Light green background
        processing_layout.addWidget(self.normalize_button)
        
        # Batch Normalize button
        self.batch_normalize_button = QPushButton("Batch Normalize Dataset")
        self.batch_normalize_button.setToolTip("Normalize all audio files in the dataset to -18 dB FS")
        self.batch_normalize_button.clicked.connect(self.on_batch_normalize_clicked)
        self.batch_normalize_button.setStyleSheet("background-color: #a8d8c8; font-weight: bold;")  # Blue-green background
        processing_layout.addWidget(self.batch_normalize_button)
        
        # Add the audio processing layout
        controls_layout.addLayout(processing_layout)
        
        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)  # Add controls to the left panel
        
        # Add refresh button
        refresh_button = QPushButton("Refresh Dataset")
        refresh_button.clicked.connect(self.load_dataset)
        left_layout.addWidget(refresh_button)
        
        # Dataset statistics label
        self.stats_label = QLabel("Dataset Statistics: Loading...")
        left_layout.addWidget(self.stats_label)
        
        # Add tree widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(['Dataset Browser'])
        self.tree_widget.itemClicked.connect(self.on_item_clicked)
        left_layout.addWidget(self.tree_widget)
        
        # Right panel: Spectrogram display only now
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Use the scrollable spectrogram widget instead of just the canvas
        self.scrollable_spectrogram = ScrollableSpectrogramWidget(right_panel)
        self.spectrogram_canvas = self.scrollable_spectrogram.spectrogram_canvas
        
        # Connect canvas signals
        self.spectrogram_canvas.audio_loaded.connect(self.on_audio_loaded)
        self.spectrogram_canvas.boundaries_modified.connect(self.on_boundaries_modified)
        self.spectrogram_canvas.phonemes_modified.connect(self.on_phonemes_modified)

        right_layout.addWidget(self.scrollable_spectrogram)
        
        # Set up media player signals
        self.media_player.stateChanged.connect(self.on_media_state_changed)
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        
        # Add widgets to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial sizes
        splitter.setSizes([300, 900])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set main widget
        self.setCentralWidget(main_widget)
    
    def on_boundaries_modified(self):
        """Handle signal that phoneme boundaries have been modified"""
        # Enable the save button
        self.save_boundaries_button.setEnabled(True)
        # Visual feedback
        self.save_boundaries_button.setStyleSheet("background-color: #f8a8a8; font-weight: bold;")  # Highlight with red
    
    def on_phonemes_modified(self):
        """Handle signal that phoneme labels have been modified"""
        # Enable the save button
        self.save_boundaries_button.setEnabled(True)
        # Visual feedback
        self.save_boundaries_button.setStyleSheet("background-color: #f8a8a8; font-weight: bold;")  # Highlight with red
        
    def on_save_boundaries_clicked(self):
        """Handle save button click to update lab file"""
        # Get the current selected item
        current_item = self.tree_widget.currentItem()
        if not current_item:
            return
            
        file_task = current_item.data(0, Qt.UserRole)
        if not file_task or not file_task.lab_file:
            return
        
        # Get modified phoneme data
        modified_data = self.spectrogram_canvas.get_modified_lab_data()
        if not modified_data:
            self.show_error("No changes to save")
            return
        
        phones, start_times, end_times = modified_data
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            'Confirm Lab File Update',
            f"This will overwrite the phoneme boundaries and labels in {os.path.basename(file_task.lab_file)}. Continue?",  # Updated message
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Update the lab file
        success = self.update_lab_file(file_task.lab_file, phones, start_times, end_times)
        
        if success:
            # Disable save button and reset style
            self.save_boundaries_button.setEnabled(False)
            self.save_boundaries_button.setStyleSheet("background-color: #f8d8a8; font-weight: bold;")
            
            # Show success message
            QMessageBox.information(
                self,
                "Save Complete",
                f"Phoneme boundaries and labels have been updated in {os.path.basename(file_task.lab_file)}."  # Updated message
            )
            
            # Reload to reflect changes
            self.display_spectrogram(file_task.wav_file, file_task.lab_file)
    
    def update_lab_file(self, lab_file, phones, start_times, end_times):
        """Update a lab file with new phoneme boundaries"""
        try:
            with open(lab_file, 'w') as f:
                for phone, start, end in zip(phones, start_times, end_times):
                    f.write(f"{start:.6f} {end:.6f} {phone}\n")
            return True
        except Exception as e:
            self.show_error(f"Error updating lab file: {str(e)}")
            return False
    
    def on_scale_slider_changed(self, value):
        """Handle width scale slider value change"""
        # Calculate scale factor (range 0.1x to 5.0x)
        self.width_scale_factor = value / 100.0
        self.width_scale_value_label.setText(f"{self.width_scale_factor:.1f}x")
        
        # Update spectrogram if canvas exists
        if self.spectrogram_canvas:
            self.spectrogram_canvas.rescale_width(self.width_scale_factor)
    
    def on_db_min_slider_changed(self, value):
        """Handle dB min slider value change"""
        # Update label text
        self.db_min_value_label.setText(f"{value} dB")
        
        # Update spectrogram canvas dB scale
        if self.spectrogram_canvas:
            self.spectrogram_canvas.update_db_scale(value, 0)  # Keep max at 0 dB FS
    
    def on_play_clicked(self):
        """Handle play button click"""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
            self.update_timer.start()
    
    def on_stop_clicked(self):
        """Handle stop button click"""
        self.media_player.stop()
        self.update_timer.stop()
        
        # Hide playback position line
        if self.spectrogram_canvas:
            self.spectrogram_canvas.hide_playback_position()
    
    def on_normalize_clicked(self):
        """Handle normalize button click"""
        # Get the current selected item
        current_item = self.tree_widget.currentItem()
        if current_item:
            file_task = current_item.data(0, Qt.UserRole)
            if file_task and file_task.wav_file:
                # Show a confirmation dialog
                reply = QMessageBox.question(
                    self, 
                    'Confirm Normalization',
                    f"This will normalize {os.path.basename(file_task.wav_file)} to -18 dB FS and overwrite the original file. Continue?",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # Perform normalization
                    success = self.normalize_to_target_db(file_task.wav_file)
                    
                    if success:
                        QMessageBox.information(
                            self,
                            "Normalization Complete",
                            f"Audio file has been normalized to -18 dB FS."
                        )
    
    def on_batch_normalize_clicked(self):
        """Handle batch normalize button click"""
        self.batch_normalize_dataset()
    
    def on_media_state_changed(self, state):
        """Handle media player state changes"""
        if state == QMediaPlayer.PlayingState:
            self.play_button.setText("Pause")
            self.stop_button.setEnabled(True)
        else:
            self.play_button.setText("Play")
            if state == QMediaPlayer.StoppedState:
                self.stop_button.setEnabled(False)
                self.update_timer.stop()
                # Hide playback position line when stopped
                if self.spectrogram_canvas:
                    self.spectrogram_canvas.hide_playback_position()
    
    def on_position_changed(self, position):
        """Handle media player position changes"""
        # Update time label
        position_ms = position
        duration_ms = self.media_player.duration()
        
        position_str = self.format_time(position_ms)
        duration_str = self.format_time(duration_ms)
        
        self.time_label.setText(f"Position: {position_str} / {duration_str}")
    
    def on_duration_changed(self, duration):
        """Handle media player duration changes"""
        # Update time label with new duration
        position_ms = self.media_player.position()
        duration_ms = duration
        
        position_str = self.format_time(position_ms)
        duration_str = self.format_time(duration_ms)
        
        self.time_label.setText(f"Position: {position_str} / {duration_str}")
    
    def format_time(self, ms):
        """Format milliseconds to MM:SS format"""
        seconds = int(ms / 1000)
        minutes = int(seconds / 60)
        seconds = seconds % 60
        return f"{minutes}:{seconds:02d}"
    
    def update_playback_position(self):
        """Update the playback position line on the spectrogram"""
        if self.spectrogram_canvas and self.media_player.state() == QMediaPlayer.PlayingState:
            position_seconds = self.media_player.position() / 1000.0
            self.spectrogram_canvas.update_playback_position(position_seconds)
    
    def on_audio_loaded(self, duration):
        """Handle audio loaded signal from spectrogram canvas"""
        # Enable play and normalize buttons
        self.play_button.setEnabled(True)
        self.normalize_button.setEnabled(True)
        
    def scan_directory(self):
        """Scan dataset directory and find WAV and LAB file pairs"""
        file_tasks = []
        
        if not os.path.exists(self.dataset_dir):
            self.show_error(f"Dataset directory not found: {self.dataset_dir}")
            return []
            
        try:
            # First find the singer directories
            singer_dirs = glob.glob(os.path.join(self.dataset_dir, '*'))
            singer_dirs = [d for d in singer_dirs if os.path.isdir(d)]
            
            for singer_dir in singer_dirs:
                singer_id = os.path.basename(singer_dir)
                singer_idx = self.singer_map.get(singer_id, -1)
                
                if singer_idx == -1:
                    continue  # Skip unmapped singers
                
                # Find language directories within this singer
                language_dirs = glob.glob(os.path.join(singer_dir, '*'))
                language_dirs = [d for d in language_dirs if os.path.isdir(d)]
                
                for language_dir in language_dirs:
                    language_id = os.path.basename(language_dir)
                    language_idx = self.language_map.get(language_id, -1)
                   
                    if language_idx == -1:
                        continue  # Skip unmapped languages
                    
                    # Find all lab files
                    lab_files = glob.glob(os.path.join(language_dir+'/lab', '*.lab'))
                    
                    for lab_file in lab_files:
                        base_name = os.path.splitext(os.path.basename(lab_file))[0]
                        wav_file = os.path.join(language_dir+'/wav', f"{base_name}.wav")
                        
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
                        
                        file_tasks.append(task)

            return file_tasks
            
        except Exception as e:
            error_message = f"Error scanning directory: {str(e)}\n{traceback.format_exc()}"
            self.show_error(error_message)
            return []
    
    def load_dataset(self):
        """Load and display the dataset structure"""
        self.tree_widget.clear()
        self.stats_label.setText("Loading dataset...")
        
        if self.spectrogram_canvas:
            self.spectrogram_canvas.clear()
            
        # Scan directory
        file_tasks = self.scan_directory()
        
        if not file_tasks:
            self.stats_label.setText("No dataset files found")
            return
            
        # Organize by singer and language
        dataset_structure = {}
        
        for task in file_tasks:
            singer_id = task.singer_id
            language_id = task.language_id
            
            if singer_id not in dataset_structure:
                dataset_structure[singer_id] = {}
            
            if language_id not in dataset_structure[singer_id]:
                dataset_structure[singer_id][language_id] = []
            
            dataset_structure[singer_id][language_id].append(task)
        
        # Populate tree widget
        for singer_id in sorted(dataset_structure.keys()):
            singer_item = QTreeWidgetItem([f"Singer: {singer_id}"])
            self.tree_widget.addTopLevelItem(singer_item)
            
            languages = dataset_structure[singer_id]
            for language_id in sorted(languages.keys()):
                files = languages[language_id]
                language_item = QTreeWidgetItem([f"Language: {language_id} ({len(files)} files)"])
                singer_item.addChild(language_item)
                
                for file_task in sorted(files, key=lambda x: x.base_name):
                    file_item = QTreeWidgetItem([os.path.basename(file_task.lab_file)])
                    file_item.setData(0, Qt.UserRole, file_task)
                    language_item.addChild(file_item)
        
        self.tree_widget.expandAll()
        
        # Update statistics
        total_singers = len(dataset_structure)
        total_languages = sum(len(languages) for languages in dataset_structure.values())
        
        self.stats_label.setText(
            f"Dataset Statistics:\n"
            f"• Singers: {total_singers}\n"
            f"• Languages: {total_languages}\n"
            f"• Files: {len(file_tasks)}"
        )
    
    def on_item_clicked(self, item, column):
        """Handle item click in the tree widget"""
        file_task = item.data(0, Qt.UserRole)
        
        if file_task:
            # Reset audio player state
            self.media_player.stop()
            self.update_timer.stop()
            self.play_button.setText("Play")
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.normalize_button.setEnabled(False)  # Disable normalize button until audio is loaded
            self.save_boundaries_button.setEnabled(False)  # Disable save button until edits are made
            self.time_label.setText("Position: 0:00 / 0:00")
            
            # Display file info
            info_text = f"Singer: {file_task.singer_id} (ID: {file_task.singer_idx})\n"
            info_text += f"Language: {file_task.language_id} (ID: {file_task.language_idx})\n"
            info_text += f"Lab file: {os.path.basename(file_task.lab_file)}\n"
            info_text += f"WAV file: {os.path.basename(file_task.wav_file)}"
            
            self.display_spectrogram(file_task.wav_file, file_task.lab_file)
        else:
            # Reset audio player
            self.media_player.stop()
            self.update_timer.stop()
            self.play_button.setText("Play")
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.normalize_button.setEnabled(False)  # Disable normalize button
            self.save_boundaries_button.setEnabled(False)  # Disable save button
            self.time_label.setText("Position: 0:00 / 0:00")
            
            if self.spectrogram_canvas:
                self.spectrogram_canvas.clear()
    
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
    
    def normalize_mel(self, mel_db):
        """Normalize mel spectrogram"""
        # Simple normalization to 0-1 range
        mel_min = torch.min(mel_db)
        mel_max = torch.max(mel_db)
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-5)
        return mel_norm
    
    def read_lab_file(self, lab_file):
        """Read phone labels from lab file"""
        phones = []
        start_times = []
        end_times = []
        
        try:
            with open(lab_file, 'r') as f:
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
            print(f"Error reading lab file: {str(e)}")
            return [], [], []
    
    def display_spectrogram(self, wav_file, lab_file):
        """Load audio file and display its spectrogram with phoneme alignment and waveform"""
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found: {wav_file}")
            
        # Set current audio file
        self.current_audio_file = wav_file
        
        # Load audio file into media player
        if self.current_audio_file:
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.current_audio_file)))
            
        # Load audio
        audio, sr = sf.read(wav_file, dtype='float32')
        
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
        
        # Calculate audio duration
        audio_duration = len(audio) / sr
        
        # Read phoneme information
        phones, start_times, end_times = self.read_lab_file(lab_file)
        
        # Check for empty phoneme data
        if not phones or not start_times or not end_times:
            logger.warning(f"No phoneme data found in {lab_file}")
        
        # Extract F0
        f0 = self.extract_f0(audio, sr, self.hop_length)
        
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
        
        # Calculate audio duration and expected mel frames
        expected_mel_frames = mel_np.shape[1]
        
        # Validation of lab file timing
        if phones and len(phones) > 0:
            lab_max_time = max(end_times)
            if abs(lab_max_time - audio_duration) > 0.5:  # More than 0.5 seconds difference
                logger.warning(f"Lab file timing mismatch: lab_max_time={lab_max_time:.2f}s, audio_duration={audio_duration:.2f}s")
        
        # Display spectrogram with phoneme alignment and waveform using current width scale factor
        self.spectrogram_canvas.plot_spectrogram(
            mel_np,
            audio=audio,  # Pass the audio data for waveform display
            f0=f0, 
            phones=phones, 
            start_times=start_times, 
            end_times=end_times,
            sample_rate=sr,
            hop_length=self.hop_length,
            width_scale_factor=self.width_scale_factor,
            audio_duration=audio_duration
        )
    
    def normalize_to_target_db(self, wav_file, target_db_fs=-18):
        """
        Normalize audio file to target dB FS level, overwrite the original file, and refresh display
        
        Args:
            wav_file (str): Path to the WAV file
            target_db_fs (float): Target dB FS level, typically -18 dB FS for EBU R128 reference
        """
        if not os.path.exists(wav_file):
            self.show_error(f"WAV file not found: {wav_file}")
            return False
            
        try:
            self.media_player.stop()
            self.media_player.setMedia(QMediaContent()) 

            # Load the audio file
            audio, sr = sf.read(wav_file, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            
            # Calculate peak amplitude
            peak_amplitude = np.max(np.abs(audio))
            
            # Current level in dB FS
            current_db_fs = 20 * np.log10(peak_amplitude) if peak_amplitude > 0 else -np.inf
            
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
            sf.write(wav_file, normalized_audio, sr)
            
            # Log the normalization
            logger.info(f"Normalized {os.path.basename(wav_file)} from {current_db_fs:.2f} dB FS to {target_db_fs:.2f} dB FS (gain: {gain_db:.2f} dB)")
            
            # Update the UI if this is the currently selected file
            if wav_file == self.current_audio_file:
                # Get the current lab file from the current file_task
                current_item = self.tree_widget.currentItem()
                if current_item:
                    file_task = current_item.data(0, Qt.UserRole)
                    if file_task and file_task.lab_file:
                        # Reload the audio and regenerate the spectrogram
                        self.display_spectrogram(wav_file, file_task.lab_file)
                        
                        # Update the media player
                        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(wav_file)))
            
            return True
        except Exception as e:
            error_message = f"Error normalizing audio: {str(e)}\n{traceback.format_exc()}"
            self.show_error(error_message)
            return False
    
    def batch_normalize_dataset(self, target_db_fs=-10):
        """
        Normalize all audio files in the dataset to target dB FS level
        
        Args:
            target_db_fs (float): Target dB FS level, typically -18 dB FS for EBU R128 reference
        """
        # Scan directory to get all files
        file_tasks = self.scan_directory()
        
        if not file_tasks:
            self.show_error("No dataset files found for batch normalization")
            return
        
        # Create progress dialog
        progress_dialog = QMessageBox(self)
        progress_dialog.setWindowTitle("Batch Normalization")
        progress_dialog.setText(f"Preparing to normalize {len(file_tasks)} files to {target_db_fs} dB FS...")
        progress_dialog.setStandardButtons(QMessageBox.Cancel)
        progress_dialog.setDefaultButton(QMessageBox.Cancel)
        
        # Show the initial dialog to confirm
        confirm_dialog = QMessageBox(self)
        confirm_dialog.setWindowTitle("Confirm Batch Normalization")
        confirm_dialog.setText(f"This will normalize all {len(file_tasks)} audio files to {target_db_fs} dB FS and overwrite the original files.\n\nDo you want to continue?")
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm_dialog.setDefaultButton(QMessageBox.No)
        
        # If user confirms, proceed with batch normalization
        if confirm_dialog.exec_() == QMessageBox.Yes:
            # Create a new progress dialog with a QProgressBar
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt
            
            progress = QProgressDialog("Normalizing audio files...", "Cancel", 0, len(file_tasks), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("Batch Normalization Progress")
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            # Lists to track results
            successful_files = []
            failed_files = []
            skipped_files = []
            
            # Process each file
            for i, file_task in enumerate(file_tasks):
                # Update progress
                progress.setValue(i)
                progress.setLabelText(f"Processing: {os.path.basename(file_task.wav_file)}")
                
                # Process events to keep UI responsive
                QApplication.processEvents()
                
                # Check if user canceled
                if progress.wasCanceled():
                    break
                
                # Check if file exists
                if not os.path.exists(file_task.wav_file):
                    skipped_files.append(file_task.wav_file)
                    continue
                
                try:
                    # Load the audio file
                    audio, sr = sf.read(file_task.wav_file, dtype='float32')
                    
                    # Convert to mono if stereo
                    if len(audio.shape) > 1 and audio.shape[1] > 1:
                        audio = audio.mean(axis=1)
                    
                    # Calculate peak amplitude
                    peak_amplitude = np.max(np.abs(audio))
                    
                    # Current level in dB FS
                    current_db_fs = 20 * np.log10(peak_amplitude) if peak_amplitude > 0 else -np.inf
                    
                    # Skip if already within 0.5 dB of target (to avoid unnecessary processing)
                    if abs(current_db_fs - target_db_fs) < 0.5:
                        skipped_files.append(file_task.wav_file)
                        logger.info(f"Skipped {os.path.basename(file_task.wav_file)} - already at {current_db_fs:.2f} dB FS (target: {target_db_fs:.2f} dB FS)")
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
                    sf.write(file_task.wav_file, normalized_audio, sr)
                    
                    # Log the normalization
                    logger.info(f"Normalized {os.path.basename(file_task.wav_file)} from {current_db_fs:.2f} dB FS to {target_db_fs:.2f} dB FS (gain: {gain_db:.2f} dB)")
                    
                    # Add to successful files
                    successful_files.append(file_task.wav_file)
                    
                except Exception as e:
                    # Log the error and add to failed files
                    error_message = f"Error normalizing {os.path.basename(file_task.wav_file)}: {str(e)}"
                    logger.error(error_message)
                    failed_files.append((file_task.wav_file, str(e)))
            
            # Complete the progress
            progress.setValue(len(file_tasks))
            
            # Update the UI if this is the currently selected file
            current_item = self.tree_widget.currentItem()
            if current_item:
                file_task = current_item.data(0, Qt.UserRole)
                if file_task and file_task.wav_file:
                    # Check if the current file was processed
                    if file_task.wav_file in successful_files:
                        # Reload the audio and regenerate the spectrogram
                        self.display_spectrogram(file_task.wav_file, file_task.lab_file)
                        
                        # Update the media player
                        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_task.wav_file)))
            
            # Show summary dialog
            summary = f"Batch Normalization Summary:\n\n"
            summary += f"Target Level: {target_db_fs} dB FS\n"
            summary += f"Files Processed: {len(successful_files) + len(failed_files) + len(skipped_files)}\n"
            summary += f"   - Successfully Normalized: {len(successful_files)}\n"
            summary += f"   - Already at Target Level: {len(skipped_files)}\n"
            summary += f"   - Failed: {len(failed_files)}\n"
            
            if failed_files:
                summary += "\nFailed Files:\n"
                for file_path, error in failed_files[:10]:  # Show first 10 failures
                    summary += f"   - {os.path.basename(file_path)}: {error}\n"
                
                if len(failed_files) > 10:
                    summary += f"   - ... and {len(failed_files) - 10} more\n"
            
            QMessageBox.information(self, "Batch Normalization Complete", summary)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetViewer()
    window.show()
    sys.exit(app.exec_())