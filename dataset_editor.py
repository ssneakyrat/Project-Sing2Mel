import sys
import os
import yaml
import traceback
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel,
                            QSplitter, QMessageBox, QPushButton, QScrollArea,
                            QSlider, QGroupBox, QProgressDialog)
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# Import our refactored modules
from ui.dataset.dataset_manager import DatasetManager
from ui.dataset.audio_processor import AudioProcessor
from ui.dataset.lab_file_handler import LabFileHandler
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
        except Exception as e:
            self.show_error(f"Error loading config: {str(e)}")
            self.config = None
        
        # Initialize components
        self.dataset_manager = DatasetManager(self.config)
        self.audio_processor = AudioProcessor(self.config)
        self.lab_file_handler = LabFileHandler(self.dataset_manager.phone_map)
        
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
        self.setWindowTitle('Dataset Editor')
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
        
        # Play Marker button
        self.play_marker_button = QPushButton("Play Markers")
        self.play_marker_button.setEnabled(False)  # Disabled until audio is loaded
        self.play_marker_button.setToolTip("Play audio between start and end markers")
        self.play_marker_button.clicked.connect(self.on_play_marker_clicked)
        self.play_marker_button.setStyleSheet("background-color: #a8f8a8;")  # Light green background
        playback_layout.addWidget(self.play_marker_button)

        # Current position and duration label
        self.time_label = QLabel("Position: 0:00 / 0:00")
        playback_layout.addWidget(self.time_label)

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
        
        # Add Sanitize Segmentation buttons
        sanitize_layout = QHBoxLayout()
        
        # Sanitize current file button
        self.sanitize_button = QPushButton("Sanitize Segmentation")
        self.sanitize_button.setEnabled(False)  # Disabled until a file is loaded
        self.sanitize_button.setToolTip("Add 'pau' at start/end of phoneme sequence and fix timing")
        self.sanitize_button.clicked.connect(self.on_sanitize_clicked)
        self.sanitize_button.setStyleSheet("background-color: #d8a8d8; font-weight: bold;")  # Light purple background
        sanitize_layout.addWidget(self.sanitize_button)
        
        # Batch Sanitize button
        self.batch_sanitize_button = QPushButton("Batch Sanitize Dataset")
        self.batch_sanitize_button.setToolTip("Sanitize all lab files in the dataset")
        self.batch_sanitize_button.clicked.connect(self.on_batch_sanitize_clicked)
        self.batch_sanitize_button.setStyleSheet("background-color: #c8a8d8; font-weight: bold;")  # Purple-blue background
        sanitize_layout.addWidget(self.batch_sanitize_button)
        
        # Add the sanitize layout
        controls_layout.addLayout(sanitize_layout)
        
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
            f"This will overwrite the phoneme boundaries and labels in {os.path.basename(file_task.lab_file)}. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Update the lab file using the lab file handler
        success = self.lab_file_handler.write_lab_file(file_task.lab_file, phones, start_times, end_times)
        
        if success:
            # Disable save button and reset style
            self.save_boundaries_button.setEnabled(False)
            self.save_boundaries_button.setStyleSheet("background-color: #f8d8a8; font-weight: bold;")
            
            # Show success message
            QMessageBox.information(
                self,
                "Save Complete",
                f"Phoneme boundaries and labels have been updated in {os.path.basename(file_task.lab_file)}."
            )
            
            # Reload to reflect changes
            self.display_spectrogram(file_task.wav_file, file_task.lab_file)

    def on_sanitize_clicked(self):
        """Handle sanitize segmentation button click for current file"""
        # Get the current selected item
        current_item = self.tree_widget.currentItem()
        if not current_item:
            return
            
        file_task = current_item.data(0, Qt.UserRole)
        if not file_task or not file_task.lab_file or not file_task.wav_file:
            return
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            'Confirm Sanitize Segmentation',
            f"This will modify the phoneme boundaries in {os.path.basename(file_task.lab_file)} "
            f"to ensure 'pau' phonemes at start and end. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Get audio duration
        try:
            audio, sr = self.audio_processor.load_audio(file_task.wav_file)
            audio_duration = len(audio) / sr
        except Exception as e:
            self.show_error(f"Error loading audio file: {str(e)}")
            return
        
        # Read lab file
        phones, start_times, end_times = self.lab_file_handler.read_lab_file(file_task.lab_file)
        
        if not phones or len(phones) == 0:
            self.show_error("No phoneme data found in lab file")
            return
        
        # Sanitize the segmentation
        phones, start_times, end_times = self.sanitize_segmentation(
            phones, start_times, end_times, audio_duration
        )
        
        # Write back to lab file
        success = self.lab_file_handler.write_lab_file(file_task.lab_file, phones, start_times, end_times)
        
        if success:
            # Show success message
            QMessageBox.information(
                self,
                "Sanitize Complete",
                f"Phoneme boundaries have been sanitized in {os.path.basename(file_task.lab_file)}."
            )
            
            # Reload to reflect changes
            self.display_spectrogram(file_task.wav_file, file_task.lab_file)
    
    def on_batch_sanitize_clicked(self):
        """Handle batch sanitize segmentation button click"""
        self.batch_sanitize_dataset()
    
    def sanitize_segmentation(self, phones, start_times, end_times, audio_duration):
        """
        Sanitize phoneme segmentation to ensure 'pau' at start and end
        
        Args:
            phones (list): List of phoneme labels
            start_times (list): List of start times
            end_times (list): List of end times
            audio_duration (float): Duration of the audio file in seconds
            
        Returns:
            tuple: (phones, start_times, end_times)
        """
        if not phones or len(phones) == 0:
            return phones, start_times, end_times
            
        # Make a copy of the lists to avoid modifying the originals
        phones = phones.copy()
        start_times = start_times.copy()
        end_times = end_times.copy()
        
        modified = False
        
        # Check if first phone is 'pau'
        if phones[0] != 'pau':
            # Split the first phone to make room for 'pau'
            original_start = start_times[0]
            split_point = min(0.1, (end_times[0] - start_times[0]) / 3)  # Use 1/3 of duration or 100ms
            
            # Insert 'pau' at the beginning
            phones.insert(0, 'pau')
            start_times.insert(0, 0.0)  # Start at 0
            end_times.insert(0, original_start + split_point)
            
            # Adjust the start time of the original first phone
            start_times[1] = original_start + split_point
            
            modified = True
        
        start_times[0] = 0.0
        
        # Check if last phone is 'pau'
        if phones[-1] != 'pau':
            # Split the last phone to make room for 'pau'
            original_end = end_times[-1]
            split_point = max(0.1, (end_times[-1] - start_times[-1]) / 3)  # Use 1/3 of duration or 100ms
            
            # Adjust the end time of the original last phone
            end_times[-1] = original_end - split_point
            
            # Add 'pau' at the end
            phones.append('pau')
            start_times.append(original_end - split_point)
            end_times.append(audio_duration)
            
            modified = True

        end_times[-1] = round(audio_duration * 10000000)
        
        return phones, start_times, end_times
    
    def batch_sanitize_dataset(self):
        """
        Sanitize all lab files in the dataset
        """
        # Get all files from dataset manager
        file_tasks = self.dataset_manager.get_file_tasks()
        
        if not file_tasks:
            self.show_error("No dataset files found for batch sanitization")
            return
        
        # Create a list of lab files with corresponding wav files
        processing_tasks = []
        for task in file_tasks:
            if os.path.exists(task.lab_file) and os.path.exists(task.wav_file):
                processing_tasks.append(task)
        
        # Show the initial dialog to confirm
        confirm_dialog = QMessageBox(self)
        confirm_dialog.setWindowTitle("Confirm Batch Sanitization")
        confirm_dialog.setText(
            f"This will sanitize all {len(processing_tasks)} lab files to ensure 'pau' "
            f"phonemes at the start and end, and proper timing alignment with audio.\n\n"
            f"Do you want to continue?"
        )
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm_dialog.setDefaultButton(QMessageBox.No)
        
        # If user confirms, proceed with batch sanitization
        if confirm_dialog.exec_() == QMessageBox.Yes:
            # Create a progress dialog
            progress = QProgressDialog("Sanitizing lab files...", "Cancel", 0, len(processing_tasks), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("Batch Sanitization Progress")
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            successful = []
            skipped = []
            failed = []
            
            # Process each file
            for i, task in enumerate(processing_tasks):
                # Update progress
                progress.setValue(i)
                progress.setLabelText(f"Processing: {os.path.basename(task.lab_file)}")
                QApplication.processEvents()
                
                # Check for cancel
                if progress.wasCanceled():
                    break
                
                try:
                    # Get audio duration
                    audio, sr = self.audio_processor.load_audio(task.wav_file)
                    audio_duration = len(audio) / sr
                    
                    # Read lab file
                    phones, start_times, end_times = self.lab_file_handler.read_lab_file(task.lab_file)
                    
                    if not phones or len(phones) == 0:
                        skipped.append((task.lab_file, "No phoneme data found"))
                        continue
                    
                    # Make a copy of original data to check if modified
                    original_phones = phones.copy()
                    original_starts = start_times.copy()
                    original_ends = end_times.copy()
                    
                    # Sanitize the segmentation
                    phones, start_times, end_times = self.sanitize_segmentation(
                        phones, start_times, end_times, audio_duration
                    )
                    
                    # Check if anything changed
                    if (phones == original_phones and 
                        start_times == original_starts and 
                        end_times == original_ends):
                        skipped.append((task.lab_file, "No changes needed"))
                        continue
                    
                    # Write back to lab file
                    success = self.lab_file_handler.write_lab_file(task.lab_file, phones, start_times, end_times)
                    
                    if success:
                        successful.append(task.lab_file)
                    else:
                        failed.append((task.lab_file, "Failed to write lab file"))
                        
                except Exception as e:
                    failed.append((task.lab_file, str(e)))
            
            # Complete the progress
            progress.setValue(len(processing_tasks))
            
            # Update the UI if this is the currently selected file
            current_item = self.tree_widget.currentItem()
            if current_item:
                file_task = current_item.data(0, Qt.UserRole)
                if file_task and file_task.lab_file:
                    # Check if the current file was processed
                    if file_task.lab_file in successful:
                        # Reload the display
                        self.display_spectrogram(file_task.wav_file, file_task.lab_file)
            
            # Show summary dialog
            summary = f"Batch Sanitization Summary:\n\n"
            summary += f"Files Processed: {len(processing_tasks)}\n"
            summary += f"   - Successfully Sanitized: {len(successful)}\n"
            summary += f"   - Already Valid: {len(skipped)}\n"
            summary += f"   - Failed: {len(failed)}\n"
            
            if failed:
                summary += "\nFailed Files:\n"
                for file_path, error in failed[:10]:  # Show first 10 failures
                    summary += f"   - {os.path.basename(file_path)}: {error}\n"
                
                if len(failed) > 10:
                    summary += f"   - ... and {len(failed) - 10} more\n"
            
            QMessageBox.information(self, "Batch Sanitization Complete", summary)
    
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
    
    def on_play_marker_clicked(self):
        """Handle play marker button click to play audio between markers"""
        if not self.current_audio_file:
            return
            
        # Get marker positions from canvas
        start_position, end_position = self.spectrogram_canvas.get_marker_positions()
        
        # Convert to milliseconds for media player
        start_ms = int(start_position * 1000)
        end_ms = int(end_position * 1000)
        
        # Check if the positions are valid
        if start_ms >= end_ms:
            QMessageBox.warning(self, "Invalid Marker Positions", 
                            "The start marker must be before the end marker.")
            return
            
        # Set position to start marker
        self.media_player.setPosition(start_ms)
        
        # Store end position to stop playback when reached
        self.playback_end_position = end_ms
        
        # Start playback
        self.media_player.play()
        self.update_timer.start()
        
        # Update button states
        self.play_button.setText("Pause")
        self.stop_button.setEnabled(True)
        
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
                    # Perform normalization using audio processor
                    success, details = self.audio_processor.normalize_audio(file_task.wav_file)
                    
                    if success:
                        QMessageBox.information(
                            self,
                            "Normalization Complete",
                            f"Audio file has been normalized to -18 dB FS."
                        )
                        
                        # Reload the current file
                        self.display_spectrogram(file_task.wav_file, file_task.lab_file)
    
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
        """Update the playback position line on the spectrogram and check markers"""
        if self.spectrogram_canvas and self.media_player.state() == QMediaPlayer.PlayingState:
            position_seconds = self.media_player.position() / 1000.0
            self.spectrogram_canvas.update_playback_position(position_seconds)
            
            # Check if playback has reached the end marker
            if hasattr(self, 'playback_end_position') and self.media_player.position() >= self.playback_end_position:
                self.media_player.pause()
                self.update_timer.stop()
                self.play_button.setText("Play")
    
    def on_audio_loaded(self, duration):
        """Handle audio loaded signal from spectrogram canvas"""
        # Enable play, normalize, and play marker buttons
        self.play_button.setEnabled(True)
        self.normalize_button.setEnabled(True)
        self.sanitize_button.setEnabled(True)
        self.play_marker_button.setEnabled(True)  # Enable play marker button
    
    def load_dataset(self):
        """Load and display the dataset structure"""
        self.tree_widget.clear()
        self.stats_label.setText("Loading dataset...")
        
        if self.spectrogram_canvas:
            self.spectrogram_canvas.clear()
            
        # Scan directory using dataset manager
        self.dataset_manager.scan_dataset()
        file_tasks = self.dataset_manager.get_file_tasks()
        dataset_structure = self.dataset_manager.get_dataset_structure()
        
        if not file_tasks:
            self.stats_label.setText("No dataset files found")
            return
            
        # Populate tree widget from dataset structure
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
        stats = self.dataset_manager.get_statistics()
        
        self.stats_label.setText(
            f"Dataset Statistics:\n"
            f"• Singers: {stats['singers']}\n"
            f"• Languages: {stats['languages']}\n"
            f"• Files: {stats['files']}"
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
            self.normalize_button.setEnabled(False)
            self.sanitize_button.setEnabled(False)  # Disable sanitize button
            self.save_boundaries_button.setEnabled(False)
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
            self.normalize_button.setEnabled(False)
            self.sanitize_button.setEnabled(False)  # Disable sanitize button
            self.save_boundaries_button.setEnabled(False)
            self.time_label.setText("Position: 0:00 / 0:00")
            
            if self.spectrogram_canvas:
                self.spectrogram_canvas.clear()
    
    def display_spectrogram(self, wav_file, lab_file):
        """Load audio file and display its spectrogram with phoneme alignment and waveform"""
        if not os.path.exists(wav_file):
            self.show_error(f"WAV file not found: {wav_file}")
            return
            
        # Set current audio file
        self.current_audio_file = wav_file
        
        # Load audio file into media player
        if self.current_audio_file:
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.current_audio_file)))
            
        try:
            # Load audio using audio processor
            audio, sr = self.audio_processor.load_audio(wav_file)
            
            # Extract features
            features = self.audio_processor.extract_features(audio, sr)
            mel_np = features['mel_spectrogram']
            f0 = features['f0']
            audio_duration = features['duration']
            
            # Read phoneme information using lab file handler
            phones, start_times, end_times = self.lab_file_handler.read_lab_file(lab_file)
            
            # Check for empty phoneme data
            if not phones or not start_times or not end_times:
                logger.warning(f"No phoneme data found in {lab_file}")
            
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
                hop_length=self.audio_processor.hop_length,
                width_scale_factor=self.width_scale_factor,
                audio_duration=audio_duration
            )
        except Exception as e:
            error_message = f"Error displaying spectrogram: {str(e)}\n{traceback.format_exc()}"
            self.show_error(error_message)
    
    def batch_normalize_dataset(self, target_db_fs=-18):
        """
        Normalize all audio files in the dataset to target dB FS level
        
        Args:
            target_db_fs (float): Target dB FS level, typically -18 dB FS for EBU R128 reference
        """
        # Get all files from dataset manager
        file_tasks = self.dataset_manager.get_file_tasks()
        
        if not file_tasks:
            self.show_error("No dataset files found for batch normalization")
            return
        
        # Create a list of wav files
        wav_files = [task.wav_file for task in file_tasks if os.path.exists(task.wav_file)]
        
        # Show the initial dialog to confirm
        confirm_dialog = QMessageBox(self)
        confirm_dialog.setWindowTitle("Confirm Batch Normalization")
        confirm_dialog.setText(
            f"This will normalize all {len(wav_files)} audio files to {target_db_fs} dB FS "
            f"and overwrite the original files.\n\nDo you want to continue?"
        )
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm_dialog.setDefaultButton(QMessageBox.No)
        
        # If user confirms, proceed with batch normalization
        if confirm_dialog.exec_() == QMessageBox.Yes:
            # Create a progress dialog
            progress = QProgressDialog("Normalizing audio files...", "Cancel", 0, len(wav_files), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("Batch Normalization Progress")
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            # Define progress callback
            def update_progress(current, total, filename):
                progress.setValue(current)
                progress.setLabelText(f"Processing: {filename}")
                QApplication.processEvents()
                return not progress.wasCanceled()
            
            # Perform batch normalization
            results = self.audio_processor.batch_normalize(wav_files, target_db_fs, update_progress)
            
            # Complete the progress
            progress.setValue(len(wav_files))
            
            # Update the UI if this is the currently selected file
            current_item = self.tree_widget.currentItem()
            if current_item:
                file_task = current_item.data(0, Qt.UserRole)
                if file_task and file_task.wav_file:
                    # Check if the current file was processed
                    if file_task.wav_file in results["successful"]:
                        # Reload the audio and regenerate the spectrogram
                        self.display_spectrogram(file_task.wav_file, file_task.lab_file)
                        
                        # Update the media player
                        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_task.wav_file)))
            
            # Show summary dialog
            summary = f"Batch Normalization Summary:\n\n"
            summary += f"Target Level: {target_db_fs} dB FS\n"
            summary += f"Files Processed: {len(wav_files)}\n"
            summary += f"   - Successfully Normalized: {len(results['successful'])}\n"
            summary += f"   - Already at Target Level: {len(results['skipped'])}\n"
            summary += f"   - Failed: {len(results['failed'])}\n"
            
            if results["failed"]:
                summary += "\nFailed Files:\n"
                for file_path, error in results["failed"][:10]:  # Show first 10 failures
                    summary += f"   - {os.path.basename(file_path)}: {error}\n"
                
                if len(results["failed"]) > 10:
                    summary += f"   - ... and {len(results['failed']) - 10} more\n"
            
            QMessageBox.information(self, "Batch Normalization Complete", summary)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetViewer()
    window.show()
    sys.exit(app.exec_())