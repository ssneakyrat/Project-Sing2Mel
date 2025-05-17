import sys
import os
import yaml
import numpy as np
import traceback
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel,
                            QSplitter, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
                            QSlider, QGroupBox)
from PyQt5.QtCore import Qt
import glob
from collections import namedtuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DatasetViewer")

# Try to import matplotlib for spectrogram display
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.patches import Rectangle
    import matplotlib.cm as cm
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

# Try to import audio processing libraries
try:
    import soundfile as sf
    import torchaudio
    import torch
    audio_processing_available = True
except ImportError:
    audio_processing_available = False

# Create a namedtuple for file metadata
FileMetadata = namedtuple('FileMetadata', [
    'wav_file', 'lab_file', 'singer_id', 'language_id', 'singer_idx', 
    'language_idx', 'base_name'
])

class SpectrogramCanvas(FigureCanvas):
    """Canvas for displaying spectrograms using Matplotlib with fixed scale"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        if not matplotlib_available:
            raise ImportError("Matplotlib is required for spectrogram display")
            
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(SpectrogramCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Set size policy to allow expansion but maintain fixed aspect ratio
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        
        # Setting a minimum size to ensure scrollbars appear as needed
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        # Store the current mel spectrogram and related data for rescaling
        self.current_mel_spec = None
        self.current_f0 = None
        self.current_phones = None
        self.current_start_times = None
        self.current_end_times = None
        self.current_sample_rate = None
        self.current_hop_length = None
        
        # Default width scale factor
        self.width_scale_factor = 1.0
        
        self.fig.tight_layout()
        
    def plot_spectrogram(self, mel_spec, f0=None, phones=None, start_times=None, end_times=None, 
                         sample_rate=None, hop_length=None, width_scale_factor=1.0):
        """Plot spectrogram with F0 contour overlay and phoneme boundaries using adjustable scale"""
        self.axes.clear()
        
        # Store current data for rescaling
        self.current_mel_spec = mel_spec
        self.current_f0 = f0
        self.current_phones = phones
        self.current_start_times = start_times
        self.current_end_times = end_times
        self.current_sample_rate = sample_rate
        self.current_hop_length = hop_length
        self.width_scale_factor = width_scale_factor
        
        # Use a fixed aspect ratio instead of 'auto'
        im = self.axes.imshow(mel_spec, aspect='auto', origin='lower')
        self.axes.set_ylim(0, mel_spec.shape[0]-1)  # From 0 to the number of mel bins

        # Use a fixed height and scale width based on time dimension and scale factor
        fixed_height = 8  # Set a consistent height
        base_width = max(10, mel_spec.shape[1]/50)  # Base width calculation
        width = base_width * self.width_scale_factor  # Apply scale factor
        
        # Disable automatic adjustments
        self.fig.set_size_inches(width, fixed_height, forward=True)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        # Force fixed pixel size
        dpi = self.fig.get_dpi()
        self.setFixedSize(int(width * dpi), int(fixed_height * dpi))
        
        # Plot F0 contour if provided
        if f0 is not None and len(f0) > 0:
            # Filter out zeros and negative values (unvoiced regions)
            x_indices = []
            y_indices = []
            
            # Convert Hz to mel scale
            def hz_to_mel(hz):
                return 2595 * np.log10(1 + hz/700)
            
            # Get frequency range (same as in display_spectrogram)
            f_min = 40  # Minimum frequency in Hz
            f_max = 12000  # Approximate Nyquist frequency for 24kHz
            n_mels = mel_spec.shape[0]  # Number of mel bins
            
            # Convert frequency range to mel scale
            mel_min = hz_to_mel(f_min)
            mel_max = hz_to_mel(f_max)
            
            # Convert each F0 value to the corresponding mel bin
            for i, freq in enumerate(f0):
                if freq > 0:  # Only consider positive frequencies (voiced)
                    # Convert frequency to mel
                    freq_mel = hz_to_mel(freq)
                    # Map to mel bin index
                    bin_idx = (freq_mel - mel_min) / (mel_max - mel_min) * (n_mels - 1)
                    x_indices.append(i)
                    y_indices.append(bin_idx)
            
            # Plot only the valid points
            if x_indices:
                self.axes.plot(x_indices, y_indices, 'r-', linewidth=1.5, label='F0 Contour')
        
        # Plot phoneme boundaries and labels if provided
        if phones and start_times and end_times and sample_rate and hop_length:
            # Scale phoneme timings to match the mel spectrogram frames
            total_mel_frames = mel_spec.shape[1]
            
            if len(end_times) > 0:
                # Get the maximum time from the lab file
                max_time = max(end_times)
                
                # Scale factor to convert time to mel frames
                scale_factor = total_mel_frames / max_time
                
                # Convert time to frames using the scale factor
                start_frames = [int(t * scale_factor) for t in start_times]
                end_frames = [int(t * scale_factor) for t in end_times]
            
            # Create unique color for each phoneme
            unique_phones = list(set(phones))
            phone_colors = {}
            cmap = cm.get_cmap('tab20', max(20, len(unique_phones)))
            
            for i, phone in enumerate(unique_phones):
                phone_colors[phone] = cmap(i % 20)  # Use modulo to handle more than 20 phonemes
            
            # Set y-position for phoneme labels (near the bottom of the spectrogram)
            label_y_pos = mel_spec.shape[0] * 0.15  # 15% from the bottom
            
            # Plot vertical lines for phoneme boundaries and phoneme labels
            for i, (phone, start, end) in enumerate(zip(phones, start_frames, end_frames)):
                if start < mel_spec.shape[1] and end > 0:  # Check if within spectrogram bounds
                    valid_start = max(0, start)
                    valid_end = min(mel_spec.shape[1], end)
                    
                    # Skip very short segments
                    #if valid_end - valid_start <= 1:
                    #    continue
                    
                    # Draw phoneme boundary as vertical lines
                    self.axes.axvline(x=valid_start, color='white', linestyle='-', alpha=0.7, linewidth=0.7)
                    
                    # Add colored background for phoneme segment
                    rect = Rectangle(
                        (valid_start, 0),                # (x, y) bottom left corner
                        valid_end - valid_start,         # width
                        label_y_pos * 1.5,               # height (just above label position)
                        alpha=0.3,                       # transparency
                        facecolor=phone_colors[phone],   # phoneme color
                        edgecolor=None                   # no edge color
                    )
                    self.axes.add_patch(rect)
                    
                    # Add phoneme label if segment is wide enough
                    segment_width = valid_end - valid_start
                    center = valid_start + segment_width/2
                    # White text with dark edge for contrast against any background
                    self.axes.text(
                        center, label_y_pos, phone,
                        ha='center', va='center',
                        fontsize=8, color='white',
                        fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1, boxstyle='round')
                    )
        
        # Add legend, title and labels
        self.axes.set_title('Mel Spectrogram with F0 Contour and Phoneme Alignment')
        self.axes.set_ylabel('Mel Bins')
        self.axes.set_xlabel('Frames')
        
        # Add a colorbar for the spectrogram
        #self.fig.colorbar(im, ax=self.axes, orientation='vertical', pad=0.01, fraction=0.05)
        
        self.fig.tight_layout()
        self.draw()
    
    def rescale_width(self, scale_factor):
        """Rescale the spectrogram width with the given scale factor"""
        if self.current_mel_spec is not None:
            self.plot_spectrogram(
                self.current_mel_spec, 
                f0=self.current_f0, 
                phones=self.current_phones, 
                start_times=self.current_start_times, 
                end_times=self.current_end_times,
                sample_rate=self.current_sample_rate,
                hop_length=self.current_hop_length,
                width_scale_factor=scale_factor
            )
        
    def clear(self):
        """Clear the spectrogram display"""
        self.axes.clear()
        self.axes.set_title('No spectrogram loaded')
        self.fig.tight_layout()
        self.draw()
        
        # Clear stored data
        self.current_mel_spec = None
        self.current_f0 = None
        self.current_phones = None
        self.current_start_times = None
        self.current_end_times = None
        self.current_sample_rate = None
        self.current_hop_length = None

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
        
        # Check dependencies
        self.check_dependencies()
        
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
        self.width_scale_factor = 1.0
        
        # Set up the UI
        self.init_ui()
        
        # Load the dataset
        self.load_dataset()
        
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        missing_deps = []
        
        if not matplotlib_available:
            missing_deps.append("matplotlib")
            
        if not audio_processing_available:
            missing_deps.append("soundfile, torchaudio, torch")
            
        if missing_deps:
            self.show_error(f"Missing dependencies: {', '.join(missing_deps)}")
            
    def show_error(self, message):
        """Show an error message box"""
        QMessageBox.critical(self, "Error", message)
        
    def init_ui(self):
        """Initialize the user interface with scrollable spectrogram"""
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
        
        # Right panel: Spectrogram display and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        if matplotlib_available:
            # Add spectrogram controls
            controls_group = QGroupBox("Spectrogram Controls")
            controls_layout = QVBoxLayout()
            
            # Add width scale slider with label
            scale_layout = QHBoxLayout()
            scale_label = QLabel("Width Scale:")
            self.width_scale_slider = QSlider(Qt.Horizontal)
            self.width_scale_slider.setRange(10, 500)  # 0.1x to 5.0x
            self.width_scale_slider.setValue(100)  # Default 1.0x
            self.width_scale_slider.setTickPosition(QSlider.TicksBelow)
            self.width_scale_slider.setTickInterval(50)
            self.width_scale_value_label = QLabel("1.0x")
            
            # Connect slider value change signal
            self.width_scale_slider.valueChanged.connect(self.on_scale_slider_changed)
            
            scale_layout.addWidget(scale_label)
            scale_layout.addWidget(self.width_scale_slider)
            scale_layout.addWidget(self.width_scale_value_label)
            
            controls_layout.addLayout(scale_layout)
            controls_group.setLayout(controls_layout)
            right_layout.addWidget(controls_group)
            
            # Use the scrollable spectrogram widget instead of just the canvas
            self.scrollable_spectrogram = ScrollableSpectrogramWidget(right_panel)
            self.spectrogram_canvas = self.scrollable_spectrogram.spectrogram_canvas
            right_layout.addWidget(self.scrollable_spectrogram)
        else:
            self.spectrogram_label = QLabel("Matplotlib is required for spectrogram display")
            right_layout.addWidget(self.spectrogram_label)
            self.spectrogram_canvas = None
        
        # Add widgets to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial sizes
        splitter.setSizes([300, 900])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set main widget
        self.setCentralWidget(main_widget)
    
    def on_scale_slider_changed(self, value):
        """Handle width scale slider value change"""
        # Calculate scale factor (range 0.1x to 5.0x)
        self.width_scale_factor = value / 100.0
        self.width_scale_value_label.setText(f"{self.width_scale_factor:.1f}x")
        
        # Update spectrogram if canvas exists
        if self.spectrogram_canvas:
            self.spectrogram_canvas.rescale_width(self.width_scale_factor)
        
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
            # Display file info
            info_text = f"Singer: {file_task.singer_id} (ID: {file_task.singer_idx})\n"
            info_text += f"Language: {file_task.language_id} (ID: {file_task.language_idx})\n"
            info_text += f"Lab file: {os.path.basename(file_task.lab_file)}\n"
            info_text += f"WAV file: {os.path.basename(file_task.wav_file)}"
            
            # Process audio for spectrogram if matplotlib is available
            if matplotlib_available and audio_processing_available and self.spectrogram_canvas:
                try:
                    self.display_spectrogram(file_task.wav_file, file_task.lab_file)
                except Exception as e:
                    error_text = f"{info_text}\nError: {str(e)}\n{traceback.format_exc()}"
                    self.spectrogram_canvas.clear()
        else:
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
        """Load audio file and display its spectrogram with phoneme alignment"""
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found: {wav_file}")
            
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
        audio_duration = len(audio) / sr
        expected_mel_frames = mel_np.shape[1]
        
        # Validation of lab file timing
        if phones and len(phones) > 0:
            lab_max_time = max(end_times)
            if abs(lab_max_time - audio_duration) > 0.5:  # More than 0.5 seconds difference
                logger.warning(f"Lab file timing mismatch: lab_max_time={lab_max_time:.2f}s, audio_duration={audio_duration:.2f}s")
        
        # Display spectrogram with phoneme alignment using current width scale factor
        self.spectrogram_canvas.plot_spectrogram(
            mel_np, 
            f0=f0, 
            phones=phones, 
            start_times=start_times, 
            end_times=end_times,
            sample_rate=sr,
            hop_length=self.hop_length,
            width_scale_factor=self.width_scale_factor
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetViewer()
    window.show()
    sys.exit(app.exec_())