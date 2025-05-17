import numpy as np

from PyQt5.QtWidgets import (QSizePolicy, QInputDialog, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor

import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

class SpectrogramCanvas(FigureCanvas):
    """Canvas for displaying spectrograms using Matplotlib with fixed scale and interactive phoneme boundaries"""
    # Signal to tell parent when audio duration is available
    audio_loaded = pyqtSignal(float)
    # Signal emitted when boundaries are modified
    boundaries_modified = pyqtSignal()
    # Signal emitted when phoneme labels are modified
    phonemes_modified = pyqtSignal()
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Create two subplots with specific height ratios (80% spectrogram, 20% waveform)
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1])  # 4:1 ratio
        self.axes = self.fig.add_subplot(self.gs[0])  # Top subplot for spectrogram
        self.waveform_axes = self.fig.add_subplot(self.gs[1])  # Bottom for waveform
        
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
        self.current_audio = None  # Store audio data for waveform
        self.current_f0 = None
        self.current_phones = None
        self.current_start_times = None
        self.current_end_times = None
        self.current_sample_rate = None
        self.current_hop_length = None
        self.current_audio_duration = None
        self.total_mel_frames = None
        
        # Default width scale factor
        self.width_scale_factor = 2.0
        
        # Fixed dB scale for waveform display (dB FS)
        self.db_min = -60  # Minimum dB FS value to display
        self.db_max = 0    # Maximum dB FS value (0 = full scale)
        
        # Playback position indicators
        self.playback_line = None
        self.waveform_playback_line = None
        
        # Add phoneme boundary editing variables
        self.dragging_boundary = None
        self.drag_start_x = None
        self.boundary_lines = []  # Keep track of boundary lines for interaction
        self.boundary_rects = []  # Keep track of rectangles
        self.boundary_waveform_lines = []  # Keep track of boundary lines in waveform
        self.boundary_indices = []  # Which boundary index is associated with each line
        self.boundary_times = []  # Store the actual times (in seconds)
        self.edits_made = False  # Track if edits have been made
        
        # Phone label editing variables
        self.phone_labels = []  # Store text objects for updating
        self.phone_label_indices = []  # Store the index of each phone label
        self.phone_labels_edits_made = False  # Track if phoneme label edits have been made
        
        # Connect to mouse events
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('pick_event', self.on_pick_event)  # For clicking on text
        
        self.fig.tight_layout()
        
    def plot_spectrogram(self, mel_spec, audio=None, f0=None, phones=None, start_times=None, end_times=None, 
                         sample_rate=None, hop_length=None, width_scale_factor=2.0, audio_duration=None):
        """Plot spectrogram with F0 contour overlay, phoneme boundaries, and fixed-scale waveform below"""
        self.axes.clear()
        self.waveform_axes.clear()
        
        # Reset boundary tracking
        self.boundary_lines = []
        self.boundary_rects = []
        self.boundary_waveform_lines = []
        self.boundary_indices = []
        self.boundary_times = []
        self.edits_made = False
        
        # Reset phone label tracking
        self.phone_labels = []
        self.phone_label_indices = []
        self.phone_labels_edits_made = False
        
        # Store current data for rescaling
        self.current_mel_spec = mel_spec
        self.current_audio = audio
        self.current_f0 = f0
        self.current_phones = phones.copy() if phones else None  # Make a copy to avoid modifying the original
        self.current_start_times = start_times.copy() if start_times else None
        self.current_end_times = end_times.copy() if end_times else None
        self.current_sample_rate = sample_rate
        self.current_hop_length = hop_length
        self.width_scale_factor = width_scale_factor
        self.current_audio_duration = audio_duration
        self.total_mel_frames = mel_spec.shape[1]
        
        # Store boundary times for editing
        if start_times:
            self.boundary_times = start_times.copy()
        
        # Use a fixed aspect ratio instead of 'auto'
        im = self.axes.imshow(mel_spec, aspect='auto', origin='lower')
        self.axes.set_ylim(0, mel_spec.shape[0]-1)  # From 0 to the number of mel bins

        # Plot waveform if audio data is provided
        if audio is not None and sample_rate is not None:
            # Convert to dB FS (Decibels relative to Full Scale)
            # Digital audio typically has a max value of 1.0 or -1.0
            # dB FS = 20 * log10(|amplitude| / 1.0)
            eps = 1e-10  # Small epsilon to avoid log(0)
            audio_db_fs = 20 * np.log10(np.abs(audio) + eps)
            
            # Create the time axis for the waveform
            time_axis = np.linspace(0, audio_duration, len(audio_db_fs)) if audio_duration else np.arange(len(audio_db_fs)) / sample_rate
            
            # Plot waveform with fixed dB FS scale
            self.waveform_axes.plot(time_axis, audio_db_fs, color='blue', linewidth=0.5)
            self.waveform_axes.set_ylim(self.db_min, self.db_max)  # Fixed scale for comparison
            self.waveform_axes.set_ylabel('Amplitude (dB FS)')
            self.waveform_axes.set_xlabel('Time (s)')
            
            # Add grid and set fixed y-ticks for better readability
            self.waveform_axes.grid(True, alpha=0.3)
            
            # Set y-ticks at reasonable intervals (-60, -50, -40, -30, -20, -10, 0)
            self.waveform_axes.set_yticks([self.db_min + i*10 for i in range((self.db_max - self.db_min)//10 + 1)])
            
            # Set x limits to match the audio duration
            if audio_duration:
                self.waveform_axes.set_xlim(0, audio_duration)
                
            # Add reference lines at common thresholds
            self.waveform_axes.axhline(y=-18, color='green', linestyle='--', alpha=0.7, linewidth=0.8)  # EBU R128 target level
            self.waveform_axes.axhline(y=-3, color='orange', linestyle='--', alpha=0.7, linewidth=0.8)  # Warning level
            self.waveform_axes.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7, linewidth=0.8)  # Clipping danger

        # Plot F0 contour if provided
        if f0 is not None and len(f0) > 0:
            # Filter out zeros and negative values (unvoiced regions)
            x_indices = []
            y_indices = []
            
            # Convert Hz to mel scale
            def hz_to_mel(hz):
                return 2595 * np.log10(1 + hz/700)
            
            # Get frequency range (same as in display_spectrogram)
            f_min = 0  # Minimum frequency in Hz
            f_max = 12000  # Approximate Nyquist frequency for 24kHz
            n_mels = mel_spec.shape[0]  # Number of mel bins
            
            # Convert frequency range to mel scale
            mel_min = hz_to_mel(f_min)
            mel_max = hz_to_mel(f_max)
            
            # Convert each F0 value to the corresponding mel bin
            for i, freq in enumerate(f0):
                if freq >= f_min:  # Only consider positive frequencies (voiced)
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
            self.phone_labels = []  # Store text objects for updating
            self.phone_label_indices = []  # Store the index of each phone label
            rect_patches = []  # Store rectangle patches
            
            for i, (phone, start, end) in enumerate(zip(phones, start_frames, end_frames)):
                if start < mel_spec.shape[1] and end > 0:  # Check if within spectrogram bounds
                    valid_start = max(0, start)
                    valid_end = min(mel_spec.shape[1], end)
                    
                    # Draw phoneme boundary as vertical lines with thicker lines for draggable boundaries
                    boundary_line = self.axes.axvline(x=valid_start, color='white', linestyle='-', 
                                                      alpha=0.8, linewidth=1.5, picker=5)
                    self.boundary_lines.append(boundary_line)
                    self.boundary_indices.append(i)
                    
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
                    self.boundary_rects.append(rect)
                    
                    # Add phoneme label if segment is wide enough
                    segment_width = valid_end - valid_start
                    center = valid_start + segment_width/2
                    # White text with dark edge for contrast against any background
                    text = self.axes.text(
                        center, label_y_pos, phone,
                        ha='center', va='center',
                        fontsize=8, color='white',
                        fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1, boxstyle='round'),
                        picker=True  # Make text pickable for editing
                    )
                    self.phone_labels.append(text)
                    self.phone_label_indices.append(i)  # Store the index for reference
                    
                    # Add phoneme boundaries in waveform plot too
                    if audio_duration and start_times and end_times:
                        # Directly use original time values for waveform plot
                        waveform_line = self.waveform_axes.axvline(
                            x=start_times[i], color='red', linestyle='-', 
                            alpha=0.7, linewidth=0.7
                        )
                        self.boundary_waveform_lines.append(waveform_line)
        
        # Use a fixed height and scale width based on time dimension and scale factor
        fixed_height = 10  # Increased to accommodate waveform
        base_width = max(10, mel_spec.shape[1]/50)  # Base width calculation
        width = base_width * self.width_scale_factor  # Apply scale factor
        
        # Disable automatic adjustments
        self.fig.set_size_inches(width, fixed_height, forward=True)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        # Force fixed pixel size
        dpi = self.fig.get_dpi()
        self.setFixedSize(int(width * dpi), int(fixed_height * dpi))
        
        # Add legend, title and labels
        self.axes.set_title('Mel Spectrogram with F0 Contour and Phoneme Alignment (Drag boundaries to adjust, Click labels to edit)')
        self.axes.set_ylabel('Mel Bins')
        self.axes.set_xlabel('Frames')
        self.waveform_axes.set_title('Audio Waveform (dB FS)')
        
        # Initialize playback position lines (hidden initially)
        self.playback_line = self.axes.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.7, visible=False)
        if audio_duration:
            self.waveform_playback_line = self.waveform_axes.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.7, visible=False)
        
        self.fig.tight_layout()
        self.draw()
        
        # Emit signal with audio duration
        if audio_duration is not None:
            self.audio_loaded.emit(audio_duration)
    
    def on_pick_event(self, event):
        """Handle pick events for phoneme label editing"""
        # Check if it's a text object (phone label)
        if hasattr(event, 'artist') and event.artist in self.phone_labels:
            # Get the index of the picked phone label
            label_idx = self.phone_labels.index(event.artist)
            phone_idx = self.phone_label_indices[label_idx]
            
            # Get the current phoneme text
            current_phone = self.current_phones[phone_idx]
            
            # Show dialog to edit the phoneme label
            new_phone, ok = QInputDialog.getText(
                self, 
                'Edit Phoneme Label', 
                'Enter new phoneme label:',
                QLineEdit.Normal,
                current_phone
            )
            
            # If user clicked OK and provided a new label
            if ok and new_phone and new_phone != current_phone:
                # Update the phoneme label text
                self.phone_labels[label_idx].set_text(new_phone)
                
                # Update the internal phoneme data
                self.current_phones[phone_idx] = new_phone
                
                # Mark that edits have been made
                self.phone_labels_edits_made = True
                
                # Redraw the canvas
                self.draw_idle()
                
                # Emit signal that phoneme labels have been modified
                self.phonemes_modified.emit()
    
    def on_mouse_press(self, event):
        """Handle mouse press events for phoneme boundary dragging"""
        if event.inaxes != self.axes or not self.boundary_lines:
            return
        
        # Check if click is near a boundary line
        for i, line in enumerate(self.boundary_lines):
            line_x = line.get_xdata()[0]
            if abs(event.xdata - line_x) < 5:  # Allow a bit of tolerance for clicking
                self.dragging_boundary = i
                self.drag_start_x = event.xdata
                if i < len(self.boundary_times):
                    self.setCursor(QCursor(Qt.SizeHorCursor))  # Change cursor to horizontal resize
                break
    
    def on_mouse_release(self, event):
        """Handle mouse release events after dragging"""
        if self.dragging_boundary is not None:
            self.dragging_boundary = None
            self.setCursor(QCursor(Qt.ArrowCursor))  # Reset cursor
            
            # Signal that edits have been made if boundaries were changed
            if not np.array_equal(self.boundary_times, self.current_start_times):
                self.edits_made = True
                self.boundaries_modified.emit()
    
    def on_mouse_move(self, event):
        """Handle mouse movement for phoneme boundary dragging"""
        if event.inaxes != self.axes:
            return
        
        # Check if hovering near a boundary for cursor change
        if self.dragging_boundary is None:
            near_boundary = False
            for line in self.boundary_lines:
                if abs(event.xdata - line.get_xdata()[0]) < 5:
                    self.setCursor(QCursor(Qt.SizeHorCursor))
                    near_boundary = True
                    break
            if not near_boundary:
                self.setCursor(QCursor(Qt.ArrowCursor))
        
        # Handle dragging
        if self.dragging_boundary is not None and event.inaxes == self.axes:
            # Get the index of the boundary being dragged
            i = self.dragging_boundary
            
            # Calculate new position with constraints
            new_frame = int(event.xdata)
            
            # Apply constraints - boundary must be between adjacent boundaries
            min_frame = 0
            if i > 0 and i-1 < len(self.boundary_lines):
                min_frame = int(self.boundary_lines[i-1].get_xdata()[0]) + 1
                
            max_frame = self.total_mel_frames
            if i < len(self.boundary_lines)-1:
                max_frame = int(self.boundary_lines[i+1].get_xdata()[0]) - 1
            
            constrained_frame = max(min_frame, min(new_frame, max_frame))
            
            # Update the boundary line position
            self.boundary_lines[i].set_xdata([constrained_frame, constrained_frame])
            
            # Update the rectangle on the left side of the dragged boundary
            if i > 0 and i-1 < len(self.boundary_rects):
                prev_start = int(self.boundary_lines[i-1].get_xdata()[0])
                width = constrained_frame - prev_start
                self.boundary_rects[i-1].set_width(width)
                
                # Update phoneme label position
                if i-1 < len(self.phone_labels):
                    center = prev_start + width/2
                    self.phone_labels[i-1].set_position((center, self.phone_labels[i-1].get_position()[1]))
            
            # Update the rectangle on the right side of the dragged boundary
            if i < len(self.boundary_rects):
                next_end = self.total_mel_frames
                if i+1 < len(self.boundary_lines):
                    next_end = int(self.boundary_lines[i+1].get_xdata()[0])
                
                width = next_end - constrained_frame
                self.boundary_rects[i].set_x(constrained_frame)
                self.boundary_rects[i].set_width(width)
                
                # Update phoneme label position
                if i < len(self.phone_labels):
                    center = constrained_frame + width/2
                    self.phone_labels[i].set_position((center, self.phone_labels[i].get_position()[1]))
            
            # Update boundary time based on frame position
            if self.current_audio_duration and i < len(self.boundary_times):
                frame_ratio = constrained_frame / self.total_mel_frames
                new_time = frame_ratio * self.current_audio_duration
                self.boundary_times[i] = new_time
                
                # Update the corresponding waveform line
                if i < len(self.boundary_waveform_lines):
                    self.boundary_waveform_lines[i].set_xdata([new_time, new_time])
            
            # Redraw
            self.draw_idle()
    
    def get_modified_lab_data(self):
        """Get the current phoneme data with modified boundaries and/or labels"""
        if not (self.edits_made or self.phone_labels_edits_made) or not self.current_phones:
            return None
        
        # For start times, use the boundary times
        new_start_times = self.boundary_times.copy()
        
        # For end times, use the next start time (for all except the last phoneme)
        new_end_times = []
        for i in range(len(new_start_times)):
            if i < len(new_start_times) - 1:
                new_end_times.append(new_start_times[i+1])
            else:
                new_end_times.append(self.current_audio_duration)
        
        # Return modified phoneme data, including any edited phoneme labels
        return self.current_phones, new_start_times, new_end_times
    
    def update_db_scale(self, db_min, db_max):
        """Update the dB scale of the waveform display"""
        self.db_min = db_min
        self.db_max = db_max
        # Redraw with the new scale if we have data
        if self.current_mel_spec is not None:
            self.plot_spectrogram(
                self.current_mel_spec, 
                audio=self.current_audio,
                f0=self.current_f0, 
                phones=self.current_phones, 
                start_times=self.current_start_times, 
                end_times=self.current_end_times,
                sample_rate=self.current_sample_rate,
                hop_length=self.current_hop_length,
                width_scale_factor=self.width_scale_factor,
                audio_duration=self.current_audio_duration
            )
    
    def rescale_width(self, scale_factor):
        """Rescale the spectrogram width with the given scale factor"""
        if self.current_mel_spec is not None:
            self.plot_spectrogram(
                self.current_mel_spec, 
                audio=self.current_audio,
                f0=self.current_f0, 
                phones=self.current_phones, 
                start_times=self.current_start_times, 
                end_times=self.current_end_times,
                sample_rate=self.current_sample_rate,
                hop_length=self.current_hop_length,
                width_scale_factor=scale_factor,
                audio_duration=self.current_audio_duration
            )
    
    def update_playback_position(self, position_seconds):
        """Update the position of the playback lines on both plots"""
        if self.playback_line and self.total_mel_frames and self.current_audio_duration:
            # For spectrogram: Convert position in seconds to frames
            position_frame = int((position_seconds / self.current_audio_duration) * self.total_mel_frames)
            
            # Update spectrogram line position
            self.playback_line.set_xdata([position_frame, position_frame])
            self.playback_line.set_visible(True)
            
            # Update waveform line position (in seconds)
            if hasattr(self, 'waveform_playback_line') and self.waveform_playback_line:
                self.waveform_playback_line.set_xdata([position_seconds, position_seconds])
                self.waveform_playback_line.set_visible(True)
            
            self.draw()
    
    def hide_playback_position(self):
        """Hide the playback position lines"""
        if self.playback_line:
            self.playback_line.set_visible(False)
        if hasattr(self, 'waveform_playback_line') and self.waveform_playback_line:
            self.waveform_playback_line.set_visible(False)
        self.draw()
        
    def clear(self):
        """Clear the spectrogram display"""
        self.axes.clear()
        if hasattr(self, 'waveform_axes'):
            self.waveform_axes.clear()
        self.axes.set_title('No spectrogram loaded')
        if hasattr(self, 'waveform_axes'):
            self.waveform_axes.set_title('No waveform loaded')
        self.fig.tight_layout()
        self.draw()
        
        # Clear stored data
        self.current_mel_spec = None
        self.current_audio = None
        self.current_f0 = None
        self.current_phones = None
        self.current_start_times = None
        self.current_end_times = None
        self.current_sample_rate = None
        self.current_hop_length = None
        self.current_audio_duration = None
        self.total_mel_frames = None
        self.playback_line = None
        self.waveform_playback_line = None
        
        # Clear boundary tracking variables
        self.boundary_lines = []
        self.boundary_rects = []
        self.boundary_waveform_lines = []
        self.boundary_indices = []
        self.boundary_times = []
        self.phone_labels = []
        self.phone_label_indices = []
        self.edits_made = False
        self.phone_labels_edits_made = False