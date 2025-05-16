#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization panel for displaying singing voice data visualizations.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter, 
    QLabel, QFrame, QGroupBox, QComboBox, QCheckBox, QToolButton,
    QPushButton, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QPalette, QColor

class VisualizationWidget(QWidget):
    """Base class for visualization widgets."""
    
    def __init__(self, parent=None, title="Visualization"):
        super().__init__(parent)
        self.title = title
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface components."""
        self.layout = QVBoxLayout(self)
        
        # Create a colored placeholder for the visualization
        self.placeholder = QFrame()
        self.placeholder.setFrameShape(QFrame.StyledPanel)
        self.placeholder.setAutoFillBackground(True)
        
        # Set a random background color for each visualization type
        palette = self.placeholder.palette()
        if self.title == "Waveform":
            palette.setColor(QPalette.Window, QColor(230, 240, 255))  # Light blue
        elif self.title == "Mel Spectrogram":
            palette.setColor(QPalette.Window, QColor(230, 255, 240))  # Light green
        elif self.title == "F0 Contour":
            palette.setColor(QPalette.Window, QColor(255, 240, 230))  # Light orange
        elif self.title == "Phoneme Alignment":
            palette.setColor(QPalette.Window, QColor(240, 230, 255))  # Light purple
        else:
            palette.setColor(QPalette.Window, QColor(240, 240, 240))  # Light gray
            
        self.placeholder.setPalette(palette)
        
        # Add a label to show this is a placeholder
        self.label = QLabel(f"{self.title} Visualization\n(Placeholder)")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 14px; color: rgba(0, 0, 0, 150);")
        
        # Use a vertical layout to position the label in the center of the placeholder
        placeholder_layout = QVBoxLayout(self.placeholder)
        placeholder_layout.addWidget(self.label)
        
        # Add the placeholder to the main layout
        self.layout.addWidget(self.placeholder)

class WaveformWidget(VisualizationWidget):
    """Widget for displaying audio waveform."""
    
    def __init__(self, parent=None):
        super().__init__(parent, title="Waveform")

class MelSpectrogramWidget(VisualizationWidget):
    """Widget for displaying mel spectrogram."""
    
    def __init__(self, parent=None):
        super().__init__(parent, title="Mel Spectrogram")

class F0ContourWidget(VisualizationWidget):
    """Widget for displaying F0 contour."""
    
    def __init__(self, parent=None):
        super().__init__(parent, title="F0 Contour")

class PhonemeAlignmentWidget(VisualizationWidget):
    """Widget for displaying phoneme alignments."""
    
    def __init__(self, parent=None):
        super().__init__(parent, title="Phoneme Alignment")

class VisualizationPanel(QWidget):
    """Panel containing all visualizations for the singing voice data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Options toolbar
        self.options_group = QGroupBox("Visualization Options")
        self.options_layout = QHBoxLayout(self.options_group)
        
        # Visualization type selection
        self.view_label = QLabel("View:")
        self.view_combo = QComboBox()
        self.view_combo.addItems(["All", "Waveform", "Spectrogram", "F0", "Phonemes"])
        self.options_layout.addWidget(self.view_label)
        self.options_layout.addWidget(self.view_combo)
        
        # Additional options
        self.show_grid_check = QCheckBox("Show Grid")
        self.show_grid_check.setChecked(True)
        self.options_layout.addWidget(self.show_grid_check)
        
        self.sync_views_check = QCheckBox("Sync Views")
        self.sync_views_check.setChecked(True)
        self.options_layout.addWidget(self.sync_views_check)
        
        # Add spacer and buttons
        self.options_layout.addStretch()
        
        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setText("🔍+")
        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setText("🔍-")
        self.reset_zoom_btn = QToolButton()
        self.reset_zoom_btn.setText("Reset")
        
        self.options_layout.addWidget(self.zoom_in_btn)
        self.options_layout.addWidget(self.zoom_out_btn)
        self.options_layout.addWidget(self.reset_zoom_btn)
        
        self.layout.addWidget(self.options_group)
        
        # Create a vertical splitter for the visualizations
        self.vis_splitter = QSplitter(Qt.Vertical)
        
        # Add visualization widgets
        self.waveform_widget = WaveformWidget()
        self.mel_widget = MelSpectrogramWidget()
        self.f0_widget = F0ContourWidget()
        self.phoneme_widget = PhonemeAlignmentWidget()
        
        self.vis_splitter.addWidget(self.waveform_widget)
        self.vis_splitter.addWidget(self.mel_widget)
        self.vis_splitter.addWidget(self.f0_widget)
        self.vis_splitter.addWidget(self.phoneme_widget)
        
        # Set initial sizes for the splitter
        self.vis_splitter.setSizes([100, 200, 100, 100])  # Higher value for spectrogram
        
        # Add the splitter to a scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.vis_splitter)
        
        self.layout.addWidget(self.scroll_area)
