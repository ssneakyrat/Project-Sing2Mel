#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistics widget for displaying dataset statistics.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QFrame, QPushButton, QComboBox, QSplitter, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

class PlaceholderPlot(QWidget):
    """Placeholder for statistics plots."""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface components."""
        self.layout = QVBoxLayout(self)
        
        # Create a colored placeholder
        self.placeholder = QFrame()
        self.placeholder.setFrameShape(QFrame.StyledPanel)
        self.placeholder.setAutoFillBackground(True)
        
        # Set a background color
        palette = self.placeholder.palette()
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        self.placeholder.setPalette(palette)
        
        # Add a label
        self.label = QLabel(f"{self.title}\n(Placeholder)")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 14px; color: rgba(0, 0, 0, 150);")
        
        placeholder_layout = QVBoxLayout(self.placeholder)
        placeholder_layout.addWidget(self.label)
        
        self.layout.addWidget(self.placeholder)

class StatsWidget(QWidget):
    """Widget for displaying dataset statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        
        # Create tab widget for different statistic categories
        self.tab_widget = QTabWidget()
        
        # Overview tab
        self.overview_tab = QWidget()
        self.overview_layout = QVBoxLayout(self.overview_tab)
        
        # Add a header with summary statistics
        self.summary_frame = QFrame()
        self.summary_frame.setFrameShape(QFrame.StyledPanel)
        self.summary_layout = QGridLayout(self.summary_frame)
        
        # Add placeholder data
        self.summary_layout.addWidget(QLabel("<b>Total Files:</b>"), 0, 0)
        self.summary_layout.addWidget(QLabel("120"), 0, 1)
        self.summary_layout.addWidget(QLabel("<b>Total Duration:</b>"), 0, 2)
        self.summary_layout.addWidget(QLabel("2h 34m"), 0, 3)
        self.summary_layout.addWidget(QLabel("<b>Singers:</b>"), 1, 0)
        self.summary_layout.addWidget(QLabel("5"), 1, 1)
        self.summary_layout.addWidget(QLabel("<b>Languages:</b>"), 1, 2)
        self.summary_layout.addWidget(QLabel("3"), 1, 3)
        
        self.overview_layout.addWidget(self.summary_frame)
        
        # Add charts for overview
        self.overview_charts_layout = QHBoxLayout()
        self.overview_charts_layout.addWidget(PlaceholderPlot("Files per Singer"))
        self.overview_charts_layout.addWidget(PlaceholderPlot("Files per Language"))
        self.overview_layout.addLayout(self.overview_charts_layout)
        
        # Add a second row of charts
        self.overview_charts_layout2 = QHBoxLayout()
        self.overview_charts_layout2.addWidget(PlaceholderPlot("Duration Distribution"))
        self.overview_charts_layout2.addWidget(PlaceholderPlot("Audio Characteristics"))
        self.overview_layout.addLayout(self.overview_charts_layout2)
        
        # Singer statistics tab
        self.singer_tab = QWidget()
        self.singer_layout = QVBoxLayout(self.singer_tab)
        
        # Add singer selection
        self.singer_selection_layout = QHBoxLayout()
        self.singer_selection_layout.addWidget(QLabel("Select Singer:"))
        self.singer_combo = QComboBox()
        self.singer_combo.addItems(["All Singers", "Singer 1", "Singer 2", "Singer 3"])
        self.singer_selection_layout.addWidget(self.singer_combo)
        self.singer_selection_layout.addStretch()
        
        self.singer_layout.addLayout(self.singer_selection_layout)
        
        # Add charts for singer stats
        self.singer_charts_layout = QHBoxLayout()
        self.singer_charts_layout.addWidget(PlaceholderPlot("Singer Language Distribution"))
        self.singer_charts_layout.addWidget(PlaceholderPlot("Singer Duration per Language"))
        self.singer_layout.addLayout(self.singer_charts_layout)
        
        # Language statistics tab
        self.language_tab = QWidget()
        self.language_layout = QVBoxLayout(self.language_tab)
        
        # Add language selection
        self.language_selection_layout = QHBoxLayout()
        self.language_selection_layout.addWidget(QLabel("Select Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["All Languages", "English", "Mandarin", "Spanish"])
        self.language_selection_layout.addWidget(self.language_combo)
        self.language_selection_layout.addStretch()
        
        self.language_layout.addLayout(self.language_selection_layout)
        
        # Add charts for language stats
        self.language_charts_layout = QHBoxLayout()
        self.language_charts_layout.addWidget(PlaceholderPlot("Phoneme Distribution"))
        self.language_charts_layout.addWidget(PlaceholderPlot("Singer Distribution"))
        self.language_layout.addLayout(self.language_charts_layout)
        
        # Phoneme statistics tab
        self.phoneme_tab = QWidget()
        self.phoneme_layout = QVBoxLayout(self.phoneme_tab)
        
        # Add phoneme selection
        self.phoneme_selection_layout = QHBoxLayout()
        self.phoneme_selection_layout.addWidget(QLabel("Select Phoneme Set:"))
        self.phoneme_combo = QComboBox()
        self.phoneme_combo.addItems(["All Phonemes", "English Phonemes", "Mandarin Phonemes", "Spanish Phonemes"])
        self.phoneme_selection_layout.addWidget(self.phoneme_combo)
        self.phoneme_selection_layout.addStretch()
        
        self.phoneme_layout.addLayout(self.phoneme_selection_layout)
        
        # Add charts for phoneme stats
        self.phoneme_charts_layout = QHBoxLayout()
        self.phoneme_charts_layout.addWidget(PlaceholderPlot("Phoneme Frequency"))
        self.phoneme_charts_layout.addWidget(PlaceholderPlot("Phoneme Duration"))
        self.phoneme_layout.addLayout(self.phoneme_charts_layout)
        
        # Add tabs to the tab widget
        self.tab_widget.addTab(self.overview_tab, "Overview")
        self.tab_widget.addTab(self.singer_tab, "Singers")
        self.tab_widget.addTab(self.language_tab, "Languages")
        self.tab_widget.addTab(self.phoneme_tab, "Phonemes")
        
        # Add the tab widget to the main layout
        self.layout.addWidget(self.tab_widget)
        
        # Add a refresh button
        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch()
        self.refresh_button = QPushButton("Refresh Statistics")
        self.button_layout.addWidget(self.refresh_button)
        
        self.layout.addLayout(self.button_layout)
