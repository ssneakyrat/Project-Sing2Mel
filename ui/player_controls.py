#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Player controls widget for audio playback.
"""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, 
    QToolButton, QComboBox, QStyle, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon

class PlayerControlsWidget(QWidget):
    """Widget for audio playback controls."""
    
    # Define signals
    playClicked = pyqtSignal()
    pauseClicked = pyqtSignal()
    stopClicked = pyqtSignal()
    positionChanged = pyqtSignal(int)
    volumeChanged = pyqtSignal(int)
    speedChanged = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        self.layout = QHBoxLayout(self)
        self.setFixedHeight(50)  # Set fixed height for the control bar
        
        # Add a border at the top
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(palette)
        
        # Add style
        self.setStyleSheet("""
            QToolButton {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f8f8;
                min-width: 32px;
                min-height: 32px;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
            }
            QToolButton:pressed {
                background-color: #d0d0d0;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #f0f0f0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #5c5c5c;
                border: 1px solid #5c5c5c;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        
        # Create playback controls
        self.play_button = QToolButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setToolTip("Play")
        
        self.pause_button = QToolButton()
        self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pause_button.setToolTip("Pause")
        
        self.stop_button = QToolButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setToolTip("Stop")
        
        # Create position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 100)
        self.position_slider.setValue(0)
        self.position_slider.setTracking(True)
        
        # Time labels
        self.current_time_label = QLabel("0:00")
        self.current_time_label.setMinimumWidth(40)
        self.current_time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        self.separator_label = QLabel("/")
        
        self.total_time_label = QLabel("0:00")
        self.total_time_label.setMinimumWidth(40)
        self.total_time_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Volume control
        self.volume_button = QToolButton()
        self.volume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        self.volume_button.setToolTip("Volume")
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)  # Default volume
        self.volume_slider.setMaximumWidth(100)
        
        # Playback speed control
        self.speed_label = QLabel("Speed:")
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(2)  # Default 1.0x
        self.speed_combo.setFixedWidth(70)
        
        # Add widgets to layout
        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.pause_button)
        self.layout.addWidget(self.stop_button)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.current_time_label)
        self.layout.addWidget(self.position_slider, 1)  # 1 is the stretch factor
        self.layout.addWidget(self.total_time_label)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.volume_button)
        self.layout.addWidget(self.volume_slider)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.speed_label)
        self.layout.addWidget(self.speed_combo)
    
    def _connect_signals(self):
        """Connect signals and slots."""
        self.play_button.clicked.connect(self.playClicked)
        self.pause_button.clicked.connect(self.pauseClicked)
        self.stop_button.clicked.connect(self.stopClicked)
        self.position_slider.valueChanged.connect(self.positionChanged)
        self.volume_slider.valueChanged.connect(self.volumeChanged)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
    
    def _on_speed_changed(self, speed_text):
        """Handle speed combobox changes."""
        speed = float(speed_text.replace('x', ''))
        self.speedChanged.emit(speed)
    
    def set_position(self, position):
        """Set the current position of the slider."""
        self.position_slider.setValue(position)
    
    def set_duration(self, duration_ms):
        """Set the total duration and update the label."""
        self.position_slider.setRange(0, duration_ms)
        minutes = duration_ms // 60000
        seconds = (duration_ms % 60000) // 1000
        self.total_time_label.setText(f"{minutes}:{seconds:02d}")
    
    def update_current_time(self, time_ms):
        """Update the current time label."""
        minutes = time_ms // 60000
        seconds = (time_ms % 60000) // 1000
        self.current_time_label.setText(f"{minutes}:{seconds:02d}")
