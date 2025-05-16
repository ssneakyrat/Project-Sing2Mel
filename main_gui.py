#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Singing Voice Dataset Visualizer - Main Application

This application provides an interactive GUI for visualizing and exploring
a singing voice dataset with mel spectrograms, F0, phoneme sequences, and audio.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui.main_window import MainWindow

def main():
    """Main application entry point."""
    # Set up the application
    app = QApplication(sys.argv)
    app.setApplicationName("Singing Voice Dataset Visualizer")
    app.setOrganizationName("SingingVoice")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    main_window = MainWindow()
    main_window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
