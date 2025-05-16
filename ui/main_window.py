#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main application window for the Singing Voice Dataset Visualizer.
"""

import os
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QDockWidget, QToolBar, QStatusBar, 
    QAction, QFileDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeView, QTabWidget, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QSize, QSettings
from PyQt5.QtGui import QIcon

from ui.dataset_browser_widget import DatasetBrowserWidget
from ui.visualization_panel import VisualizationPanel
from ui.player_controls import PlayerControlsWidget
from ui.stats_widget import StatsWidget

class MainWindow(QMainWindow):
    """Main window for the Singing Voice Dataset Visualizer application."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize window properties
        self.setWindowTitle("Singing Voice Dataset Visualizer")
        self.resize(1200, 800)
        
        # Initialize UI components
        self._init_ui()
        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_statusbar()
        
        # Connect signals and slots
        self._connect_signals()
        
        # Restore window settings (if available)
        self._restore_settings()
    
    def _init_ui(self):
        """Initialize the user interface components."""
        # Central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create the main horizontal splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Dataset browser on the left side
        self.dataset_browser = DatasetBrowserWidget()
        self.main_splitter.addWidget(self.dataset_browser)
        
        # Right side widget with visualization panel and controls
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.main_splitter.addWidget(self.right_widget)
        
        # Visualization panel
        self.visualization_panel = VisualizationPanel()
        self.right_layout.addWidget(self.visualization_panel)
        
        # Player controls
        self.player_controls = PlayerControlsWidget()
        self.right_layout.addWidget(self.player_controls)
        
        # Set the splitter sizes
        self.main_splitter.setSizes([int(self.width() * 0.3), int(self.width() * 0.7)])
        
        # Create dock widget for statistics
        self.stats_dock = QDockWidget("Dataset Statistics", self)
        self.stats_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.stats_widget = StatsWidget()
        self.stats_dock.setWidget(self.stats_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.stats_dock)
    
    def _create_actions(self):
        """Create application actions."""
        # File menu actions
        self.action_open_dataset = QAction("&Open Dataset...", self)
        self.action_open_dataset.setStatusTip("Open a dataset directory")
        
        self.action_load_cache = QAction("Load Dataset &Cache...", self)
        self.action_load_cache.setStatusTip("Load a cached dataset file")
        
        self.action_export_visualization = QAction("&Export Visualization...", self)
        self.action_export_visualization.setStatusTip("Export the current visualization")
        self.action_export_visualization.setEnabled(False)
        
        self.action_export_audio = QAction("Export &Audio...", self)
        self.action_export_audio.setStatusTip("Export the current audio clip")
        self.action_export_audio.setEnabled(False)
        
        self.action_quit = QAction("&Quit", self)
        self.action_quit.setShortcut("Ctrl+Q")
        self.action_quit.setStatusTip("Quit the application")
        self.action_quit.triggered.connect(self.close)
        
        # View menu actions
        self.action_show_stats = QAction("Show &Statistics", self)
        self.action_show_stats.setCheckable(True)
        self.action_show_stats.setChecked(True)
        self.action_show_stats.triggered.connect(self._toggle_stats_dock)
        
        # Help menu actions
        self.action_about = QAction("&About", self)
        self.action_about.setStatusTip("Show the application's About box")
        self.action_about.triggered.connect(self._show_about_dialog)
    
    def _create_menus(self):
        """Create application menus."""
        # File menu
        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addAction(self.action_open_dataset)
        self.file_menu.addAction(self.action_load_cache)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.action_export_visualization)
        self.file_menu.addAction(self.action_export_audio)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.action_quit)
        
        # View menu
        self.view_menu = self.menuBar().addMenu("&View")
        self.view_menu.addAction(self.action_show_stats)
        
        # Help menu
        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_menu.addAction(self.action_about)
    
    def _create_toolbars(self):
        """Create application toolbars."""
        # Main toolbar
        self.main_toolbar = QToolBar("Main", self)
        self.main_toolbar.setMovable(False)
        self.main_toolbar.addAction(self.action_open_dataset)
        self.main_toolbar.addAction(self.action_load_cache)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self.action_export_visualization)
        self.main_toolbar.addAction(self.action_export_audio)
        
        self.addToolBar(self.main_toolbar)
    
    def _create_statusbar(self):
        """Create application status bar."""
        self.statusBar().showMessage("Ready")
        
    def _connect_signals(self):
        """Connect signals and slots."""
        # Connect file open action
        self.action_open_dataset.triggered.connect(self._open_dataset)
        self.action_load_cache.triggered.connect(self._load_cache)
        
    def _open_dataset(self):
        """Open a dataset directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Open Dataset Directory", os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            self.statusBar().showMessage(f"Opening dataset from {directory}...")
            # TODO: Implement dataset loading
            QMessageBox.information(self, "Not Implemented", 
                                   "Dataset loading will be implemented in Phase 2.")
    
    def _load_cache(self):
        """Load a cached dataset file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Dataset Cache", os.path.expanduser("~"),
            "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            self.statusBar().showMessage(f"Loading cached dataset from {file_path}...")
            # TODO: Implement cache loading
            QMessageBox.information(self, "Not Implemented", 
                                  "Cache loading will be implemented in Phase 2.")
    
    def _toggle_stats_dock(self, checked):
        """Toggle the statistics dock widget."""
        self.stats_dock.setVisible(checked)
        self.action_show_stats.setChecked(checked)
    
    def _show_about_dialog(self):
        """Show the application's About dialog."""
        QMessageBox.about(
            self,
            "About Singing Voice Dataset Visualizer",
            "<h3>Singing Voice Dataset Visualizer</h3>"
            "<p>An interactive visualization tool for exploring singing voice datasets.</p>"
            "<p>Version: 0.1.0</p>"
        )
    
    def _restore_settings(self):
        """Restore saved application settings."""
        settings = QSettings()
        
        # Restore window geometry if settings exist
        if settings.contains("geometry"):
            self.restoreGeometry(settings.value("geometry"))
        
        if settings.contains("windowState"):
            self.restoreState(settings.value("windowState"))
    
    def closeEvent(self, event):
        """Handle window close event and save settings."""
        # Save window settings
        settings = QSettings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        
        super().closeEvent(event)
