#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset browser widget for navigating the singing voice dataset.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeView, QHeaderView, QComboBox,
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QToolButton,
    QFrame, QGroupBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon

class DatasetBrowserWidget(QWidget):
    """Widget for browsing and navigating the dataset."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create search box
        self.search_box = QGroupBox("Search")
        self.search_layout = QVBoxLayout(self.search_box)
        
        self.search_input_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search dataset...")
        self.search_button = QToolButton()
        self.search_button.setText("🔍")
        self.search_input_layout.addWidget(self.search_input)
        self.search_input_layout.addWidget(self.search_button)
        
        self.filter_layout = QHBoxLayout()
        self.filter_layout.addWidget(QLabel("Filter by:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Singer", "Language", "Phoneme"])
        self.filter_layout.addWidget(self.filter_combo)
        
        self.search_layout.addLayout(self.search_input_layout)
        self.search_layout.addLayout(self.filter_layout)
        
        self.layout.addWidget(self.search_box)
        
        # Create dataset tree view
        self.tree_group = QGroupBox("Dataset Contents")
        self.tree_layout = QVBoxLayout(self.tree_group)
        
        self.tree_view = QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setHeaderHidden(False)
        self.tree_view.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tree_view.header().setStretchLastSection(True)
        
        # Create and set the model
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(["Name", "Type"])
        self.tree_view.setModel(self.tree_model)
        
        # Add some placeholder data
        self._add_placeholder_data()
        
        self.tree_layout.addWidget(self.tree_view)
        self.layout.addWidget(self.tree_group)
        
        # Selection controls
        self.selection_group = QGroupBox("Selection")
        self.selection_layout = QVBoxLayout(self.selection_group)
        
        self.selection_info_layout = QVBoxLayout()
        self.selected_item_label = QLabel("No item selected")
        self.selection_info_layout.addWidget(self.selected_item_label)
        
        self.selection_buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Selection")
        self.load_button.setEnabled(False)
        self.selection_buttons_layout.addWidget(self.load_button)
        
        self.selection_layout.addLayout(self.selection_info_layout)
        self.selection_layout.addLayout(self.selection_buttons_layout)
        
        self.layout.addWidget(self.selection_group)
        
        # Connect signals
        self.tree_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
    
    def _add_placeholder_data(self):
        """Add placeholder data to the tree model for UI mockup."""
        # Add singers
        for i in range(1, 4):
            singer_item = QStandardItem(f"Singer {i}")
            singer_item.setData("singer", Qt.UserRole)
            type_item = QStandardItem("Singer")
            
            # Add languages for each singer
            for j in range(1, 3):
                lang_item = QStandardItem(f"Language {j}")
                lang_item.setData("language", Qt.UserRole)
                lang_type_item = QStandardItem("Language")
                
                # Add song files for each language
                for k in range(1, 4):
                    file_item = QStandardItem(f"Song {k}")
                    file_item.setData("file", Qt.UserRole)
                    file_type_item = QStandardItem("Audio")
                    
                    lang_item.appendRow([file_item, file_type_item])
                
                singer_item.appendRow([lang_item, lang_type_item])
            
            self.tree_model.appendRow([singer_item, type_item])
        
        # Expand the first singer
        first_index = self.tree_model.index(0, 0)
        self.tree_view.expand(first_index)
    
    def _on_selection_changed(self, selected, deselected):
        """Handle tree view selection changes."""
        indexes = selected.indexes()
        if indexes:
            # Get the first selected index (column 0 contains the name)
            index = indexes[0]
            item = self.tree_model.itemFromIndex(index)
            
            if item:
                item_type = item.data(Qt.UserRole)
                item_name = item.text()
                
                self.selected_item_label.setText(f"Selected: {item_name} ({item_type})")
                self.load_button.setEnabled(item_type == "file")
        else:
            self.selected_item_label.setText("No item selected")
            self.load_button.setEnabled(False)
