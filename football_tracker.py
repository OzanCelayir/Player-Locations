# This is your main script - copy the entire FootballTracker class here
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
from pathlib import Path
import json
from datetime import datetime


class FootballTracker:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
        """
        Initialize the football tracker
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.field_dimensions = (105, 68)  # Standard field size in meters
        self.video_dimensions = None

        # Define class mappings
        self.class_names = {
            0: 'player',
            1: 'goalkeeper',
            2: 'referee',
            3: 'ball'
        }

        # Colors for different classes
        self.colors = {
            'player_team1': (255, 0, 0),  # Blue
            'player_team2': (0, 0, 255),  # Red
            'goalkeeper': (0, 255, 255),  # Yellow
            'referee': (255, 255, 0),  # Cyan
            'ball': (0, 255, 0)  # Green
        }

        # Tracking data storage
        self.tracking_data = []

        # Create output directories
        self.setup_output_directories()

    def setup_output_directories(self):
        """Create organized output directory structure on desktop"""
        # Get desktop path
        desktop_path = Path.home() / "Desktop"
        project_name = "FootballTrackingProject"

        # Main project directory
        self.project_dir = desktop_path / project_name
        self.output_dir = self.project_dir / "output"
        self.dataset_dir = self.project_dir / "dataset"
        self.exports_dir = self.project_dir / "exports"

        # Create directories
        for directory in [self.project_dir, self.output_dir, self.dataset_dir, self.exports_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        print(f"Project directory created: {self.project_dir}")

    # ... (include all your existing methods here)
    # Copy the rest of your FootballTracker class implementation

# The rest of your existing code...
