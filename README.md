
# Football Analysis Tracker

A comprehensive computer vision system for tracking football players, detecting positions, and analyzing gameplay using YOLO object detection.

## Features

- **Real-time Player Tracking**: Track players, goalkeepers, referees, and ball
- **Position Mapping**: Convert pixel coordinates to field coordinates (meters)
- **Team Classification**: Automatically assign players to teams based on position
- **Zone Analysis**: Determine player positions (Defensive/Midfield/Attacking)
- **Data Export**: Export tracking data to CSV, JSON, and video formats
- **Field Coordinate System**: Standard football field dimensions (105x68 meters)

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO

### Quick Install
```bash
git clone https://github.com/yourusername/football-analysis-tracker.git
cd football-analysis-tracker
pip install -r requirements.txt
