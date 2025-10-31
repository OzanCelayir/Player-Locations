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

    def get_center_of_bbox(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def get_bbox_width(self, bbox):
        """Get width of bounding box"""
        return bbox[2] - bbox[0]

    def get_foot_position(self, bbox):
        """Get foot position (bottom center) of bounding box"""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    def map_to_field_coordinates(self, image_point, frame_shape):
        """
        Map image coordinates to field coordinates in meters
        """
        if self.video_dimensions is None:
            self.video_dimensions = (frame_shape[1], frame_shape[0])

        x_img, y_img = image_point

        # Map to field coordinates (0-105m for x, 0-68m for y)
        x_field = (x_img / self.video_dimensions[0]) * self.field_dimensions[0]
        y_field = (y_img / self.video_dimensions[1]) * self.field_dimensions[1]

        return (round(x_field, 2), round(y_field, 2))

    def get_distance_from_center(self, field_coords):
        """
        Calculate distance from field center
        """
        center_x, center_y = self.field_dimensions[0] / 2, self.field_dimensions[1] / 2
        distance = np.sqrt((field_coords[0] - center_x) ** 2 + (field_coords[1] - center_y) ** 2)
        return round(distance, 2)

    def get_zone_position(self, field_coords):
        """
        Determine which zone the player is in (defensive, midfield, attacking)
        """
        x, y = field_coords

        # Define field zones based on x-coordinate
        if x < 35:  # Defensive third
            zone = "Defensive"
        elif x < 70:  # Middle third
            zone = "Midfield"
        else:  # Attacking third
            zone = "Attacking"

        return zone

    def draw_ellipse(self, frame, bbox, color, track_id=None, is_goalkeeper=False):
        """
        Draw ellipse for player/referee tracking
        """
        y2 = int(bbox[3])
        x_center, _ = self.get_center_of_bbox(bbox)
        width = self.get_bbox_width(bbox)

        if is_goalkeeper:
            width = int(width * 1.2)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=3 if is_goalkeeper else 2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          (0, 0, 0),
                          1)

            track_id_str = str(track_id)
            x1_text = x1_rect + 12
            if len(track_id_str) > 2:
                x1_text -= 10

            cv2.putText(
                frame,
                track_id_str,
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame

    def draw_position_info(self, frame, bbox, field_coords, track_id, team, is_goalkeeper=False):
        """
        Draw position information near the player
        """
        x1, y1, x2, y2 = map(int, bbox)

        text_y = y1 - 10
        if text_y < 20:
            text_y = y2 + 25

        position_text = f"({field_coords[0]}, {field_coords[1]})"

        zone = self.get_zone_position(field_coords)

        text_color = self.colors['player_team1'] if team == 'team1' else self.colors['player_team2']
        if is_goalkeeper:
            text_color = self.colors['goalkeeper']

        cv2.putText(frame, position_text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        cv2.putText(frame, zone, (x1, text_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        return frame

    def draw_field_overlay(self, frame):
        """
        Draw field coordinate overlay on the frame
        """
        overlay = frame.copy()
        alpha = 0.7

        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def draw_frame_stats(self, frame, frame_data):
        """
        Draw frame statistics and position information
        """
        frame = self.draw_field_overlay(frame)

        cv2.putText(frame, f"Frame: {frame_data['frame_number']}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {frame_data['timestamp']:.1f}s", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        players = [obj for obj_key, obj in frame_data['objects'].items() if
                   'player' in obj_key and 'goalkeeper' not in obj_key]
        goalkeepers = [obj for obj_key, obj in frame_data['objects'].items() if 'goalkeeper' in obj_key]

        cv2.putText(frame, f"Players: {len(players)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Goalkeepers: {len(goalkeepers)}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Field Coordinates Active", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draw triangle for ball tracking
        """
        y = int(bbox[1])
        x, _ = self.get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def detect_objects(self, frame):
        """
        Detect objects in the frame using YOLO
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        return results[0] if results else None

    def classify_objects(self, detections):
        """
        Classify detected objects into players, goalkeepers, referees, and ball
        """
        classified_objects = {
            'players': [],
            'goalkeepers': [],
            'referees': [],
            'ball': []
        }

        if detections is None:
            return classified_objects

        boxes = detections.boxes
        if boxes is None:
            return classified_objects

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()

            if class_id in self.class_names:
                obj_type = self.class_names[class_id]
                if obj_type == 'player':
                    classified_objects['players'].append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'center': self.get_center_of_bbox(bbox),
                        'is_goalkeeper': False
                    })
                elif obj_type == 'goalkeeper':
                    classified_objects['goalkeepers'].append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'center': self.get_center_of_bbox(bbox),
                        'is_goalkeeper': True
                    })
                elif obj_type == 'referee':
                    classified_objects['referees'].append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'center': self.get_center_of_bbox(bbox)
                    })
                elif obj_type == 'ball':
                    classified_objects['ball'].append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'center': self.get_center_of_bbox(bbox)
                    })

        return classified_objects

    def assign_team_colors(self, players, frame_width):
        """
        Assign team colors based on player positions
        """
        for player in players:
            bbox = player['bbox']
            x_center, _ = self.get_center_of_bbox(bbox)

            if x_center < frame_width / 2:
                player['team_color'] = self.colors['player_team1']
                player['team'] = 'team1'
            else:
                player['team_color'] = self.colors['player_team2']
                player['team'] = 'team2'

        return players

    def track_players(self, frame, frame_number):
        """
        Main tracking function with position measurement
        """
        detections = self.detect_objects(frame)
        classified_objects = self.classify_objects(detections)

        frame_data = {
            'frame_number': frame_number,
            'timestamp': frame_number / 30,
            'objects': {}
        }

        frame_width = frame.shape[1]
        classified_objects['players'] = self.assign_team_colors(classified_objects['players'], frame_width)

        for i, player in enumerate(classified_objects['players']):
            bbox = player['bbox']
            color = player['team_color']
            team = player['team']
            is_goalkeeper = player['is_goalkeeper']

            center = self.get_center_of_bbox(bbox)
            field_coords = self.map_to_field_coordinates(center, frame.shape)
            distance_from_center = self.get_distance_from_center(field_coords)
            zone = self.get_zone_position(field_coords)

            frame = self.draw_ellipse(frame, bbox, color, track_id=i + 1, is_goalkeeper=is_goalkeeper)
            frame = self.draw_position_info(frame, bbox, field_coords, i + 1, team, is_goalkeeper)

            obj_data = {
                'bbox': bbox,
                'center_pixel': center,
                'center_field': field_coords,
                'confidence': player['confidence'],
                'team': team,
                'is_goalkeeper': is_goalkeeper,
                'distance_from_center': distance_from_center,
                'zone': zone
            }
            frame_data['objects'][f'player_{i + 1}'] = obj_data

        for i, goalkeeper in enumerate(classified_objects['goalkeepers']):
            bbox = goalkeeper['bbox']
            center = self.get_center_of_bbox(bbox)
            field_coords = self.map_to_field_coordinates(center, frame.shape)
            distance_from_center = self.get_distance_from_center(field_coords)
            zone = self.get_zone_position(field_coords)

            frame = self.draw_ellipse(frame, bbox, self.colors['goalkeeper'], track_id=100 + i, is_goalkeeper=True)
            frame = self.draw_position_info(frame, bbox, field_coords, 100 + i, 'goalkeeper', True)

            obj_data = {
                'bbox': bbox,
                'center_pixel': center,
                'center_field': field_coords,
                'confidence': goalkeeper['confidence'],
                'team': 'goalkeeper',
                'is_goalkeeper': True,
                'distance_from_center': distance_from_center,
                'zone': zone
            }
            frame_data['objects'][f'goalkeeper_{i + 1}'] = obj_data

        for i, referee in enumerate(classified_objects['referees']):
            bbox = referee['bbox']
            center = self.get_center_of_bbox(bbox)
            field_coords = self.map_to_field_coordinates(center, frame.shape)

            frame = self.draw_ellipse(frame, bbox, self.colors['referee'], track_id=200 + i)

            obj_data = {
                'bbox': bbox,
                'center_pixel': center,
                'center_field': field_coords,
                'confidence': referee['confidence']
            }
            frame_data['objects'][f'referee_{i + 1}'] = obj_data

        for i, ball in enumerate(classified_objects['ball']):
            bbox = ball['bbox']
            center = self.get_center_of_bbox(bbox)
            field_coords = self.map_to_field_coordinates(center, frame.shape)

            frame = self.draw_triangle(frame, bbox, self.colors['ball'])

            obj_data = {
                'bbox': bbox,
                'center_pixel': center,
                'center_field': field_coords,
                'confidence': ball['confidence']
            }
            frame_data['objects'][f'ball_{i + 1}'] = obj_data

        frame = self.draw_frame_stats(frame, frame_data)

        self.tracking_data.append(frame_data)
        return frame, frame_data

    def process_video(self, video_path, max_frames=None):
        """
        Process video file for tracking with position measurement
        """
        if not os.path.exists(video_path):
            print(f"Error: Input video not found: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False

        # Generate output video path on desktop
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = self.output_dir / f"football_tracking_{timestamp}.avi"

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        print(f"Starting video processing...")
        print(f"Output will be saved to: {output_video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            processed_frame, frame_data = self.track_players(frame, frame_count)

            cv2.imshow('Football Tracker - Position Measurement', processed_frame)
            out.write(processed_frame)

            if frame_count % 30 == 0:
                print(f"Processed frame {frame_count}")
                self.print_detailed_position_info(frame_data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Processing complete. Processed {frame_count} frames.")
        print(f"Output video saved: {output_video_path}")
        return True

    def print_detailed_position_info(self, frame_data):
        """
        Print detailed position information for current frame
        """
        print(f"\n=== Frame {frame_data['frame_number']} Position Data ===")
        print(f"Timestamp: {frame_data['timestamp']:.1f}s")

        players = [obj for obj_key, obj in frame_data['objects'].items() if
                   'player' in obj_key and 'goalkeeper' not in obj_key]
        goalkeepers = [obj for obj_key, obj in frame_data['objects'].items() if 'goalkeeper' in obj_key]

        print(f"\nPlayers ({len(players)}):")
        for i, (obj_key, obj) in enumerate(
                [(k, v) for k, v in frame_data['objects'].items() if 'player' in k and 'goalkeeper' not in k]):
            print(
                f"  Player {i + 1}: {obj['team']} - Position: {obj['center_field']} - Zone: {obj['zone']} - Dist from center: {obj['distance_from_center']}m")

    def create_player_location_dataset(self):
        """
        Create comprehensive dataset of players' locations
        """
        if not self.tracking_data:
            print("No tracking data available. Please process a video first.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. CSV Dataset
        csv_path = self.dataset_dir / f"player_locations_{timestamp}.csv"
        data_rows = []

        for frame_data in self.tracking_data:
            for obj_key, obj in frame_data['objects'].items():
                if 'player' in obj_key or 'goalkeeper' in obj_key:
                    data_rows.append({
                        'frame_number': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'object_type': obj_key.split('_')[0],
                        'object_id': obj_key.split('_')[1] if '_' in obj_key else '1',
                        'x_pixel': obj['center_pixel'][0],
                        'y_pixel': obj['center_pixel'][1],
                        'x_field': obj['center_field'][0],
                        'y_field': obj['center_field'][1],
                        'confidence': obj['confidence'],
                        'team': obj.get('team', ''),
                        'is_goalkeeper': obj.get('is_goalkeeper', False),
                        'distance_from_center': obj.get('distance_from_center', 0),
                        'zone': obj.get('zone', '')
                    })

        df = pd.DataFrame(data_rows)
        df.to_csv(csv_path, index=False)

        # 2. JSON Dataset (structured format)
        json_path = self.dataset_dir / f"player_locations_{timestamp}.json"
        dataset_json = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_frames': len(self.tracking_data),
                'total_detections': len(data_rows),
                'field_dimensions': self.field_dimensions
            },
            'frames': []
        }

        for frame_data in self.tracking_data:
            frame_info = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp'],
                'players': []
            }

            for obj_key, obj in frame_data['objects'].items():
                if 'player' in obj_key or 'goalkeeper' in obj_key:
                    player_info = {
                        'object_id': obj_key,
                        'type': obj_key.split('_')[0],
                        'position_pixel': obj['center_pixel'],
                        'position_field': obj['center_field'],
                        'team': obj.get('team', ''),
                        'zone': obj.get('zone', ''),
                        'confidence': obj['confidence']
                    }
                    frame_info['players'].append(player_info)

            dataset_json['frames'].append(frame_info)

        with open(json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)

        # 3. Summary Statistics
        stats_path = self.dataset_dir / f"dataset_summary_{timestamp}.txt"
        with open(stats_path, 'w') as f:
            f.write("=== Football Player Location Dataset Summary ===\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {len(self.tracking_data)}\n")
            f.write(f"Total Player Detections: {len(data_rows)}\n")
            f.write(f"Average Players Per Frame: {len(data_rows) / len(self.tracking_data):.2f}\n")

            if len(data_rows) > 0:
                player_data = df[df['object_type'] == 'player']
                f.write(f"Team Distribution:\n")
                f.write(f"  Team 1: {len(player_data[player_data['team'] == 'team1'])} detections\n")
                f.write(f"  Team 2: {len(player_data[player_data['team'] == 'team2'])} detections\n")
                f.write(f"  Goalkeepers: {len(df[df['is_goalkeeper'] == True])} detections\n")

        print(f"\n=== Dataset Created ===")
        print(f"CSV Data: {csv_path}")
        print(f"JSON Data: {json_path}")
        print(f"Summary: {stats_path}")
        print(f"Total player location records: {len(data_rows)}")

        return {
            'csv': csv_path,
            'json': json_path,
            'summary': stats_path
        }


# Main execution
if __name__ == "__main__":
    # Initialize tracker
    tracker = FootballTracker()

    # Process video (update with your video path)
    input_video = "/Users/ozancelayir/Desktop/08fd33_4.mp4"  # Change this to your video path

    try:
        # Process video
        success = tracker.process_video(input_video, max_frames=100)

        if success:
            # Create dataset
            dataset_files = tracker.create_player_location_dataset()

            print(f"\n=== Project Output Summary ===")
            print(f"Project Location: {tracker.project_dir}")
            print(f"Output Video: {tracker.output_dir}")
            print(f"Dataset Files: {tracker.dataset_dir}")
            print(f"All files have been saved to your desktop!")

        else:
            print("Video processing failed. Please check the input video path.")

    except Exception as e:
        print(f"An error occurred: {e}")
