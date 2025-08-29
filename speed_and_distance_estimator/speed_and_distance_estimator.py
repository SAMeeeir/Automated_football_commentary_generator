
import cv2
import numpy as np
import sys 
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24    

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            for frame_num in range(0, len(object_tracks), self.frame_window):
                last_frame = min(frame_num + self.frame_window, len(object_tracks) - 1)
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']
                    if start_position is None or end_position is None:
                        continue
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_mps = distance_covered / (time_elapsed + 1)
                    speed_kph = speed_mps * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0       
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_kph
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_top5_table(self, frame, top5_distance, top5_speed, tracks):
        overlay = frame.copy()
        height, width, _ = frame.shape
        x_start = width - 250  # Right corner padding
        y_start = 50  # Top padding
        spacing = 25  # Spacing between lines
        
        cv2.rectangle(overlay, (x_start - 10, y_start - 10), (width - 10, y_start + 140), (0, 0, 0), -1)  # Distance box
        cv2.rectangle(overlay, (x_start - 10, y_start + 160), (width - 10, y_start + 320), (0, 0, 0), -1)  # Speed box
        
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        y_offset = y_start
        cv2.putText(frame, "Top 5 Distance (m)", (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += spacing
        for track_id, dist in top5_distance[:5]:  # Ensure only 5 entries
            team_no = "Unknown"
            for frame_data in tracks.get('players', []):  # Iterate through frames
                if track_id in frame_data:
                    team_no = frame_data[track_id].get('team', 'Unknown')
                    break
            cv2.putText(frame, f"ID {track_id} (T{team_no}): {dist:.2f} m", (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += spacing
        
        y_offset += 40  # Extra space between tables
        cv2.putText(frame, "Top 5 Speed (km/h)", (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += spacing
        for track_id, speed in top5_speed[:5]:  # Ensure only 5 entries
            team_no = "Unknown"
            for frame_data in tracks.get('players', []):  # Iterate through frames
                if track_id in frame_data:
                    team_no = frame_data[track_id].get('team', 'Unknown')
                    break
            cv2.putText(frame, f"ID {track_id} (T{team_no}): {speed:.2f} km/h", (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += spacing

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        total_distances = {}
        max_speeds = {}
        
        for frame_num, frame in enumerate(frames):
            if frame_num % 30 == 0:  # Update every 30th frame
                total_distances.clear()
                max_speeds.clear()
                for object, object_tracks in tracks.items():
                    if object == "ball" or object == "referees":
                        continue 
                    for track_id, track_info in object_tracks[frame_num].items():
                        if "distance" in track_info:
                            total_distances[track_id] = track_info["distance"]
                        if "speed" in track_info:
                            max_speeds[track_id] = max(max_speeds.get(track_id, 0), track_info["speed"])
                
                top5_distance = sorted(total_distances.items(), key=lambda x: x[1], reverse=True)[:5]
                top5_speed = sorted(max_speeds.items(), key=lambda x: x[1], reverse=True)[:5]
            
            self.draw_top5_table(frame, top5_distance, top5_speed, tracks)
            output_frames.append(frame)
        
        return output_frames


