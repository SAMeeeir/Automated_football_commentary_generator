

import supervision as sv
import numpy as np
import cv2
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
)
from inference import get_model
from ultralytics import YOLO


class PitchLocalization:

    def draw_pitch_localization(self, tracks, frames, model_path):
      
        ROBOFLOW_API_KEY = "ZX86c6n6IQSkFSESSazC"
        FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"
        FIELD_DETECTION_MODEL = get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

        print("Running pitch localization...")
        model_pitch = YOLO(model_path)
        output_frames = []
        CONFIG = SoccerPitchConfiguration()
        key_points_detected=0
        previous_filter = None
        previous_key_points = None
        frame_count1=0


        for frame_count, players in enumerate(tracks['players']):
            players_team1_xy = []
            players_team2_xy = []
            colors = {}  # Reset colors dictionary for each frame
        
            # Process player coordinates
            for player_id, info in players.items():
                x1, y1, x2, y2 = info['bbox']
                center_x = (x1 + x2) / 2
                
                # Convert team_color to tuple for comparison
                team_color_tuple = tuple(info['team_color'])
        
                if info['team'] == 1:
                    players_team1_xy.append([center_x, y2])
                    
                    
                    if '1' not in colors or colors['1'] != team_color_tuple:
                        colors['1'] = team_color_tuple  # Store color for team 1
                
                else:
                    players_team2_xy.append([center_x, y2])
                    
                    if '2' not in colors or colors['2'] != team_color_tuple:
                        colors['2'] = team_color_tuple  # Store color for team 2
                            
                

            ball_xy = []
            r1,g1,b1=colors['1']
            r2,g2,b2=colors['2']

            players_team1_xy = np.array(players_team1_xy, dtype=np.float32)
            players_team2_xy = np.array(players_team2_xy, dtype=np.float32)

            x1, y1, x2, y2 = tracks['ball'][frame_count][1]['bbox']
            center_x = (x1 + x2) / 2
            ball_xy.append([center_x, y2])
            ball_xy = np.array(ball_xy, dtype=np.float32)
            frame = frames[frame_count]
            if frame_count1==0:
            # Predict keypoints on the pitch
                result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
                key_points = sv.KeyPoints.from_inference(result)
            else:
                if frame_count1==25:
                    frame_count1=0
                else:
                    frame_count1=frame_count1+1
                
            if key_points.confidence is not None:
                key_points_detected=1
                filter = key_points.confidence[0] > 0.5
                previous_filter = filter
                previous_key_points = key_points
            else:
                filter = previous_filter
                key_points = previous_key_points
            if key_points_detected==0:
                continue

            frame_reference_points = key_points.xy[0][filter]
            pitch_reference_points = np.array(CONFIG.vertices)[filter]

            # Create a transformation object
            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )

            # Transform player and ball coordinates to pitch coordinates
            pitch_players_team1_xy = transformer.transform_points(points=players_team1_xy)
            pitch_players_team2_xy = transformer.transform_points(points=players_team2_xy)
            pitch_ball_xy = transformer.transform_points(points=ball_xy)

            # Annotate the pitch
            annotated_frame = draw_pitch(CONFIG )
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=annotated_frame
            )
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_team1_xy,
                face_color=sv.Color(r1,g1,b1),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_frame
            )
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_team2_xy,
                face_color=sv.Color(r2,g2,b2),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_frame
            )

            frame_height, frame_width, _ = frame.shape
            pitch_height, pitch_width, _ = annotated_frame.shape

            # Adjust scale to fit within a smaller area in the top-right corner
            scale = min(frame_width / pitch_width * 0.3, frame_height / pitch_height * 0.2)
            resized_pitch = cv2.resize(annotated_frame, (int(pitch_width * scale), int(pitch_height * scale)))

            # Calculate position for top-right corner placement
            x_offset = (frame_width - resized_pitch.shape[1]) // 2
            y_offset = frame_height - resized_pitch.shape[0]-10

            # Overlay resized pitch onto the main frame
            alpha = 0.7  # Adjust transparency level (0: fully transparent, 1: fully opaque)
            overlay = frame.copy()
            overlay[y_offset:y_offset + resized_pitch.shape[0], x_offset:x_offset + resized_pitch.shape[1]] = resized_pitch

            # Blend the overlay with the original frame
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

           
            

            output_frames.append(frame)

        return output_frames

