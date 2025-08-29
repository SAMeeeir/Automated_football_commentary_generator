
from utils import read_video, save_video,sample_frames
from trackers import Tracker
import cv2
import json
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from jersey import JerseyNumberRecognition
import pickle
import cloudpickle
from pass_compute import ComputePass
from pitch_view import PitchLocalization
from action_detection import ActionDetectionModel
import pandas
from gpt import Commentary


def main():
    # Read Video
    video_frames,fps = read_video('/kaggle/working/project/input_videos/match.mp4')
    fps=int(fps)
    tracker = Tracker('/kaggle/working/project/models/best1.pt')
    #sample_video_frames=sample_frames(fps,video_frames,5)
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='/kaggle/working/project/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)
    
 
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                               stub_path='/kaggle/working/project/stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    print("Calculating speed and distancce ")
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                     tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
         team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                              player_id)
         tracks['players'][frame_num][player_id]['team'] = team 
         tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
         

    
    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
    
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
    
        for player_id, info in player_track.items():
            info['has_ball'] = False 
    
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # Check if team_ball_control is empty before accessing the last element
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])  # Use previous frame's value
            else:
                team_ball_control.append(-1)  # Default value if no previous data exists

    team_ball_control = np.array(team_ball_control)



# Detect Jerseys in Each Frame
    #for frame_num, player_track in enumerate(tracks['players']):
     #   for player_id, track in player_track.items():
      #      jersey_no, jersey_conf = jersey_assigner.detect_jersey_number(
       #                              video_frames[frame_num], track['bbox'])
        #    tracks['players'][frame_num][player_id]['jersey'] = jersey_no
         #   tracks['players'][frame_num][player_id]['jersey_conf']=jersey_conf
# Assign Most Confident Jerseys
 
    #jersey_assigner.assign_best_jersey_for_players(tracks)



    pitch_draw=PitchLocalization()

    
    pass_accuracy=ComputePass()
    tracks=pass_accuracy.compute_pass_player(tracks)
    tracks=pass_accuracy.compute_pass_team(tracks)
    print("Drawing annotation")
    output_video_frames=speed_and_distance_estimator.draw_speed_and_distance(video_frames,tracks)
    output_video_frames,tracks = tracker.draw_annotations(output_video_frames, tracks,team_ball_control)
    output_video_frames=pitch_draw.draw_pitch_localization(tracks,output_video_frames,'/kaggle/working/project/models/best_pitch.pt')
    
    
    with open('/kaggle/working/project/first.pkl', 'wb') as f:
        cloudpickle.dump(tracks, f)
    
    # Save video
    save_video(output_video_frames,fps, '/kaggle/working/project/output_videos/output_video1.avi')
    
    action_detection_model = ActionDetectionModel()
    action_weights_path = "/kaggle/working/project/models/swin.weights.h5"


    print("Running action detection...")
    actions = action_detection_model.sliding_window_inference_direct(
        video_path='/kaggle/working/project/input_videos/match.mp4',
        weight_path=action_weights_path,
        window_duration=4,
        stride_duration=2,
        fps=fps,
        confidence_threshold=0.80
    )

    audio_generator=Commentary(actions, tracks, fps)
    audio_generator.text_to_speech()
    audio_generator.combine_audio_video()
   

      
if __name__ == '__main__':
    main()

