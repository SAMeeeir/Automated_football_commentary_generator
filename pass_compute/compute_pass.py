

from collections import defaultdict

class ComputePass:
    def compute_pass_player(self, tracks):
        previous_frame = None
        count=0
        print('Running Pass Computation')
        # Initialize stats for players in all frames
        for frame_num, players in enumerate(tracks['players']):
            for player_id, info in players.items():
                info.setdefault("accurate_passes", 0)
                info.setdefault("inaccurate_passes", 0)
                info.setdefault("pass_accuracy", 0)

        # Iterate over frames
        for frame_num, player_tracks in enumerate(tracks['players']):
            if previous_frame is not None:
                # Identify the ball holder in the previous frame
                previous_ball_holder = None
                previous_team_color = None
                for player_id, info in previous_frame.items():
                    if info.get('has_ball'):
                        previous_ball_holder = player_id
                        previous_team_color = tuple(info.get('team_color', []))
                        break

                # Identify the ball holder in the current frame
                current_ball_holder = None
                current_team_color = None
                for player_id, info in player_tracks.items():
                    if info.get('has_ball'):
                        current_ball_holder = player_id
                        current_team_color = tuple(info.get('team_color', []))
                        break

                # Check for a pass and update stats
                for player_id in player_tracks:
                    # Copy previous frame stats to the current frame
                    if player_id in tracks['players'][frame_num - 1]:
                        tracks['players'][frame_num][player_id]['accurate_passes'] = tracks['players'][frame_num - 1][player_id]['accurate_passes']
                        tracks['players'][frame_num][player_id]['inaccurate_passes'] = tracks['players'][frame_num - 1][player_id]['inaccurate_passes']
                        tracks['players'][frame_num][player_id]['pass_accuracy'] = tracks['players'][frame_num - 1][player_id]['pass_accuracy']

                if previous_ball_holder is not None and current_ball_holder is not None:
                    if previous_ball_holder != current_ball_holder:
                        if previous_team_color == current_team_color:
                            if previous_ball_holder in player_tracks:
                                tracks['players'][frame_num][previous_ball_holder]['accurate_passes'] += 1
                            else:
                                tracks['players'][frame_num - 1][previous_ball_holder]['accurate_passes'] += 1
                        else:
                            if previous_ball_holder in player_tracks:
                                tracks['players'][frame_num][previous_ball_holder]['inaccurate_passes'] += 1
                            else:
                                tracks['players'][frame_num - 1][previous_ball_holder]['inaccurate_passes'] += 1

            # Update the previous frame
            previous_frame = player_tracks

        # Calculate pass accuracy for players
        for frame_num, players in enumerate(tracks['players']):
            for player_id, info in players.items():
                total_passes = info["accurate_passes"] + info["inaccurate_passes"]
                info['pass_accuracy'] = (
                    (info["accurate_passes"] / total_passes * 100) if total_passes > 0 else 0
                )
        return tracks


    
    def compute_pass_team(self, tracks):

        if 'team_status' not in tracks:
            tracks['team_status'] = []
        
        # Initialize team statistics for all frames
        for _ in range(len(tracks['players'])):
            tracks['team_status'].append({
                1: {  # Team 1
                    'accurate_passes': 0,
                    'inaccurate_passes': 0,
                    'pass_accuracy': 0,
                    'ball_control':0,
                },
                2: {  # Team 2
                    'accurate_passes': 0,
                    'inaccurate_passes': 0,
                    'pass_accuracy': 0,
                    'ball_control':0,
                },
            })
        accurate_passes_team_1 = 0
        inaccurate_passes_team_1 = 0
        accurate_passes_team_2 = 0
        inaccurate_passes_team_2 = 0
        
        # Compute team-level statistics for each frame
        for frame_num, players in enumerate(tracks['players']):
            

            # Aggregate individual player stats by team
            for player_id, player_info in players.items():
                team = player_info.get('team', -1)  # Default to -1 if team is missing
                if team == 1:  # Team 1
                    accurate_passes_team_1 += player_info.get('accurate_passes', 0)
                    inaccurate_passes_team_1 += player_info.get('inaccurate_passes', 0)
                elif team == 2:  # Team 2
                    accurate_passes_team_2 += player_info.get('accurate_passes', 0)
                    inaccurate_passes_team_2 += player_info.get('inaccurate_passes', 0)
            
            # Calculate pass accuracy for both teams
            total_passes_team_1 = accurate_passes_team_1 + inaccurate_passes_team_1
            pass_accuracy_team_1 = (
                (accurate_passes_team_1 / total_passes_team_1 * 100)
                if total_passes_team_1 > 0 else 0
            )
            
            total_passes_team_2 = accurate_passes_team_2 + inaccurate_passes_team_2
            pass_accuracy_team_2 = (
                (accurate_passes_team_2 / total_passes_team_2 * 100)
                if total_passes_team_2 > 0 else 0
            )
            
            # Populate team stats for the frame
            tracks['team_status'][frame_num][1]['accurate_passes'] = accurate_passes_team_1
            tracks['team_status'][frame_num][1]['inaccurate_passes'] = inaccurate_passes_team_1
            tracks['team_status'][frame_num][1]['pass_accuracy'] = pass_accuracy_team_1
            
            tracks['team_status'][frame_num][2]['accurate_passes'] = accurate_passes_team_2
            tracks['team_status'][frame_num][2]['inaccurate_passes'] = inaccurate_passes_team_2
            tracks['team_status'][frame_num][2]['pass_accuracy'] = pass_accuracy_team_2

        return tracks




