import pandas as pd
import openai
import numpy as np
import os
import time
from pydub import AudioSegment
import json
from pathlib import Path
from openai import OpenAI
import moviepy
from moviepy.editor import VideoFileClip, AudioFileClip

class Commentary:

    def __init__(self,action_data, match_info, yolo_data,fps):
        self.action_data=action_data
        self.match_info=match_info
        self.yolo_data=yolo_data
        self.fps=fps
        self.api_key=""

    def format_data(self):
        count = 1
        player_records=[]
        for frame_num, frame in enumerate(self.yolo_data['players']):
            if (frame_num+1) % self.fps == 0:
                for player_id, info in frame.items():
                    player_records.append({
                        'player_id': player_id,
                        'frame_no': count,
                        'speed': info.get('speed', 0),
                        'distance': info.get('distance', 0),
                        'team': info.get('team', 'unknown'),
                        'acc_passes': info.get('accurate_passes', 0),
                        'inacc_passes': info.get('inaccurate_passes', 0),
                        'pass_acc': info.get('pass_accuracy', 0.0),
                        'has_ball': info.get('has_ball', False)

                    })
                count += 1
        player_df = pd.DataFrame(player_records)
        player_df = player_df[player_df['frame_no'] % 10 == 0]

        players_with_ball = [record for record in player_records if record['has_ball']]

        df_with_ball = pd.DataFrame(players_with_ball)

        df_with_ball = df_with_ball[df_with_ball['frame_no'] % 10 == 0]
        print(df_with_ball.head())
        team_count=1
        team_records=[]
# Process each frame's team status
        for frame_num, frame in enumerate(self.yolo_data['team_status']):
            if (frame_num+1) % 150 == 0:
                record = {'frame_no': team_count}
                for team, stats in frame.items():
                    record[f'accurate_passes_team{team}'] = stats.get('accurate_passes', 0)
                    record[f'inaccurate_passes_team{team}'] = stats.get('inaccurate_passes', 0)
                    record[f'pass_accuracy_team{team}'] = stats.get('pass_accuracy', 0)
                    record[f'team_ball_control{team}']=stats.get('ball_control',0)

                team_records.append(record)
                team_count=team_count+1

            # Create DataFrame
        team_df = pd.DataFrame(team_records)
        
        return df_with_ball,team_df



    def generate_football_commentary(self,action_data, player_data, team_data, external_info):
        prompt = f"""


            ### Data Descriptions:
            1. `player_data`: Contains information about individual player which has ball in each frame. Columns:
            2. `team_data`: Contains aggregated statistics for each team.
            3. 'action_data' (Highest Priority) :Contains critical in-game actions detected at specific timestamps.
            4. 'external_info': General match-related details.

            ### Critical Instructions:
            1. **Introduction**: Start the commentary with an introduction using `external_info` to describe the teams, venue.
            2. **TTS Constraints**: Generate sentences upto 19-24 words (2.33 words/sec × 10 sec = ~23 words).
            3. **Numerical Integration**: Always include specific numbers from datasets
            4. **Frame Management**:
            - 6 fixed frames at [10,20,30,40,50,60] seconds
            - Actions override next available frame (25s action → 30s frame)
            5. **Data Balance**:
            - Use external_info in middle non-consecutive frames beyond introduction
            6. **Priority Order**:
            1. Action_data triggers
            2. Numerical team/player stats
            3. External context weaving
            7. **No Reuse of Frames**: Once a frame is used (e.g., frame 10), it cannot be reused in subsequent commentary.
            8. **Do not mention the time in generated text commentary**

            ### Match Data:
            - **Action Frames**: {[a['frameno'] for a in action_data]}
            - **Team Data Frames**: 10s, 20s, 30s, 40s, 50s, 60s.
            - **Player Data Frames**: 10s, 20s, 30s, 40s, 50s, 60s.
            - **External Info**: {external_info}

            ### **Match Data Inputs:**
            - `action_data`: {action_data}
            - `player_data`: {player_data}
            - `team_data`: {team_data}
            - `external_info`: {external_info}


            **You will provide the output in json format with the following properties:**
            1.TimeStart which will be the time when the commentary start
            2.Text will be your generated commentary.
          



            """


        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {'role': 'system', 'content': 'You are an expert football commentator providing real-time commentary for a live match based on structured data.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=2000,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    
    def text_to_speech(self):
        df_with_ball, team_df = self.format_data()
        openai.api_key=self.api_key
        commentary_text = self.generate_football_commentary(self.action_data, df_with_ball, team_df, self.match_info)
        print(team_df )
        print(self.match_info)
        print(commentary_text)
        
        os.makedirs("/kaggle/working/commentary_audio", exist_ok=True)
        client=OpenAI(api_key=self.api_key)

        audio_files = []
        prev_end_time = 0  # Track the end time of the last commentary
        lines=commentary_text.strip().split("\n")

        for i,line in enumerate(lines):
            if ":" in line:
                start_time,text = line.split(":", 1) # Already in seconds

            # Generate speech from text
            audio_path = f"/kaggle/working/commentary_audio/commentary{i}.mp3"
            response = client.audio.speech.create(
                model="tts-1",
                voice="ash",
                input=text
            )

            with open(audio_path, "wb") as f:
                f.write(response.content)  # response.content contains the binary audio data

            # Load generated audio and get duration
            commentary_audio = AudioSegment.from_mp3(audio_path)
            duration = len(commentary_audio) / 1000  # Convert ms to seconds

            end_time = start_time + duration

            # Check if the next commentary overlaps
            if i < len(commentary_text) - 1:
                next_start_time,next_text= lines[i+1].split(":",1)
                if end_time > next_start_time:
                    # Cut the current commentary short to avoid overlap
                    trim_duration = next_start_time - start_time
                    commentary_audio = commentary_audio[:int(trim_duration * 1000)]
                    end_time = next_start_time  # Update new end time
                    silent_gap_duration = 0  # No silent gap after trimming
                else:
                    if(start_time % 10==0):
                        silent_gap_duration = 10 - duration
                    else:
                        silent_gap_duration = 5+max(0, 10 - duration)

      

            # Silent gap calculation: Ensure no overlap
            silent_gap = AudioSegment.silent(duration=int(silent_gap_duration * 1000))

        

            # Store details for merging
            audio_files.append((audio_path, commentary_audio, silent_gap))
            prev_end_time = end_time  # Update previous end time

        # Merge the audio files with correct silent gaps **after** each audio
        merged_audio = AudioSegment.silent(duration=0)
        for audio_file, commentary_audio, silent_gap in audio_files:
            merged_audio += commentary_audio + silent_gap  # Add commentary and THEN silence

        # Save the merged file
        merged_audio.export("/kaggle/working/commentary_audio/merged_commentary.mp3", format="mp3")

        print("✅ AI Commentary with Correct Timing Generated Successfully!")

    def combine_audio_video(self):
               
        # Load your silent video and audio file
        video_path =  "/kaggle/working/project/output_videos/output_video1.avi"# Replace with your video file path
        audio_path = "/kaggle/working/commentary_audio/merged_commentary.mp3"       

        # Load video and audio clips
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        # Trim the video to match the audio duration
        video_clip = video_clip.subclip(0, audio_clip.duration)

        # Set the audio of the video
        video_with_audio = video_clip.set_audio(audio_clip)

        # Output the final video with audio
        output_path = '/kaggle/working/project/output_videos/demo.avi'
        video_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac')

        print(f"Video with audio saved as {output_path}")



