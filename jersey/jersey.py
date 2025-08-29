
import easyocr
import cv2
import numpy as np
from collections import defaultdict, Counter

class JerseyNumberRecognition:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        
    def detect_jersey_number(self, frame, bbox):
        # Crop and preprocess the image
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0] / 2), :]
        
        # Resize with aspect ratio preservation
        h, w = top_half_image.shape[:2]
        scale_factor = 128
        resized_image = cv2.resize(top_half_image, (scale_factor, scale_factor))

        # Perform OCR
        results = self.reader.readtext(resized_image)

        if results:
            _, text, confidence = results[0]
            if text.isdigit() and confidence>0.60:
                return text, confidence
        
        return 'x', 0



    def assign_best_jersey_for_players(self,tracks):
        # Dictionary to hold jersey numbers and confidence for each player across frames
        player_jersey_data = defaultdict(list)

        # Collect all jersey numbers and confidences for each player
        for frame_num, player_tracks in enumerate(tracks['players']):
            for player_id, track in player_tracks.items():
                jersey_no = track['jersey']
                jersey_conf=track['jersey_conf']
                player_jersey_data[player_id].append((jersey_no, jersey_conf))

        # Iterate over all players and find the highest confidence jersey
        for player_id, jersey_data in player_jersey_data.items():
            if jersey_data:
                # Get the jersey with the maximum confidence
                best_jersey = max(jersey_data, key=lambda x: x[1])  # (jersey_no, confidence)
                best_jersey_no, best_confidence = best_jersey

                # Assign the best jersey number to all frames for that player
                for frame_num, player_tracks in enumerate(tracks['players']):
                    if player_id in player_tracks:
                        tracks['players'][frame_num][player_id]['jersey'] = best_jersey_no
                        tracks['players'][frame_num][player_id]['jersey_conf'] = best_confidence

