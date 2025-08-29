import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decord import VideoReader
import tensorflow as tf
from huggingface_hub import hf_hub_download
from mpl_toolkits.axes_grid1 import ImageGrid
import keras
from tensorflow.keras import layers
import torch # or any backend.
from videoswin import VideoSwinB
import cv2
import numpy as np
import json
from tensorflow.keras import mixed_precision


class ActionDetectionModel:
    


	processing_model = keras.Sequential(
		[
			layers.Normalization(
				mean=[123.675, 116.28, 103.53],
				variance=[np.square(58.395), np.square(57.12), np.square(57.375)]
			)
		]
	)
	mixed_precision.set_global_policy('mixed_float16')

	def vswin_base(self, weights_path):
		self.model = VideoSwinB(
			num_classes=6,
			include_rescaling=False,
			activation=None
		)
		self.model.load_weights(
			weights_path
		)
		return self.model

	def extract_frames_direct(self, video_path, start_time, duration, fps, target_size=(224, 224)):

		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise ValueError(f"Unable to open video file: {video_path}")

		start_frame = int(start_time * fps)
		end_frame = int((start_time + duration) * fps)

		# Initialize the video capture to the starting frame
		cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

		frames = []
		total_frames_needed = 32
		frame_interval = max((end_frame - start_frame) // total_frames_needed, 1)

		for i in range(total_frames_needed):
			frame_pos = start_frame + i * frame_interval
			if frame_pos >= end_frame:
				break
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
			ret, frame = cap.read()
			if not ret:
				print(f"Warning: Could not read frame at position {frame_pos}.")
				break
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
			frame = cv2.resize(frame, target_size)  # Resize to target dimensions
			frames.append(frame)

		cap.release()

		# If fewer than 32 frames were extracted, pad with the last frame
		while len(frames) < total_frames_needed:
			frames.append(frames[-1])  # Pad with the last available frame

		return np.array(frames[:32], dtype=np.uint8)

	def sliding_window_inference_direct(self, video_path, weight_path, window_duration=4, stride_duration=2, fps=None, confidence_threshold=0.90):

		cap = cv2.VideoCapture(video_path)

		id2label={0: 'goal',
 				  1: 'substitution',
 				  2: 'throw-in',
 				  3: 'background',
 				  4: 'corner',
 				  5: 'foul'}

		if fps is None:
			fps = cap.get(cv2.CAP_PROP_FPS)

		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		video_duration = total_frames / fps
		inference_results = []
		start_time = 0
		print('inferencing action')
		while start_time + window_duration <= video_duration:
			# Extract frames directly
			clip_frames = self.extract_frames_direct(video_path, start_time, window_duration, fps)
			clip_frames = self.processing_model(clip_frames)
			clip_frames = tf.expand_dims(clip_frames, axis=0)
			model = self.vswin_base(weights_path=weight_path)
			y_pred = model(clip_frames, training=False)
			y_pred = y_pred.numpy()

			probabilities = tf.nn.softmax(y_pred)

			# Get top predictions filtered by confidence
			top_indices = np.argsort(probabilities[0])[::-1]
			top_classes = {
				id2label[i]: float(probabilities[0][i])
				for i in top_indices if probabilities[0][i] > confidence_threshold
			}

			start_frame = int(start_time * fps)

			if top_classes:
				inference_result = {
					"time": f"{start_time:.2f}-{start_time + window_duration:.2f}",
					"frame_number": start_frame,
					"class": top_classes
				}
				inference_results.append(inference_result)


			# Move to the next window
			start_time += stride_duration

		cap.release()
		return inference_results

	    
	    
    
    