

import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    count=0
    frames = []
    while True:
        ret, frame = cap.read()
        count=count+1
        if not ret:
            break
        frames.append(frame)
    print(count)
    return frames,fps

def save_video(ouput_video_frames,fps,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
    print('saved')
    

def sample_frames(orginal_fps,video_frames, fps=3):
    interval = max(int(orginal_fps / fps), 1)  
    video_frames_sampled = video_frames[::interval]
    return video_frames_sampled
    

