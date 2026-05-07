import cv2
import global_var
import os
from scenedetect import open_video, SceneManager, StatsManager
from scenedetect.detectors import AdaptiveDetector

input_folder = global_var.input_folder
output_folder = global_var.good_cuts
target_frames = global_var.frames

def split_with_buffer(video_path, filename):
    if filename.startswith('.') or not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return

    try:
        video = open_video(video_path)
    except Exception as e:
        print(f"Critic error on {filename}: {e}")
        return

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(AdaptiveDetector())
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Critic error on {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    base_name = os.path.splitext(filename)[0]

    for i, (start_time, end_time) in enumerate(scene_list):
        if i == 0: continue

        cut_frame = start_time.get_frames()
        
        start_idx = cut_frame - target_frames
        start_idx = max(0, start_idx)

        name_of_file = f"{i+1:03d}_{base_name}.mp4"
        full_output_path = os.path.join(output_folder, name_of_file)
        
        out = cv2.VideoWriter(full_output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames_written = 0
        frames_read = 0

        while frames_written < target_frames:
            ret, frame = cap.read()
            if not ret:
                break 
            
            if frames_read % 2 == 0:
                out.write(frame)
                frames_written += 1
                
            frames_read += 1

        out.release()
        # print(f"Exported : {name_of_file}")

    cap.release()
     
if __name__ == "__main__":
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        full_input_path = os.path.join(input_folder, filename)
        split_with_buffer(full_input_path, filename)