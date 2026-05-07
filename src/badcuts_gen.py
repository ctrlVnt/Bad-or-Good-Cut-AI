import cv2
import global_var
import numpy as np
import os
import random
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector

input_folder = global_var.input_folder
output_folder = global_var.bad_cuts
target_frames = global_var.frames

def extract_frames(cap, start_frame_idx, num_frames_to_extract):
    start_frame_idx = max(0, int(start_frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    
    frames_list = []
    for _ in range(num_frames_to_extract):
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(frame)
    return frames_list

def create_bad_cuts(video_path, filename):
    if filename.startswith('.') or not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return
    try:
        video = open_video(video_path)
    except Exception as e:
        print(f"Critic error on {filename}: {e}")
        return

    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    type_options = ["bad_jump", "bad_mix"]
    choice_i = 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    base_name = os.path.splitext(filename)[0]

    for i, (start, end) in enumerate(scene_list):
        if i == 0: continue

        choice = type_options[choice_i]
        cut_frame = start.get_frames()
    
        start_frame_A = max(0, cut_frame - target_frames)
        
        jump_size = target_frames 
        start_frame_B = cut_frame + jump_size

        if choice == "bad_jump":
            out_name_jump = f"bad_jump_frame_{i}_{base_name}.mp4"
            full_path_jump = os.path.join(output_folder, out_name_jump)
            
            frames_A = extract_frames(cap, start_frame_A, target_frames)
            frames_B = extract_frames(cap, start_frame_B, target_frames)
            
            combined_frames = frames_A + frames_B
            final_frames = combined_frames[::2] 
            
            if final_frames:
                out = cv2.VideoWriter(full_path_jump, fourcc, fps, (width, height))
                for f in final_frames[:target_frames]:
                    out.write(f)
                out.release()

        elif choice == "bad_mix":
            other_scene = random.choice(scene_list)
            other_time = other_scene[0].get_frames()
            
            out_name_mix = f"bad_mix_{i}_{base_name}.mp4"
            full_path_mix = os.path.join(output_folder, out_name_mix)
            
            frames_A = extract_frames(cap, start_frame_A, target_frames)
            frames_B = extract_frames(cap, other_time, target_frames)
            
            combined_frames = frames_A + frames_B
            final_frames = combined_frames[::2]
            
            if final_frames:
                out = cv2.VideoWriter(full_path_mix, fourcc, fps, (width, height))
                for f in final_frames[:target_frames]:
                    out.write(f)
                out.release()

        choice_i = 1 if choice_i == 0 and len(scene_list) > 2 else 0

    cap.release()

if __name__ == "__main__":
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        create_bad_cuts(os.path.join(input_folder, filename), filename)