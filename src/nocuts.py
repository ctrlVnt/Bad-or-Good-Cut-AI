import cv2
import os
import shutil
from pathlib import Path
from scenedetect import AdaptiveDetector, detect

def extract_first_32_frames(input_folder, output_folder, frame_count=32):
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_path = Path(input_folder)
    valid_ext = {'.mp4', '.avi', '.mov', '.mkv'}

    for video_file in input_path.iterdir():
        if video_file.suffix.lower() not in valid_ext:
            continue

        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Salta i video troppo corti
        if total_frames < frame_count:
            print(f"Saltato {video_file.name}: troppo corto ({total_frames} frame)")
            cap.release()
            continue

        print(f"Analizzando primi {frame_count} frame di: {video_file.name}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        temp_segment = "temp_segment.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_segment, fourcc, fps, (width, height))
        
        # Estraiamo solo i primi 32 frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(frame_count):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
        out.release()
        cap.release() # Possiamo chiudere il file originale subito

        # Analizziamo se c'è un taglio nei primi 32 frame
        scene_list = detect(temp_segment, AdaptiveDetector())

        if len(scene_list) <= 1: 
            output_filename = f"{video_file.stem}_start_0.mp4"
            final_path = os.path.join(output_folder, output_filename)
            shutil.move(temp_segment, final_path) 
            print(f" -> OK: Segmento iniziale pulito salvato.")
        else:
            if os.path.exists(temp_segment):
                os.remove(temp_segment)
            print(f" -> SCARTATO: Rilevato taglio nei primi {frame_count} frame.")

# Utilizzo dello script
extract_first_32_frames(input_folder='/Volumes/Z Slim/_sample/', output_folder='/Volumes/Z Slim/No_cuts')