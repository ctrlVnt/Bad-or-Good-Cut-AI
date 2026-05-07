import os
import shutil
import torch
import torch.nn as nn
from torchvision.io import read_video
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# --- 1. CONFIGURATION ---
# Insert here the name of your saved file
MODEL_PATH = "good-bad-weights.pth" 
# Insert the path to the folder with the new videos (the converted ones)
VIDEOS_TO_TEST = "test_folder"

#OUTPUT_BAD_FOLDER = "_validation/output_bad"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Working on: {device}")

# --- 2. MODEL PREPARATION ---
def get_model(weights_path):
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

preprocess = R2Plus1D_18_Weights.DEFAULT.transforms()

# --- 3. SINGLE PREDICTION FUNCTION ---
def predict_video(video_path, model):
    try:
        video, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")
        
        target = 16
        current = video.shape[0]
        if current > target:
            video = video[:target]
        else:
            diff = target - current
            padding = video[-1:].repeat(diff, 1, 1, 1)
            video = torch.cat((video, padding), dim=0)

        video = video.to(torch.float32) / 255.0
        video = preprocess(video)
        
        input_tensor = video.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(probs, 0)
            
        return pred.item(), conf.item()
    except Exception as e:
        return None, str(e)

# --- 4. EXECUTION ON FOLDER ---
def main():


    model = get_model(MODEL_PATH)
    labels = {
        0: "BAD CUT ❌", 
        1: "GOOD CUT ✅", 
        2: "NO CUT ⚪"
    }
    count_good = 0
    count_bad = 0
    
    print(f"\nAnalyzing videos in folder: {VIDEOS_TO_TEST}")
    print("-" * 60)
    
    for filename in os.listdir(VIDEOS_TO_TEST):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            path = os.path.join(VIDEOS_TO_TEST, filename)
            result, score = predict_video(path, model)
            
            if result is not None:
                if result == 1:
                    count_good += 1
                else:
                    count_bad += 1
                print(f"Video: {filename[:30]:<30} | Result: {labels[result]:<12} | Confidence: {score:.2%}")
            else:
                print(f"Video: {filename[:30]:<30} | Error: {score}")

    print(f"\nFinal results:")
    print(f"Good cuts: {count_good}")
    print(f"Bad cuts: {count_bad}")

if __name__ == "__main__":
    main()