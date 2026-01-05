import os
import cv2
import glob
import torch
import torchvision

import numpy as np

from PIL import Image
from pathlib import Path

from src.configs import transform, MAX_FRAMES, ROOT_DIR, WEIGHTS_PATH

from src.model.model import VideoClassificationModel, SimpleAttention


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg", ".mpg"}

def is_video_path(path):
    p = Path(path)
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS

def load_model(model_path, device='cuda'):
    model = VideoClassificationModel(hidden_size=256, num_classes=2)

    with torch.serialization.safe_globals([VideoClassificationModel, torch.nn.modules.container.Sequential, torch.nn.modules.conv.Conv2d, torch.nn.modules.batchnorm.BatchNorm2d,
                                        torch.nn.modules.activation.ReLU, torch.nn.modules.pooling.MaxPool2d, torchvision.models.resnet.BasicBlock,
                                        torch.nn.modules.pooling.AdaptiveAvgPool2d, torch.nn.modules.rnn.LSTM, SimpleAttention, torch.nn.modules.linear.Linear,
                                        ]):
        if device == 'cpu':
            model = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
        else:
            model = torch.load(model_path, weights_only=True)

    model.to(device)
    model.eval()
    return model

def _transform_frame(frames):
    transformed_frames = []
    for frame in frames:
        transformed_frame = transform(frame)
        transformed_frames.append(transformed_frame)

    return torch.stack(transformed_frames)
    


def frames_from_video(path, num_frames=16, size=(224, 224)):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Invalid video or no frames found")

    # Pick evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        frames.append(pil_image)

    cap.release()

    # If video is shorter, pad with last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames   # shape: (num_frames, H, W, 3)


def inference(path):
    if os.path.isdir(path):
        frames_path = glob.glob(f'{path}/*.*g')[-300:]
        frames = []
        for f in frames_path:
            img = Image.open(f)
            frames.append(img)
        
        if not frames:
            raise ValueError(f"No Frames Found at {path}")

    else:
        if not is_video_path(path):
            raise ValueError(f"Please add valid video path. {path}")

        else:
            frames = frames_from_video(path=path, num_frames=MAX_FRAMES, size=(224, 224))
    
    frames = _transform_frame(frames=frames)  # (T, H, W, C)
    frames = frames.unsqueeze(0)  # (1, T, C, H, W)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames = frames.to(device)
    model = load_model(model_path=WEIGHTS_PATH, device=device)
    
    output = model(frames)
    
    output = output.detach().cpu().numpy()
    mapping_ = {0: 'No Fall', 1: 'Fall'}
    predicted_class = np.argmax(output, axis=1)[0]
    
    return mapping_[predicted_class]
    

    
            