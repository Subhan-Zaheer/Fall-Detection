import os
import glob
import torch

from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg", ".mpg"}

def is_video_path(path):
    p = Path(path)
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def frames_from_video(path):



def inference(path):
    if os.path.isdir(path):
        frames = glob.glob(f'{path}/*.*g'}[-300:]
        
        if not frames:
            raise ValueError(f"No Frames Found at {path}")

    else:
        if not is_video_path(path):
            raise ValueError(f"Please add valid video path. {path}")

        else:
            

    
            