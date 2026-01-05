import os
import torch
import random

from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset

from src.configs import ROOT_DIR, transform


class FallVideoDataset(Dataset):
    class_samples = []
    def __init__(self, root_dir, transform=None, max_frames=200):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Transform to be applied on a frame.
            max_frames (int): Number of frames per video to use (for padding/trimming).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []        
        self.max_frames = max_frames
        


        # # Label: 1 for falling, 0 for not falling
        for subject in os.listdir(root_dir):
            subject_folder = os.path.join(root_dir, subject)
            
            for each_action in os.listdir(subject_folder):
                action_folder = os.path.join(subject_folder, each_action)
                label = 1 if 'fall' in each_action.lower() else 0
                
                video_folder = action_folder # as each action folder is made up of frames from a video
                
                if os.path.isdir(video_folder):
                    self.samples.append((video_folder, label))
                    FallVideoDataset.class_samples.append((video_folder, label))
                
                     

    def __len__(self):
        return len(self.samples)
    
    @classmethod
    def detailed_sample_stats(cls):
        all_files = [os.listdir(video_folder) for video_folder, _ in cls.class_samples]
        all_files = [item for sublist in all_files for item in sublist]
        frames_only = [file for file in all_files if file.endswith('g')]
        
        temp = {}
        
        for video_folder, _ in cls.class_samples:
            if not str(os.path.basename(video_folder)) in temp:
                temp[str(os.path.basename(video_folder))] = {}
            temp[str(os.path.basename(video_folder))]['Original'] = len([frame for frame in os.listdir(video_folder) if not '_aug_' in frame])
            temp[str(os.path.basename(video_folder))]['Total'] = len(os.listdir(video_folder))
        
        return {
            'videos': len(cls.class_samples), 
            'frames': len(frames_only),
            'frames_per_video': temp
        }

    def load_frames(self, video_folder):
        frames = sorted(os.listdir(video_folder))  
        frame_groups = {}

        # group by prefix before "_aug"
        for f in frames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                base = f.split("_aug")[0]  # e.g. "frame_001"
                frame_groups.setdefault(base, []).append(f)

        frame_tensors = []
        selected_keys = sorted(frame_groups.keys())[:self.max_frames]

        for key in selected_keys:
            variants = frame_groups[key]
            frame_choice = random.choice(variants)  # pick one randomly
            frame_path = os.path.join(video_folder, frame_choice)

            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frame_tensors.append(image)

        # padding
        while len(frame_tensors) < self.max_frames:
            frame_tensors.append(torch.zeros_like(frame_tensors[0]))

        return torch.stack(frame_tensors)  # [T, C, H, W]

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video_tensor = self.load_frames(video_path)
        return video_tensor, torch.tensor(label, dtype=torch.long)
    


dataset = FallVideoDataset(root_dir=ROOT_DIR, transform=transform)

