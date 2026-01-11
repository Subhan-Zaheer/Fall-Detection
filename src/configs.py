import os
import sys

import numpy as np
import pandas as pd


from PIL import Image
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



ROOT_DIR = "src"
WEIGHTS_PATH = "src\\weights\\full_model.pth"
MAX_FRAMES = 300
# ROOT_DIR = "/home/ubuntu/addtitional_drive/temp_Training/Dataset_CAUCAFall/CAUCAFall"

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])