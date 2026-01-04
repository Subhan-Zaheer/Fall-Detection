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



# ROOT_DIR = "D:\Fall Dataset\Dataset CAUCAFall_1\Dataset CAUCAFall\CAUCAFall"

ROOT_DIR = "/home/ubuntu/addtitional_drive/temp_Training/Dataset_CAUCAFall/CAUCAFall"