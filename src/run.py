from src import Trainer, VideoClassificationModel, FallVideoDataset

class Runner:
    def __init__(self):
        pass
    
    def run(self,):
        print("Provide the following details: ")
        
        pass
    


import os
import torch
import logging
from logging import Logger
from torchvision import transforms  
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src import transform
from src import Trainer, VideoClassificationModel, FallVideoDataset
from src.configs import ROOT_DIR

formatted_Logger = logging.getLogger(__name__)
formatted_Logger.setLevel(logging.INFO)

# Optional: Add a handler if not already configured
if not formatted_Logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    formatted_Logger.addHandler(handler)


class Runner:
    def __init__(self, 
                 data_dir="",
                 max_frames=200,
                 batch_size=4,
                 lr=1e-4,
                 epochs=10,
                 num_classes=2,
                 train_test_split=[0.85, 0.15],
                 device="cuda"
                 ):
        self.data_dir = data_dir if data_dir else ROOT_DIR
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.train_test_split = train_test_split
        self.device = device
        
        formatted_Logger.info("Dataset Stats:\n%s", FallVideoDataset.detailed_sample_stats())

        # Define transforms (can be extended with augmentations)
        self.transform = transform

    def run(self):
        print("🚀 Initializing pipeline...")

        # Prepare datasets
        dataset = FallVideoDataset(
            root_dir=f"{self.data_dir}",
            transform=self.transform,
            max_frames=self.max_frames
        )
        
        # Split: e.g., 85% train, 15% val
        train_split, val_split = self.train_test_split
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        formatted_Logger.info(f"Train videos: {len(train_dataset)}")
        formatted_Logger.info(f"Val videos:   {len(val_dataset)}")

        # Wrap datasets in loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # Initialize model
        model = VideoClassificationModel(hidden_size=256, num_classes=self.num_classes)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes = self.num_classes, 
            batch_size = self.batch_size, 
            lr=self.lr,
            num_epochs=self.epochs,
            device=self.device
        )

        # Train
        formatted_Logger.info("🏋️ Training started...")
        trainer.train_model()
        formatted_Logger.info("✅ Training completed!")
        
        # Save
        formatted_Logger.info("🏋️ Saving model and weights...")
        weights_Dir = os.path.join(ROOT_DIR, 'weights')
        os.makedirs(weights_Dir, exist_ok=True)
        trainer.save_model(weights_Dir)
        formatted_Logger.info("✅ Training completed!")
        
        