from torch.utils.data import DataLoader

from .retriever_class  import dataset

from torch.utils.data import random_split

from logging import Logger

# Split: e.g., 85% train, 15% val
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

Logger.info(f"Train videos: {len(train_dataset)}")
Logger.info(f"Val videos:   {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
