import os
import torch
import torch.nn as nn
import torch.optim as optim


from tqdm import tqdm
from torch.utils.data import DataLoader

print(">>> Training class ...")

class Trainer:
    
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 num_classes, 
                 batch_size, 
                 num_epochs, 
                 lr=1e-1, 
                 device="cuda"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        
    def early_stopping(self, losses, accuracies, loss_thres, accuracy_thresh, patience):
        needed_losses = losses[-patience:]
        needed_accuracies = accuracies[-patience:]
        pass
        
        
        
    def save_model(self, path_root):
        """Save only the model weights (state_dict)."""
        weight_path = os.path.join(path_root, 'model_weights.pth')
        torch.save(self.model.state_dict(), weight_path)
        print(f"[INFO] Model weights saved at: {weight_path}")

        """Save the full model including architecture."""
        model_path = os.path.join(path_root, 'full_model.pth')
        torch.save(self.model, model_path)
        print(f"[INFO] Full model saved at: {model_path}")

    def train_model(self,):
        

        # Loss + Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        losses, accuracies = [], []

        # Move model to device
        self.model = self.model.to(self.device)

        for epoch in tqdm(range(self.num_epochs), desc="Iteration"):
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            
            # --------------------
            # Training Phase
            # --------------------
            self.model.train()
            train_loss, correct, total = 0, 0, 0

            for videos, labels in tqdm(self.train_loader, desc="Training"):
                videos, labels = videos.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(videos)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Stats
                train_loss += loss.item() * videos.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            train_loss /= total

            # --------------------
            # Validation Phase
            # --------------------
            self.model.eval()
            val_loss, correct, total = 0, 0, 0

            with torch.no_grad():
                for videos, labels in tqdm(self.val_loader, desc="Validation"):
                    videos, labels = videos.to(self.device), labels.to(self.device)

                    outputs = self.model(videos)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * videos.size(0)
                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            val_loss /= total
            losses.append(val_loss)
            accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        return self.model
    
    
