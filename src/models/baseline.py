import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from pathlib import Path
import sys
sys.path.append('src')
from utils.metrics import MetricsTracker
from utils.config import TRAIN_CONFIG, MODEL_CONFIG

class SyntheticDataset(Dataset):
    """Simple synthetic dataset for baseline training"""
    
    def __init__(self, num_samples=1000, img_size=(64, 128)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_chars = 67  # Chinese + alphanumeric
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic plate image
        img = np.ones((*self.img_size, 3), dtype=np.uint8) * 255
        noise = np.random.randint(0, 50, (*self.img_size, 3))
        img = np.clip(img - noise, 200, 255).astype(np.uint8)
        
        # Random character labels (7 characters)
        char_indices = np.random.randint(0, self.num_chars, 7)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(char_indices).long()
        
        return img_tensor, labels

class BaselineCNN(nn.Module):
    """Simple CNN for license plate recognition"""
    
    def __init__(self, num_chars=67, sequence_length=7):
        super(BaselineCNN, self).__init__()
        self.num_chars = num_chars
        self.sequence_length = sequence_length
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((4, 8))
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, sequence_length * num_chars)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.view(-1, self.sequence_length, self.num_chars)
        return x

class Trainer:
    """Training utilities"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        metrics = MetricsTracker()
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            
            # Calculate loss for all character positions
            loss = 0
            for i in range(labels.size(1)):
                loss += self.criterion(outputs[:, i, :], labels[:, i])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metrics.update(outputs, labels, loss.item())
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: {metrics.get_summary()}')
        
        return metrics.compute()['avg_loss']
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        metrics = MetricsTracker()
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                metrics.update(outputs, labels)
        
        eval_metrics = metrics.compute()
        return eval_metrics['sequence_accuracy'] * 100
    
    def train(self, train_loader, val_loader, epochs=5):
        """Full training loop"""
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            
            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)
            
            print(f'Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        
        total_time = time.time() - start_time
        print(f'Training completed in {total_time:.2f} seconds')
        
        # Save model
        Path('models').mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), 'models/baseline_model.pth')
        print('Model saved to models/baseline_model.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f'Using device: {device}')
    
    model = BaselineCNN()
    trainer = Trainer(model, device)
    
    if args.train:
        print("Creating synthetic dataset...")
        train_dataset = SyntheticDataset(num_samples=1000)
        val_dataset = SyntheticDataset(num_samples=200)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        trainer.train(train_loader, val_loader, epochs=args.epochs)
    
    if args.evaluate:
        model_path = 'models/baseline_model.pth'
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            test_dataset = SyntheticDataset(num_samples=100)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
            accuracy = trainer.evaluate(test_loader)
            print(f'Test Accuracy: {accuracy:.2f}%')
        else:
            print('Model not found. Train first.')

if __name__ == "__main__":
    main()