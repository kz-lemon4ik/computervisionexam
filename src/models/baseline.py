import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import time
from pathlib import Path
import sys

sys.path.append("src")
from data.data_loader import CCPDDataLoader, CCPDDataset


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
            nn.AdaptiveAvgPool2d((4, 8)),
        )

        # Classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, sequence_length * num_chars),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.view(-1, self.sequence_length, self.num_chars)
        return x


class Trainer:
    """Training utilities"""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            if isinstance(batch_data, dict):
                images = batch_data["image"]
                labels = batch_data["chars"]
            else:
                images, labels = batch_data

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Handle image tensor dimensions
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            if images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2)

            outputs = self.model(images)

            # Calculate loss for all character positions
            loss = 0
            for i in range(labels.size(1)):
                loss += self.criterion(outputs[:, i, :], labels[:, i])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        return total_loss / num_batches if num_batches > 0 else 0

    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        correct_sequences = 0
        total_sequences = 0

        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, dict):
                    images = batch_data["image"]
                    labels = batch_data["chars"]
                else:
                    images, labels = batch_data

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Handle image tensor dimensions
                if len(images.shape) == 3:
                    images = images.unsqueeze(0)
                if images.shape[1] != 3:
                    images = images.permute(0, 3, 1, 2)

                outputs = self.model(images)

                # Calculate sequence accuracy
                predicted = torch.argmax(outputs, dim=2)
                sequence_correct = torch.all(predicted == labels, dim=1)
                correct_sequences += sequence_correct.sum().item()
                total_sequences += labels.size(0)

        accuracy = (
            (correct_sequences / total_sequences) * 100 if total_sequences > 0 else 0
        )
        return accuracy

    def train(self, train_loader, val_loader, epochs=5):
        """Full training loop"""
        print("Starting training...")
        start_time = time.time()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)

            print(f"Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

        # Save model
        Path("models").mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), "models/baseline_model.pth")
        print("Model saved to models/baseline_model.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = BaselineCNN()
    trainer = Trainer(model, device)

    if args.train:
        data_path = "data/processed/ccpd_subset"
        if Path(data_path).exists():
            print("Loading CCPD dataset...")
            loader = CCPDDataLoader(data_path)
            annotations = loader.load_dataset(max_samples=2000)

            if annotations:
                train_data, val_data = loader.get_train_val_split(
                    annotations, val_ratio=0.2
                )

                train_dataset = CCPDDataset(train_data)
                val_dataset = CCPDDataset(val_data)

                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=args.batch_size, shuffle=False
                )

                print(f"Training samples: {len(train_data)}")
                print(f"Validation samples: {len(val_data)}")

                trainer.train(train_loader, val_loader, epochs=args.epochs)
            else:
                print("No annotations found, using synthetic data...")
                train_dataset = SyntheticDataset(num_samples=1000)
                val_dataset = SyntheticDataset(num_samples=200)

                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=args.batch_size, shuffle=False
                )

                trainer.train(train_loader, val_loader, epochs=args.epochs)
        else:
            print("CCPD data not found, using synthetic dataset...")
            train_dataset = SyntheticDataset(num_samples=1000)
            val_dataset = SyntheticDataset(num_samples=200)

            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False
            )

            trainer.train(train_loader, val_loader, epochs=args.epochs)

    if args.evaluate:
        model_path = "models/baseline_model.pth"
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            test_dataset = SyntheticDataset(num_samples=100)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
            accuracy = trainer.evaluate(test_loader)
            print(f"Test Accuracy: {accuracy:.2f}%")
        else:
            print("Model not found. Train first.")


if __name__ == "__main__":
    main()
