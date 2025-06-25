#!/usr/bin/env python3
"""
Train PDLPR model for character recognition
"""

import sys
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

sys.path.append('src')
from recognition.pdlpr_model import create_pdlpr_model, PDLPRTrainer
from models.baseline import SyntheticDataset
from utils.metrics import MetricsTracker
from utils.config import TRAIN_CONFIG, ALL_CHARS

def train_pdlpr(epochs=20, batch_size=16, save_dir='models'):
    """Train PDLPR model"""
    print("Training PDLPR model...")
    
    # Setup device
    device = torch.device('cpu')  # For VM compatibility
    print(f"Using device: {device}")
    
    # Create model
    model = create_pdlpr_model(
        num_classes=len(ALL_CHARS),
        sequence_length=7
    )
    trainer = PDLPRTrainer(model, device)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SyntheticDataset(num_samples=2000, img_size=(64, 128))
    val_dataset = SyntheticDataset(num_samples=400, img_size=(64, 128))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_metrics = MetricsTracker()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            loss = trainer.train_step(images, labels)
            
            # For metrics, we need predictions
            with torch.no_grad():
                outputs = model(images.to(device))
                train_metrics.update(outputs, labels, loss)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss={loss:.4f}")
        
        # Validation phase
        model.eval()
        val_metrics = MetricsTracker()
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_metrics.update(outputs, labels)
        
        # Print metrics
        train_results = train_metrics.compute()
        val_results = val_metrics.compute()
        
        print(f"Train - {train_metrics.get_summary()}")
        print(f"Val   - {val_metrics.get_summary()}")
        
        # Save best model
        val_acc = val_results.get('sequence_accuracy', 0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(save_dir) / 'pdlpr_best.pth'
            trainer.save_model(save_path)
            print(f"New best model saved: {val_acc:.3f}")
        
        print("-" * 50)
    
    # Save final model
    final_path = Path(save_dir) / 'pdlpr_final.pth'
    trainer.save_model(final_path)
    
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Final model saved to: {final_path}")

def main():
    parser = argparse.ArgumentParser(description='Train PDLPR model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='models', help='Save directory')
    args = parser.parse_args()
    
    # Ensure save directory exists
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    train_pdlpr(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()