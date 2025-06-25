"""
PDLPR (Parallel Detection and Language Parsing Recognition) model
Character recognition for license plates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class PDLPRModel(nn.Module):
    """PDLPR architecture for character recognition"""
    
    def __init__(self, num_classes=67, sequence_length=7, input_channels=3):
        super(PDLPRModel, self).__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Sequence modeling with LSTM
        self.lstm_input_size = 512 * 8  # Assuming width becomes 8 after pooling
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=256,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )
        
        # Character classification heads
        self.classifier = nn.Linear(256 * 2, num_classes)  # *2 for bidirectional
        
        # Attention mechanism (simplified)
        self.attention = nn.MultiheadAttention(256 * 2, num_heads=8, batch_first=True)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.backbone(x)  # (B, 512, H/8, W/8)
        
        # Reshape for sequence processing
        B, C, H, W = features.size()
        features = features.permute(0, 3, 1, 2)  # (B, W, C, H)
        features = features.reshape(B, W, C * H)  # (B, W, C*H)
        
        # LSTM sequence modeling
        lstm_out, (h_n, c_n) = self.lstm(features)  # (B, W, 512)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Character prediction for each position
        predictions = []
        for i in range(self.sequence_length):
            if i < attn_out.size(1):
                char_logits = self.classifier(attn_out[:, i, :])
            else:
                # Pad with last available features if sequence is shorter
                char_logits = self.classifier(attn_out[:, -1, :])
            predictions.append(char_logits)
        
        # Stack predictions
        output = torch.stack(predictions, dim=1)  # (B, seq_len, num_classes)
        
        return output

class PDLPRTrainer:
    """Training utilities for PDLPR model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def train_step(self, images, labels):
        """Single training step"""
        self.model.train()
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate loss for each character position
        total_loss = 0
        for i in range(labels.size(1)):
            loss = self.criterion(outputs[:, i, :], labels[:, i])
            total_loss += loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def predict(self, image):
        """Predict characters from image"""
        self.model.eval()
        
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            image = image.to(self.device)
            outputs = self.model(image)
            
            # Get predicted characters
            predicted_indices = torch.argmax(outputs, dim=2)
            
            return predicted_indices.cpu().numpy()
    
    def save_model(self, path):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

def create_pdlpr_model(num_classes=67, sequence_length=7):
    """Create PDLPR model instance"""
    model = PDLPRModel(
        num_classes=num_classes,
        sequence_length=sequence_length
    )
    return model

if __name__ == "__main__":
    # Test PDLPR model
    print("Testing PDLPR model...")
    
    model = create_pdlpr_model()
    trainer = PDLPRTrainer(model)
    
    # Test with random input
    test_input = torch.randn(2, 3, 64, 128)  # Batch of 2 images
    test_labels = torch.randint(0, 67, (2, 7))  # Random character labels
    
    # Forward pass
    outputs = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Training step
    loss = trainer.train_step(test_input, test_labels)
    print(f"Training loss: {loss:.4f}")
    
    # Prediction
    predictions = trainer.predict(test_input[0])
    print(f"Predictions: {predictions}")
    
    print("PDLPR model test completed successfully!")