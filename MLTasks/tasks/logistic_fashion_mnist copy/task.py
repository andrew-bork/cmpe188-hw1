"""
Softmax Regression (Multiclass) using PyTorch
Implements multiclass classification with CrossEntropyLoss
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.utils import shuffle
from torchvision import datasets, transforms

from tqdm import tqdm

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "softmax_regression_multiclass",
        "task_type": "classification",
        "num_classes": 3,
        "input_dim": 2,
        "description": "Multiclass softmax regression using torch.nn.Module + CrossEntropyLoss"
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataloaders(batch_size=32, device=None):
    """Load data from the csv."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_ds = datasets.FashionMNIST("data", train=True,
                                     download=True, transform=tfm)
    val_ds   = datasets.FashionMNIST("data", train=False,
                                     download=True, transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)
    return train_dl, val_dl



class SoftmaxRegressionModel(nn.Module):
    """Simple softmax regression model (multiclass logistic regression)."""
    
    def __init__(self, input_dim, num_classes):
        super(SoftmaxRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Return logits (raw scores) - CrossEntropyLoss applies softmax internally
        return self.linear(self.flatten(x))


def build_model(input_dim, num_classes, device):
    """Build and return the model."""
    model = SoftmaxRegressionModel(input_dim, num_classes)
    model = model.to(device)
    return model


def train(model, train_loader, criterion, optimizer, device, epochs=100, verbose=True):
    """Train the model."""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total += y_batch.size(0)
            correct += (outputs.argmax(1) == y_batch).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model and return metrics.
    
    Returns dict with:
    - loss: average cross-entropy loss
    - accuracy: classification accuracy
    - f1_macro: macro F1 score
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            # Move data to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1_macro = f1_score(all_targets, all_predictions, average='macro')
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro
    }
    
    


def predict(model, X, device):
    """Predict class labels for samples in X."""
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.cpu().numpy()


def save_artifacts(model, metrics, X_train, y_train, X_val, y_val, 
                   output_dir="output", filename_prefix="logistic_fashion"):
    """Save model artifacts and visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{filename_prefix}_model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{filename_prefix}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Artifacts saved to {output_dir}")

def main():
    """Main function to run the softmax regression task."""
    print("=" * 60)
    print("Softmax Regression (Multiclass) - PyTorch Implementation")
    print("=" * 60)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Input dimension: {metadata['input_dim']}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders(
        batch_size=32,
        device=device
    )
    
    # print(f"Training samples: {len(X_train)}")
    # print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(
        input_dim=28*28,
        num_classes=10,
        device=device
    )
    print(f"Model architecture: {model}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    
    # Train model
    print("\nTraining model...")
    train(model, train_loader, criterion, optimizer, device, 
                  epochs=100, verbose=True)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, criterion, device)
    print("Train Metrics:")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {train_metrics['f1_macro']:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, criterion, device)
    print("Validation Metrics:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {val_metrics['f1_macro']:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    all_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "metadata": metadata
    }
    # save_artifacts(model, all_metrics, X_train, y_train, X_val, y_val)
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nTrain Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Val Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"Train F1 Macro:  {train_metrics['f1_macro']:.4f}")
    print(f"Val F1 Macro:    {val_metrics['f1_macro']:.4f}")
    
    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)
    
    quality_passed = True
    
    # Check 1: Train accuracy > 0.80
    check1 = train_metrics['accuracy'] > 0.80
    status1 = "✓" if check1 else "✗"
    print(f"{status1} Train Accuracy > 0.80: {train_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and check1
    
    # Check 2: Validation accuracy > 0.80
    check2 = val_metrics['accuracy'] > 0.80
    status2 = "✓" if check2 else "✗"
    print(f"{status2} Val Accuracy > 0.80: {val_metrics['accuracy']:.4f}")
    quality_passed = quality_passed and check2
    
    # Check 3: Validation F1 Macro > 0.75 (as specified)
    check3 = val_metrics['f1_macro'] > 0.75
    status3 = "✓" if check3 else "✗"
    print(f"{status3} Val F1 Macro > 0.75: {val_metrics['f1_macro']:.4f}")
    quality_passed = quality_passed and check3
    
    # Check 4: Loss decreased during training
    check4 = train_metrics['loss'] < 1.0
    status4 = "✓" if check4 else "✗"
    print(f"{status4} Final Train Loss < 1.0: {train_metrics['loss']:.4f}")
    quality_passed = quality_passed and check4
    
    # Check 5: Small gap between train and val performance
    accuracy_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
    check5 = accuracy_gap < 0.15
    status5 = "✓" if check5 else "✗"
    print(f"{status5} Accuracy gap < 0.15: {accuracy_gap:.4f}")
    quality_passed = quality_passed and check5
    
    # Final summary
    print("\n" + "=" * 60)
    if quality_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)
    
    # Exit with appropriate code
    return 0 if quality_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)