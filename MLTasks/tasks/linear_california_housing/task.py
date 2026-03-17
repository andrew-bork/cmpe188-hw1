import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import pandas as pd

VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.03
REGULARIZATION_RATE = 0.0001

# Set seeds for reproducibility
def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "linear_california_housing",
        "task_type": "regression",
        "input_type": "continuous",
        "output_type": "continuous",
        "description": "Linear regression on California housing data using Adam"
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    

def get_device():
    """Get device (cuda/cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(validation_split=0.2, batch_size=32, device=None):
    """Load data from the csv."""
    if device is None:
        device = get_device()
    
    
    csv = pd.read_csv('data/california_houses.csv')
    
    # Normalize datapoints
    csv = (csv - csv.mean()) / csv.std()
    print(csv)
    
    # Shuffle
    csv = shuffle(csv)
    
    n_data = len(csv)
    n_validation = int(n_data * (validation_split))
    training, validation = csv.iloc[n_validation:], csv.iloc[0:n_validation]
    
    print(csv.columns)

    training_target, training_features = torch.as_tensor(training[csv.columns[0]].to_numpy().reshape((-1, 1))).to(device).to(torch.float32), torch.as_tensor(training[csv.columns[1:]].to_numpy()).to(device).to(torch.float32)
    validation_target, validation_features = torch.as_tensor(validation[csv.columns[0]].to_numpy().reshape((-1, 1))).to(device).to(torch.float32), torch.as_tensor(validation[csv.columns[1:]].to_numpy()).to(device).to(torch.float32)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(training_features, training_target)
    val_dataset = TensorDataset(validation_features, validation_target)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, training_features.cpu(), validation_features.cpu(), training_target.cpu(), validation_target.cpu()

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)
    
    @property
    def device(self):
        return next(self.parameters()).device

def build_model(input_dim, device=None):
    if device is None:
        device = get_device()
    
    model = LinearRegressionModel(input_dim).to(device)
    return model

def train(model, train_loader, val_loader, device=None, epochs=100, lr=0.1, weight_decay=0.01):
    if device is None:
        device = get_device()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Move to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate(model, data_loader, device=None):
    """Evaluate the model and return metrics."""
    if device is None:
        device = get_device()
    
    model.eval()
    criterion = nn.MSELoss()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'mse': mse,
        'r2': r2
    }
    
    return metrics

def predict(model, X, device=None):
    """Make predictions."""
    if device is None:
        device = get_device()
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        return predictions.cpu().numpy()

def save_artifacts(model, train_losses, val_losses, X_train, y_train, X_val, y_val, 
                   output_dir="output", filename_prefix="linear_california_housing"):
    """Save model artifacts and visualization."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, f"{filename_prefix}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, f"{filename_prefix}_history.json")
    history = {
        'train_losses': [float(l) for l in train_losses],
        'val_losses': [float(l) for l in val_losses]
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"{filename_prefix}_fit.png")
    plot_fit(model, X_train, y_train, X_val, y_val, train_losses, val_losses, viz_path)
    
    print(f"Artifacts saved to {output_dir}")

def plot_fit(model, X_train, y_train, X_val, y_val, train_losses, val_losses, save_path):
    """Create and save visualization of the fit."""
    # Sort by x for smooth curves
    train_sorted = np.argsort(X_train[:, 1])  # Sort by x (second column)
    val_sorted = np.argsort(X_val[:, 1])
    
    x_train_sorted = X_train[train_sorted, 1]
    y_train_sorted = y_train[train_sorted]
    x_val_sorted = X_val[val_sorted, 1]
    y_val_sorted = y_val[val_sorted]
    
    # Get predictions
    y_train_pred = predict(model, X_train)
    y_val_pred = predict(model, X_val)
    
    y_train_pred_sorted = y_train_pred[train_sorted]
    y_val_pred_sorted = y_val_pred[val_sorted]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and fit
    ax1.scatter(x_train_sorted, y_train_sorted, alpha=0.5, label='Train data', s=15)
    ax1.scatter(x_val_sorted, y_val_sorted, alpha=0.5, label='Val data', s=15)
    ax1.plot(x_train_sorted, y_train_pred_sorted, 'r-', linewidth=2, label='Train fit')
    ax1.plot(x_val_sorted, y_val_pred_sorted, 'g--', linewidth=2, label='Val fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Polynomial Regression Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    ax2.plot(train_losses, 'b-', linewidth=2, label='Train Loss')
    ax2.plot(val_losses, 'r--', linewidth=2, label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training History')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("California Housing (Ridge + Adam + GPU)")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, device=device
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Build model
    print("\nBuilding model...")
    input_dim = X_train.shape[1]
    model = build_model(input_dim, device=device)
    print(f"Model architecture: {model}")
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, 
        device=device,
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        weight_decay=REGULARIZATION_RATE  # L2 regularization
    )
    
    # Evaluate on both splits
    print("\nEvaluating model...")
    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)
    
    print("\nMetrics:")
    print(f"  Train - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, R2: {train_metrics['r2']:.4f}")
    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, R2: {val_metrics['r2']:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, train_losses, val_losses, X_train, y_train, X_val, y_val)
    
    # Quality checks
    print("\n" + "=" * 60)
    print("Quality Checks:")
    print("=" * 60)
    
    checks_passed = True
    
    # Check 1: Train R2 > 0.6
    if train_metrics['r2'] > 0.6:
        print(f"✓ Train R2 > 0.6: {train_metrics['r2']:.4f}")
    else:
        print(f"✗ Train R2 > 0.6: {train_metrics['r2']:.4f}")
        checks_passed = False
    print(f"This data is likely not linearly correlated.")
    
    # Check 2: Val R2 > 0.6
    if val_metrics['r2'] > 0.6:
        print(f"✓ Val R2 > 0.6: {val_metrics['r2']:.4f}")
    else:
        print(f"✗ Val R2 > 0.6: {val_metrics['r2']:.4f}")
        checks_passed = False
    print(f"This data is likely not linearly correlated.")
    
    # Check 3: Val MSE < 2.0
    if val_metrics['mse'] < 2.0:
        print(f"✓ Val MSE < 2.0: {val_metrics['mse']:.4f}")
    else:
        print(f"✗ Val MSE < 2.0: {val_metrics['mse']:.4f}")
        checks_passed = False
    
    # Check 4: R2 difference < 0.15 (avoid overfitting)
    r2_diff = abs(train_metrics['r2'] - val_metrics['r2'])
    if r2_diff < 0.15:
        print(f"✓ R2 difference < 0.15: {r2_diff:.4f}")
    else:
        print(f"✗ R2 difference < 0.15: {r2_diff:.4f}")
        checks_passed = False
    
    # Final summary
    print("=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)
    
    # Exit with appropriate code
    return 0 if checks_passed else 1

if __name__ == '__main__':
    exit(main())