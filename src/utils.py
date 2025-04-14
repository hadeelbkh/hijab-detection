# utils.py
import os
import torch
from src.config import Config

def save_checkpoint(model, optimizer, epoch, is_best=False):
    """Save model and optimizer state to a checkpoint file."""
    try:
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        path = os.path.join(Config.MODEL_PATH, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch}: {path}")
        if is_best:
            best_path = os.path.join(Config.MODEL_PATH, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    except Exception as e:
        print(f"Failed to save checkpoint at epoch {epoch}: {e}")

def load_checkpoint(model, path):
    """Load model state from a checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}: {path}")
    return model

def validate(model, val_loader, criterion):
    """Validates the model on the validation dataset."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for images, labels in val_loader:
            images, labels = images.to(Config.DEVICE), labels.float().view(-1, 1).to(Config.DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # Compute average loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_val_loss, accuracy