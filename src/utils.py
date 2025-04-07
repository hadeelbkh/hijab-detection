# utils.py
import torch
from src.config import Config

def save_checkpoint(model, optimizer, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, Config.MODEL_PATH)
    print(f"Checkpoint saved at epoch {epoch}")

def validate(model, val_loader, criterion):
    """
    Validates the model on the validation dataset.
    Args:
        model: The PyTorch model to validate.
        val_loader: DataLoader for the validation dataset.
        criterion: Loss function (e.g., BCEWithLogitsLoss).
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in val_loader:
            images, labels = images.to(Config.DEVICE), labels.float().unsqueeze(1).to(Config.DEVICE)
            
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
    accuracy = correct / total
    
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")