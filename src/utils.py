# utils.py
import os
import torch
from src.config import Config
from sklearn.metrics import precision_recall_fscore_support
from torchvision.models.densenet import DenseNet
import torch.serialization

# Allowlist DenseNet class for safe loading
torch.serialization.add_safe_globals([DenseNet])

def save_checkpoint(model, optimizer, epoch, is_best=False, run_id=None):
    """Save model and optimizer state to a checkpoint file."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    checkpoint_path = os.path.join(Config.MODEL_PATH, "last_model.pt")
    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        if is_best:
            best_checkpoint_path = os.path.join(Config.MODEL_PATH, "best_model.pt")
            torch.save(checkpoint, best_checkpoint_path)  # Save same checkpoint dict
            print(f"Saved best model: {best_checkpoint_path}")
    except Exception as e:
        print(f"Failed to save checkpoint at epoch {epoch}: {e}")

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)  # Handle case where checkpoint is state_dict
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}: {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        raise


def validate(model, data_loader, criterion, dataset_name="Validation"):
    """Evaluates the model on a dataset, computing loss, accuracy, precision, recall, F1-score, and confusion matrix.

    Args:
        model: The PyTorch model to evaluate.
        data_loader: DataLoader for the dataset (train, val, test).
        criterion: Loss function (BCEWithLogitsLoss).
        dataset_name (str): Name for logging ('Train', 'Validation', 'Test').

    Returns:
        Tuple: (avg_loss, accuracy, precision, recall, f1, TP, TN, FP, FN).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient computation
        for images, labels in data_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE).float().view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', labels=[0, 1], zero_division=0)
    # cm = confusion_matrix(all_labels, all_preds)
    # TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    #print(f"{dataset_name} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
    #    f"Recall: {recall:.4f}, F1-Score: {f1:.4f}, TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    # return avg_loss, accuracy, precision, recall, f1, TP, TN, FP, FN
    return avg_loss, accuracy, precision, recall, f1