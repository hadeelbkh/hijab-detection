# test.py
import torch
from torch.nn import BCEWithLogitsLoss
from src.config import Config
from src.model import build_model
from src.dataset import get_dataloaders
from src.utils import load_checkpoint
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
import os

def evaluate():
    """Evaluate the trained model on the test set."""
    try:
        # Validate config
        from src.train import validate_config
        validate_config()

        # Initialize model and load best checkpoint
        model = build_model()
        # Convert MODEL_PATH to str if Path object
        model_path = str(Config.MODEL_PATH) if isinstance(Config.MODEL_PATH, Path) else Config.MODEL_PATH
        best_model_path = os.path.join(model_path, "best_model.pt")
        model = load_checkpoint(model, best_model_path)
        model.eval()

        # Load test data
        try:
            _, _, test_loader = get_dataloaders()
            if test_loader is None:
                raise ValueError("Test loader is not provided by get_dataloaders()")
        except Exception as e:
            raise RuntimeError(f"Failed to load test data: {e}")

        # Evaluation
        criterion = BCEWithLogitsLoss()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():  
            for images, labels in test_loader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE).float().view(-1, 1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5 # Convert logits to binary predictions
                correct += (preds == labels).sum().item() # Count correct predictions
                total += labels.size(0) # Total number of samples
                all_preds.extend(preds.cpu().numpy()) # Store predictions
                all_labels.extend(labels.cpu().numpy()) # Store true labels

        # Compute metrics
        avg_test_loss = test_loss / len(test_loader) 
        accuracy = correct / total if total > 0 else 0.0 
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        # Report results
        print(f"Test Results:")
        print(f"  Loss: {avg_test_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        return avg_test_loss, accuracy, precision, recall, f1

    except Exception as e:
        print(f"Testing failed: {e}")
        raise

if __name__ == "__main__":
    evaluate()