# test.py
import torch
from torch.nn import BCEWithLogitsLoss
from src.config import Config
from src.model import build_model
from src.dataset import get_dataloaders
from src.utils import validate
import os

def load_checkpoint(model, path):
    """Load model state from a checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}: {path}")
    return model

def test():
    try:
        # Validate config
        from src.train import validate_config
        validate_config()

        # Initialize model and load best checkpoint
        model = build_model()
        best_model_path = f"{Config.MODEL_PATH}/best_model.pt"
        model = load_checkpoint(model, best_model_path)
        model.eval()

        # Load test data
        _, _, test_loader = get_dataloaders()
        if test_loader is None:
            raise ValueError("Test loader not provided by get_dataloaders()")

        # Evaluate
        criterion = BCEWithLogitsLoss()
        test_loss, test_acc = validate(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    except Exception as e:
        print(f"Testing failed: {e}")
        raise

if __name__ == "__main__":
    test()