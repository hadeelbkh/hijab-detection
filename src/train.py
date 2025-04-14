# train.py
import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.amp import GradScaler, autocast
from src.config import Config
from src.model import build_model
from src.dataset import get_dataloaders
from src.utils import save_checkpoint, validate
import os

def validate_config():
    """Validate Config attributes."""
    required_attrs = {
        "LEARNING_RATE": float,
        "WEIGHT_DECAY": float,
        "EPOCHS": int,
        "DEVICE": str,
        "MODEL_PATH": str,
    }
    for attr, expected_type in required_attrs.items():
        if not hasattr(Config, attr):
            raise AttributeError(f"Config missing required attribute: {attr}")
        value = getattr(Config, attr)
        if not isinstance(value, expected_type):
            raise TypeError(f"Config.{attr} must be {expected_type}, got {type(value)}")
    if not os.path.isdir(Config.MODEL_PATH):
        os.makedirs(Config.MODEL_PATH, exist_ok=True)
        print(f"Created directory: {Config.MODEL_PATH}")
    # I'm using AMP so I'll check for CUDA
    if "cuda" in Config.DEVICE.type and not torch.cuda.is_available():
        raise RuntimeError("CUDA device specified but not available")

def train():
    try:
        # Validate config
        validate_config()

        # Initialize model, data, and optimizer
        model = build_model()
        try:
            train_loader, val_loader, _ = get_dataloaders()
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

        optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        criterion = BCEWithLogitsLoss()
        scaler = GradScaler('cuda')
        
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0.0
            try:
                for images, labels in train_loader:
                    images = images.to(Config.DEVICE)
                    labels = labels.to(Config.DEVICE).float().view(-1, 1)

                    # Mixed precision training
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    train_loss += loss.item()

                    # Backpropagation with mixed precision
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)  # Update weights
                    scaler.update()  # Update the scale for next iteration
            except Exception as e:
                raise RuntimeError(f"Training failed at epoch {epoch}: {e}")

            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{Config.EPOCHS}, Train Loss: {avg_train_loss:.4f}")

            # Validation
            val_loss, val_acc = validate(model, val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, is_best=True)
            else:
                patience_counter += 1
                save_checkpoint(model, optimizer, epoch, is_best=False)

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    except Exception as e:
        print(f"Training terminated due to error: {e}")
        raise

if __name__ == "__main__":
    train()