import itertools
import pandas as pd
import os
from src.config import Config
from src.train import train
from src.utils import validate
from src.model import build_model
from src.dataset import get_dataloaders
from src.utils import load_checkpoint
from torch.nn import BCEWithLogitsLoss
import csv
import torch

def run_grid_search():
    """Perform Grid Search over hyperparameters for DenseNet-121."""
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [1e-5, 1e-4, 1e-3],
        'batch_size': [4, 8, 16],
        'dropout': [0.3, 0.5, 0.7],
        'weight_decay': [1e-6, 1e-5, 1e-4]
    }

    # Create results directory
    base_model_path = Config.MODEL_PATH
    results_dir = base_model_path / "grid_search"
    os.makedirs(results_dir, exist_ok=True)
    results_csv = results_dir / "grid_search_results.csv"

    # Define CSV headers
    csv_headers = [
        "learning_rate", "batch_size", "dropout", "weight_decay",
        "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"
    ]

    # Initialize results list
    results = []

    # Get validation loader for evaluation
    _, val_loader, _ = get_dataloaders()

    # Iterate over all combinations
    for lr, bs, dropout, wd in itertools.product(
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['dropout'],
        param_grid['weight_decay']
    ):
        print(f"\nTesting: lr={lr}, batch_size={bs}, dropout={dropout}, weight_decay={wd}")

        # Update Config
        Config.LEARNING_RATE = lr
        Config.BATCH_SIZE = bs
        Config.DROPOUT = dropout
        Config.WEIGHT_DECAY = wd
        Config.MODEL_PATH = results_dir / f"run_lr_{lr}_bs_{bs}_do_{dropout}_wd_{wd}"

        # Train model
        try:
            run_id = f"lr_{lr}_bs_{bs}_do_{dropout}_wd_{wd}"
            train(run_id=run_id)
            # Evaluate on validation set
            model = build_model()
            checkpoint_path = Config.MODEL_PATH / "best_model.pt"
            model = load_checkpoint(model, checkpoint_path)
            model.eval()
            criterion = BCEWithLogitsLoss()
            val_metrics = validate(model, val_loader, criterion, dataset_name="Validation")
            val_loss, val_acc, val_precision, val_recall, val_f1 = val_metrics

            # Save results
            results.append({
                "learning_rate": lr,
                "batch_size": bs,
                "dropout": dropout,
                "weight_decay": wd,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            })
            print(f"Results: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

            # Save intermediate results
            with open(results_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(results)
        except Exception as e:
            print(f"Failed for lr={lr}, bs={bs}, do={dropout}, wd={wd}: {e}")
            continue

    # Save final results
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(results)
    print(f"Grid Search complete. Results saved to {results_csv}")

    # Find best combination
    best_result = max(results, key=lambda x: x['val_f1'])  # Optimize for F1-score
    print("\nBest Hyperparameters:")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"Dropout: {best_result['dropout']}")
    print(f"Weight Decay: {best_result['weight_decay']}")
    print(f"Validation F1-Score: {best_result['val_f1']:.4f}")
    print(f"Validation Accuracy: {best_result['val_accuracy']:.4f}")

if __name__ == "__main__":
    run_grid_search()