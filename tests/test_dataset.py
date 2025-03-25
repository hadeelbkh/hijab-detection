# tests/test_dataset.py
'''Test HijabDataset class.'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import HijabDataset
from config import Config

def test_hijab_dataset():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = HijabDataset(root_dir=Config.DATA_DIR / "train", transform=transform)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate through a batch
    for images, labels in dataloader:
        print("Images shape:", images.shape)
        print("Labels:", labels)
        break

if __name__ == "__main__":
    test_hijab_dataset()