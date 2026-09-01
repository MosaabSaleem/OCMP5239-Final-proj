"""
data_cifar10.py
===============
CIFAR-10 data loaders with the standard augmentation recipe used for both the teacher
and the students. Keeping the data pipeline identical across every condition is essential
for a clean ablation: only the teacher signal should vary, never the data.
"""

import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# CIFAR-10 channel statistics
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def default_data_root():
    """
    Pick a sensible CIFAR-10 location for whatever platform we are on.

    * Kaggle  -> /kaggle/working/data   (this folder persists as notebook output).
    * Colab / local -> ./data.

    On Kaggle you must turn ON 'Internet' in the notebook settings so torchvision can
    download CIFAR-10 the first time. Once downloaded it is cached in /kaggle/working.
    """
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/data"
    return "./data"


def build_loaders(data_root=None, batch_size=128, num_workers=2, download=True):
    if data_root is None:
        data_root = default_data_root()
    """Return (train_loader, test_loader) for CIFAR-10."""
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=download, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=download, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    """Top-1 accuracy (%) of a model on a loader."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total
