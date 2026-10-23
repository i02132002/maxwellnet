"""MNIST as a drop-in stand-in for ShapeDataset, for sanity-checking the DDPM.

Each 28x28 digit is duplicated into both channels, so items look like
ShapeDataset's: {"optical_constant_image": (2, 28, 28) in [0, 1]}.
"""
import torch
from torchvision.datasets import MNIST


class MnistImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, n_samples=None):
        ds = MNIST(root, train=train, download=True)
        data = ds.data if n_samples is None else ds.data[:n_samples]
        self.data = data  # (N, 28, 28) uint8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].float().div(255)[None]  # (1, 28, 28)
        return {"optical_constant_image": img.repeat(2, 1, 1)}


def load_mnist(root, n_samples=None, valid_fraction=0.0):
    """Train = first n_samples of MNIST train. Valid = round(n_train * valid_fraction)
    images from MNIST test (so no overlap with train)."""
    train = MnistImageDataset(root, True, n_samples)
    n_valid = round(len(train) * valid_fraction)
    return train, MnistImageDataset(root, False, n_valid)
