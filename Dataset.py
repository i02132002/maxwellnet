# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

import torch
import numpy as np
import os
from datasets import load_dataset


class LensDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode):
        self.dataset = np.load(os.path.join(
            data_path, mode + ".npz"))['sample']
        self.n = np.load(os.path.join(data_path, mode + ".npz"))['n']

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        sample = self.dataset[idx, :, :, :]
        return sample, self.n, idx


class ShapeDataset(torch.utils.data.Dataset):
    """Loads shapes from the "als-rixs/latent-image-training" dataset for
    training MaxwellNet against the Helmholtz residual loss (no ground-truth
    E-field is needed, so it is not part of this dataset)."""

    def __init__(self, config=None, split=None, hf_dataset=None):
        if hf_dataset is not None:
            self.dataset = hf_dataset
        else:
            self.dataset = load_dataset(
                "als-rixs/latent-image-training", config, split=split)

    @classmethod
    def load_train_valid(cls, config, mode, valid_fraction=0.1, seed=0):
        """Load the HF dataset filtered to the polarization matching `mode`
        ('te' -> 's', 'tm' -> 'p'), then split into train/valid ShapeDatasets."""
        pol = 's' if mode == 'te' else 'p'
        full = load_dataset(
            "als-rixs/latent-image-training", config, split="train"
        ).filter(lambda r: r["pol"] == pol)
        splits = full.train_test_split(test_size=valid_fraction, seed=seed)
        return cls(hf_dataset=splits["train"]), cls(hf_dataset=splits["test"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r = self.dataset[idx]

        optical_constant = (np.asarray(r["optical_constant_real"])
                            + 1j * np.asarray(r["optical_constant_imag"]))

        nm_per_pixel = r["wavelength_nm"] / r["pixels_per_wavelength"]
        delta_x_a = (r["grid_nx"] * nm_per_pixel / (r["grid_nx"] - 1)) * 10.0
        delta_z_a = (r["grid_ny"] * nm_per_pixel / (r["grid_ny"] - 1)) * 10.0

        return {
            "optical_constant": torch.from_numpy(optical_constant),
            "wavelength_nm": r["wavelength_nm"],
            "pixels_per_wavelength": r["pixels_per_wavelength"],
            "x0_nm": r["x0_nm"],
            "y0_nm": r["y0_nm"],
            "grid_nx": r["grid_nx"],
            "grid_ny": r["grid_ny"],
            "pol": r["pol"],
            "delta_x_a": delta_x_a,
            "delta_z_a": delta_z_a,
        }
