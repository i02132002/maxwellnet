# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

import torch
import numpy as np
import os
import random
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

    # Columns __getitem__ actually reads; the dataset also carries several
    # unused (Ny, Nx, 3) float64 arrays (e_inc, p_inc, e_field) that are
    # dropped via select_columns since converting them from Arrow per-row is
    # the dominant cost of loading a sample otherwise.
    _COLUMNS = ['sample_id', 'optical_constant_real', 'optical_constant_imag', 'wavelength_nm',
                'x0_A', 'y0_A', 'z0_A', 'grid_nx', 'grid_ny', 'pol', 'theta']

    def __init__(self, config=None, split=None, hf_dataset=None):
        if hf_dataset is not None:
            self.dataset = hf_dataset
        else:
            self.dataset = load_dataset(
                "als-rixs/latent-image-training", config, split=split, revision="fixed-dimension",
            ).select_columns(self._COLUMNS)

    @classmethod
    def load_train_valid(cls, config, mode, valid_fraction=0.1, seed=0):
        """Load the HF dataset filtered to the polarization matching `mode`
        ('te' -> 's', 'tm' -> 'p'), then split into train/valid ShapeDatasets,
        grouped by `sample_id` so rows sharing a sample_id (e.g. its s/p-pol
        pair) always land in the same split — otherwise the same underlying
        structure could leak across train and valid."""
        pol = 's' if mode == 'te' else 'p'
        full = load_dataset(
            "als-rixs/latent-image-training", config, split="train", revision="fixed-dimension",
        ).select_columns(cls._COLUMNS).filter(lambda r: r["pol"] == pol)

        sample_ids = full["sample_id"]
        unique_ids = sorted(set(sample_ids))
        random.Random(seed).shuffle(unique_ids)
        n_valid = round(len(unique_ids) * valid_fraction)
        valid_ids = set(unique_ids[:n_valid])

        is_valid = np.fromiter((sid in valid_ids for sid in sample_ids), dtype=bool)
        train_indices = np.flatnonzero(~is_valid).tolist()
        valid_indices = np.flatnonzero(is_valid).tolist()
        return cls(hf_dataset=full.select(train_indices)), cls(hf_dataset=full.select(valid_indices))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r = self.dataset[idx]

        optical_constant = (np.asarray(r["optical_constant_real"])
                            + 1j * np.asarray(r["optical_constant_imag"]))

        delta_x_a = 1.0
        delta_z_a = 1.0

        return {
            "sample_id": r["sample_id"],
            "optical_constant": torch.from_numpy(optical_constant),
            "wavelength_nm": r["wavelength_nm"],
            "x0_A": r["x0_A"],
            "y0_A": r["y0_A"],
            "z0_A": r["z0_A"],
            "grid_nx": r["grid_nx"],
            "grid_ny": r["grid_ny"],
            "pol": r["pol"],
            "theta": r["theta"],
            "delta_x_a": delta_x_a,
            "delta_z_a": delta_z_a,
        }
