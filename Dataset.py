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
    def load_train_valid(cls, config, mode, valid_fraction=0.1, seed=0, n_samples=None, sample_id=None, theta=None):
        """Load the HF dataset filtered to the polarization matching `mode`
        ('te' -> 's', 'tm' -> 'p'), then split into train/valid ShapeDatasets,
        grouped by `sample_id` so rows sharing a sample_id (e.g. its s/p-pol
        pair) always land in the same split — otherwise the same underlying
        structure could leak across train and valid.

        `n_samples`, if given, caps the training split to the first
        `n_samples` unique sample_ids (post-shuffle, pre-valid-split) --
        e.g. for overfitting sanity checks on a handful of samples.

        `sample_id`, if given (a single sample_id string, or a list of
        them), bypasses the shuffle/split entirely and routes every row
        matching that id (e.g. all of its incidence angles) into the train
        split, with an empty valid split -- for targeting one specific,
        known structure directly rather than whichever one happens to land
        first after shuffling. `n_samples`/`valid_fraction` are ignored
        when this is given. `theta`, if also given alongside `sample_id`,
        further restricts to just the row(s) at that incidence angle
        (degrees) rather than every angle for that sample_id."""
        pol = 's' if mode == 'te' else 'p'
        raw = load_dataset(
            "als-rixs/latent-image-training", config, split="train", revision="fixed-dimension",
        )
        # .filter(lambda r: ...) materializes every selected column (Python
        # dict per row) just to evaluate the predicate -- even restricted to
        # cls._COLUMNS, that still includes the large (Ny, Nx, 3) float64
        # optical_constant arrays, making this take minutes for a predicate
        # that only needs 'pol'. Pull that one lightweight column via fast
        # columnar access first to find matching row indices, then select
        # only those rows' worth of the actual needed columns.
        matching_indices = [i for i, p in enumerate(raw["pol"]) if p == pol]
        full = raw.select_columns(cls._COLUMNS).select(matching_indices)

        sample_ids = full["sample_id"]

        if sample_id is not None:
            wanted = {sample_id} if isinstance(sample_id, str) else set(sample_id)
            if theta is not None:
                thetas = full["theta"]
                train_indices = [i for i, (sid, th) in enumerate(zip(sample_ids, thetas))
                                 if sid in wanted and abs(th - theta) < 1e-6]
            else:
                train_indices = [i for i, sid in enumerate(sample_ids) if sid in wanted]
            return cls(hf_dataset=full.select(train_indices)), cls(hf_dataset=full.select([]))

        unique_ids = sorted(set(sample_ids))
        random.Random(seed).shuffle(unique_ids)
        n_valid = round(len(unique_ids) * valid_fraction)
        valid_ids = set(unique_ids[:n_valid])
        train_ids = unique_ids[n_valid:]
        if n_samples is not None:
            train_ids = train_ids[:n_samples]
        train_ids = set(train_ids)

        is_valid = np.fromiter((sid in valid_ids for sid in sample_ids), dtype=bool)
        is_train = np.fromiter((sid in train_ids for sid in sample_ids), dtype=bool)
        train_indices = np.flatnonzero(is_train).tolist()
        valid_indices = np.flatnonzero(is_valid).tolist()
        return cls(hf_dataset=full.select(train_indices)), cls(hf_dataset=full.select(valid_indices))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r = self.dataset[idx]

        optical_constant = (np.asarray(r["optical_constant_real"])
                            + 1j * np.asarray(r["optical_constant_imag"]))
        # Crop off the last row/column (e.g. 257x257 -> 256x256) so the grid
        # size is a clean power of two -- both ShapeNet (UNet pooling) and
        # helmholtz_checker (periodic-x/PML-z residual) consume this same
        # cropped array, so they see a consistent grid.
        optical_constant = optical_constant[:-1, :-1, :]

        #optical_constant = 1.0 + 10.0 * np.abs(optical_constant - 1.0)

        delta_x_a = 1.0
        delta_z_a = 1.0

        return {
            "sample_id": r["sample_id"],
            "optical_constant": torch.from_numpy(optical_constant),
            "wavelength_nm": r["wavelength_nm"],
            "x0_A": r["x0_A"],
            "y0_A": r["y0_A"],
            "z0_A": r["z0_A"],
            "grid_nx": r["grid_nx"] - 1,
            "grid_ny": r["grid_ny"] - 1,
            "pol": r["pol"],
            "theta": r["theta"],
            "delta_x_a": delta_x_a,
            "delta_z_a": delta_z_a,
        }
