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
    def load_train_valid(cls, config, mode, valid_fraction=0.1, seed=0, n_samples=None):
        """Load the HF dataset filtered to the polarization matching `mode`
        ('te' -> 's', 'tm' -> 'p'), then split into train/valid ShapeDatasets,
        grouped by `sample_id` so rows sharing a sample_id (e.g. its s/p-pol
        pair) always land in the same split — otherwise the same underlying
        structure could leak across train and valid.

        `n_samples`, if given, caps the training split to its first
        `n_samples` rows (in the dataset's natural order, after excluding
        the valid split) -- e.g. for overfitting sanity checks on a handful
        of samples. Note this counts *rows*, not unique sample_ids: a
        sample_id can have multiple rows at this pol (e.g. several incidence
        angles), so `n_samples=1` takes just the row that happens to come
        first for whichever sample_id is first, not necessarily "one full
        sample_id worth" of rows."""
        pol = 's' if mode == 'te' else 'p'
        full = load_dataset(
            "als-rixs/latent-image-training", config, split="train", revision="fixed-dimension",
        ).select_columns(cls._COLUMNS).filter(lambda r: r["pol"] == pol)

        sample_ids = full["sample_id"]
        unique_ids = sorted(set(sample_ids))
        random.Random(seed).shuffle(unique_ids)
        n_valid = round(len(unique_ids) * valid_fraction)
        valid_ids = set(unique_ids[:n_valid])
        train_ids = set(unique_ids[n_valid:])

        is_valid = np.fromiter((sid in valid_ids for sid in sample_ids), dtype=bool)
        is_train = np.fromiter((sid in train_ids for sid in sample_ids), dtype=bool)
        train_indices = np.flatnonzero(is_train).tolist()
        valid_indices = np.flatnonzero(is_valid).tolist()
        if n_samples is not None:
            train_indices = train_indices[:n_samples]
        return cls(hf_dataset=full.select(train_indices)), cls(hf_dataset=full.select(valid_indices))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r = self.dataset[idx]

        optical_constant = (np.asarray(r["optical_constant_real"])
                            + 1j * np.asarray(r["optical_constant_imag"]))
        # Crop off the last row/column (e.g. 257x257 -> 256x256) so the grid
        # size is a clean power of two for the UNet's pooling.
        optical_constant = optical_constant[:-1, :-1, :]

        # optical_constant's last dim is 3 components ordered [Ez, Ex, Ey];
        # 's' (te) only needs Ey (index 2). 'p' (tm) would need both Ez, Ex,
        # which MaxwellNet.forward's single scat_pot/ri_value pair doesn't
        # support -- only 's'/'te' is exercised below.
        pol_index = 2 if r["pol"] == 's' else 0
        n_map = optical_constant[..., pol_index]

        #n_map = 1.0 + 10.0 * np.abs(n_map - 1.0)

        # MaxwellNet's tensor convention has dim-2 ("Nx") as the axis its
        # x-difference operators act on and dim-1 ("Nz") as the one its
        # incident wave propagates along (see the gradient_h_x/gradient_h_z
        # kernel shapes in MaxwellNet.py) -- the transpose of this array's
        # (row=z, col=x) layout, so transpose to match.
        n_map = n_map.T

        # MaxwellNet.forward reconstructs epsilon as scat_pot * ri_value**2
        # (then clips to a floor of 1) -- ri_value is the map's peak
        # refractive index, and scat_pot the (0-1) fraction of that peak
        # each pixel represents, so this exactly reproduces n_map.
        ri_value = float(n_map.max())
        scat_pot = (n_map / ri_value) ** 2

        delta_x_a = 1.0
        delta_z_a = 1.0

        return {
            "sample_id": r["sample_id"],
            "scat_pot": torch.from_numpy(scat_pot.astype(np.float32)).unsqueeze(0),
            "ri_value": ri_value,
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
