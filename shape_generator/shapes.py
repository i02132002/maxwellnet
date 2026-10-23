"""Loader for the newer flat-schema shards of als-rixs/latent-image-training.

Reads only sample_id + optical_constant_{real,imag} from the first `num_files`
new parquet shards (the two original, old-schema files are skipped), keeps one
row per sample_id, and exposes each shape as {"optical_constant_image": (2, 256, 256)}
(channel 0 = real part, 1 = imag part of one component of optical_constant).
"""
import random
import re

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download

REPO = "als-rixs/latent-image-training"
REVISION = "fixed-dimension"
GRID = 256
OLD_FILES = {"training/train-00000-of-00001.parquet", "training/train-00001-of-00002.parquet"}


def list_new_shards():
    files = [f.path for f in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION,
                                                     path_in_repo="training")
             if re.fullmatch(r"training/train-\d+-of-\d+\.parquet", f.path)]
    return sorted(p for p in files if p not in OLD_FILES)


class NewShapeDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images  # (N, 2, 256, 256) float32

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"optical_constant_image": self.images[idx]}


def load_new_shapes(num_files=None, component=2, n_samples=None, valid_fraction=0.0, seed=0, image_size=GRID):
    """`num_files=None` uses every new shard. Returns (train, valid). Shapes are unique by sample_id, shuffled with `seed`;
    the last `valid_fraction` go to valid. `n_samples` caps the train split.
    `component` indexes optical_constant's 3 components (flat layout is (3, 256, 256)).
    `image_size` < 256 subsamples with a stride of 256 // image_size (no averaging)."""
    if GRID % image_size:
        raise ValueError(f"image_size ({image_size}) must divide {GRID}")
    stride = GRID // image_size
    shards = list_new_shards()[:num_files]
    print(f"loading {len(shards)} shards: {shards[0]} ... {shards[-1]}")
    seen, images = set(), []
    for shard in shards:
        path = hf_hub_download(REPO, shard, repo_type="dataset", revision=REVISION)
        table = pq.read_table(path, columns=["sample_id", "optical_constant_real", "optical_constant_imag"])
        ids = table.column("sample_id").to_pylist()
        parts = []
        for name in ("optical_constant_real", "optical_constant_imag"):
            flat = table.column(name).combine_chunks().flatten().to_numpy()
            parts.append(flat.reshape(len(ids), 3, GRID, GRID)[:, component])
        both = np.stack(parts, axis=1)[:, :, ::stride, ::stride]  # (rows, 2, image_size, image_size)
        for i, sid in enumerate(ids):
            if sid not in seen:
                seen.add(sid)
                images.append(both[i])
    images = torch.from_numpy(np.stack(images)).float()
    order = list(range(len(images)))
    random.Random(seed).shuffle(order)
    n_valid = round(len(order) * valid_fraction)
    train_idx, valid_idx = order[:len(order) - n_valid], order[len(order) - n_valid:]
    if n_samples is not None:
        train_idx = train_idx[:n_samples]
    return NewShapeDataset(images[train_idx]), NewShapeDataset(images[valid_idx])
