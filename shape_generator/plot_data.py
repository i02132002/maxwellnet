"""Plot a 3x3 grid of training-data samples, in the same style as infer.py's generated grid.

Run from the repo root:  python -m shape_generator.plot_data [--config path/to/config.json]
    [--split train|valid] [--seed N] [--out path.png]
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch

from shape_generator.config import DEFAULT_PATH, load_config
from shape_generator.infer import plot_grid
from shape_generator.mnist import load_mnist
from shape_generator.shapes import load_new_shapes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_PATH)
    p.add_argument("--split", choices=("train", "valid"), default="train")
    p.add_argument("--seed", type=int, default=None, help="seed for picking which samples to show")
    p.add_argument("--out", default="shape_generator/data_samples.png",
                   help="output path; _real / _imag is inserted before the extension")
    args = p.parse_args()
    cfg = load_config(args.config)
    d, t, m = cfg.data, cfg.train, cfg.model

    if d.dataset == "mnist":
        train_ds, valid_ds = load_mnist(d.mnist_root, d.n_samples, d.valid_fraction)
    else:
        train_ds, valid_ds = load_new_shapes(
            d.num_files, d.component, d.n_samples, d.valid_fraction, t.seed if t.seed is not None else 0,
            m.image_size)
    ds = train_ds if args.split == "train" else valid_ds
    if len(ds) == 0:
        raise ValueError(f"the {args.split} split is empty (data.valid_fraction={d.valid_fraction})")

    gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    idx = torch.randperm(len(ds), generator=gen)[:9].tolist()
    images = torch.stack([ds[i]["optical_constant_image"] for i in idx])

    root, ext = os.path.splitext(args.out)
    for channel in ("real", "imag"):
        fig = plot_grid(images, channel, f"{root}_{channel}{ext}")
        fig.suptitle(f"Training data: {args.split} samples ({channel})")
        fig.savefig(f"{root}_{channel}{ext}", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
