"""Dataset loading + input normalization shared by train.py and train_vae.py."""
import torch

from shape_generator.mnist import load_mnist
from shape_generator.model import Normalizer
from shape_generator.shapes import load_general_shapes


def load_data(cfg):
    """Returns (train_ds, valid_ds) for cfg.data at cfg.model.image_size."""
    d, m = cfg.data, cfg.model
    if d.dataset == "mnist":
        if m.image_size != 28:
            raise ValueError(f"data.dataset='mnist' has 28x28 images but model.image_size={m.image_size}")
        train_ds, valid_ds = load_mnist(d.mnist_root, d.n_samples, d.valid_fraction)
    else:
        seed = cfg.train.seed if cfg.train.seed is not None else 0
        train_ds, valid_ds = load_general_shapes(
            d.num_files, d.component, d.n_samples, d.valid_fraction, seed, m.image_size, d.max_oc_value)
    print(f"train shapes: {len(train_ds)}, valid shapes: {len(valid_ds)}")
    return train_ds, valid_ds


def compute_normalizer(dataset, mode, n=256):
    idx = torch.randperm(len(dataset))[:min(n, len(dataset))].tolist()
    imgs = torch.stack([dataset[i]["optical_constant_image"] for i in idx])
    return Normalizer.fit(imgs, mode)
