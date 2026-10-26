"""Sample shapes from a trained DDPM and plot them in a 3x3 grid.

Run from the repo root:  python -m shape_generator.infer [--config path/to/config.json]
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from shape_generator.config import DEFAULT_PATH, load_config
from shape_generator.model import load_checkpoint


@torch.no_grad()
def sample(unet, scheduler, normalizer, n, device, steps=None, seed=None):
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    size = unet.config.sample_size
    x = torch.randn(n, unet.config.in_channels, size, size, generator=gen).to(device)
    if steps is not None:
        scheduler.set_timesteps(steps)
    else:
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    for t in tqdm(scheduler.timesteps, desc="sampling"):
        eps = unet(x, t).sample
        x = scheduler.step(eps, t, x).prev_sample
    return normalizer.decode(x).cpu()


def plot_grid(images, channel, path=None, show=False):
    """images: (9, 2, H, W). channel: 'real' | 'imag'. Returns the figure."""
    c = {"real": 0, "imag": 1}[channel]
    # axes region is 0.9W x 0.95H, so W = 9.5 / H = 9 makes the cells square
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9))
    fig.subplots_adjust(left=0, right=0.9, bottom=0, top=0.95, wspace=0.01, hspace=0.01)  # thin white lines between tiles
    vmin, vmax = images[:, c].min().item(), images[:, c].max().item()
    for ax in axes.flat:
        ax.axis("off")  # also hides unused tiles when fewer than 9 images are given
    for ax, img in zip(axes.flat, images):
        im = ax.imshow(img[c], cmap="viridis", vmin=vmin, vmax=vmax)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cax, label=f"optical_constant_{channel}")
    fig.suptitle(f"Generated shapes ({channel})")
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print("saved", path)
    if show:
        plt.show()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_PATH)
    c = load_config(p.parse_args().config).inference

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    unet, scheduler, normalizer = load_checkpoint(c.ckpt, device)
    images = sample(unet, scheduler, normalizer, 9, device, c.steps, c.seed)
    root, ext = os.path.splitext(c.out)
    for channel in ("real", "imag"):
        plot_grid(images, channel, f"{root}_{channel}{ext}")
    plt.show()


if __name__ == "__main__":
    main()
