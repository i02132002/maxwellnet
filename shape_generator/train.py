"""Train a small DDPM UNet on optical_constant_{real,imag} images (shapes or MNIST).

With model.vae_ckpt set, the UNet instead diffuses in the latent space of that (frozen) VAE; train it first with
python -m shape_generator.train_vae. Images are encoded on the fly (a fresh posterior sample per batch).

Run from the repo root:  python -m shape_generator.train [--config path/to/config.json]
"""
import argparse
from dataclasses import asdict
import os
import shutil

import matplotlib
matplotlib.use("Agg")  # eval plots are saved/logged, never shown
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from shape_generator.config import DEFAULT_PATH, load_config  # noqa: E402
from shape_generator.data import compute_normalizer, load_data  # noqa: E402
from shape_generator.infer import plot_grid, sample  # noqa: E402
from shape_generator.model import (  # noqa: E402
    LatentCodec, build_scheduler, build_unet, save_checkpoint,
)
from shape_generator.schedule import make_lr_scheduler  # noqa: E402


def run_eval(unet, scheduler, normalizer, device, t, epoch):
    """Sample 9 shapes from the same fixed noise every epoch; save + return the 3x3 figures."""
    unet.eval()
    images = sample(unet, scheduler, normalizer, 9, device, t.eval_diffusion_timesteps, t.eval_seed)
    unet.train()
    figs = {}
    for channel in ("real", "imag"):
        path = os.path.join(t.out_dir, "samples", f"epoch_{epoch:03d}_{channel}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        figs[channel] = plot_grid(images, channel, path)
    return figs


@torch.no_grad()
def run_valid(unet, scheduler, normalizer, loader, device, num_timesteps, seed):
    """Mean noise-prediction MSE over the validation set. Timesteps and noise
    come from a seeded generator, so the number is comparable across epochs."""
    unet.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    total, n = 0.0, 0
    for batch in loader:
        x0 = normalizer.encode(batch["optical_constant_image"]).to(device)
        noise = torch.randn(x0.shape, generator=gen).to(device)
        ts = torch.randint(0, num_timesteps, (x0.shape[0],), generator=gen).to(device)
        loss = F.mse_loss(unet(scheduler.add_noise(x0, noise, ts), ts).sample, noise, reduction="sum")
        total += loss.item()
        n += x0.numel()
    unet.train()
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_PATH)
    config_path = p.parse_args().config
    cfg = load_config(config_path)
    d, m, t = cfg.data, cfg.model, cfg.train

    lg = cfg.logging
    os.makedirs(t.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    if t.seed is not None:
        torch.manual_seed(t.seed)

    train_ds, valid_ds = load_data(cfg)
    valid_loader = None
    if len(valid_ds) > 0:
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=t.batch_size, shuffle=False,
                                                   num_workers=d.num_workers)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=t.batch_size, shuffle=True,
                                         num_workers=d.num_workers, drop_last=True)
    if m.vae_ckpt is not None:
        normalizer = LatentCodec.from_checkpoint(m.vae_ckpt, device)
        f, vk = normalizer.downsample, normalizer.vae_kwargs
        if vk["image_size"] != m.image_size:
            raise ValueError(f"VAE {m.vae_ckpt} was trained at image_size={vk['image_size']} "
                             f"but model.image_size={m.image_size}")
        net_size, net_channels = m.image_size // f, normalizer.latent_channels
        print(f"latent diffusion: {m.image_size}px -> {net_channels}x{net_size}x{net_size} latents "
              f"(scale_factor {normalizer.scale_factor:.3f})")
    else:
        normalizer = compute_normalizer(train_ds, d.normalization)
        net_size, net_channels = m.image_size, 2
    print("normalizer:", normalizer.describe())
    if net_size % 2 ** (len(m.block_out_channels) - 1):
        raise ValueError(f"UNet input size ({net_size}) must be divisible by 2**(levels-1) = "
                         f"{2 ** (len(m.block_out_channels) - 1)}")

    unet_kwargs = {"block_out_channels": m.block_out_channels, "image_size": net_size,
                   "layers_per_block": m.layers_per_block, "channels": net_channels,
                   "attention_levels": m.attention_levels}
    unet = build_unet(**unet_kwargs).to(device)
    scheduler = build_scheduler(m.timesteps, normalizer.clip_range)
    print(f"UNet params: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(unet.parameters(), lr=t.lr)
    lr_sched = make_lr_scheduler(opt, t.lr_schedule, t.lr, t.min_lr, t.warmup_steps, t.epochs * len(loader))

    if lg.wandb:
        wandb.init(entity=lg.entity, project=lg.project, name=lg.run_name, config=asdict(cfg))
        wandb.save(config_path, base_path=os.path.dirname(os.path.abspath(config_path)), policy="now")

    step = 0
    for epoch in range(t.epochs):
        unet.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            x0 = normalizer.encode(batch["optical_constant_image"], sample=True).to(device)
            noise = torch.randn_like(x0)
            ts = torch.randint(0, m.timesteps, (x0.shape[0],), device=device)
            xt = scheduler.add_noise(x0, noise, ts)
            loss = F.mse_loss(unet(xt, ts).sample, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if t.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), t.grad_clip)
            opt.step()
            lr_sched.step()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if lg.wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr_sched.get_last_lr()[0]}, step=step)
        if (epoch + 1) % t.eval_epochs == 0 or epoch + 1 == t.epochs:
            log = {"epoch": epoch}
            if valid_loader is not None:
                log["valid/loss"] = run_valid(unet, scheduler, normalizer, valid_loader, device,
                                              m.timesteps, t.eval_seed)
                print(f"epoch {epoch} valid loss: {log['valid/loss']:.4f}")
            figs = run_eval(unet, scheduler, normalizer, device, t, epoch)
            if lg.wandb:
                wandb.log({**log, **{f"samples/{c}": wandb.Image(f) for c, f in figs.items()}}, step=step)
            for f in figs.values():
                plt.close(f)
        if (epoch + 1) % t.save_every == 0 or epoch + 1 == t.epochs:
            ckpt_path = os.path.join(t.out_dir, f"ckpt_epoch_{epoch + 1:04d}.pt")
            save_checkpoint(ckpt_path, unet, normalizer, unet_kwargs, m.timesteps, step)
            shutil.copyfile(ckpt_path, os.path.join(t.out_dir, "ckpt.pt"))  # latest, the default for infer.py
    if lg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
