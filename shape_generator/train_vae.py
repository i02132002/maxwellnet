"""Stage 1 of latent diffusion: train a KL autoencoder on optical_constant_{real,imag} images.

Run from the repo root:  python -m shape_generator.train_vae [--config path/to/config.json]
Then set model.vae_ckpt to <vae.out_dir>/vae.pt and run shape_generator.train.

Loss = recon (l1 | mse) in normalized pixel space + kl_weight * per-element mean KL(q(z|x) || N(0, 1)).
Every vae.ckpt_every steps (the same steps vae_step_XXXXXX.pt / vae.pt are saved) the script logs reconstruction / latent-health metrics (see run_metrics) and
input / recon / |error| plots, on the validation split (or a fixed train subset if there is none).
"""
import argparse
from dataclasses import asdict
import os
import shutil

import matplotlib
matplotlib.use("Agg")  # plots are saved/logged, never shown
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import wandb  # noqa: E402
from tqdm import tqdm  # noqa: E402

from shape_generator.config import DEFAULT_PATH, load_config  # noqa: E402
from shape_generator.data import compute_normalizer, load_data  # noqa: E402
from shape_generator.model import Normalizer, build_vae, save_vae_checkpoint  # noqa: E402
from shape_generator.schedule import make_lr_scheduler  # noqa: E402

CHANNELS = ("real", "imag")
EDGE_EPS = 1e-3          # a pixel is an "edge" if its normalized value differs from a 4-neighbour by more than this
ACTIVE_KL = 0.01         # a latent channel counts as "active" if its mean KL (nats / element) exceeds this
N_PLOT = 6               # images shown in the reconstruction plots
N_SCALE = 256            # images used to estimate the latent scale factor


def kl_terms(post):
    """Elementwise KL(N(mean, var) || N(0, 1)), shape (B, C_latent, h, w)."""
    return 0.5 * (post.mean.pow(2) + post.var - 1 - post.logvar)


def recon_loss(kind, rec, x):
    """Per-channel reconstruction loss, shape (C,). Its mean equals the plain mean over all pixels."""
    err = (rec - x).abs() if kind == "l1" else (rec - x).pow(2)
    return err.mean(dim=(0, 2, 3))


def edge_mask(x):
    """(B, C, H, W) bool: True where x differs from a 4-neighbour (shape boundaries and region steps)."""
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs() > EDGE_EPS
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs() > EDGE_EPS
    m = torch.zeros_like(x, dtype=torch.bool)
    m[..., :, 1:] |= dx
    m[..., :, :-1] |= dx
    m[..., 1:, :] |= dy
    m[..., :-1, :] |= dy
    return m


@torch.no_grad()
def estimate_scale_factor(vae, normalizer, dataset, device, n=N_SCALE):
    """1 / std of sampled latents over (up to) n train images: what the DDPM multiplies latents by."""
    vae.eval()
    idx = torch.randperm(len(dataset))[:min(n, len(dataset))].tolist()
    x = normalizer.encode(torch.stack([dataset[i]["optical_constant_image"] for i in idx])).to(device)
    z = torch.cat([vae.encode(c).latent_dist.sample() for c in x.split(32)])
    return 1.0 / z.std().item()


@torch.no_grad()
def run_metrics(vae, normalizer, loader, device):
    """Metrics over `loader` (deterministic order). Returns (flat dict of scalars for logging, first batch (x, rec)).

    Reconstruction (decoding the posterior mean; normalized space unless stated):
      recon/mse_{ch}, recon/r2_{ch}          MSE, and 1 - MSE / var(x): 1.0 is perfect, 0 = predicting the mean
      recon/mae_raw_{ch}                     mean |error| in physical optical-constant units
      recon/mae_edge_{ch}, mae_flat_{ch}     |error| on boundary pixels vs. on flat-region pixels
      recon/rel_err_{median,p90}_{ch}        per-image rmse / image std: error relative to each image's own
                                             contrast, so faint low-contrast shapes are not hidden by the mean
      recon/mse_sampled                      MSE when decoding a *sampled* z; the gap to recon/mse_mean is how
                                             much the decoder suffers from latent noise (diffusion output is noisy)
    Latent health:
      latent/kl_mean, kl_ch{i}, active_channels   KL in nats/element; a channel with KL ~ 0 has collapsed
      latent/mean_ch{i}, std_ch{i}                posterior-mean statistics per channel
      latent/post_std                             mean posterior sigma (-> 1 means z carries no information)
      latent/scale_factor                         1 / std of sampled z (the DDPM latent scaling)
    """
    vae.eval()
    nc = len(CHANNELS)
    lc = vae.config.latent_channels
    s = {k: torch.zeros(nc, dtype=torch.float64) for k in
         ("sse", "sse_sampled", "sx", "sxx", "abs_raw", "abs_edge", "n_edge", "abs_flat", "n_flat", "n")}
    rel = [[] for _ in range(nc)]
    kl_sum = torch.zeros(lc, dtype=torch.float64)
    zm_sum, zm_sq = torch.zeros(lc, dtype=torch.float64), torch.zeros(lc, dtype=torch.float64)
    zs_sum, zs_sq, post_std_sum, n_lat = 0.0, 0.0, 0.0, 0
    first = None
    for batch in loader:
        raw = batch["optical_constant_image"]
        x = normalizer.encode(raw).to(device)
        post = vae.encode(x).latent_dist
        rec = vae.decode(post.mean).sample
        rec_s = vae.decode(post.sample()).sample
        # accumulate in float64 on the CPU (MPS has no float64)
        x_c, rec_c, rec_sc = x.cpu(), rec.cpu(), rec_s.cpu()
        if first is None:
            first = (x_c, rec_c)
        err = (rec_c - x_c).double()
        edge = edge_mask(x_c)
        flat_mask = ~edge
        raw_err = (normalizer.decode(rec_c) - raw).abs().double()
        for c in range(nc):
            xc, ec = x_c[:, c].double(), err[:, c]
            s["sse"][c] += ec.pow(2).sum().item()
            s["sse_sampled"][c] += (rec_sc[:, c] - x_c[:, c]).double().pow(2).sum().item()
            s["sx"][c] += xc.sum().item()
            s["sxx"][c] += xc.pow(2).sum().item()
            s["n"][c] += xc.numel()
            s["abs_raw"][c] += raw_err[:, c].sum().item()
            e, f = edge[:, c], flat_mask[:, c]
            s["abs_edge"][c] += ec.abs()[e].sum().item()
            s["n_edge"][c] += e.sum().item()
            s["abs_flat"][c] += ec.abs()[f].sum().item()
            s["n_flat"][c] += f.sum().item()
            std_i = xc.flatten(1).std(dim=1)
            rmse_i = ec.flatten(1).pow(2).mean(dim=1).sqrt()
            keep = std_i > 1e-3  # blank images (no shape) have no contrast to be relative to
            rel[c].append((rmse_i[keep] / std_i[keep]))
        kl = kl_terms(post).cpu().double()
        kl_sum += kl.sum(dim=(0, 2, 3))
        zm = post.mean.cpu().double()
        zm_sum += zm.sum(dim=(0, 2, 3))
        zm_sq += zm.pow(2).sum(dim=(0, 2, 3))
        z = post.sample().cpu().double()
        zs_sum += z.sum().item()
        zs_sq += z.pow(2).sum().item()
        post_std_sum += post.std.cpu().double().sum().item()
        n_lat += z.numel()

    out = {}
    for c, name in enumerate(CHANNELS):
        n = s["n"][c].item()
        mse = s["sse"][c].item() / n
        var = s["sxx"][c].item() / n - (s["sx"][c].item() / n) ** 2
        out[f"recon/mse_{name}"] = mse
        out[f"recon/r2_{name}"] = 1 - mse / max(var, 1e-12)
        out[f"recon/mae_raw_{name}"] = s["abs_raw"][c].item() / n
        out[f"recon/mae_edge_{name}"] = s["abs_edge"][c].item() / max(s["n_edge"][c].item(), 1)
        out[f"recon/mae_flat_{name}"] = s["abs_flat"][c].item() / max(s["n_flat"][c].item(), 1)
        r = torch.cat(rel[c]) if rel[c] else torch.zeros(1)
        out[f"recon/rel_err_median_{name}"] = r.median().item()
        out[f"recon/rel_err_p90_{name}"] = torch.quantile(r, 0.9).item() if len(r) > 1 else r.item()
    n_pix = s["n"].sum().item()
    out["recon/mse_mean"] = s["sse"].sum().item() / n_pix
    out["recon/mse_sampled"] = s["sse_sampled"].sum().item() / n_pix
    per_ch_n = n_lat / lc
    kl_ch = kl_sum / per_ch_n
    out["latent/kl_mean"] = kl_ch.mean().item()
    out["latent/active_channels"] = int((kl_ch > ACTIVE_KL).sum())
    zm_mean = zm_sum / per_ch_n
    zm_std = (zm_sq / per_ch_n - zm_mean ** 2).clamp_min(0).sqrt()
    for i in range(lc):
        out[f"latent/kl_ch{i}"] = kl_ch[i].item()
        out[f"latent/mean_ch{i}"] = zm_mean[i].item()
        out[f"latent/std_ch{i}"] = zm_std[i].item()
    out["latent/post_std"] = post_std_sum / n_lat
    z_mean = zs_sum / n_lat
    out["latent/scale_factor"] = 1.0 / max((zs_sq / n_lat - z_mean ** 2), 1e-12) ** 0.5
    return out, first


def plot_recon(x, rec, normalizer, channel, path):
    """Rows: input / reconstruction / |error|, in raw units; columns: the first N_PLOT eval images."""
    c = CHANNELS.index(channel)
    xr, rr = normalizer.decode(x)[:N_PLOT, c], normalizer.decode(rec)[:N_PLOT, c]
    err = (rr - xr).abs()
    n = len(xr)
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6.4), squeeze=False)
    fig.subplots_adjust(left=0.04, right=1, bottom=0, top=0.93, wspace=0.01, hspace=0.01)
    vmin, vmax = xr.min().item(), xr.max().item()
    for j in range(n):
        axes[0, j].imshow(xr[j], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, j].imshow(rr[j], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[2, j].imshow(err[j], cmap="magma", vmin=0, vmax=max(err.max().item(), 1e-12))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for i, label in enumerate(("input", "recon", "|error|")):
        axes[i, 0].set_ylabel(label)
    fig.suptitle(f"VAE reconstruction ({channel}); input/recon share a color scale, error has its own")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_PATH)
    config_path = p.parse_args().config
    cfg = load_config(config_path)
    d, m, v, lg = cfg.data, cfg.model, cfg.vae, cfg.logging

    factor = 2 ** (len(v.block_out_channels) - 1)
    if m.image_size % factor:
        raise ValueError(f"model.image_size ({m.image_size}) must be divisible by the VAE downsample factor {factor}")
    os.makedirs(v.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    if v.seed is not None:
        torch.manual_seed(v.seed)

    train_ds, valid_ds = load_data(cfg)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=v.batch_size, shuffle=True,
                                         num_workers=d.num_workers, drop_last=True)
    if len(loader) == 0:
        raise ValueError(f"vae.batch_size ({v.batch_size}) is larger than the train set ({len(train_ds)})")
    eval_ds = valid_ds if len(valid_ds) > 0 else torch.utils.data.Subset(train_ds, range(min(256, len(train_ds))))
    eval_name = "valid" if len(valid_ds) > 0 else "train_subset"
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=v.batch_size, shuffle=False,
                                              num_workers=d.num_workers)
    vae_kwargs = {"block_out_channels": v.block_out_channels, "latent_channels": v.latent_channels,
                  "layers_per_block": v.layers_per_block, "image_size": m.image_size}
    start_step = 0
    if v.resume:
        ckpt = torch.load(v.resume, map_location="cpu")
        if {k: ckpt["vae_kwargs"][k] for k in vae_kwargs} != {k: list(x) if isinstance(x, tuple) else x
                                                              for k, x in vae_kwargs.items()}:
            raise ValueError(f"vae.resume {v.resume} was built with {ckpt['vae_kwargs']}, config has {vae_kwargs}")
        normalizer = Normalizer.from_state_dict(ckpt["pixel_normalizer"])  # must match what the weights were trained on
        start_step = ckpt["step"]
        print(f"resuming from {v.resume} (step {start_step}); optimizer and lr schedule restart")
    else:
        normalizer = compute_normalizer(train_ds, d.normalization)
    print("pixel normalizer:", normalizer.describe())

    vae = build_vae(**vae_kwargs).to(device)
    if v.resume:
        vae.load_state_dict(ckpt["vae"])
    lat = m.image_size // factor
    print(f"VAE params: {sum(q.numel() for q in vae.parameters()) / 1e6:.1f}M; "
          f"{2}x{m.image_size}x{m.image_size} -> {v.latent_channels}x{lat}x{lat} "
          f"({2 * m.image_size ** 2 / (v.latent_channels * lat ** 2):.0f}x compression)")
    opt = torch.optim.AdamW(vae.parameters(), lr=v.lr)
    lr_sched = make_lr_scheduler(opt, v.lr_schedule, v.lr, v.min_lr, v.warmup_steps, v.epochs * len(loader))

    if lg.wandb:
        wandb.init(entity=lg.entity, project=lg.project, name=lg.run_name, job_type="vae", config=asdict(cfg))
        wandb.save(config_path, base_path=os.path.dirname(os.path.abspath(config_path)), policy="now")

    def run_eval_and_ckpt(step):
        metrics, (x_first, rec_first) = run_metrics(vae, normalizer, eval_loader, device)
        print(f"step {step} [{eval_name}] mse {metrics['recon/mse_mean']:.5f} "
              f"(sampled z {metrics['recon/mse_sampled']:.5f}) "
              f"r2 real/imag {metrics['recon/r2_real']:.4f}/{metrics['recon/r2_imag']:.4f} "
              f"kl {metrics['latent/kl_mean']:.3f} active {metrics['latent/active_channels']}/{v.latent_channels} "
              f"scale {metrics['latent/scale_factor']:.3f}")
        figs = {c: plot_recon(x_first, rec_first, normalizer, c,
                              os.path.join(v.out_dir, "recon", f"step_{step:06d}_{c}.png"))
                for c in CHANNELS}
        if lg.wandb:
            wandb.log({**{f"vae_{eval_name}/{k}": val for k, val in metrics.items()},
                       **{f"vae_recon/{c}": wandb.Image(f) for c, f in figs.items()}}, step=step)
        for f in figs.values():
            plt.close(f)
        scale = estimate_scale_factor(vae, normalizer, train_ds, device)
        ckpt_path = os.path.join(v.out_dir, f"vae_step_{step:06d}.pt")
        save_vae_checkpoint(ckpt_path, vae, vae_kwargs, normalizer, scale, step)
        shutil.copyfile(ckpt_path, os.path.join(v.out_dir, "vae.pt"))  # latest, what model.vae_ckpt points at

    # step numbers (checkpoint names, logs) continue from the resumed run
    step = start_step
    last_step = start_step + v.epochs * len(loader)
    for epoch in range(v.epochs):
        vae.train()
        pbar = tqdm(loader, desc=f"vae epoch {epoch}")
        for batch in pbar:
            x = normalizer.encode(batch["optical_constant_image"]).to(device)
            post = vae.encode(x).latent_dist
            rec = vae.decode(post.sample()).sample
            rl_ch, kl = recon_loss(v.recon_loss, rec, x), kl_terms(post).mean()
            rl = rl_ch.mean()
            loss = rl + v.kl_weight * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if v.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), v.grad_clip)
            opt.step()
            lr_sched.step()
            step += 1
            pbar.set_postfix(**{f"recon_{c}": f"{r:.4f}" for c, r in zip(CHANNELS, rl_ch.tolist())}, kl=f"{kl.item():.3f}")
            if lg.wandb:
                wandb.log({"vae_train/loss": loss.item(), "vae_train/recon": rl.item(), "vae_train/kl": kl.item(),
                           **{f"vae_train/recon_{c}": r for c, r in zip(CHANNELS, rl_ch.tolist())},
                           "vae_train/lr": lr_sched.get_last_lr()[0]}, step=step)

            if step % v.ckpt_every == 0 or step == last_step:  # eval and checkpoint at the same steps
                run_eval_and_ckpt(step)
                vae.train()  # run_metrics / estimate_scale_factor leave the VAE in eval mode
    if lg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
