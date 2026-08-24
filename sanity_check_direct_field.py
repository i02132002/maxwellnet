# Diagnostic: is the UNet the bottleneck, or is it the residual loss itself?
#
# For the single fixed sample used in the current overfitting run, the
# network always sees the same input -- millions of UNet parameters are
# being spent to fit one 256x256 complex field. This script replaces
# PeriodicMaxwellNet's UNet with a bare per-pixel parameter tensor (the same
# shape/interpretation the UNet's output would have) and runs the *real*
# `net.forward()` on top of it -- so padding/cropping, epsilon selection,
# and incident-wave construction are all exactly the production code path,
# not a reimplementation of it. Only the UNet itself is swapped out.
# `net.forward()` returns the envelope `a` directly (E = (1+a) * E_i is
# reconstructed by the loss function / caller, not by the model).
#
# Interpretation of the result:
#   - RMS|a| grows well above 0 (and doesn't sit near 1): the residual
#     landscape permits real scattering -- so the UNet optimization is what's
#     slow, not the loss/discretization.
#   - RMS|a| stays near 0: even direct, UNet-free optimization can't develop
#     scattering, so the issue is the residual landscape/discretization, not
#     dataset size or CNN capacity.
#   - a approaches -1 (RMS|a - (-1)| small, i.e. E approaches 0): the loss
#     still has a pathological minimum at zero total field.

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn

from Dataset import ShapeDataset
from ShapeNet import PeriodicMaxwellNet
from losses.helmholtz_checker import helmholtz_residual_loss_periodic_pml


class DirectField(nn.Module):
    """Drop-in replacement for UNet: ignores its inputs and just returns a
    free parameter tensor of the same (B, 2k, Nz, Nx) shape/channel
    convention UNet's output has (first k channels: envelope real part minus
    1; next k: imaginary part) -- so PeriodicMaxwellNet.forward's envelope
    reconstruction (`torch.complex(out[:, :k] + 1, out[:, k:2*k])`) treats it
    identically to a real UNet output."""

    def __init__(self, channels, nz, nx):
        super().__init__()
        self.out = nn.Parameter(torch.zeros(1, channels, nz, nx))

    def forward(self, x, cond=None):
        return self.out


def plot_residual(residual, step, out_dir):
    loss_map = residual.detach().abs().pow(2).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(loss_map, origin='lower', aspect='equal', cmap='magma')
    fig.colorbar(im, ax=ax, label='|residual|^2')
    ax.set_title(f'Helmholtz residual (step {step})')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'residual_step{step:06d}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(steps, lr, log_every, plot_every, clip, plot_dir):
    assert steps > 0
    specs = json.load(open('specs_maxwell.json'))
    mode = specs['PhysicalSpecs']['mode']
    pml_thickness = specs['PhysicalSpecs']['pml_thickness']
    pol = 's' if mode == 'te' else 'p'
    object_weight = specs.get('ObjectWeight', 1.0)
    object_threshold = specs.get('ObjectThreshold', 1e-4)

    train_ds, _ = ShapeDataset.load_train_valid(
        specs.get('HFConfig'), mode, specs.get('ValidFraction', 0.1),
        specs.get('Seed', 0), n_samples=specs.get('NumTrainSamples'))
    sample = train_ds[0]

    net = PeriodicMaxwellNet(**specs['NetworkSpecs'], mode=mode)
    nz, nx = sample['optical_constant'].shape[0], sample['optical_constant'].shape[1]
    channels = 2 if mode == 'te' else 4
    net.model = DirectField(channels, nz, nx)

    optical_constant = sample['optical_constant'].unsqueeze(0).to(torch.complex64)
    theta = torch.tensor([sample['theta']], dtype=torch.float32)
    wavelength_nm = torch.tensor([sample['wavelength_nm']], dtype=torch.float32)
    delta_x_a = torch.tensor([sample['delta_x_a']], dtype=torch.float32)
    delta_z_a = torch.tensor([sample['delta_z_a']], dtype=torch.float32)
    wavelength_a = float(sample['wavelength_nm']) * 10.0

    optimizer_field = torch.optim.Adam(net.model.parameters(), lr=lr)

    print(f"Directly optimizing {net.model.out.numel()} free real parameters "
          f"(grid {nz}x{nx}) through net.forward(), for {steps} steps, lr={lr}"
          + (f", grad_clip={clip}" if clip else ""))

    for step in range(steps):
        # net() (ShapeNet.forward) now returns the envelope `a` directly
        # (E_total = incident * (1+a)), not the reconstructed total field --
        # helmholtz_residual_loss_periodic_pml also now takes the envelope
        # directly, so no reconstruction is needed for the loss itself.
        envelope, eps, incident, kz, kx = net(
            optical_constant, theta, wavelength_nm, delta_x_a, delta_z_a)

        residual = helmholtz_residual_loss_periodic_pml(
            envelope, eps, incident, kz, kx, pol, wavelength_a,
            float(sample['delta_x_a']), float(sample['delta_z_a']), pml_thickness)

        is_object = (eps - 1.0).abs() > object_threshold
        weight = torch.where(is_object, object_weight, 1.0)
        loss = torch.mean(weight * residual.abs().pow(2))

        optimizer_field.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(net.model.parameters(), clip)
        optimizer_field.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                a_rms = envelope.abs().pow(2).mean().sqrt().item()
                a_max = envelope.abs().max().item()
            print(f"step {step:6d}  loss={loss.item():.6e}  RMS|a|={a_rms:.4e}  max|a|={a_max:.4e}")

        if step == steps - 1 or (plot_every and step % plot_every == 0):
            path = plot_residual(residual, step, plot_dir)
            print(f"  saved {path}")

    with torch.no_grad():
        rms_a = envelope.abs().pow(2).mean().sqrt().item()
        rms_a_vs_minus1 = (envelope + 1.0).abs().pow(2).mean().sqrt().item()

    print("\n" + "=" * 70)
    print(f"final RMS|a|        = {rms_a:.4e}")
    print(f"final RMS|a - (-1)| = {rms_a_vs_minus1:.4e}  (small => a approx -1, i.e. E approx 0)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--plot_every", type=int, default=0,
                        help="Save a |residual|^2 heatmap every N steps (0 = only at the end).")
    parser.add_argument("--plot_dir", type=str, default="sanity_check_plots")
    parser.add_argument("--clip", type=float, default=None,
                        help="Optional grad-norm clip, for stabilizing higher lr.")
    args = parser.parse_args()
    main(args.steps, args.lr, args.log_every, args.plot_every, args.clip, args.plot_dir)
