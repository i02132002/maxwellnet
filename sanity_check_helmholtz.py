# Pre-training sanity check for the Helmholtz residual loss
# (losses/helmholtz_checker.py) against known-exact solutions in a
# homogeneous medium (eps=1), using the real ShapeNet incident-wave
# convention and physical parameters (wavelength, grid spacing, PML
# thickness) pulled from specs_maxwell.json / the actual dataset.
#
# In a true vacuum (no scatterer) the only physically correct field is
# exactly the incident wave, E = E_i. This script checks:
#   - L(E=0) is NOT a free minimum: it should be large, since both the
#     z-PML anchor and the Bloch-periodic x boundary are tied to the known
#     incident wave (see helmholtz_checker._lap_z_pml_scattered and
#     _bloch_pad_x).
#   - L(E=E_i) << L(E=0).
#   - Scanning E = alpha * E_i over alpha, the loss-minimizing alpha should
#     land close to 1 (the field shouldn't want to shrink towards 0, nor
#     blow up away from the true incident amplitude).
#   - Splitting E=E_i's residual energy into x-edge / z-PML / interior
#     regions should each be small -- just the numerical-dispersion floor,
#     with no spurious boundary-handling artifact.

import json
import math

import torch

from Dataset import ShapeDataset
from losses.helmholtz_checker import helmholtz_residual_loss_periodic_pml


def region_energies(residual, pml_thickness, x_edge_width=4):
    energy = residual.abs().square()

    x_edge = torch.cat(
        [energy[..., :x_edge_width], energy[..., -x_edge_width:]],
        dim=-1,
    ).mean()

    z_edge = torch.cat(
        [
            energy[..., :pml_thickness, :],
            energy[..., -pml_thickness:, :],
        ],
        dim=-2,
    ).mean()

    interior = energy[
        ...,
        pml_thickness:-pml_thickness,
        x_edge_width:-x_edge_width,
    ].mean()

    return x_edge.item(), z_edge.item(), interior.item()


def run_case(label, wavelength_a, delta_x_a, delta_z_a, theta_deg, nz, nx, pml_thickness):
    print(f"\n=== {label} (theta={theta_deg:.2f} deg, wavelength_a={wavelength_a:.3f}, "
          f"dx={delta_x_a:.4f}, dz={delta_z_a:.4f}, grid={nz}x{nx}, pml={pml_thickness}) ===")

    k = 2 * math.pi / wavelength_a
    theta_rad = math.radians(theta_deg)
    kx_val = -k * math.sin(theta_rad)
    kz_val = -k * math.cos(theta_rad)

    z = torch.arange(nz, dtype=torch.float32) * delta_z_a
    x = torch.arange(nx, dtype=torch.float32) * delta_x_a
    phase = kz_val * z[:, None] + kx_val * x[None, :]
    incident = torch.complex(torch.cos(phase), torch.sin(phase)).to(torch.complex64)

    kz = torch.tensor(kz_val)
    kx = torch.tensor(kx_val)
    eps_vacuum = torch.ones(nz, nx, dtype=torch.complex64)

    def loss_for(E):
        residual = helmholtz_residual_loss_periodic_pml(
            E, eps_vacuum, incident, kz, kx, 's', wavelength_a, delta_x_a, delta_z_a, pml_thickness)
        return residual.abs().pow(2).mean().item(), residual

    loss_zero, _ = loss_for(torch.zeros_like(incident))
    loss_inc, res_inc = loss_for(incident)

    ratio = loss_inc / loss_zero if loss_zero > 0 else float('nan')
    print(f"L(E=0)      = {loss_zero:.6e}")
    print(f"L(E=E_i)    = {loss_inc:.6e}")
    print(f"L(E_i)/L(0) = {ratio:.3e}   (want << 1)")

    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    losses = [loss_for(alpha * incident)[0] for alpha in alphas]
    best_idx = min(range(len(alphas)), key=lambda i: losses[i])

    print("\nalpha sweep (E = alpha * E_i):")
    for alpha, loss_a in zip(alphas, losses):
        marker = "  <-- min" if alpha == alphas[best_idx] else ""
        print(f"  alpha={alpha:5.2f}  L={loss_a:.6e}{marker}")
    print(f"argmin alpha = {alphas[best_idx]}   (want close to 1.0)")

    x_edge, z_edge, interior = region_energies(res_inc, pml_thickness)
    print("\nE=E_i region-wise residual energy (want all small -- pure numerical-dispersion floor):")
    print(f"  x-edge   mean|res|^2 = {x_edge:.6e}")
    print(f"  z-PML    mean|res|^2 = {z_edge:.6e}   (E_s=0 exactly here -- should be smallest)")
    print(f"  interior mean|res|^2 = {interior:.6e}")

    return dict(loss_zero=loss_zero, loss_inc=loss_inc, ratio=ratio,
                best_alpha=alphas[best_idx], x_edge=x_edge, z_edge=z_edge, interior=interior)


def main():
    specs = json.load(open('specs_maxwell.json'))
    mode = specs['PhysicalSpecs']['mode']
    pml_thickness = specs['PhysicalSpecs']['pml_thickness']

    train_ds, _ = ShapeDataset.load_train_valid(
        specs.get('HFConfig'), mode, specs.get('ValidFraction', 0.1), specs.get('Seed', 0))

    # Pull a handful of real samples spanning distinct incidence angles, so
    # this validates at the actual physical parameters used in training
    # (including normal incidence, if present, and oblique).
    seen_thetas = []
    cases = []
    for i in range(len(train_ds)):
        item = train_ds[i]
        theta = float(item['theta'])
        if any(abs(theta - t) < 1e-6 for t in seen_thetas):
            continue
        seen_thetas.append(theta)
        wavelength_a = float(item['wavelength_nm']) * 10.0
        nz, nx = item['optical_constant'].shape[0], item['optical_constant'].shape[1]
        cases.append((f"sample #{i}", wavelength_a, float(item['delta_x_a']), float(item['delta_z_a']),
                      theta, nz, nx, pml_thickness))
        if len(cases) >= 4:
            break

    results = [run_case(*case) for case in cases]

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    all_ok = True
    for (label, *_rest), r in zip(cases, results):
        ok_ratio = r['ratio'] < 1e-2
        ok_alpha = abs(r['best_alpha'] - 1.0) <= 0.25
        ok = ok_ratio and ok_alpha
        all_ok &= ok
        status = "OK" if ok else "CHECK"
        print(f"[{status}] {label}: L(E_i)/L(0)={r['ratio']:.2e}, argmin alpha={r['best_alpha']}")
    print("\nOverall:", "PASS" if all_ok else "SOME CASES NEED ATTENTION")


if __name__ == '__main__':
    main()
