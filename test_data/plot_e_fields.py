from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm
from numpy import dtype, floating, ndarray
from numpy._typing import _64Bit

from periodictable.xsf import index_of_refraction
from refloxide.pxr.tjf4x4 import hc
from refloxide.pxr.stacks import Layer, Material, stack_slabs, stack_tensor
from losses.helmholtz_checker import helmholtz_residual_loss

ENERGY_EV = 250.0
FILM_NM = 200.0
FILM_A = FILM_NM * 10.0

def c8h8_n(energy_ev: float) -> complex:
    return index_of_refraction("C8H8", density=1.0, energy=energy_ev * 1e-3)


def si_n(energy_ev: float) -> complex:
    return index_of_refraction("Si", density=2.33, energy=energy_ev * 1e-3)


def build_layers(energy_ev: float = ENERGY_EV) -> list[Layer]:
    """Return the smooth vacuum / C8H8 / Si stack used for field maps."""
    return [
        Layer(
            thickness=0.0,
            roughness=0.0,
            material=Material("scalar"),
            sld=complex(1.0, 0.0),
        ),
        Layer(
            thickness=FILM_A,
            roughness=0.0,
            material=Material("uniaxial"),
            sld=c8h8_n,
        ),
        Layer(
            thickness=0.0,
            roughness=0.0,
            material=Material("scalar"),
            sld=si_n,
        ),
    ]

n_vac = complex(1.0, 0.0)
n_film = c8h8_n(ENERGY_EV)
n_si = si_n(ENERGY_EV)
wavelength_a = hc / ENERGY_EV
k0 = 2.0 * np.pi / wavelength_a

def plot_efield_maps(
        x_nm, z_nm,
        Ey_s, Ex_p, Ez_p,
        film_nm=None, suptitle=None,
        layer_labels=("vacuum", "C8H8", "Si"),
):
    I_s = np.abs(Ey_s) ** 2
    I_p = np.abs(Ex_p) ** 2 + np.abs(Ez_p) ** 2

    vmax_s = I_s.max()
    vmax_p = I_p.max()

    rows = [
        (r"$|E_y|^2$ (s-pol)", "magma", I_s, LogNorm(vmin=max(vmax_s * 1e-3, 1e-4), vmax=vmax_s), "s-pol"),
        (r"$|E_x|^2+|E_z|^2$ (p-pol)", "magma", I_p, LogNorm(vmin=max(vmax_p * 1e-3, 1e-4), vmax=vmax_p), "p-pol"),
    ]

    fig, axes = plt.subplots(len(rows), 1, figsize=(7, 4 * len(rows)), sharex=True, sharey=True)

    for ax, (ylabel, cmap, data, norm, row_label) in zip(axes, rows):
        im = ax.pcolormesh(x_nm, z_nm, data, shading="auto", cmap=cmap, norm=norm)
        if film_nm is not None:
            ax.axhline(0.0, color="w", lw=0.8, alpha=0.8)
            ax.axhline(film_nm, color="w", lw=0.8, alpha=0.8)
            ax.text(0.02, 0.97, layer_labels[0], transform=ax.transAxes, va="top", color="w", fontsize=8)
            ax.text(0.02, 0.55, layer_labels[1], transform=ax.transAxes, va="top", color="w", fontsize=8)
            ax.text(0.02, 0.08, layer_labels[2], transform=ax.transAxes, va="bottom", color="w", fontsize=8)
        ax.set_ylabel(f"{row_label}\n$z$ (nm)")
        fig.colorbar(im, ax=ax, label=ylabel, shrink=0.85)

    axes[-1].set_xlabel(r"$x$ (nm)")

    if suptitle:
        fig.suptitle(suptitle, y=1.01)

    fig.tight_layout()
    plt.show()


def plot_epsilon(x_a, z_lab_a, film_a, eps_map):
    """
    Plot a 3x3 grid of epsilon tensor components over the (x, z) spatial domain.

    Parameters
    ----------
    x_a : np.ndarray
        1-D array of x positions in angstroms.
    z_lab_a : np.ndarray
        1-D array of z positions in angstroms (z=0 is the vacuum/film interface,
        z=film_a is the film/substrate interface).
    film_a : float
        Film thickness in angstroms.
    epsilon_tensor : np.ndarray
        Shape (N_layers, 3, 3) complex array. Layer order is
        [0] vacuum (above film), [1] film, [2] substrate (below z=0).
    """

    # ------------------------------------------------------------------
    # Plot – one subplot per tensor component
    # ------------------------------------------------------------------
    x_nm = x_a / 10.0
    z_nm = z_lab_a / 10.0
    film_nm = film_a / 10.0

    component_labels = [
        (r"$\varepsilon_{xx}$", r"$\varepsilon_{xy}$", r"$\varepsilon_{xz}$"),
        (r"$\varepsilon_{yx}$", r"$\varepsilon_{yy}$", r"$\varepsilon_{yz}$"),
        (r"$\varepsilon_{zx}$", r"$\varepsilon_{zy}$", r"$\varepsilon_{zz}$"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            data = eps_map[row, col]          # shape (nz, nx)

            # Split into real and imaginary – plot real part by default;
            # imaginary part is overlaid as a contour if non-trivial.
            real_part = data.real

            im = ax.pcolormesh(
                x_nm, z_nm, real_part,
                shading="auto",
                cmap="RdBu_r",
                vmin=real_part.min(),
                vmax=real_part.max(),
            )
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

            # Mark layer boundaries
            ax.axhline(0.0,      color="k", lw=0.8, ls="--", alpha=0.7)
            ax.axhline(film_nm,  color="k", lw=0.8, ls="--", alpha=0.7)

            ax.set_title(component_labels[row][col], fontsize=11)

            if col == 0:
                ax.set_ylabel(r"$z$ (nm)")
            if row == 2:
                ax.set_xlabel(r"$x$ (nm)")

    # Layer annotations on the left-most column
    for ax, label, ypos in zip(
            axes[:, 0],
            ["vacuum", "C₈H₈ film", "Si substrate"],
            [0.92, 0.50, 0.08],
    ):
        ax.text(
            0.02, ypos, label,
            transform=ax.transAxes,
            va="top", fontsize=7,
            bbox=dict(fc="white", ec="none", alpha=0.6),
        )

    fig.suptitle(r"Dielectric tensor $\varepsilon$ (real part)", y=1.01, fontsize=13)
    fig.tight_layout()
    plt.show()


def get_epsilon_map(epsilon_tensor: ndarray, film_a: float, x_a: ndarray,
                    z_lab_a: ndarray) -> ndarray[Any, dtype[floating[_64Bit]]] | \
                                         ndarray[Any, dtype[Any]]:
    # ------------------------------------------------------------------
    # Build the spatial epsilon grid: shape (3, 3, nz, nx)
    # ------------------------------------------------------------------
    nz = z_lab_a.size
    nx = x_a.size

    eps_map = np.zeros((3, 3, nz, nx), dtype=np.complex128)

    # Layer masks along z (broadcast over x later)
    # Layer 0 – vacuum:    z > film_a
    # Layer 1 – film:      0 <= z <= film_a
    # Layer 2 – substrate: z < 0
    layer_mask = np.empty(nz, dtype=int)
    layer_mask[z_lab_a > film_a] = 0
    layer_mask[(z_lab_a >= 0) & (z_lab_a <= film_a)] = 1
    layer_mask[z_lab_a < 0] = 2

    for iz, layer_idx in enumerate(layer_mask):
        eps_map[:, :, iz, :] = epsilon_tensor[layer_idx, :, :, np.newaxis]
    return eps_map


if __name__ == "__main__":
    e_fields = np.load("test_data/e_fields_qc.npz")

    # (z, x): (360, 320)
    plot_efield_maps(
        x_nm=e_fields['x_nm'],
        z_nm=e_fields['z_nm'],
        Ey_s=e_fields['Ey_s'],
        Ex_p=e_fields['Ex_p'],
        Ez_p=e_fields['Ez_p'],
        suptitle=rf"$q={e_fields['q']:.4f}\ \mathrm{{Å}}^{{-1}},\ \theta={e_fields['theta_deg']:.2f}^\circ,\ E={e_fields['energy_ev']:.1f}\ \mathrm{{eV}}$",
    )

    layers = build_layers()

    # slabs: [thickness, delta, beta, roughness] (N_layers, 4)
    slabs = np.asarray(stack_slabs(layers, energy=ENERGY_EV), dtype=np.float64)
    # tensor: epsilon = delta - j * beta (N_layers, 3, 3)
    tensor = np.asarray(stack_tensor(layers, energy=ENERGY_EV), dtype=np.complex128)

    x_nm = np.linspace(-80.0, 80.0, 320)
    z_nm = np.linspace(-80.0, 400.0, 360)
    x_a = x_nm * 10.0
    z_lab_a = z_nm * 10.0

    eps_map = get_epsilon_map(tensor, FILM_A, x_a, z_lab_a)
    plot_epsilon(x_a, z_lab_a, FILM_A, eps_map)

    wavelength_a = hc / ENERGY_EV
    delta_z_a = z_lab_a[1] - z_lab_a[0]
    delta_x_a = x_a[1] - x_a[0]

    Ey_s = torch.Tensor(e_fields['Ey_s'])
    Ex_p = torch.Tensor(e_fields['Ex_p'])
    Ez_p = torch.Tensor(e_fields['Ez_p'])
    E_p = torch.stack([Ez_p, Ex_p], dim=0)

    eps_map = torch.Tensor(eps_map)
    error_s = helmholtz_residual_loss(Ey_s, eps_map[0,0], "s", wavelength_a, delta_z_a, delta_x_a)
    error_p = helmholtz_residual_loss(E_p, eps_map[0], "p", wavelength_a, delta_z_a, delta_x_a)

    # trim the axes to match the (Nz-4, Nx-4) interior output
    x_nm_interior = x_nm[2:-2]
    z_nm_interior = z_nm[2:-2]

    error_s_abs = (error_s.abs()**2).cpu()
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        error_s_abs,
        extent=[x_nm_interior[0], x_nm_interior[-1], z_nm_interior[-1], z_nm_interior[0]],
        aspect="auto",
    )
    ax.invert_yaxis()
    fig.colorbar(im, ax=ax, label=r"$|\mathrm{residual}|^2$")
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$z$ (nm)")
    ax.set_title(r"Helmholtz residual — s-pol ($|E_y|$)")
    fig.tight_layout()
    plt.show()


    error_p_abs = (error_p[0].abs()**2 + error_p[1].abs()**2).cpu()
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        error_p_abs,
        extent=[x_nm_interior[0], x_nm_interior[-1], z_nm_interior[-1], z_nm_interior[0]],
        aspect="auto",
    )
    ax.invert_yaxis()
    fig.colorbar(im, ax=ax, label=r"$|\mathrm{residual}|^2$")
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$z$ (nm)")
    ax.set_title(r"Helmholtz residual — p-pol ($|E_z|^2 + |E_x|^2$)")
    fig.tight_layout()
    plt.show()




