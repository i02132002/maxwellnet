"""Type stubs for the Rust-backed ``refloxide.rust`` extension module.

The native implementation is produced from ``src/lib.rs`` via PyO3 and
``maturin``. Function shapes and conventions mirror
``refloxide.pxr.tjf4x4.uniaxial_reflectivity`` so the two implementations
can be exchanged without changing call sites.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def uniaxial_reflectivity(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy: float,
    parallel: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Compute uniaxial 4x4 reflection and transmission for a stratified medium.

    Parameters
    ----------
    q
        Scattering wavevectors in inverse angstroms. Shape ``(numpnts,)``.
    layers
        Per-layer rows ``[d, sld_real, sld_imag, sigma]`` of shape
        ``(nlayers, 4)``. The first and last rows describe the fronting and
        the backing; the fronting and backing thicknesses are ignored.
    tensor
        Per-layer 3x3 dispersion tensor of shape ``(nlayers, 3, 3)``. The
        Berreman dielectric is built as ``eps = conj(I - 2 * tensor)``.
    energy
        Photon energy in eV.
    parallel
        When ``True`` (the default), the q-point loop runs on rayon's global
        thread pool. Pass ``False`` from within fitting routines that are
        themselves multi-threaded or multi-process (refnx workers, emcee
        walkers, ``multiprocessing.Pool``) to avoid CPU oversubscription.
        The pool size can also be capped globally with the environment
        variable ``RAYON_NUM_THREADS``.

    Returns
    -------
    refl
        Real power reflectance with shape ``(numpnts, 2, 2)``. Index
        layout matches ``refloxide.pxr.tjf4x4.uniaxial_reflectivity``:
        ``refl[:, 0, 0] = R_ss``, ``refl[:, 1, 1] = R_pp``,
        ``refl[:, 0, 1] = R_sp``, ``refl[:, 1, 0] = R_ps``.
    tran
        Complex amplitude transmission with the same index layout.

    Raises
    ------
    ValueError
        Layer count mismatch, fewer than two slabs, malformed input shapes,
        or non-finite or non-positive ``energy``.
    RuntimeError
        Dynamic matrix is singular at some (layer, q-index), with the
        offending indices reported in the exception message.
    """
    ...

def interp_ooc_linear(
    energy_ev: NDArray[np.float64],
    n_xx: NDArray[np.float64],
    n_ixx: NDArray[np.float64],
    n_zz: NDArray[np.float64],
    n_izz: NDArray[np.float64],
    query_ev: float,
) -> tuple[float, float, float, float]:
    """Piecewise-linear OOC lookup at one photon energy (eV).

    Returns ``(n_xx, n_ixx, n_zz, n_izz)``. Out-of-range ``query_ev`` clamps to
    the tabulated endpoints.
    """
    ...

def lab_tensor_diagonals_batch(
    n_mol_xx: complex,
    n_mol_zz: complex,
    orientations_rad: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Batch laboratory ``(3, 3)`` tensors for uniaxial molecular constants.

    Parameters
    ----------
    n_mol_xx, n_mol_zz
        Complex molecular indices along principal axes after density scaling.
    orientations_rad
        Polar rotations in radians, shape ``(n_sub,)``.

    Returns
    -------
    NDArray[np.complex128]
        Shape ``(n_sub, 3, 3)`` diagonal laboratory tensors.
    """
    ...

def isotropic_lab_tensor(n: complex) -> NDArray[np.complex128]:
    """Build a ``(3, 3)`` isotropic tensor with scalar index ``n`` on the diagonal."""
    ...

def bookended_uniaxial_reflectivity(
    q: NDArray[np.float64],
    energy_ev: NDArray[np.float64],
    n_xx: NDArray[np.float64],
    n_ixx: NDArray[np.float64],
    n_zz: NDArray[np.float64],
    n_izz: NDArray[np.float64],
    query_ev: float,
    total_thick: float,
    surface_roughness: float,
    tau_si: float,
    tau_vac: float,
    alpha_bulk: float,
    alpha_si: float,
    alpha_vac: float,
    density_bulk: float,
    density_si: float,
    density_vac: float,
    num_slabs: int,
    mesh_constant: float,
    fronting: NDArray[np.float64],
    backing: NDArray[np.float64],
    parallel: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Fused book-ended film + substrate reflectivity (GIL released).

    Builds the adaptive microslab mesh, OOC lookup, laboratory tensors, and
    uniaxial transfer-matrix solve entirely in Rust. ``fronting`` is one row
    ``[d, delta, beta, sigma]``; ``backing`` has shape ``(n_backing, 4)``.
    """
    ...
