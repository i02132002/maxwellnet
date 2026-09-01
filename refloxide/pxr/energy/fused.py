"""Fused Rust evaluation for book-ended graded uniaxial stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from refloxide.pxr.energy.bookended import EnergyBookendedOrientationDensityProfile

if TYPE_CHECKING:
    from numpy.typing import NDArray


def find_bookended_profile(
    structure: Any,
) -> tuple[int, EnergyBookendedOrientationDensityProfile] | None:
    """Locate the single energy book-ended profile in a pyref/refnx structure."""
    for i, component in enumerate(structure.components):
        if isinstance(component, EnergyBookendedOrientationDensityProfile):
            return i, component
    return None


def _component_slab_rows(component: Any) -> NDArray[np.float64]:
    slabs = component.slabs() if hasattr(component, "slabs") else None
    if slabs is None:
        return np.empty((0, 4), dtype=np.float64)
    arr = np.asarray(slabs, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, 4)
    return arr


def stack_fronting_row(structure: Any) -> NDArray[np.float64]:
    """Fronting slab row ``[d, delta, beta, sigma]`` from the first component."""
    rows = _component_slab_rows(structure.components[0])
    if rows.size == 0:
        return np.zeros(4, dtype=np.float64)
    return rows[0]


def stack_backing_rows(structure: Any, bookended_index: int) -> NDArray[np.float64]:
    """Concatenate slab rows for all components after the graded film."""
    parts: list[NDArray[np.float64]] = []
    for component in structure.components[bookended_index + 1 :]:
        rows = _component_slab_rows(component)
        if rows.size:
            parts.append(rows)
    if not parts:
        return np.zeros((0, 4), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def bookended_profile_params(
    profile: EnergyBookendedOrientationDensityProfile,
) -> dict[str, float | int]:
    """Scalar book-ended parameters for the Rust fused kernel."""
    return {
        "total_thick": float(profile.total_thick.value or 0.0),
        "surface_roughness": float(profile.surface_roughness.value or 0.0),
        "tau_si": float(profile.tau_si.value or 0.0),
        "tau_vac": float(profile.tau_vac.value or 0.0),
        "alpha_bulk": float(profile.alpha_bulk.value or 0.0),
        "alpha_si": float(profile.alpha_si.value or 0.0),
        "alpha_vac": float(profile.alpha_vac.value or 0.0),
        "density_bulk": float(profile.density_bulk.value or 1.0),
        "density_si": float(profile.density_si.value or 0.0),
        "density_vac": float(profile.density_vac.value or 0.0),
        "num_slabs": int(profile.num_slabs),
        "mesh_constant": float(profile.mesh_constant),
    }


def effective_energy_ev(
    profile: EnergyBookendedOrientationDensityProfile,
    base_energy_ev: float,
    structure_energy_offset: float = 0.0,
) -> float:
    """Photon energy (eV) including structure and profile offsets."""
    probe = profile.probe_at(base_energy_ev + structure_energy_offset)
    return probe.effective_ev


def evaluate_fused_bookended_reflectivity(
    q: NDArray[np.float64],
    structure: Any,
    energy: float,
    *,
    structure_energy_offset: float = 0.0,
    parallel: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]] | None:
    """Evaluate reflectivity on the fused Rust path when the stack qualifies.

    Returns ``None`` when ``structure`` is not a single
    :class:`~refloxide.pxr.energy.bookended.EnergyBookendedOrientationDensityProfile`
    between a fronting slab and one or more backing slabs.
    """
    located = find_bookended_profile(structure)
    if located is None:
        return None
    idx, profile = located
    if profile.anchor.interp != "linear":
        return None
    from refloxide.rust import bookended_uniaxial_reflectivity

    anchor = profile.anchor
    query_ev = effective_energy_ev(profile, float(energy), structure_energy_offset)
    profile.cache_ooc_at(query_ev)
    params = bookended_profile_params(profile)
    fronting = stack_fronting_row(structure)
    backing = stack_backing_rows(structure, idx)
    if backing.shape[0] < 1:
        return None
    num_slabs = int(params["num_slabs"])
    refl, tran = bookended_uniaxial_reflectivity(
        np.asarray(q, dtype=np.float64),
        anchor.energy_ev,
        anchor.n_xx,
        anchor.n_ixx,
        anchor.n_zz,
        anchor.n_izz,
        query_ev,
        fronting=np.asarray(fronting, dtype=np.float64),
        backing=np.asarray(backing, dtype=np.float64),
        parallel=parallel,
        total_thick=params["total_thick"],
        surface_roughness=params["surface_roughness"],
        tau_si=params["tau_si"],
        tau_vac=params["tau_vac"],
        alpha_bulk=params["alpha_bulk"],
        alpha_si=params["alpha_si"],
        alpha_vac=params["alpha_vac"],
        density_bulk=params["density_bulk"],
        density_si=params["density_si"],
        density_vac=params["density_vac"],
        num_slabs=num_slabs,
        mesh_constant=params["mesh_constant"],
    )
    return np.asarray(refl, dtype=np.float64), np.asarray(tran, dtype=np.complex128)
