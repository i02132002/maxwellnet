"""Book-ended adaptive microslabs with deferred OOC lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from refnx.analysis import possibly_create_parameter

from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.energy.probe import EnergyProbe
from refloxide.pxr.plugin.structure import PXR_Component

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


def orientation_profile_bookended(
    total_thick: float,
    depth: float | NDArray[np.float64],
    tau_si: float,
    tau_vac: float,
    alpha_bulk: float,
    alpha_si: float,
    alpha_vac: float,
) -> float | NDArray[np.float64]:
    """Tilt angle (rad) vs depth for a vacuum|film|substrate book-ended stack."""
    if isinstance(depth, np.ndarray):
        z = np.asarray(depth, dtype=np.float64)
        term_vac = (alpha_vac - alpha_bulk) * np.exp(-z / tau_vac)
        term_si = (alpha_si - alpha_bulk) * np.exp(-(total_thick - z) / tau_si)
        return alpha_bulk + term_vac + term_si
    z = float(depth)
    term_vac = (alpha_vac - alpha_bulk) * np.exp(-z / tau_vac)
    term_si = (alpha_si - alpha_bulk) * np.exp(-(total_thick - z) / tau_si)
    return float(alpha_bulk + term_vac + term_si)


def density_profile_bookended(
    total_thick: float,
    depth: float | NDArray[np.float64],
    tau_si: float,
    tau_vac: float,
    rho_bulk: float,
    rho_si: float,
    rho_vac: float,
) -> float | NDArray[np.float64]:
    """Mass density vs depth for the same book-ended functional form as orientation."""
    if isinstance(depth, np.ndarray):
        z = np.asarray(depth, dtype=np.float64)
        term_vac = (rho_vac - rho_bulk) * np.exp(-z / tau_vac)
        term_si = (rho_si - rho_bulk) * np.exp(-(total_thick - z) / tau_si)
        return rho_bulk + term_vac + term_si
    z = float(depth)
    term_vac = (rho_vac - rho_bulk) * np.exp(-z / tau_vac)
    term_si = (rho_si - rho_bulk) * np.exp(-(total_thick - z) / tau_si)
    return float(rho_bulk + term_vac + term_si)


def adaptive_microslab_thicknesses(
    total_thick: float,
    num_slabs: int,
    mesh_constant: float,
) -> NDArray[np.float64]:
    """Refining mesh symmetric about mid-film; sums to ``total_thick``."""
    if num_slabs <= 1:
        return np.array([total_thick], dtype=np.float64)
    n_half = num_slabs // 2
    half_thick = total_thick / 2.0
    r = mesh_constant ** (1 / n_half)
    if num_slabs % 2 == 0:
        a = half_thick * (r - 1) / (r**n_half - 1)
        mesh_half = a * r ** np.arange(n_half)
        mesh = np.concatenate([mesh_half[::-1], mesh_half])
    else:
        center_share = total_thick / num_slabs
        half_sum = (total_thick - center_share) / 2.0
        a = half_sum * (r - 1) / (r**n_half - 1)
        mesh_half = a * r ** np.arange(n_half)
        center = total_thick - 2 * mesh_half.sum()
        mesh = np.concatenate([mesh_half[::-1], np.array([center]), mesh_half])
    remainder = total_thick - mesh.sum()
    mesh = mesh.copy()
    mesh[0] += remainder
    return mesh


def _assemble_diagonal_tensor(
    n_o: NDArray[np.complex128],
    n_e: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    tensor = np.zeros((n_o.size, 3, 3), dtype=np.complex128)
    tensor[:, 0, 0] = n_o
    tensor[:, 1, 1] = n_o
    tensor[:, 2, 2] = n_e
    return tensor


def _lab_tensor_diagonals(
    n_mol_xx: NDArray[np.complex128],
    n_mol_zz: NDArray[np.complex128],
    orientation_rad: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    cos2 = np.square(np.cos(orientation_rad))
    sin2 = np.square(np.sin(orientation_rad))
    n_o = (n_mol_xx * (1.0 + cos2) + n_mol_zz * sin2) / 2.0
    n_e = n_mol_xx * sin2 + n_mol_zz * cos2
    return n_o, n_e


class EnergyBookendedOrientationDensityProfile(PXR_Component):
    """Adaptive ZnPc film with book-ended angle and density on a refined grid.

    Drop-in replacement for ``AdaptiveBookendedOrientationDensityProfile`` that stores
    the full OOC table in an :class:`~refloxide.pxr.energy.ooc.OocAnchor` and resolves
    optical constants at evaluation energy via the Rust linear interpolator.

    Parameters match the refl-analysis profile helper; angles ``alpha_*`` are radians.
    """

    def __init__(
        self,
        ooc: pd.DataFrame | OocAnchor,
        total_thick: float,
        surface_roughness: float,
        density_bulk: float,
        density_si: float,
        density_vac: float,
        tau_si: float,
        tau_vac: float,
        alpha_bulk: float,
        alpha_si: float,
        alpha_vac: float,
        energy: float,
        energy_offset: float = 0.0,
        name: str = "",
        num_slabs: int = 20,
        mesh_constant: float = 0.1,
        *,
        interp: Literal["linear", "pchip"] = "linear",
    ) -> None:
        super().__init__(name=name)
        self.mesh_constant = mesh_constant
        self.num_slabs = int(num_slabs)
        if isinstance(ooc, OocAnchor):
            self._anchor = ooc
        else:
            self._anchor = OocAnchor.from_dataframe(ooc, interp=interp)
        self.total_thick = possibly_create_parameter(total_thick, name="total_thick")
        self.surface_roughness = possibly_create_parameter(
            surface_roughness,
            name="surface_roughness",
        )
        self.density_bulk = possibly_create_parameter(density_bulk, name="density_bulk")
        self.density_si = possibly_create_parameter(density_si, name="density_si")
        self.density_vac = possibly_create_parameter(density_vac, name="density_vac")
        self.tau_si = possibly_create_parameter(tau_si, name="tau_si")
        self.tau_vac = possibly_create_parameter(tau_vac, name="tau_vac")
        self.alpha_bulk = possibly_create_parameter(alpha_bulk, name="alpha_bulk")
        self.alpha_si = possibly_create_parameter(alpha_si, name="alpha_si")
        self.alpha_vac = possibly_create_parameter(alpha_vac, name="alpha_vac")
        self.energy_offset = possibly_create_parameter(
            energy_offset,
            name="energy_offset",
        )
        self._nominal_energy_ev = float(energy)
        self._ooc_cache_energy: float | None = None
        self._ooc_cache_values: tuple[float, float, float, float] | None = None
        self._parameters = super().parameters
        self._parameters.extend(
            [
                self.total_thick,
                self.surface_roughness,
                self.density_bulk,
                self.density_si,
                self.density_vac,
                self.tau_si,
                self.tau_vac,
                self.alpha_bulk,
                self.alpha_si,
                self.alpha_vac,
                self.energy_offset,
            ]
        )

    @property
    def anchor(self) -> OocAnchor:
        """Tabulated optical constants."""
        return self._anchor

    def probe_at(self, base_energy_ev: float | None = None) -> EnergyProbe:
        """Build the energy probe including this component's offset."""
        if base_energy_ev is None:
            base = float(self._nominal_energy_ev)
        else:
            base = float(base_energy_ev)
        off = float(self.energy_offset.value or 0.0)
        return EnergyProbe(base_energy_ev=base, component_offset_ev=off)

    @property
    def slab_thick(self) -> NDArray[np.float64]:
        """Microslab thicknesses (angstrom)."""
        return adaptive_microslab_thicknesses(
            float(self.total_thick.value or 0.0),
            self.num_slabs,
            self.mesh_constant,
        )

    @property
    def mid_points(self) -> NDArray[np.float64]:
        """Depth of each microslab center from the vacuum interface (angstrom)."""
        thicknesses = self.slab_thick
        cumulative = np.cumsum(thicknesses)
        return cumulative - thicknesses / 2.0

    @property
    def parameters(self) -> Any:
        return self._parameters

    def orientation(
        self,
        depth: NDArray[np.float64] | float,
    ) -> NDArray[np.float64] | float:
        """Polar angle (rad) at ``depth``."""
        return orientation_profile_bookended(
            float(self.total_thick.value or 0.0),
            depth,
            float(self.tau_si.value or 0.0),
            float(self.tau_vac.value or 0.0),
            float(self.alpha_bulk.value or 0.0),
            float(self.alpha_si.value or 0.0),
            float(self.alpha_vac.value or 0.0),
        )

    def local_density(
        self,
        depth: NDArray[np.float64] | float,
    ) -> NDArray[np.float64] | float:
        """Mass density (g/cm^3) at ``depth``."""
        return density_profile_bookended(
            float(self.total_thick.value or 0.0),
            depth,
            float(self.tau_si.value or 0.0),
            float(self.tau_vac.value or 0.0),
            float(self.density_bulk.value or 1.0),
            float(self.density_si.value or 1.0),
            float(self.density_vac.value or 1.0),
        )

    def cache_ooc_at(
        self,
        energy: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Precompute OOC indices at ``energy`` for repeated hot-loop calls."""
        probe = self.probe_at(energy)
        eff = probe.effective_ev
        if self._ooc_cache_energy == eff and self._ooc_cache_values is not None:
            return self._ooc_cache_values
        values = self._anchor.values_at(eff)
        self._ooc_cache_energy = eff
        self._ooc_cache_values = values
        return values

    def clear_ooc_cache(self) -> None:
        """Drop cached OOC values after anchor or energy-offset changes."""
        self._ooc_cache_energy = None
        self._ooc_cache_values = None

    def tensor(self, energy: float | None = None) -> NDArray[np.complex128]:
        """Microslab tensors ``(num_slabs, 3, 3)`` at ``energy`` (eV)."""
        n_xx, n_ixx, n_zz, n_izz = self.cache_ooc_at(energy)
        depth = self.mid_points
        ori = np.asarray(self.orientation(depth), dtype=np.float64)
        rho_local = np.asarray(self.local_density(depth), dtype=np.float64)
        n_mol_xx = rho_local * complex(n_xx, n_ixx)
        n_mol_zz = rho_local * complex(n_zz, n_izz)
        n_o, n_e = _lab_tensor_diagonals(
            np.asarray(n_mol_xx, dtype=np.complex128),
            np.asarray(n_mol_zz, dtype=np.complex128),
            ori,
        )
        return _assemble_diagonal_tensor(n_o, n_e)

    def slabs(self, structure=None) -> NDArray[np.float64]:  # noqa: ARG002
        """Refnx slab rows ``[d, delta, beta, sigma]`` for each microslab."""
        thicknesses = self.slab_thick
        tens = self.tensor()
        iso = np.trace(tens, axis1=1, axis2=2)
        out = np.zeros((self.num_slabs, 4), dtype=np.float64)
        out[:, 0] = thicknesses
        out[:, 1] = np.real(iso)
        out[:, 2] = np.imag(iso)
        out[0, 3] = float(self.surface_roughness.value or 0.0)
        return out


def bookended_from_three_slabs(
    vacuum_slab,
    surface_slab,
    bulk_slab,
    interface_slab,
    ooc: pd.DataFrame | OocAnchor,
    *,
    energy: float,
    energy_offset: float = 0.0,
    num_slabs: int = 24,
    mesh_constant: float = 0.1,
    name: str = "",
    interp: Literal["linear", "pchip"] = "linear",
) -> EnergyBookendedOrientationDensityProfile:
    """Build a book-ended film from the three legacy ``UniTensorSLD`` slabs.

    Parameters
    ----------
    vacuum_slab, surface_slab, bulk_slab, interface_slab
        Template slabs in vacuum-to-substrate order (surface = vacuum side).
    ooc
        Full OOC table; not cropped at construction.
    energy, energy_offset, num_slabs, mesh_constant, name, interp
        Forwarded to :class:`EnergyBookendedOrientationDensityProfile`.
    """
    del vacuum_slab
    total_thick = (
        float(surface_slab.thick.value or 0.0)
        + float(bulk_slab.thick.value or 0.0)
        + float(interface_slab.thick.value or 0.0)
    )
    return EnergyBookendedOrientationDensityProfile(
        ooc,
        total_thick=total_thick,
        surface_roughness=float(surface_slab.rough.value or 0.0),
        density_bulk=float(bulk_slab.sld.density.value or 1.0),
        density_si=float(interface_slab.sld.density.value or 1.0),
        density_vac=float(surface_slab.sld.density.value or 1.0),
        tau_si=float(interface_slab.thick.value or 1.0),
        tau_vac=float(surface_slab.thick.value or 1.0),
        alpha_bulk=float(bulk_slab.sld.rotation.value or 0.0),
        alpha_si=float(interface_slab.sld.rotation.value or 0.0),
        alpha_vac=float(surface_slab.sld.rotation.value or 0.0),
        energy=float(energy),
        energy_offset=energy_offset,
        name=name or f"ZnPc_{energy:.1f}",
        num_slabs=num_slabs,
        mesh_constant=mesh_constant,
        interp=interp,
    )
