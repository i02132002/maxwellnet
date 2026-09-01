"""Orientation profiles with deferred OOC lookup at evaluation energy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from refnx.analysis import Parameters, possibly_create_parameter

from refloxide.pxr.energy.scatterers import EnergyDependentUniTensorSLD
from refloxide.pxr.energy.structure import (
    EnergyDependentStructure,
    EnergyOrientationSlab,
)

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from refloxide.pxr.energy.ooc import OocAnchor


class AdaptiveOrientationScatterer(EnergyDependentUniTensorSLD):
    """Uniaxial material plus a depth-varying polar angle profile for fitting.

    Parameters
    ----------
    ooc
        Optical-constant table or :class:`~refloxide.pxr.energy.ooc.OocAnchor`.
    orientations_rad
        Initial polar angles in radians, one per sub-layer in the slab.
    density, rotation, energy_offset, name, interp
        Forwarded to ``EnergyDependentUniTensorSLD`` constructor arguments.
    """

    def __init__(
        self,
        ooc: pd.DataFrame | OocAnchor,
        orientations_rad: NDArray[np.float64],
        *,
        density: float = 1.0,
        rotation: float = 0.0,
        energy_offset: float = 0.0,
        name: str = "",
        interp: Literal["linear", "pchip"] = "linear",
    ) -> None:
        super().__init__(
            ooc,
            rotation=rotation,
            density=density,
            energy_offset=energy_offset,
            name=name,
            interp=interp,
        )
        angles = np.asarray(orientations_rad, dtype=np.float64).ravel()
        self._orientation_parameters = Parameters(name=f"{name}_orientations")
        for i, theta in enumerate(angles):
            p = possibly_create_parameter(
                float(theta),
                name=f"{name}_theta_{i}",
                vary=True,
                bounds=(-np.pi, np.pi),
            )
            self._orientation_parameters.append(p)
        self._parameters.extend(self._orientation_parameters)

    @property
    def orientations_rad(self) -> NDArray[np.float64]:
        """Current polar angles (rad), one per sub-layer."""
        return np.array(
            [float(p.value or 0.0) for p in self._orientation_parameters],
            dtype=np.float64,
        )

    def orientation_slab(
        self,
        thick: float,
        rough: float,
        *,
        name: str = "",
    ) -> EnergyOrientationSlab:
        """Build an orientation slab for stacking with ``|``."""
        return EnergyOrientationSlab(
            thick,
            self,
            self.orientations_rad,
            rough,
            name=name or self.name,
        )


def bookended_orientation_angles(
    n_sub: int,
    theta_start: float,
    theta_end: float,
) -> NDArray[np.float64]:
    """Linear polar-angle grid between ``theta_start`` and ``theta_end`` inclusive."""
    if n_sub < 1:
        msg = "n_sub must be positive"
        raise ValueError(msg)
    return np.linspace(theta_start, theta_end, n_sub, dtype=np.float64)


def attach_to_structure(
    structure: EnergyDependentStructure,
    slab: EnergyOrientationSlab,
) -> EnergyDependentStructure:
    """Return ``structure`` with ``slab`` appended (refnx ``|`` semantics)."""
    structure.append(slab)
    return structure
