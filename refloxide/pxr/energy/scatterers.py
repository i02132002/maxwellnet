"""Energy-deferred scatterers compatible with refloxide plugin structures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import periodictable as pt
import periodictable.xsf as xsf
from refnx.analysis import Parameters, possibly_create_parameter

from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.energy.probe import EnergyProbe
from refloxide.pxr.plugin.structure import Scatterer

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


class EnergyDependentScatterer(Scatterer):
    """Scatterer that resolves optical tensors only when an energy is supplied."""

    def tensor_at(
        self,
        probe: EnergyProbe,
    ) -> NDArray[np.complex128]:
        """Return the ``(3, 3)`` laboratory tensor at ``probe.effective_ev``."""
        raise NotImplementedError


class EnergyDependentMaterialSLD(EnergyDependentScatterer):
    """Dispersive isotropic material from a chemical formula (periodictable)."""

    def __init__(
        self,
        formula: str,
        density: float | None = None,
        *,
        energy_offset: float = 0.0,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self._formula = pt.formula(formula)
        self._compound = formula
        if density is None:
            from refloxide.pxr.plugin.structure import compound_density

            density = compound_density(formula)
        self.density = possibly_create_parameter(  # type: ignore[assignment]
            density, name=f"{name}_rho", vary=True, bounds=(0.0, 5.0 * float(density))
        )
        self.energy_offset = possibly_create_parameter(  # type: ignore[assignment]
            energy_offset, name=f"{name}_energy_offset", vary=True, bounds=(-1.0, 1.0)
        )
        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density, self.energy_offset])

    def tensor_at(self, probe: EnergyProbe) -> NDArray[np.complex128]:
        eff = EnergyProbe(
            base_energy_ev=probe.base_energy_ev,
            structure_offset_ev=probe.structure_offset_ev,
            component_offset_ev=float(self.energy_offset.value or 0.0),
        )
        energy_kev = eff.effective_ev * 1e-3
        sldc = xsf.index_of_refraction(
            self._formula,
            density=float(self.density.value or 1.0),
            energy=energy_kev,
        )
        if hasattr(sldc, "item"):
            sldc = sldc.item()
        n = complex(1.0) - complex(sldc)
        from refloxide.rust import isotropic_lab_tensor

        return np.asarray(isotropic_lab_tensor(n), dtype=np.complex128)

    def __repr__(self) -> str:
        return f"EnergyDependentMaterialSLD({self._compound!r}, name={self.name!r})"


class EnergyDependentUniTensorSLD(EnergyDependentScatterer):
    """Uniaxial OOC table with density, rotation, and per-material energy offset."""

    def __init__(
        self,
        ooc: pd.DataFrame | OocAnchor,
        *,
        rotation: float = 0.0,
        density: float = 1.0,
        energy_offset: float = 0.0,
        name: str = "",
        interp: Literal["linear", "pchip"] = "linear",
    ) -> None:
        super().__init__(name=name)
        if isinstance(ooc, OocAnchor):
            anchor = ooc
        else:
            anchor = OocAnchor.from_dataframe(ooc, interp=interp)
        self._anchor = anchor
        self.density = possibly_create_parameter(  # type: ignore[assignment]
            density,
            name=f"{name}_density",
            vary=True,
            bounds=(0.0, 5.0 * float(density)),
        )
        self.rotation = possibly_create_parameter(  # type: ignore[assignment]
            rotation, name=f"{name}_rotation", vary=True, bounds=(-np.pi, np.pi)
        )
        self.energy_offset = possibly_create_parameter(  # type: ignore[assignment]
            energy_offset, name=f"{name}_energy_offset", vary=True, bounds=(-0.01, 0.01)
        )
        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density, self.rotation, self.energy_offset])

    @property
    def anchor(self) -> OocAnchor:
        """Tabulated optical constants backing this scatterer."""
        return self._anchor

    def tensor_at(self, probe: EnergyProbe) -> NDArray[np.complex128]:
        eff = EnergyProbe(
            base_energy_ev=probe.base_energy_ev,
            structure_offset_ev=probe.structure_offset_ev,
            component_offset_ev=float(self.energy_offset.value or 0.0),
        )
        rho = float(self.density.value or 1.0)
        theta = float(self.rotation.value or 0.0)
        n_mol_xx, n_mol_zz = self._anchor.molecular_index(eff.effective_ev, rho)
        cos2 = float(np.cos(theta) ** 2)
        sin2 = 1.0 - cos2
        n_o = (n_mol_xx * (1.0 + cos2) + n_mol_zz * sin2) / 2.0
        n_e = n_mol_xx * sin2 + n_mol_zz * cos2
        return np.diag([n_o, n_o, n_e]).astype(np.complex128)

    def tensor_batch_at(
        self,
        probe: EnergyProbe,
        orientations_rad: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        """Vectorized laboratory tensors for a depth profile at one energy."""
        eff = EnergyProbe(
            base_energy_ev=probe.base_energy_ev,
            structure_offset_ev=probe.structure_offset_ev,
            component_offset_ev=float(self.energy_offset.value or 0.0),
        )
        rho = float(self.density.value or 1.0)
        n_mol_xx, n_mol_zz = self._anchor.molecular_index(eff.effective_ev, rho)
        from refloxide.rust import lab_tensor_diagonals_batch

        return np.asarray(
            lab_tensor_diagonals_batch(n_mol_xx, n_mol_zz, orientations_rad),
            dtype=np.complex128,
        )

    def __repr__(self) -> str:
        return f"EnergyDependentUniTensorSLD(name={self.name!r})"


class FixedTensorScatterer(EnergyDependentScatterer):
    """Constant ``(3, 3)`` reference tensor independent of energy."""

    def __init__(
        self,
        tensor: NDArray[np.complex128],
        *,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self._tensor = np.asarray(tensor, dtype=np.complex128)
        if self._tensor.shape != (3, 3):
            msg = "FixedTensorScatterer requires a (3, 3) array"
            raise ValueError(msg)
        self._parameters = Parameters(name=name)

    def tensor_at(self, probe: EnergyProbe) -> NDArray[np.complex128]:
        del probe
        tensor = self._tensor
        assert tensor is not None
        return tensor.copy()
