"""Photon-energy evaluation context for deferred optical models."""

from __future__ import annotations

from dataclasses import dataclass

from refnx.analysis import Parameter, possibly_create_parameter


@dataclass(frozen=True, slots=True)
class EnergyProbe:
    """Effective photon energy for tabulated or dispersive optical models.

    Parameters
    ----------
    base_energy_ev
        Nominal experiment energy in eV (for example from a batched term or
        :class:`~refloxide.pxr.plugin.model.ReflectModel`).
    structure_offset_ev
        Global offset shared by the whole stack, typically
        :class:`~refloxide.pxr.plugin.model.ReflectModel` ``energy_offset``.
    component_offset_ev
        Per-scatterer offset in eV added on top of the structure offset.
    """

    base_energy_ev: float
    structure_offset_ev: float = 0.0
    component_offset_ev: float = 0.0

    @property
    def effective_ev(self) -> float:
        """Photon energy used for OOC lookup and dispersive formulas."""
        return (
            float(self.base_energy_ev)
            + float(self.structure_offset_ev)
            + float(self.component_offset_ev)
        )

    @classmethod
    def from_parameters(
        cls,
        base_energy_ev: float,
        structure_offset: float | Parameter | None = None,
        component_offset: float | Parameter | None = None,
    ) -> EnergyProbe:
        """Build a probe from refnx :class:`~refnx.analysis.Parameter` values."""
        struct_off = 0.0
        if structure_offset is not None:
            p = possibly_create_parameter(structure_offset, name="energy_offset")
            struct_off = float(p.value) if p.value is not None else 0.0
        comp_off = 0.0
        if component_offset is not None:
            p = possibly_create_parameter(component_offset, name="energy_offset")
            comp_off = float(p.value) if p.value is not None else 0.0
        return cls(
            base_energy_ev=float(base_energy_ev),
            structure_offset_ev=struct_off,
            component_offset_ev=comp_off,
        )
