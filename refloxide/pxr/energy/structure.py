"""Deferred-energy structures that materialize slabs and tensors at evaluation time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from refnx.analysis import Parameter, possibly_create_parameter

from refloxide.pxr.energy.probe import EnergyProbe
from refloxide.pxr.energy.scatterers import EnergyDependentScatterer
from refloxide.pxr.plugin.structure import PXR_Component, Slab, Stack, Structure

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class StackSnapshot:
    """Materialized layer geometry and optical tensors at one photon energy.

    Parameters
    ----------
    layers
        Rows ``[thickness_A, delta, beta, roughness_A]`` with shape ``(n_layers, 4)``.
    tensors
        Per-layer ``(3, 3)`` complex index tensors, shape ``(n_layers, 3, 3)``.
    energy_ev
        Effective photon energy used for the materialization (eV).
    """

    layers: NDArray[np.float64]
    tensors: NDArray[np.complex128]
    energy_ev: float


def _tensor_to_slab_row(
    thick: float,
    rough: float,
    tensor: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Convert a diagonal tensor to refnx ``[d, delta, beta, sigma]`` layout."""
    n_avg = (tensor[0, 0] + tensor[1, 1] + tensor[2, 2]) / 3.0
    delta = float((1.0 - n_avg).real)
    beta = float((1.0 - n_avg).imag)
    return np.array([thick, delta, beta, rough], dtype=np.float64)


class EnergyDependentStructure(Structure):
    """Structure stack whose scatterers resolve optical data only at evaluation energy.

    Parameters
    ----------
    components
        Initial :class:`~refloxide.pxr.plugin.structure.PXR_Component` sequence.
    name
        Human-readable label.
    reverse_structure
        When ``True``, reverse slab order like the plugin ``Structure`` class.
    structure_energy_offset
        Global offset in eV added to every probe before OOC lookup. Mirrors
        :class:`~refloxide.pxr.plugin.model.ReflectModel` ``energy_offset`` when set.
    """

    def __init__(
        self,
        *components: PXR_Component,
        name: str = "",
        reverse_structure: bool = False,
        structure_energy_offset: float | Parameter = 0.0,
    ) -> None:
        super().__init__(*components, name=name, reverse_structure=reverse_structure)
        self._structure_energy_offset = possibly_create_parameter(
            structure_energy_offset,
            name="structure_energy_offset",
            vary=True,
            bounds=(-1.0, 1.0),
        )

    @property
    def structure_energy_offset(self) -> Parameter:
        """Global energy offset (eV) applied before per-scatterer offsets."""
        return self._structure_energy_offset

    @structure_energy_offset.setter
    def structure_energy_offset(self, value: float | Parameter) -> None:
        self._structure_energy_offset = possibly_create_parameter(
            value,
            name="structure_energy_offset",
            vary=True,
            bounds=(-1.0, 1.0),
        )

    @property
    def energy_offset(self) -> Parameter:
        """Alias compatible with :class:`~refloxide.pxr.plugin.model.ReflectModel`."""
        return self.structure_energy_offset

    @energy_offset.setter
    def energy_offset(self, value: Parameter) -> None:
        self.structure_energy_offset = value
        for component in self.data:
            if hasattr(component, "sld") and hasattr(component.sld, "energy_offset"):
                component.sld.energy_offset.setp(
                    value=value.value,
                    vary=None,
                    bounds=value.bounds,
                    constraint=value,
                )

    def probe_at(self, base_energy_ev: float) -> EnergyProbe:
        """Build the probe for ``base_energy_ev`` plus structure offset."""
        off = self._structure_energy_offset.value
        struct_off = float(off) if off is not None else 0.0
        return EnergyProbe(
            base_energy_ev=float(base_energy_ev),
            structure_offset_ev=struct_off,
        )

    def _component_tensors(
        self,
        component: PXR_Component,
        probe: EnergyProbe,
    ) -> NDArray[np.complex128]:
        if isinstance(component, EnergyOrientationSlab):
            return component.tensors_at(probe)
        if isinstance(component, Slab):
            sld = component.sld
            if isinstance(sld, EnergyDependentScatterer):
                return np.asarray([sld.tensor_at(probe)], dtype=np.complex128)
            if energy := probe.effective_ev:
                t = component.tensor(energy=energy)
                return np.asarray([t], dtype=np.complex128)
            return np.asarray([component.tensor()], dtype=np.complex128)
        if isinstance(component, Stack):
            parts = [self._component_tensors(c, probe) for c in component.components]
            tensor = np.concatenate(parts, axis=0)
            repeats = round(abs(component.repeats.value))  # type: ignore[union-attr]
            if repeats > 1:
                tensor = np.concatenate([tensor] * repeats, axis=0)
            return tensor
        if hasattr(component, "tensor"):
            t = component.tensor(energy=probe.effective_ev)
            return np.asarray(t, dtype=np.complex128)
        msg = (
            "Unsupported component type for energy materialization: "
            f"{type(component)!r}"
        )
        raise TypeError(msg)

    def _component_layers(
        self,
        component: PXR_Component,
        probe: EnergyProbe,
    ) -> NDArray[np.float64]:
        if isinstance(component, EnergyOrientationSlab):
            return component.layers_at(probe)
        if isinstance(component, Slab):
            thick = float(component.thick.value or 0.0)
            rough = float(component.rough.value or 0.0)
            sld = component.sld
            if isinstance(sld, EnergyDependentScatterer):
                tensor = sld.tensor_at(probe)
                return np.asarray(
                    [_tensor_to_slab_row(thick, rough, tensor)],
                    dtype=np.float64,
                )
            row = component.slabs()
            return row if row is not None else np.empty((0, 4))
        if isinstance(component, Stack):
            parts = [self._component_layers(c, probe) for c in component.components]
            layers = np.concatenate(parts, axis=0)
            repeats = round(abs(component.repeats.value))  # type: ignore[union-attr]
            if repeats > 1:
                layers = np.concatenate([layers] * repeats, axis=0)
            return layers
        slabs = component.slabs(structure=self) if hasattr(component, "slabs") else None
        if slabs is None:
            return np.empty((0, 4))
        return np.asarray(slabs, dtype=np.float64)

    def materialize(self, base_energy_ev: float) -> StackSnapshot:
        """Resolve every layer at ``base_energy_ev`` plus configured offsets."""
        if not len(self):
            msg = "EnergyDependentStructure has no components"
            raise ValueError(msg)
        probe = self.probe_at(base_energy_ev)
        layer_parts = [self._component_layers(c, probe) for c in self.components]
        tensor_parts = [self._component_tensors(c, probe) for c in self.components]
        layers = np.concatenate(layer_parts, axis=0)
        tensors = np.concatenate(tensor_parts, axis=0)
        if self.reverse_structure:
            roughnesses = layers[1:, 3].copy()
            layers = np.flipud(layers)
            tensors = np.flip(tensors, axis=0)
            layers[1:, 3] = roughnesses[::-1]
            layers[0, 3] = 0.0
        return StackSnapshot(
            layers=layers,
            tensors=tensors,
            energy_ev=probe.effective_ev,
        )

    def tensor(self, energy: float | None = None) -> NDArray[np.complex128]:
        """Return tensors at ``energy`` (eV) with all configured offsets applied."""
        if energy is None:
            energy = 250.0
        return self.materialize(float(energy)).tensors

    def slabs(self) -> NDArray[np.float64] | None:
        """Slab rows at the structure's nominal substrate energy offset only.

        Prefer :meth:`materialize` when the experiment energy is known.
        """
        if not len(self):
            return None
        probe = self.probe_at(250.0)
        return self.materialize(probe.base_energy_ev).layers


class EnergyOrientationSlab(PXR_Component):
    """Depth-resolved uniaxial slab with orientations evaluated at one energy.

    Parameters
    ----------
    thick
        Total thickness in angstrom.
    scatterer
        ``EnergyDependentUniTensorSLD`` supplying tabulated OOC data.
    orientations_rad
        Polar angles in radians, one per sub-layer; length sets sub-layer count.
    rough
        Nevot-Croce roughness between this block and the layer below (angstrom).
    name
        Component label.
    """

    def __init__(
        self,
        thick: float,
        scatterer: EnergyDependentScatterer,
        orientations_rad: NDArray[np.float64],
        rough: float,
        *,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        from refloxide.pxr.energy.scatterers import EnergyDependentUniTensorSLD

        if not isinstance(scatterer, EnergyDependentUniTensorSLD):
            msg = "EnergyOrientationSlab requires EnergyDependentUniTensorSLD"
            raise TypeError(msg)
        self.thick = possibly_create_parameter(thick, name=f"{name}_thick")
        self.scatterer = scatterer
        self.orientations_rad = np.asarray(orientations_rad, dtype=np.float64).ravel()
        self.rough = possibly_create_parameter(rough, name=f"{name}_rough")
        n = self.orientations_rad.size
        if n < 1:
            msg = "orientations_rad must have at least one sample"
            raise ValueError(msg)
        self._sub_thick = float(thick) / float(n)
        self._parameters = scatterer.parameters

    def tensors_at(self, probe: EnergyProbe) -> NDArray[np.complex128]:
        """Sub-layer tensors with shape ``(n_sub, 3, 3)``."""
        return self.scatterer.tensor_batch_at(probe, self.orientations_rad)

    def layers_at(self, probe: EnergyProbe) -> NDArray[np.float64]:
        """Sub-layer slab rows; roughness applies on the bottom sub-layer only."""
        tensors = self.tensors_at(probe)
        rows = [
            _tensor_to_slab_row(
                self._sub_thick,
                float(self.rough.value or 0.0) if i == 0 else 0.0,
                tensors[i],
            )
            for i in range(tensors.shape[0])
        ]
        return np.stack(rows, axis=0)
