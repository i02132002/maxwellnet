"""Convert legacy fixed-energy scatterers and structures to deferred-energy types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.energy.scatterers import (
    EnergyDependentMaterialSLD,
    EnergyDependentUniTensorSLD,
    FixedTensorScatterer,
)
from refloxide.pxr.energy.structure import EnergyDependentStructure
from refloxide.pxr.plugin.structure import (
    MaterialSLD,
    Scatterer,
    Slab,
    Structure,
    UniTensorSLD,
)

if TYPE_CHECKING:
    from refloxide.pxr.plugin.structure import PXR_Component


def upgrade_scatterer(scatterer: Scatterer):
    """Map a plugin scatterer to its energy-deferred counterpart.

    Parameters
    ----------
    scatterer
        :class:`~refloxide.pxr.plugin.structure.MaterialSLD`,
        :class:`~refloxide.pxr.plugin.structure.UniTensorSLD`, or an existing
        energy-dependent scatterer (returned unchanged).

    Returns
    -------
    EnergyDependentMaterialSLD, EnergyDependentUniTensorSLD, or FixedTensorScatterer
        Deferred-energy scatterer preserving parameters where possible.
    """
    if isinstance(
        scatterer,
        (EnergyDependentMaterialSLD, EnergyDependentUniTensorSLD, FixedTensorScatterer),
    ):
        return scatterer
    if isinstance(scatterer, MaterialSLD):
        density = float(scatterer.density.value or 1.0)
        off = float(scatterer.energy_offset.value or 0.0)
        out = EnergyDependentMaterialSLD(
            scatterer.formula,
            density=density,
            energy_offset=off,
            name=scatterer.name,
        )
        out.density.setp(vary=scatterer.density.vary, bounds=scatterer.density.bounds)
        out.energy_offset.setp(
            vary=scatterer.energy_offset.vary,
            bounds=scatterer.energy_offset.bounds,
        )
        return out
    if isinstance(scatterer, UniTensorSLD):
        anchor = OocAnchor(
            energy_ev=np.asarray(scatterer.n_xx.x, dtype=np.float64),
            n_xx=np.asarray(scatterer.n_xx.y, dtype=np.float64),
            n_ixx=np.asarray(scatterer.n_ixx.y, dtype=np.float64),
            n_zz=np.asarray(scatterer.n_zz.y, dtype=np.float64),
            n_izz=np.asarray(scatterer.n_izz.y, dtype=np.float64),
        )
        density = float(scatterer.density.value or 1.0)
        rotation = float(scatterer.rotation.value or 0.0)
        off = float(scatterer.energy_offset.value or 0.0)
        out = EnergyDependentUniTensorSLD(
            anchor,
            rotation=rotation,
            density=density,
            energy_offset=off,
            name=scatterer.name,
        )
        out.density.setp(vary=scatterer.density.vary, bounds=scatterer.density.bounds)
        out.rotation.setp(
            vary=scatterer.rotation.vary,
            bounds=scatterer.rotation.bounds,
        )
        out.energy_offset.setp(
            vary=scatterer.energy_offset.vary,
            bounds=scatterer.energy_offset.bounds,
        )
        return out
    if scatterer._tensor is not None:
        return FixedTensorScatterer(np.asarray(scatterer.tensor), name=scatterer.name)
    msg = f"Cannot migrate scatterer type {type(scatterer)!r}"
    raise TypeError(msg)


def _upgrade_component(component: PXR_Component) -> PXR_Component:
    if isinstance(component, Slab) and isinstance(component.sld, Scatterer):
        upgraded = upgrade_scatterer(component.sld)
        if upgraded is component.sld:
            return component
        return Slab(
            component.thick.value,
            upgraded,
            component.rough.value,
            name=component.name,
        )
    return component


def upgrade_structure(structure: Structure) -> EnergyDependentStructure:
    """Clone a plugin :class:`~refloxide.pxr.plugin.structure.Structure`.

    Scatterers are converted to deferred-energy types.

    Parameters
    ----------
    structure
        Source stack; fronting and backing must remain slabs.

    Returns
    -------
    EnergyDependentStructure
        New structure with migrated scatterers and preserved ``reverse_structure``.
    """
    components = [_upgrade_component(c) for c in structure.components]
    out = EnergyDependentStructure(
        *components,
        name=structure.name,
        reverse_structure=structure.reverse_structure,
    )
    try:
        sub_off = structure.energy_offset
        out.energy_offset = sub_off
    except (ValueError, AttributeError):
        pass
    return out
