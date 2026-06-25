"""Energy-deferred optical models for refloxide PXR stacks.

This package separates **definition** of materials and orientation profiles from
**evaluation** at a photon energy. Structure-level and per-scatterer
``energy_offset`` parameters shift tabulated OOC lookup; hot interpolation and
batch tensor assembly call into :mod:`refloxide.rust`.
"""

from refloxide.pxr.energy.bookended import (
    EnergyBookendedOrientationDensityProfile,
    bookended_from_three_slabs,
)
from refloxide.pxr.energy.fused import evaluate_fused_bookended_reflectivity
from refloxide.pxr.energy.migrate import upgrade_scatterer, upgrade_structure
from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.energy.orientation import (
    AdaptiveOrientationScatterer,
    attach_to_structure,
    bookended_orientation_angles,
)
from refloxide.pxr.energy.probe import EnergyProbe
from refloxide.pxr.energy.scatterers import (
    EnergyDependentMaterialSLD,
    EnergyDependentScatterer,
    EnergyDependentUniTensorSLD,
    FixedTensorScatterer,
)
from refloxide.pxr.energy.structure import (
    EnergyDependentStructure,
    EnergyOrientationSlab,
    StackSnapshot,
)

__all__ = [
    "AdaptiveOrientationScatterer",
    "EnergyBookendedOrientationDensityProfile",
    "EnergyDependentMaterialSLD",
    "EnergyDependentScatterer",
    "EnergyDependentStructure",
    "EnergyDependentUniTensorSLD",
    "EnergyOrientationSlab",
    "EnergyProbe",
    "FixedTensorScatterer",
    "OocAnchor",
    "StackSnapshot",
    "attach_to_structure",
    "bookended_from_three_slabs",
    "bookended_orientation_angles",
    "evaluate_fused_bookended_reflectivity",
    "upgrade_scatterer",
    "upgrade_structure",
]
