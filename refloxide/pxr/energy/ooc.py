"""Tabulated optical-constant curves with deferred energy lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

_OOC_COLUMNS = ("energy", "n_xx", "n_ixx", "n_zz", "n_izz")


@dataclass(slots=True)
class OocAnchor:
    """Sorted OOC table stored as contiguous ``float64`` arrays for fast lookup.

    Parameters
    ----------
    energy_ev
        Photon energies in eV, strictly increasing.
    n_xx, n_ixx, n_zz, n_izz
        Real optical constant components aligned with ``energy_ev``.
    interp
        ``'linear'`` uses the Rust kernel; ``'pchip'`` defers to SciPy for
        interactive refinement (slower in hot loops).
    """

    energy_ev: np.ndarray
    n_xx: np.ndarray
    n_ixx: np.ndarray
    n_zz: np.ndarray
    n_izz: np.ndarray
    interp: Literal["linear", "pchip"] = "linear"

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        interp: Literal["linear", "pchip"] = "linear",
    ) -> OocAnchor:
        """Build an anchor from a pandas table with standard OOC columns."""
        missing = [c for c in _OOC_COLUMNS if c not in frame.columns]
        if missing:
            msg = f"OOC dataframe missing columns: {missing}"
            raise ValueError(msg)
        ordered = frame.sort_values("energy").drop_duplicates(subset=["energy"])
        return cls(
            energy_ev=np.asarray(ordered["energy"], dtype=np.float64),
            n_xx=np.asarray(ordered["n_xx"], dtype=np.float64),
            n_ixx=np.asarray(ordered["n_ixx"], dtype=np.float64),
            n_zz=np.asarray(ordered["n_zz"], dtype=np.float64),
            n_izz=np.asarray(ordered["n_izz"], dtype=np.float64),
            interp=interp,
        )

    def values_at(self, energy_ev: float) -> tuple[float, float, float, float]:
        """Return ``(n_xx, n_ixx, n_zz, n_izz)`` at ``energy_ev``."""
        if self.interp == "pchip":
            from scipy.interpolate import PchipInterpolator

            return (
                float(PchipInterpolator(self.energy_ev, self.n_xx)(energy_ev)),
                float(PchipInterpolator(self.energy_ev, self.n_ixx)(energy_ev)),
                float(PchipInterpolator(self.energy_ev, self.n_zz)(energy_ev)),
                float(PchipInterpolator(self.energy_ev, self.n_izz)(energy_ev)),
            )
        from refloxide.rust import interp_ooc_linear

        n_xx, n_ixx, n_zz, n_izz = interp_ooc_linear(
            self.energy_ev,
            self.n_xx,
            self.n_ixx,
            self.n_zz,
            self.n_izz,
            float(energy_ev),
        )
        return float(n_xx), float(n_ixx), float(n_zz), float(n_izz)

    def to_dataframe(self) -> pd.DataFrame:
        """Export the anchor as a pandas table with standard OOC column names."""
        import pandas as pd

        return pd.DataFrame(
            {
                "energy": self.energy_ev,
                "n_xx": self.n_xx,
                "n_ixx": self.n_ixx,
                "n_zz": self.n_zz,
                "n_izz": self.n_izz,
            }
        )

    def molecular_index(
        self,
        energy_ev: float,
        density: float,
    ) -> tuple[complex, complex]:
        """Scaled uniaxial molecular indices ``(n_xx, n_zz)`` at ``energy_ev``."""
        n_xx, n_ixx, n_zz, n_izz = self.values_at(energy_ev)
        n_mol_xx = density * complex(n_xx, n_ixx)
        n_mol_zz = density * complex(n_zz, n_izz)
        return n_mol_xx, n_mol_zz
