"""Public Python interface for refloxide.

The native extension lives in :mod:`refloxide.rust` (Rust / PyO3). This
package re-exports the small surface intended to replace ad hoc Python
pipelines such as :func:`refloxide.pxr.uniaxial_reflectivity` once the
stack representation and Passler pipeline are wired through.

* :func:`compute_amplitudes` — one-shot amplitude solve (stack, frequency,
  incidence).
* :func:`compute_field` — field reconstruction at a depth inside a layer.
* :mod:`refloxide.batch` — NumPy broadcasting over frequency and incidence (and
  optional legacy *q* grids).

Legacy EMpy-style helpers remain under :mod:`refloxide.pxr`.
"""

from __future__ import annotations

from refloxide import pxr

__all__ = [
    "__version__",
    "pxr",
]
__version__ = "0.1.4"
