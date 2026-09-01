"""Laboratory s/p channel layout for polarized ``(n_q, 2, 2)`` power matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def reflectmodel_layout(matrix: NDArray[Any]) -> NDArray[Any]:
    """Reorder native Jones powers onto legacy pyref ``ReflectModel`` diagonals.

    Native refloxide stores ``R_ss`` at ``[:,0,0]`` and ``R_pp`` at ``[:,1,1]``.
    Stock :class:`pyref.fitting.model.ReflectModel` reads ``pol='s'`` from
    ``[:,1,1]`` and ``pol='p'`` from ``[:,0,0]``. This permutation matches that
    legacy layout without changing :meth:`ReflectModel.model`.

    Prefer :func:`reflectivity_for_pol` with an unpermuted laboratory matrix when
    patching pyref for refloxide (preserves ``pol='sp'`` / ``pol='ps'`` ordering).
    """
    out = np.empty_like(matrix)
    out[:, 0, 0] = matrix[:, 1, 1]
    out[:, 0, 1] = matrix[:, 1, 0]
    out[:, 1, 0] = matrix[:, 0, 1]
    out[:, 1, 1] = matrix[:, 0, 0]
    return out


def apply_laboratory_scales(
    refl: NDArray[np.float64],
    scale_s: float,
    scale_p: float,
) -> None:
    """Apply ``scale_s`` and ``scale_p`` to ``R_ss`` and ``R_pp`` diagonals in place."""
    refl[:, 0, 0] = scale_s * refl[:, 0, 0]
    refl[:, 1, 1] = scale_p * refl[:, 1, 1]


def reflectivity_for_pol(
    pol: str,
    refl: NDArray[Any],
    qvals: NDArray[np.float64],
    qvals_1: NDArray[np.float64],
    qvals_2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract reflectivity for ``pol`` from a native refloxide Jones power matrix.

    Parameters
    ----------
    pol
        ``'s'``, ``'p'``, ``'sp'``, or ``'ps'``. For ``'sp'``, the first segment
        of the concatenated ``q`` axis (``qvals_1``) is s-in/s-out; for ``'ps'``,
        the first segment is p-in/p-out.
    refl
        Power reflectance ``(n_q, 2, 2)`` as returned by the uniaxial kernel
        (``[:,0,0] = R_ss``, ``[:,1,1] = R_pp``). Channel extraction follows
        :class:`pyref.fitting.model.ReflectModel` so ``pol='s'`` reads
        ``[:,1,1]`` and ``pol='p'`` reads ``[:,0,0]``, matching pyref datasets
        and combined ``sp`` / ``ps`` objectives.
    qvals, qvals_1, qvals_2
        Q grids from :meth:`pyref.fitting.model.ReflectModel._model`.

    Returns
    -------
    np.ndarray
        Reflectivity samples aligned with the requested ``pol`` mode.
    """
    if pol == "s":
        return np.asarray(refl[:, 1, 1], dtype=np.float64)
    if pol == "p":
        return np.asarray(refl[:, 0, 0], dtype=np.float64)
    if pol == "sp":
        spol = np.interp(qvals_1, qvals, refl[:, 1, 1])
        ppol = np.interp(qvals_2, qvals, refl[:, 0, 0])
        return np.concatenate([spol, ppol])
    if pol == "ps":
        spol = np.interp(qvals_2, qvals, refl[:, 1, 1])
        ppol = np.interp(qvals_1, qvals, refl[:, 0, 0])
        return np.concatenate([ppol, spol])
    msg = f"reflectivity_for_pol supports pol in ('s', 'p', 'sp', 'ps'); got {pol!r}"
    raise ValueError(msg)


apply_reflectmodel_scales = apply_laboratory_scales
