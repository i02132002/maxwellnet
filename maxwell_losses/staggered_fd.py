"""Injected staggered first-derivative kernels for Helmholtz residuals.

This module owns MaxwellNet-style Yee taps (order 2 / 4 / 6) as a frozen
value object. It does not score Helmholtz residuals or own boundary
conditions. Order 2 matches ``helmholtz_checker``; order 4 matches
MaxwellNet ``high_order='fourth'``; order 6 uses the same staggered
pattern with Fornberg weights.

Callers pass a ``StaggeredFD`` into ``helmholtz_ordered`` rather than
editing ``helmholtz_checker``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_E_COEFFS: dict[int, tuple[float, ...]] = {
    2: (0.0, -1.0, 1.0),
    4: (0.0, 1.0 / 24.0, -9.0 / 8.0, 9.0 / 8.0, -1.0 / 24.0),
    6: (
        0.0,
        -3.0 / 640.0,
        25.0 / 384.0,
        -75.0 / 64.0,
        75.0 / 64.0,
        -25.0 / 384.0,
        3.0 / 640.0,
    ),
}
_H_COEFFS: dict[int, tuple[float, ...]] = {
    2: (-1.0, 1.0, 0.0),
    4: (1.0 / 24.0, -9.0 / 8.0, 9.0 / 8.0, -1.0 / 24.0, 0.0),
    6: (
        -3.0 / 640.0,
        25.0 / 384.0,
        -75.0 / 64.0,
        75.0 / 64.0,
        -25.0 / 384.0,
        3.0 / 640.0,
        0.0,
    ),
}


@dataclass(frozen=True, slots=True)
class StaggeredFD:
    """Staggered first-derivative taps and the pad width they require.

    Parameters
    ----------
    order : int
        Formal accuracy order (``2``, ``4``, or ``6``).
    pad : int
        Samples to ghost on each side so two sequential convolutions
        restore the original axis length. Equal to ``order`` for these
        stencils.
    e_coeffs : tuple of float
        Leading-zero MaxwellNet ``gradient_e`` taps (unscaled).
    h_coeffs : tuple of float
        Trailing-zero MaxwellNet ``gradient_h`` taps (unscaled).
    """

    order: int
    pad: int
    e_coeffs: tuple[float, ...]
    h_coeffs: tuple[float, ...]

    def kernel_e_z(self, dz: float, device: torch.device | str | None = None) -> Tensor:
        """Builds the staggered forward-like ``d/dz`` kernel, shape ``(1,1,K,1)``.

        Parameters
        ----------
        dz : float
            Sample spacing along z. Must be nonzero.
        device : torch.device or str or None, optional
            Device for the returned kernel.

        Returns
        -------
        Tensor
            Complex ``(1, 1, K, 1)`` convolution kernel scaled by ``1/dz``.
        """
        return (
            torch.tensor(
                [[[[c] for c in self.e_coeffs]]],
                dtype=torch.complex64,
                device=device,
            )
            / dz
        )

    def kernel_h_z(self, dz: float, device: torch.device | str | None = None) -> Tensor:
        """Builds the staggered backward-like ``d/dz`` kernel, shape ``(1,1,K,1)``.

        Parameters
        ----------
        dz : float
            Sample spacing along z. Must be nonzero.
        device : torch.device or str or None, optional
            Device for the returned kernel.

        Returns
        -------
        Tensor
            Complex ``(1, 1, K, 1)`` convolution kernel scaled by ``1/dz``.
        """
        return (
            torch.tensor(
                [[[[c] for c in self.h_coeffs]]],
                dtype=torch.complex64,
                device=device,
            )
            / dz
        )

    def kernel_e_x(self, dx: float, device: torch.device | str | None = None) -> Tensor:
        """Builds the staggered forward-like ``d/dx`` kernel, shape ``(1,1,1,K)``.

        Parameters
        ----------
        dx : float
            Sample spacing along x. Must be nonzero.
        device : torch.device or str or None, optional
            Device for the returned kernel.

        Returns
        -------
        Tensor
            Complex ``(1, 1, 1, K)`` convolution kernel scaled by ``1/dx``.
        """
        return (
            torch.tensor(
                [[[[*self.e_coeffs]]]],
                dtype=torch.complex64,
                device=device,
            )
            / dx
        )

    def kernel_h_x(self, dx: float, device: torch.device | str | None = None) -> Tensor:
        """Builds the staggered backward-like ``d/dx`` kernel, shape ``(1,1,1,K)``.

        Parameters
        ----------
        dx : float
            Sample spacing along x. Must be nonzero.
        device : torch.device or str or None, optional
            Device for the returned kernel.

        Returns
        -------
        Tensor
            Complex ``(1, 1, 1, K)`` convolution kernel scaled by ``1/dx``.
        """
        return (
            torch.tensor(
                [[[[*self.h_coeffs]]]],
                dtype=torch.complex64,
                device=device,
            )
            / dx
        )


def staggered_fd(order: int) -> StaggeredFD:
    """Returns the frozen staggered stencil for ``order`` in ``(2, 4, 6)``.

    Parameters
    ----------
    order : int
        ``2``, ``4``, or ``6``.

    Returns
    -------
    StaggeredFD
        Taps and pad width for two-step Laplacian composition.

    Raises
    ------
    ValueError
        When ``order`` is not one of ``(2, 4, 6)``.
    """
    if order not in _E_COEFFS:
        raise ValueError(f"order must be one of {tuple(_E_COEFFS)}, got {order}")
    return StaggeredFD(
        order=order,
        pad=order,
        e_coeffs=_E_COEFFS[order],
        h_coeffs=_H_COEFFS[order],
    )
