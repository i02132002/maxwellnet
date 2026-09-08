"""Order-6 TE Helmholtz residual (Fornberg staggered taps).

Binds ``helmholtz_ordered.helmholtz_residual_loss_periodic_pml`` to
``staggered_fd(6)``. Does not change ``helmholtz_checker``.
"""

from torch import Tensor

from maxwell_losses.helmholtz_ordered import (
    helmholtz_residual_loss_periodic_pml as _residual,
)
from maxwell_losses.staggered_fd import staggered_fd

_FD = staggered_fd(6)


def helmholtz_residual_loss_periodic_pml(
    envelope: Tensor,
    epsilon_map: Tensor,
    incident: Tensor,
    kz,
    kx,
    mode: str,
    wavelength_a: float,
    delta_x_a: float,
    delta_z_a: float,
    pml_thickness: int,
    pml_order: float = 4.0,
) -> Tensor:
    """Scores the TE Helmholtz residual with staggered order-6 kernels.

    Parameters match ``helmholtz_checker.helmholtz_residual_loss_periodic_pml``.
    ``fd`` is bound to order 6 and is not a caller argument.

    Returns
    -------
    Tensor
        Complex residual, same shape as ``envelope``.

    Raises
    ------
    NotImplementedError
        When ``mode == 'p'``.
    ValueError
        When ``mode`` is not ``'s'`` or ``'p'``.
    """
    return _residual(
        envelope,
        epsilon_map,
        incident,
        kz,
        kx,
        mode,
        wavelength_a,
        delta_x_a,
        delta_z_a,
        pml_thickness,
        pml_order,
        fd=_FD,
    )
