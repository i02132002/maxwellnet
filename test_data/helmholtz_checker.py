import torch
from torch import Tensor
import torch.nn.functional as F
import math


def calculate_helmholtz_diff(
        E_field: Tensor,
        epsilon_map: Tensor,
        mode: str,
        wavelength_a: float,
        delta_x_a: float,
        delta_z_a: float,
) -> Tensor:
    """
    Compute the Helmholtz residual for a given total E-field and epsilon map,
    using staggered (Yee-grid) finite differences following MaxwellNet.

    The second derivative is computed as two sequential first-order steps:
        d_h( d_e(f) )
    where d_e uses a forward difference [0, -1, +1]/d
    and   d_h uses a backward difference [-1, +1, 0]/d.
    Each size-3 kernel with no padding shrinks the spatial dimension by 2,
    so two sequential steps shrink by 4 total. The residual is trimmed to
    the interior where all stencil points are valid.

    Args:
        E_field: Complex tensor of shape (Nz, Nx) or (batch, Nz, Nx).
                 - s-pol: Ey
                 - p-pol: (Ez, Ex) stacked as (2, Nz, Nx) or (batch, 2, Nz, Nx)
        epsilon_map: Complex tensor of shape (Nz, Nx) for s-pol,
                     or (2, Nz, Nx) [εz, εx] for p-pol.
        mode: 's' for s-polarisation (TE), 'p' for p-polarisation (TM).
        wavelength_a: Free-space wavelength in angstroms.
        delta_x_a: Grid spacing in angstroms along x (cols).
        delta_z_a: Grid spacing in angstroms along z (rows).

    Returns:
        Complex residual tensor of shape (Nz-4, Nx-4) for s-pol, or
        (2, Nz-4, Nx-4) for p-pol. Should be ~0 everywhere in the interior
        if E_field satisfies the Helmholtz equation for the given epsilon_map.
    """
    k = 2 * math.pi / wavelength_a  # Å⁻¹
    dx = delta_x_a
    dz = delta_z_a

    # ------------------------------------------------------------------
    # Staggered first-order kernels — conv2d called with padding=0
    # Each kernel has size 3, so each application shrinks that dimension by 2.
    # d_e: forward difference  [0, -1, +1] / d   (E-field nodes)
    # d_h: backward difference [-1, +1,  0] / d  (H-field nodes)
    # ------------------------------------------------------------------

    # along z (rows): kernel shape (1, 1, 3, 1)
    k_e_z = torch.tensor([[[[0.0], [-1.0], [1.0]]]], dtype=torch.complex64) / dz
    k_h_z = torch.tensor([[[[-1.0], [1.0], [0.0]]]], dtype=torch.complex64) / dz

    # along x (cols): kernel shape (1, 1, 1, 3)
    k_e_x = torch.tensor([[[[0.0, -1.0, 1.0]]]], dtype=torch.complex64) / dx
    k_h_x = torch.tensor([[[[-1.0, 1.0, 0.0]]]], dtype=torch.complex64) / dx

    def _d_e_z(f: Tensor) -> Tensor:
        """Forward ∂/∂z — no padding, output shrinks rows by 2: (Nz-2, Nx)."""
        return F.conv2d(f, k_e_z.to(f.device), padding=0)

    def _d_h_z(f: Tensor) -> Tensor:
        """Backward ∂/∂z — no padding, output shrinks rows by 2: (Nz-2, Nx)."""
        return F.conv2d(f, k_h_z.to(f.device), padding=0)

    def _d_e_x(f: Tensor) -> Tensor:
        """Forward ∂/∂x — no padding, output shrinks cols by 2: (Nz, Nx-2)."""
        return F.conv2d(f, k_e_x.to(f.device), padding=0)

    def _d_h_x(f: Tensor) -> Tensor:
        """Backward ∂/∂x — no padding, output shrinks cols by 2: (Nz, Nx-2)."""
        return F.conv2d(f, k_h_x.to(f.device), padding=0)

    def _lap_z(f: Tensor) -> Tensor:
        """∂²f/∂z² = d_h_z( d_e_z(f) ), output shape (Nz-4, Nx)."""
        return _d_h_z(_d_e_z(f))

    def _lap_x(f: Tensor) -> Tensor:
        """∂²f/∂x² = d_h_x( d_e_x(f) ), output shape (Nz, Nx-4)."""
        return _d_h_x(_d_e_x(f))

    def _d_x_d_z(f: Tensor) -> Tensor:
        """∂²f/∂x∂z = d_h_x( d_e_z(f) ), output shape (Nz-2, Nx-2)."""
        return _d_h_x(_d_e_z(f))

    def _d_z_d_x(f: Tensor) -> Tensor:
        """∂²f/∂z∂x = d_h_z( d_e_x(f) ), output shape (Nz-2, Nx-2)."""
        return _d_h_z(_d_e_x(f))

    def _ensure_4d(t: Tensor) -> Tensor:
        """(Nz, Nx) -> (1, 1, Nz, Nx)"""
        if t.dim() == 2:
            return t.unsqueeze(0).unsqueeze(0)
        if t.dim() == 3:
            return t.unsqueeze(0)
        return t

    # Interior slice matching the (Nz-4, Nx-4) output of the Laplacians
    interior = (slice(None), slice(None), slice(2, -2), slice(2, -2))

    if mode == 's':
        # ∂²Ey/∂z² + ∂²Ey/∂x² + k²·ε·Ey = 0
        Ey  = _ensure_4d(E_field.to(torch.complex64))
        eps = _ensure_4d(epsilon_map.to(torch.complex64))

        # _lap_z(Ey): (Nz-4, Nx)   -> trim cols by 2 each side -> (Nz-4, Nx-4)
        # _lap_x(Ey): (Nz,   Nx-4) -> trim rows by 2 each side -> (Nz-4, Nx-4)
        diff = (
                _lap_z(Ey)[:, :, :, 2:-2]
                + _lap_x(Ey)[:, :, 2:-2, :]
                + k**2 * eps[interior] * Ey[interior]
        )
        return diff.squeeze()

    elif mode == 'p':
        # Ez: ∂²Ez/∂x² - ∂²Ex/∂x∂z + k²·εz·Ez = 0
        # Ex: ∂²Ex/∂z² - ∂²Ez/∂z∂x + k²·εx·Ex = 0
        if E_field.dim() == 2:
            raise ValueError("p-pol E_field must have shape (2, Nz, Nx) [Ez, Ex]")

        E_field     = E_field.to(torch.complex64)
        epsilon_map = epsilon_map.to(torch.complex64)

        Ez    = _ensure_4d(E_field[0])
        Ex    = _ensure_4d(E_field[1])
        eps_z = _ensure_4d(epsilon_map[0])
        eps_x = _ensure_4d(epsilon_map[1])

        # _lap_x(Ez):   (Nz,   Nx-4) -> trim rows by 2 each side -> (Nz-4, Nx-4)
        # _d_x_d_z(Ex): (Nz-2, Nx-2) -> trim both by 1 each side -> (Nz-4, Nx-4)
        diff_z = (
                _lap_x(Ez)[:, :, 2:-2, :]
                - _d_x_d_z(Ex)[:, :, 1:-1, 1:-1]
                + k**2 * eps_z[interior] * Ez[interior]
        )

        # _lap_z(Ex):   (Nz-4, Nx)   -> trim cols by 2 each side -> (Nz-4, Nx-4)
        # _d_z_d_x(Ez): (Nz-2, Nx-2) -> trim both by 1 each side -> (Nz-4, Nx-4)
        diff_x = (
                _lap_z(Ex)[:, :, :, 2:-2]
                - _d_z_d_x(Ez)[:, :, 1:-1, 1:-1]
                + k**2 * eps_x[interior] * Ex[interior]
        )

        return torch.stack([diff_z.squeeze(), diff_x.squeeze()])

    else:
        raise ValueError(f"mode must be 's' or 'p', got '{mode}'")
