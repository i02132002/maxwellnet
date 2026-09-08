"""Discrete Helmholtz residuals for MaxwellNet-style staggered finite differences.

This module scores ``∇²E + k² ε E`` on complex fields using the same Yee-style
staggered first-derivative composition as training (orders 2, 4, and 6 on TE).
It owns interior residuals, periodic-x / transparent-z residuals with PML, and
a plane-wave Laplacian validator. It does not own network training, FEM
solvers, or dataset I/O.

Collaborators must keep kernel taps, pad widths, and Bloch/transparent ghosts
consistent with the chosen ``fd_order``. p-pol mixed derivatives remain
second-order only.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


_FD_ORDERS = (2, 4, 6)


def _fd_pad(fd_order: int) -> int:
    """Ghost/pad width for staggered composition at the given FD order.

    Parameters
    ----------
    fd_order : int
        ``2``, ``4``, or ``6`` (MaxwellNet-style staggered first derivatives).

    Returns
    -------
    int
        Samples to pad on each side so two sequential first derivatives
        restore the original axis length (``2`` / ``4`` / ``6``).
    """
    if fd_order not in _FD_ORDERS:
        raise ValueError(f"fd_order must be one of {_FD_ORDERS}, got {fd_order}")
    return int(fd_order)


def _staggered_e_coeffs(fd_order: int) -> list[float]:
    """Leading-zero staggered first-derivative taps (MaxwellNet ``gradient_e``)."""
    if fd_order == 2:
        return [0.0, -1.0, 1.0]
    if fd_order == 4:
        return [0.0, 1.0 / 24.0, -9.0 / 8.0, 9.0 / 8.0, -1.0 / 24.0]
    if fd_order == 6:
        return [
            0.0,
            -3.0 / 640.0,
            25.0 / 384.0,
            -75.0 / 64.0,
            75.0 / 64.0,
            -25.0 / 384.0,
            3.0 / 640.0,
        ]
    raise ValueError(f"fd_order must be one of {_FD_ORDERS}, got {fd_order}")


def _staggered_h_coeffs(fd_order: int) -> list[float]:
    """Trailing-zero staggered first-derivative taps (MaxwellNet ``gradient_h``)."""
    if fd_order == 2:
        return [-1.0, 1.0, 0.0]
    if fd_order == 4:
        return [1.0 / 24.0, -9.0 / 8.0, 9.0 / 8.0, -1.0 / 24.0, 0.0]
    if fd_order == 6:
        return [
            -3.0 / 640.0,
            25.0 / 384.0,
            -75.0 / 64.0,
            75.0 / 64.0,
            -25.0 / 384.0,
            3.0 / 640.0,
            0.0,
        ]
    raise ValueError(f"fd_order must be one of {_FD_ORDERS}, got {fd_order}")


def _kernel_e_z(dz: float, device=None, fd_order: int = 2) -> Tensor:
    """Staggered forward-like ∂/∂z kernel, shape ``(1,1,K,1)``."""
    coeffs = _staggered_e_coeffs(fd_order)
    return (
        torch.tensor([[[[c] for c in coeffs]]], dtype=torch.complex64, device=device)
        / dz
    )


def _kernel_h_z(dz: float, device=None, fd_order: int = 2) -> Tensor:
    """Staggered backward-like ∂/∂z kernel, shape ``(1,1,K,1)``."""
    coeffs = _staggered_h_coeffs(fd_order)
    return (
        torch.tensor([[[[c] for c in coeffs]]], dtype=torch.complex64, device=device)
        / dz
    )


def _kernel_e_x(dx: float, device=None, fd_order: int = 2) -> Tensor:
    """Staggered forward-like ∂/∂x kernel, shape ``(1,1,1,K)``."""
    coeffs = _staggered_e_coeffs(fd_order)
    return torch.tensor([[[[*coeffs]]]], dtype=torch.complex64, device=device) / dx


def _kernel_h_x(dx: float, device=None, fd_order: int = 2) -> Tensor:
    """Staggered backward-like ∂/∂x kernel, shape ``(1,1,1,K)``."""
    coeffs = _staggered_h_coeffs(fd_order)
    return torch.tensor([[[[*coeffs]]]], dtype=torch.complex64, device=device) / dx


def _ensure_4d(t: Tensor) -> Tensor:
    """(Nz, Nx) -> (1, 1, Nz, Nx); (B, Nz, Nx) -> (B, 1, Nz, Nx)."""
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 3:
        return t.unsqueeze(1)
    return t


def _pml_stretch_1d(
    n: int,
    delta: float,
    pml_thickness: int,
    pml_order: float = 4.0,
    pml_strength: float = 5.0,
    device=None,
) -> Tensor:
    """
    Complex coordinate-stretching PML profile 1/s(z), following the same
    polynomial-graded construction as MaxwellNet's `rz_inverse` (m=4, const=5
    by default), generalized to an arbitrary axis length `n` and physical
    sample spacing `delta`. The interior (more than `pml_thickness` samples
    from either edge) is exactly 1 (no effect); each edge grades smoothly to
    an absorbing profile over the outer `pml_thickness` samples.

    The polynomial is built from d/d_pml — the distance into the PML band
    divided by the PML's own physical width (pml_thickness * delta) — not
    the absolute distance `d` to the `pml_order` power. That ratio is
    dimensionless by construction, so `pml_strength` (sigma_max, the
    stretching factor exactly at the outer edge) is a portable, unitless
    number regardless of what length unit `delta` happens to be in.
    Building it from absolute `d` instead (MaxwellNet's own `rz_inverse`
    does this, in micrometers) makes the constant implicitly carry units of
    [length]^-pml_order: reusing pml_strength=5 with `delta` in angstroms
    (this module) rather than micrometers overshoots by (1e4)**4 = 1e16,
    since 1 um = 1e4 angstrom — the profile then jumps from ~1 to
    ~1 + 3125i within the first few angstroms of the PML rather than
    grading smoothly across its full width.

    Args:
        n: axis length in samples.
        delta: physical sample spacing (same length unit as pml_thickness*delta).
        pml_thickness: PML width, in samples, at each edge.
        pml_order: grading polynomial order (MaxwellNet default: 4).
        pml_strength: dimensionless stretching factor at the outer edge
            (sigma_max; MaxwellNet default: 5).
    """
    idx = torch.arange(n, device=device, dtype=torch.float32) - (n - 1) / 2.0
    z = idx * delta
    d_pml = pml_thickness * delta

    r_p = 1 + 1j * pml_strength * ((z - z[-1] + d_pml) / d_pml) ** pml_order
    r_p[: n - pml_thickness] = 0

    r_n = 1 + 1j * pml_strength * ((z - z[0] - d_pml) / d_pml) ** pml_order
    r_n[pml_thickness:] = 0

    r = r_p + r_n
    r[pml_thickness : n - pml_thickness] = 1
    return (1.0 / r).to(torch.complex64)


def helmholtz_residual_loss(
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

    def _d_e_z(f: Tensor) -> Tensor:
        """Forward ∂/∂z — no padding, output shrinks rows by 2: (Nz-2, Nx)."""
        return F.conv2d(f, _kernel_e_z(dz, f.device), padding=0)

    def _d_h_z(f: Tensor) -> Tensor:
        """Backward ∂/∂z — no padding, output shrinks rows by 2: (Nz-2, Nx)."""
        return F.conv2d(f, _kernel_h_z(dz, f.device), padding=0)

    def _d_e_x(f: Tensor) -> Tensor:
        """Forward ∂/∂x — no padding, output shrinks cols by 2: (Nz, Nx-2)."""
        return F.conv2d(f, _kernel_e_x(dx, f.device), padding=0)

    def _d_h_x(f: Tensor) -> Tensor:
        """Backward ∂/∂x — no padding, output shrinks cols by 2: (Nz, Nx-2)."""
        return F.conv2d(f, _kernel_h_x(dx, f.device), padding=0)

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

    # Interior slice matching the (Nz-4, Nx-4) output of the Laplacians
    interior = (slice(None), slice(None), slice(2, -2), slice(2, -2))

    if mode == "s":
        # ∂²Ey/∂z² + ∂²Ey/∂x² + k²·ε·Ey = 0
        Ey = _ensure_4d(E_field.to(torch.complex64))
        eps = _ensure_4d(epsilon_map.to(torch.complex64))

        # _lap_z(Ey): (Nz-4, Nx)   -> trim cols by 2 each side -> (Nz-4, Nx-4)
        # _lap_x(Ey): (Nz,   Nx-4) -> trim rows by 2 each side -> (Nz-4, Nx-4)
        diff = (
            _lap_z(Ey)[:, :, :, 2:-2]
            + _lap_x(Ey)[:, :, 2:-2, :]
            + k**2 * eps[interior] * Ey[interior]
        )
        return diff.squeeze()

    elif mode == "p":
        # Ez: ∂²Ez/∂x² - ∂²Ex/∂x∂z + k²·εz·Ez = 0
        # Ex: ∂²Ex/∂z² - ∂²Ez/∂z∂x + k²·εx·Ex = 0
        if E_field.dim() == 2:
            raise ValueError("p-pol E_field must have shape (2, Nz, Nx) [Ez, Ex]")

        E_field = E_field.to(torch.complex64)
        epsilon_map = epsilon_map.to(torch.complex64)

        Ez = _ensure_4d(E_field[0])
        Ex = _ensure_4d(E_field[1])
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
    fd_order: int = 2,
) -> Tensor:
    """
    Helmholtz residual with periodic-x (Bloch) and transparent-z boundaries
    for TE, matching the training loss path in ``train_maxwellnet``.

    Reconstructs ``E = incident * (1 + envelope)`` and returns the residual of
    ``∇²E + k² ε E`` over the full domain. The Laplacian uses staggered
    first-derivative composition of order ``fd_order`` (``2``, ``4``, or ``6``).
    Order ``4`` matches MaxwellNet ``high_order='fourth'`` taps; order ``6``
    extends the same staggered pattern via Fornberg weights.

    Parameters
    ----------
    envelope, epsilon_map, incident, kz, kx, mode, wavelength_a, delta_x_a,
    delta_z_a, pml_thickness, pml_order
        Same contracts as the historical second-order implementation.
    fd_order : int, optional
        Staggered FD order for TE Laplacians (default ``2``).

    Returns
    -------
    Tensor
        Complex residual with the same shape as ``envelope``.

    Raises
    ------
    NotImplementedError
        When ``mode == 'p'`` and ``fd_order != 2``.
    ValueError
        When ``fd_order`` is not in ``(2, 4, 6)`` or ``mode`` is unknown.
    """
    k = 2 * math.pi / wavelength_a
    dx = delta_x_a
    dz = delta_z_a
    pad = _fd_pad(fd_order)
    pml_strength = 5

    def _pad_z(f: Tensor, n: int) -> Tensor:
        return F.pad(f, (0, 0, n, n), mode="replicate")

    def _bloch_pad_x(f: Tensor, n: int, kx_) -> Tensor:
        Lx = f.shape[-1] * dx
        kx_t = torch.as_tensor(kx_, dtype=torch.float32, device=f.device)
        if kx_t.dim() > 0:
            kx_t = kx_t.reshape(-1, 1, 1, 1)
        angle = kx_t * Lx
        phase = torch.complex(torch.cos(angle), torch.sin(angle)).to(f.dtype)
        left = f[..., -n:] * torch.conj(phase)
        right = f[..., :n] * phase
        return torch.cat([left, f, right], dim=-1)

    def _inc_z_ghosts(inc: Tensor, kz_, n: int) -> Tensor:
        kz_t = torch.as_tensor(kz_, dtype=torch.float32, device=inc.device)
        if kz_t.dim() > 0:
            kz_t = kz_t.reshape(-1, 1, 1, 1)
        angle = kz_t * dz
        step = torch.complex(torch.cos(angle), torch.sin(angle)).to(inc.dtype)
        below = [inc[..., :1, :] * torch.conj(step) ** p for p in range(n, 0, -1)]
        above = [inc[..., -1:, :] * step**p for p in range(1, n + 1)]
        return torch.cat([*below, inc, *above], dim=-2)

    def _lap_z_pml_scattered(f: Tensor, inc: Tensor, kz_) -> Tensor:
        scattered = f - inc
        padded = F.pad(scattered, (0, 0, pad, pad), mode="constant", value=0)
        rz_inv = _pml_stretch_1d(
            padded.shape[-2],
            dz,
            pml_thickness,
            pml_order,
            pml_strength,
            device=f.device,
        )
        half = pad // 2
        e = F.conv2d(padded, _kernel_e_z(dz, f.device, fd_order=fd_order), padding=0)
        e = e * rz_inv[half:-half].reshape(1, 1, -1, 1)
        h = F.conv2d(e, _kernel_h_z(dz, f.device, fd_order=fd_order), padding=0)
        h = h * rz_inv[pad:-pad].reshape(1, 1, -1, 1)
        inc_padded = _inc_z_ghosts(inc, kz_, pad)
        e_inc = F.conv2d(
            inc_padded, _kernel_e_z(dz, f.device, fd_order=fd_order), padding=0
        )
        correction = F.conv2d(
            e_inc, _kernel_h_z(dz, f.device, fd_order=fd_order), padding=0
        )
        return h + correction

    def _lap_z_transparent(f: Tensor, inc: Tensor, kz_, kx_) -> Tensor:
        scattered = f - inc
        nx = scattered.shape[-1]
        kx_t = torch.as_tensor(kx_, dtype=torch.float32, device=f.device)
        if kx_t.dim() > 0:
            kx_t = kx_t.reshape(-1, 1, 1, 1)
        x_idx = torch.arange(nx, device=f.device, dtype=torch.float32)
        carrier_angle = kx_t * x_idx * dx
        carrier = torch.complex(torch.cos(carrier_angle), torch.sin(carrier_angle)).to(
            f.dtype
        )
        s_periodic = scattered * torch.conj(carrier)
        spectrum = torch.fft.fft(s_periodic, dim=-1)
        n_idx = torch.fft.fftfreq(nx, d=dx, device=f.device) * 2 * math.pi
        kx_n = kx_t + n_idx.reshape(1, 1, 1, nx)
        kz_n = torch.sqrt((k**2 - kx_n**2).to(torch.complex64))

        def _ghost(spectrum_row: Tensor, p: int, kz_edge: Tensor) -> Tensor:
            phase = torch.exp(1j * kz_edge * p * dz)
            return torch.fft.ifft(spectrum_row * phase, dim=-1) * carrier

        top = [_ghost(spectrum[..., :1, :], p, kz_n) for p in range(pad, 0, -1)]
        bot = [_ghost(spectrum[..., -1:, :], p, -kz_n) for p in range(1, pad + 1)]
        padded = torch.cat([*top, scattered, *bot], dim=-2).to(f.dtype)
        e = F.conv2d(padded, _kernel_e_z(dz, f.device, fd_order=fd_order), padding=0)
        h = F.conv2d(e, _kernel_h_z(dz, f.device, fd_order=fd_order), padding=0)
        inc_padded = _inc_z_ghosts(inc, kz_, pad)
        e_inc = F.conv2d(
            inc_padded, _kernel_e_z(dz, f.device, fd_order=fd_order), padding=0
        )
        correction = F.conv2d(
            e_inc, _kernel_h_z(dz, f.device, fd_order=fd_order), padding=0
        )
        return h + correction

    def _lap_x_periodic(f: Tensor, kx_) -> Tensor:
        padded = _bloch_pad_x(f, pad, kx_)
        e = F.conv2d(padded, _kernel_e_x(dx, f.device, fd_order=fd_order), padding=0)
        h = F.conv2d(e, _kernel_h_x(dx, f.device, fd_order=fd_order), padding=0)
        return h

    def _d_x_d_z_pml(f: Tensor, kx_) -> Tensor:
        padded = _bloch_pad_x(_pad_z(f, 1), 1, kx_)
        rz_inv = _pml_stretch_1d(
            padded.shape[-2],
            dz,
            pml_thickness,
            pml_order,
            pml_strength,
            device=f.device,
        )
        e = F.conv2d(padded, _kernel_e_z(dz, f.device, fd_order=2), padding=0)
        e = e * rz_inv[1:-1].reshape(1, 1, -1, 1)
        return F.conv2d(e, _kernel_h_x(dx, f.device, fd_order=2), padding=0)

    def _d_z_d_x_pml(f: Tensor, kx_) -> Tensor:
        padded = _bloch_pad_x(_pad_z(f, 1), 1, kx_)
        rz_inv = _pml_stretch_1d(
            padded.shape[-2],
            dz,
            pml_thickness,
            pml_order,
            pml_strength,
            device=f.device,
        )
        e = F.conv2d(padded, _kernel_e_x(dx, f.device, fd_order=2), padding=0)
        h = F.conv2d(e, _kernel_h_z(dz, f.device, fd_order=2), padding=0)
        return h * rz_inv[1:-1].reshape(1, 1, -1, 1)

    if mode == "s":
        is_batched = envelope.dim() >= 3
        a_y = _ensure_4d(envelope.to(torch.complex64))
        eps = _ensure_4d(epsilon_map.to(torch.complex64))
        inc = _ensure_4d(incident.to(torch.complex64))
        Ey = (1.0 + a_y) * inc
        diff = (
            _lap_z_transparent(Ey, inc, kz, kx)
            + _lap_x_periodic(Ey, kx)
            + k**2 * eps * Ey
        )
        return diff.squeeze(1) if is_batched else diff.squeeze()

    elif mode == "p":
        if fd_order != 2:
            raise NotImplementedError(
                "p-pol Helmholtz residual currently supports fd_order=2 only"
            )
        if envelope.dim() == 3:
            is_batched = False
            az_raw, ax_raw = envelope[0], envelope[1]
            eps_z_raw, eps_x_raw = epsilon_map[0], epsilon_map[1]
        elif envelope.dim() == 4:
            is_batched = True
            az_raw, ax_raw = envelope[:, 0], envelope[:, 1]
            eps_z_raw, eps_x_raw = epsilon_map[:, 0], epsilon_map[:, 1]
        else:
            raise ValueError(
                "p-pol envelope must have shape (2, Nz, Nx) [Ez, Ex] or (batch, 2, Nz, Nx)"
            )
        a_z = _ensure_4d(az_raw.to(torch.complex64))
        a_x = _ensure_4d(ax_raw.to(torch.complex64))
        eps_z = _ensure_4d(eps_z_raw.to(torch.complex64))
        eps_x = _ensure_4d(eps_x_raw.to(torch.complex64))
        inc = _ensure_4d(incident.to(torch.complex64))
        Ez = (1.0 + a_z) * inc
        Ex = (1.0 + a_x) * inc
        diff_z = (
            _lap_x_periodic(Ez, kx) - _d_x_d_z_pml(Ex, kx) + k**2 * eps_z * Ez
        ).squeeze(1)
        diff_x = (
            _lap_z_pml_scattered(Ex, inc, kz) - _d_z_d_x_pml(Ez, kx) + k**2 * eps_x * Ex
        ).squeeze(1)
        if is_batched:
            return torch.stack([diff_z, diff_x], dim=1)
        return torch.stack([diff_z.squeeze(0), diff_x.squeeze(0)])

    else:
        raise ValueError(f"mode must be 's' or 'p', got '{mode}'")


def validate_laplacian_orders(
    field: Tensor,
    dx: float,
    dz: float,
    kx: float,
    kz: float,
    orders: tuple[int, ...] = (2, 4, 6),
) -> dict[int, dict[str, float]]:
    """Score staggered Laplacian orders against an analytic plane-wave curvature.

    Applies Bloch-periodic ``∂²/∂x²`` and exact phase-ghost ``∂²/∂z²`` (no PML)
    to ``field ≈ exp(i(kx x + kz z))`` and compares to ``-(kx²+kz²) field``.
    Higher ``fd_order`` should reduce interior relative error for a resolved
    plane wave.

    Parameters
    ----------
    field : Tensor
        Complex ``(Nz, Nx)`` plane-wave sample.
    dx, dz : float
        Grid spacings matching the wavevector units.
    kx, kz : float
        Plane-wave wavevector components.
    orders : tuple of int
        Subset of ``(2, 4, 6)`` to evaluate.

    Returns
    -------
    dict
        ``order -> {mean_rel, max_rel, mean_abs}`` on the interior crop
        ``[pad:-pad, pad:-pad]`` where ``pad = fd_order``.
    """
    if field.dim() != 2:
        raise ValueError(f"expected (Nz, Nx) field, got shape {tuple(field.shape)}")
    f = _ensure_4d(field.to(torch.complex64))
    analytic = -(kx**2 + kz**2) * f
    out: dict[int, dict[str, float]] = {}
    for order in orders:
        pad = _fd_pad(order)
        Lx = f.shape[-1] * dx
        angle = torch.tensor(kx * Lx, dtype=torch.float32, device=f.device)
        phase = torch.complex(torch.cos(angle), torch.sin(angle)).to(f.dtype)
        px = torch.cat(
            [f[..., -pad:] * torch.conj(phase), f, f[..., :pad] * phase], dim=-1
        )
        e_x = F.conv2d(px, _kernel_e_x(dx, f.device, fd_order=order), padding=0)
        lap_x = F.conv2d(e_x, _kernel_h_x(dx, f.device, fd_order=order), padding=0)

        step = torch.exp(
            torch.complex(
                torch.tensor(0.0, device=f.device),
                torch.tensor(kz * dz, dtype=torch.float32, device=f.device),
            )
        ).to(f.dtype)
        below = [f[..., :1, :] * torch.conj(step) ** p for p in range(pad, 0, -1)]
        above = [f[..., -1:, :] * step**p for p in range(1, pad + 1)]
        pz = torch.cat([*below, f, *above], dim=-2)
        e_z = F.conv2d(pz, _kernel_e_z(dz, f.device, fd_order=order), padding=0)
        lap_z = F.conv2d(e_z, _kernel_h_z(dz, f.device, fd_order=order), padding=0)

        err = (lap_x + lap_z - analytic).abs()
        rel = err / analytic.abs().clamp_min(1e-30)
        crop = (slice(None), slice(None), slice(pad, -pad), slice(pad, -pad))
        out[order] = {
            "mean_rel": float(rel[crop].mean()),
            "max_rel": float(rel[crop].max()),
            "mean_abs": float(err[crop].mean()),
        }
    return out
