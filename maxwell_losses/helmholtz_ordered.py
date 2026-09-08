"""TE Helmholtz residual with an injected staggered Laplacian.

This module copies the TE training path of
``helmholtz_checker.helmholtz_residual_loss_periodic_pml`` as used on
``origin/master`` and ``origin/fl/supervised``: Bloch-periodic ``x`` and
transparent (Rayleigh) ``z``, with TF/SF incident curvature added back.
It does not use PML. p-pol is not implemented here; use
``helmholtz_checker``.

Order-2 callers should keep importing ``helmholtz_checker``. Orders 4 and
6 should use ``helmholtz_checker_o4`` / ``helmholtz_checker_o6``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from maxwell_losses.helmholtz_checker import _ensure_4d
from maxwell_losses.staggered_fd import StaggeredFD


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
    *,
    fd: StaggeredFD,
) -> Tensor:
    """Scores the TE training residual with injected staggered taps.

    Reconstructs ``E = incident * (1 + envelope)`` and returns
    ``Laplacian_FD(E) + k^2 epsilon E`` over the full domain. Boundaries
    match ``helmholtz_checker`` TE on main / ``fl/supervised``: Bloch
    ghosts in ``x``, transparent Rayleigh ghosts in ``z`` (same
    ``kz_n`` with ``Im >= 0`` at both edges), incident stencil correction
    added back. ``pml_thickness`` and ``pml_order`` are ignored; they
    exist only so callers can keep the training signature.

    Parameters
    ----------
    envelope : Tensor
        Predicted complex envelope ``a`` with ``E = incident * (1 + a)``.
        s-pol shape ``(Nz, Nx)`` or ``(batch, Nz, Nx)``.
    epsilon_map : Tensor
        Complex permittivity, same spatial shape as ``envelope``.
    incident : Tensor
        Incident plane wave, same spatial shape as ``envelope``.
    kz, kx : tensor_like
        Incident wavevector components, scalar or ``(batch,)``.
    mode : {'s', 'p'}
        Polarisation. Only ``'s'`` is implemented.
    wavelength_a, delta_x_a, delta_z_a : float
        Wavelength and grid spacings in the same units as training
        (``train_maxwellnet._compute_loss`` uses ``wavelength_a = 1``).
    pml_thickness : int
        Ignored. Kept so the call matches training.
    pml_order : float, optional
        Ignored. Kept so the call matches training.
    fd : StaggeredFD
        Injected staggered stencil (keyword-only).

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
    if mode == "p":
        raise NotImplementedError(
            "p-pol Helmholtz residual with injected FD is not implemented; "
            "use maxwell_losses.helmholtz_checker for order 2"
        )
    if mode != "s":
        raise ValueError(f"mode must be 's' or 'p', got {mode!r}")

    k = 2 * math.pi / wavelength_a
    dx = delta_x_a
    dz = delta_z_a
    pad = fd.pad
    del pml_thickness, pml_order

    def _bloch_pad_x(f: Tensor, n: int, kx_) -> Tensor:
        lx = f.shape[-1] * dx
        kx_t = torch.as_tensor(kx_, dtype=torch.float32, device=f.device)
        if kx_t.dim() > 0:
            kx_t = kx_t.reshape(-1, 1, 1, 1)
        angle = kx_t * lx
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

        def _ghost(spectrum_row: Tensor, p: int) -> Tensor:
            phase = torch.exp(1j * kz_n * p * dz)
            return torch.fft.ifft(spectrum_row * phase, dim=-1) * carrier

        top = [_ghost(spectrum[..., :1, :], p) for p in range(pad, 0, -1)]
        bot = [_ghost(spectrum[..., -1:, :], p) for p in range(1, pad + 1)]
        padded = torch.cat([*top, scattered, *bot], dim=-2).to(f.dtype)
        e = F.conv2d(padded, fd.kernel_e_z(dz, f.device), padding=0)
        h = F.conv2d(e, fd.kernel_h_z(dz, f.device), padding=0)
        inc_padded = _inc_z_ghosts(inc, kz_, pad)
        e_inc = F.conv2d(inc_padded, fd.kernel_e_z(dz, f.device), padding=0)
        correction = F.conv2d(e_inc, fd.kernel_h_z(dz, f.device), padding=0)
        return h + correction

    def _lap_x_periodic(f: Tensor, kx_) -> Tensor:
        padded = _bloch_pad_x(f, pad, kx_)
        e = F.conv2d(padded, fd.kernel_e_x(dx, f.device), padding=0)
        h = F.conv2d(e, fd.kernel_h_x(dx, f.device), padding=0)
        return h

    is_batched = envelope.dim() >= 3
    a_y = _ensure_4d(envelope.to(torch.complex64))
    eps = _ensure_4d(epsilon_map.to(torch.complex64))
    inc = _ensure_4d(incident.to(torch.complex64))
    ey = (1.0 + a_y) * inc
    diff = (
        _lap_z_transparent(ey, inc, kz, kx) + _lap_x_periodic(ey, kx) + k**2 * eps * ey
    )
    return diff.squeeze(1) if is_batched else diff.squeeze()


def validate_laplacian(
    field: Tensor,
    dx: float,
    dz: float,
    kx: float,
    kz: float,
    fd: StaggeredFD,
    *,
    wavelength: float = 1.0,
) -> dict[str, float]:
    """Scores the training TE residual on a vacuum plane wave.

    Applies the same operator MaxwellNet is trained with on main and
    ``fl/supervised``: Bloch-periodic ``x``, transparent ``z``, envelope
    ``a = 0``, ``epsilon = 1``, ``E = field``. The discrete residual of a
    resolved vacuum plane wave should approach 0; it is not a PML check.

    Parameters
    ----------
    field : Tensor
        Complex ``(Nz, Nx)`` incident plane-wave sample.
    dx, dz : float
        Grid spacings in the same units as ``wavelength``.
    kx, kz : float
        Plane-wave wavevector components in those units.
    fd : StaggeredFD
        Injected staggered stencil.
    wavelength : float, optional
        Free-space wavelength. Default ``1.0`` matches
        ``train_maxwellnet._compute_loss``.

    Returns
    -------
    dict
        ``mean_rel``, ``max_rel``, and ``mean_abs`` of ``|r|`` over the
        full domain (the training reduction is a domain mean of
        ``|r|^2``). Relative scale is ``k^2 |E|``.

    Raises
    ------
    ValueError
        When ``field`` is not rank 2.
    """
    if field.dim() != 2:
        raise ValueError(f"expected (Nz, Nx) field, got shape {tuple(field.shape)}")
    wave = field.to(torch.complex64)
    residual = helmholtz_residual_loss_periodic_pml(
        torch.zeros_like(wave),
        torch.ones_like(wave),
        wave,
        kz,
        kx,
        "s",
        wavelength,
        dx,
        dz,
        0,
        fd=fd,
    )
    k2 = (2 * math.pi / wavelength) ** 2
    err = residual.abs()
    rel = err / (k2 * wave.abs()).clamp_min(1e-30)
    return {
        "mean_rel": float(rel.mean()),
        "max_rel": float(rel.max()),
        "mean_abs": float(err.mean()),
    }
