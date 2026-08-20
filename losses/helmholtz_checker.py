import torch
from torch import Tensor
import torch.nn.functional as F
import math

torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)


def _kernel_e_z(dz: float, device=None) -> Tensor:
    """Forward-difference kernel along z: [0, -1, +1] / dz, shape (1,1,3,1)."""
    return torch.tensor([[[[0.0], [-1.0], [1.0]]]], dtype=torch.complex64, device=device) / dz


def _kernel_h_z(dz: float, device=None) -> Tensor:
    """Backward-difference kernel along z: [-1, +1, 0] / dz, shape (1,1,3,1)."""
    return torch.tensor([[[[-1.0], [1.0], [0.0]]]], dtype=torch.complex64, device=device) / dz


def _kernel_e_x(dx: float, device=None) -> Tensor:
    """Forward-difference kernel along x: [0, -1, +1] / dx, shape (1,1,1,3)."""
    return torch.tensor([[[[0.0, -1.0, 1.0]]]], dtype=torch.complex64, device=device) / dx


def _kernel_h_x(dx: float, device=None) -> Tensor:
    """Backward-difference kernel along x: [-1, +1, 0] / dx, shape (1,1,1,3)."""
    return torch.tensor([[[[-1.0, 1.0, 0.0]]]], dtype=torch.complex64, device=device) / dx


def _ensure_4d(t: Tensor) -> Tensor:
    """(Nz, Nx) -> (1, 1, Nz, Nx); (B, Nz, Nx) -> (B, 1, Nz, Nx)."""
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 3:
        return t.unsqueeze(1)
    return t


def _pml_stretch_1d(n: int, delta: float, pml_thickness: int, pml_order: float = 4.0,
                    pml_strength: float = 5.0, device=None) -> Tensor:
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
    r[pml_thickness: n - pml_thickness] = 1
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


def helmholtz_residual_loss_periodic_pml(
        E_field: Tensor,
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
    """
    Helmholtz residual with a periodic (circular) boundary along x and a
    complex-coordinate-stretching PML absorbing boundary along z, following
    the same stretched-coordinate construction as MaxwellNet's
    `rz_inverse`/`dd_z_pml`, generalized to an arbitrary physical z-spacing
    and domain length (see `_pml_stretch_1d`).

    Unlike `helmholtz_residual_loss`, nothing is trimmed. The x second
    derivative wraps around with a Bloch/Floquet phase correction
    (exp(∓i*kx*Lx), Lx = domain width) rather than plain repetition: the
    field is only periodic *up to this phase factor* (E(x+Lx) = E(x)*e^{i kx
    Lx}) unless kx*Lx happens to be an integer multiple of 2*pi, which is
    generically false for an arbitrary oblique incidence angle. Plain
    circular wraparound silently assumes phase = 1 and introduces a large,
    spurious error with no real physical meaning (verified: >100% relative
    error against the true incident wave's own analytic curvature, for a
    generic non-commensurate angle) — this affects both `'s'` and `'p'`
    mode's periodic-x derivatives, not just a PML-boundary-region artifact.

    The z second derivative is evaluated on the *scattered* field
    (E_field - incident), zero-padded at each z-edge before PML grading —
    mirroring MaxwellNet's total-field/scattered-field split (`ey_s`,
    `padding_zero`) — with the incident wave's own z-curvature (computed
    with the same un-stretched finite-difference stencil, not the analytic
    -kz**2*incident — see `_lap_z_pml_scattered`) added back in. This
    anchors the total field to the known incident wave at the z-truncation
    boundary: without it (i.e. differencing E_field directly with replicate
    padding, the prior behavior of this function), the bare residual is
    exactly zero for the trivial E_field ≡ 0 solution regardless of
    epsilon_map, and training can collapse the predicted field to near-zero
    amplitude while still driving this loss to ~0. The returned residual
    therefore covers the full input domain.

    Args:
        E_field, epsilon_map, mode, wavelength_a, delta_x_a, delta_z_a: same
            shapes/conventions as `helmholtz_residual_loss` (s-pol: Ey;
            p-pol: (Ez, Ex) / [εz, εx]).
        incident: incident plane wave e^(i k·r), same shape convention as
            E_field but without the p-pol component axis — (Nz, Nx) or
            (batch, Nz, Nx) — as returned by `ShapeNet.forward`.
        kz, kx: incident wave's z- and x-wavevector components, scalar or
            (batch,), as returned by `ShapeNet.forward`.
        pml_thickness: number of grid samples, at each z-edge, over which the
            PML grades from transparent (interior) to absorbing.
        pml_order, pml_strength: PML grading polynomial order and absorption
            strength — defaults (4, 5) match MaxwellNet's hardcoded values.

    Returns:
        Complex residual tensor, same shape as E_field ((Nz, Nx) for s-pol,
        (2, Nz, Nx) for p-pol).

    Note: the p-pol path mirrors MaxwellNet's `dd_zx_pml`/`dd_xz_pml`
    cross-derivative treatment (single PML stretch factor at the z-derivative
    stage of each mixed term, x Bloch-periodic since it's periodic-up-to-a-
    phase), but is not covered by the (s-pol only) test data used to
    validate this module.
    """
    k = 2 * math.pi / wavelength_a
    dx = delta_x_a
    dz = delta_z_a

    pml_strength = 5

    def _pad_z(f: Tensor, n: int) -> Tensor:
        return F.pad(f, (0, 0, n, n), mode='replicate')

    def _bloch_pad_x(f: Tensor, n: int, kx_) -> Tensor:
        """Pad n columns on each side along x with a Bloch/Floquet phase
        correction (exp(∓i*kx*Lx), Lx = f's own x-extent) instead of plain
        repetition — the field is periodic only up to this phase factor
        unless kx*Lx happens to be an integer multiple of 2*pi, which is
        generically false for an oblique incidence angle."""
        Lx = f.shape[-1] * dx
        kx_t = torch.as_tensor(kx_, dtype=torch.float32, device=f.device)
        if kx_t.dim() > 0:
            kx_t = kx_t.reshape(-1, 1, 1, 1)
        angle = kx_t * Lx
        phase = torch.complex(torch.cos(angle), torch.sin(angle)).to(f.dtype)
        left = f[..., -n:] * torch.conj(phase)
        right = f[..., :n] * phase
        return torch.cat([left, f, right], dim=-1)

    def _lap_z_pml_scattered(f: Tensor, inc: Tensor, kz_) -> Tensor:
        """∂²f/∂z² with PML absorption, computed on the scattered field
        (f - inc) with a hard zero boundary (instead of replicating f
        itself), plus the incident wave's own z-curvature added back —
        anchoring f to the known incident wave at the z-truncation boundary.

        The incident-curvature correction is computed with the *same*
        (un-stretched) finite-difference stencil used on the scattered
        field, not the analytic -kz**2*inc — by linearity of the stencil,
        [stencil(scattered)] + [stencil(incident)] == [stencil(total field)]
        exactly, whatever the stencil's own numerical-dispersion error is.
        Using the exact analytic value instead would mix a dispersion-free
        term with a dispersion-imperfect one, creating a spurious residual
        (unrelated to any real scatterer) the network would have to spend
        capacity compensating for — mirroring MaxwellNet's `dd_z_fast`,
        which is likewise a finite difference of the precomputed incident
        field rather than an analytic expression. Output z-length == f's
        z-length."""
        scattered = f - inc
        padded = F.pad(scattered, (0, 0, 2, 2), mode='constant', value=0)
        rz_inv = _pml_stretch_1d(padded.shape[-2], dz, pml_thickness, pml_order,
                                 pml_strength, device=f.device)
        e = F.conv2d(padded, _kernel_e_z(dz, f.device), padding=0)
        e = e * rz_inv[1:-1].reshape(1, 1, -1, 1)
        h = F.conv2d(e, _kernel_h_z(dz, f.device), padding=0)
        h = h * rz_inv[2:-2].reshape(1, 1, -1, 1)

        # Extend `inc` by 2 exact ghost rows on each z-edge via its own
        # closed-form phase advance (exp(i*kz*dz) per step) rather than
        # replicate-padding, then difference it with the same stencil, no
        # PML stretching (matching dd_z_fast, which isn't PML-stretched
        # either — the incident wave isn't what the PML is meant to absorb).
        kz_t = torch.as_tensor(kz_, dtype=torch.float32, device=f.device)
        if kz_t.dim() > 0:
            kz_t = kz_t.reshape(-1, 1, 1, 1)
        angle = kz_t * dz
        step = torch.complex(torch.cos(angle), torch.sin(angle)).to(inc.dtype)
        below2 = inc[..., :1, :] * torch.conj(step) ** 2
        below1 = inc[..., :1, :] * torch.conj(step)
        above1 = inc[..., -1:, :] * step
        above2 = inc[..., -1:, :] * step ** 2
        inc_padded = torch.cat([below2, below1, inc, above1, above2], dim=-2)
        e_inc = F.conv2d(inc_padded, _kernel_e_z(dz, f.device), padding=0)
        correction = F.conv2d(e_inc, _kernel_h_z(dz, f.device), padding=0)

        return h + correction

    def _lap_x_periodic(f: Tensor, kx_) -> Tensor:
        """∂²f/∂x² with Bloch-periodic wraparound, output x-length == f's
        x-length."""
        padded = _bloch_pad_x(f, 2, kx_)
        e = F.conv2d(padded, _kernel_e_x(dx, f.device), padding=0)
        h = F.conv2d(e, _kernel_h_x(dx, f.device), padding=0)
        return h

    def _d_x_d_z_pml(f: Tensor, kx_) -> Tensor:
        """∂²f/∂x∂z = d_h_x( pml_z(d_e_z(f)) ), output shape == f's shape."""
        padded = _bloch_pad_x(_pad_z(f, 1), 1, kx_)
        rz_inv = _pml_stretch_1d(padded.shape[-2], dz, pml_thickness, pml_order,
                                 pml_strength, device=f.device)
        e = F.conv2d(padded, _kernel_e_z(dz, f.device), padding=0)
        e = e * rz_inv[1:-1].reshape(1, 1, -1, 1)
        return F.conv2d(e, _kernel_h_x(dx, f.device), padding=0)

    def _d_z_d_x_pml(f: Tensor, kx_) -> Tensor:
        """∂²f/∂z∂x = pml_z( d_h_z(d_e_x(f)) ), output shape == f's shape."""
        padded = _bloch_pad_x(_pad_z(f, 1), 1, kx_)
        rz_inv = _pml_stretch_1d(padded.shape[-2], dz, pml_thickness, pml_order,
                                 pml_strength, device=f.device)
        e = F.conv2d(padded, _kernel_e_x(dx, f.device), padding=0)
        h = F.conv2d(e, _kernel_h_z(dz, f.device), padding=0)
        return h * rz_inv[1:-1].reshape(1, 1, -1, 1)

    if mode == 's':
        Ey  = _ensure_4d(E_field.to(torch.complex64))
        eps = _ensure_4d(epsilon_map.to(torch.complex64))
        inc = _ensure_4d(incident.to(torch.complex64))

        diff = _lap_z_pml_scattered(Ey, inc, kz) + _lap_x_periodic(Ey, kx) + k**2 * eps * Ey
        return diff.squeeze()

    elif mode == 'p':
        if E_field.dim() == 3:
            is_batched = False
            Ez_raw, Ex_raw = E_field[0], E_field[1]
            eps_z_raw, eps_x_raw = epsilon_map[0], epsilon_map[1]
        elif E_field.dim() == 4:
            is_batched = True
            Ez_raw, Ex_raw = E_field[:, 0], E_field[:, 1]
            eps_z_raw, eps_x_raw = epsilon_map[:, 0], epsilon_map[:, 1]
        else:
            raise ValueError(
                "p-pol E_field must have shape (2, Nz, Nx) [Ez, Ex] or (batch, 2, Nz, Nx)")

        Ez    = _ensure_4d(Ez_raw.to(torch.complex64))
        Ex    = _ensure_4d(Ex_raw.to(torch.complex64))
        eps_z = _ensure_4d(eps_z_raw.to(torch.complex64))
        eps_x = _ensure_4d(eps_x_raw.to(torch.complex64))
        inc   = _ensure_4d(incident.to(torch.complex64))

        diff_z = (_lap_x_periodic(Ez, kx) - _d_x_d_z_pml(Ex, kx) + k**2 * eps_z * Ez).squeeze(1)
        diff_x = (_lap_z_pml_scattered(Ex, inc, kz) - _d_z_d_x_pml(Ez, kx) + k**2 * eps_x * Ex).squeeze(1)

        if is_batched:
            return torch.stack([diff_z, diff_x], dim=1)  # (B, 2, Nz, Nx)
        return torch.stack([diff_z.squeeze(0), diff_x.squeeze(0)])  # (2, Nz, Nx)

    else:
        raise ValueError(f"mode must be 's' or 'p', got '{mode}'")
