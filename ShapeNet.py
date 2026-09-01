import math

import torch
from torch import nn
import torch.nn.functional as F

from UNet import UNet


class PeriodicMaxwellNet(nn.Module):
    """
    UNet-based Helmholtz field predictor for training against ShapeDataset.

    Unlike MaxwellNet, there is no incident/scattered-field decomposition in
    the governing equation: the network's output is a complex envelope for
    the *total* field, which (after adding 1 and multiplying by the incident
    plane wave, see `forward`) must satisfy the *homogeneous* Helmholtz
    equation for the given epsilon map (checked via
    `losses.helmholtz_checker.helmholtz_residual_loss_periodic_pml`, which
    is periodic along x and PML-absorbing along z).

    The incident wave-vector (kx, kz), computed from ShapeDataset's `theta`
    (incidence angle, degrees) and `wavelength_nm`, is fed into the UNet at
    the bottleneck (concatenated to the latent feature map) so the network
    is conditioned on the incidence geometry, in addition to being used to
    build the plane-wave carrier itself.

    Grid size varies per sample (ShapeDataset has no fixed Nx/Nz), so nothing
    about the physical domain is baked into buffers at construction time. The
    UNet's pooling depth does require the spatial dims to be a multiple of
    `2**(depth-1)`, so the input is padded up to that multiple (circular in
    x, replicate in z — matching UNet.py's own internal convolution padding)
    and the output is cropped back down to the sample's native grid.
    """

    # Number of incident-wave-vector components (kx, kz) fed to the UNet
    # bottleneck as conditioning.
    _COND_CHANNELS = 2

    # optical_constant's last dim is 3 components ordered [Ez, Ex, Ey];
    # 'te' (s-pol) only needs Ey (index 2), 'tm' (p-pol) needs [Ez, Ex].
    _EPS_INDEX = {'te': [2], 'tm': [0, 1]}

    def __init__(self, depth=6, filter=16, norm='weight', up_mode='upconv', mode='te',
                contrast_scale=100.0):
        super().__init__()
        if mode not in ('te', 'tm'):
            raise ValueError(f"mode must be 'te' or 'tm', got {mode!r}")
        self.mode = mode
        channels = 2 if mode == 'te' else 4
        self.model = UNet(channels, channels, depth, filter, norm, up_mode,
                          cond_channels=self._COND_CHANNELS)
        # Zero-init the last layer so the network starts at the zeroth-order
        # Born approximation (scattered envelope a=0 -> E=E_i) rather than
        # an arbitrary random scattered field.
        nn.init.zeros_(self.model.last_conv.weight)
        self._divisor = 2 ** (depth - 1)
        # X-ray refractive-index contrast (eps - 1) is tiny (~1e-2 or smaller)
        # in this dataset, easily swamped by the eps=1 background when fed
        # to the UNet directly. contrast_scale amplifies just the UNet's
        # *input* representation of that contrast; the true, unscaled eps is
        # still what's returned for the physics loss (see forward()).
        self.contrast_scale = contrast_scale

    def _select_epsilon(self, optical_constant: torch.Tensor) -> torch.Tensor:
        """optical_constant: (B, Ny, Nx, 3) complex -> eps: (B, k, Ny, Nx) complex."""
        n = optical_constant[..., self._EPS_INDEX[self.mode]]
        n = n.movedim(-1, 1)
        return (n ** 2).to(torch.complex64)

    def _pad_to_divisor(self, x: torch.Tensor):
        """Pad (append) the last two dims up to a multiple of self._divisor:
        circular along x (last dim), replicate along z (second-to-last)."""
        d = self._divisor
        pad_z = (-x.shape[-2]) % d
        pad_x = (-x.shape[-1]) % d
        if pad_x:
            x = F.pad(x, (0, pad_x, 0, 0), mode='circular')
        if pad_z:
            x = F.pad(x, (0, 0, 0, pad_z), mode='replicate')
        return x, pad_z, pad_x

    def _incident_k_vector(self, theta: torch.Tensor, wavelength_nm: torch.Tensor):
        """
        Incident-wave-vector components (kx, kz), in Å⁻¹, from ShapeDataset's
        `theta` (incidence angle from the z axis, in degrees) and
        `wavelength_nm` (vacuum wavelength). Both args and the returned
        tensors have shape (B,).

        Sign convention is matched empirically to the (row=z, col=x) layout
        of ShapeDataset's field arrays (verified against ground-truth
        e_field samples via local phase-gradient measurement) — this is
        *not* the same sign convention as the dataset's own `k_vector`
        column, whose x-component is flipped relative to the array's column
        axis.
        """
        wavelength_a = wavelength_nm.to(torch.float32) * 10.0
        k = 2 * math.pi / wavelength_a
        theta_rad = torch.deg2rad(theta.to(torch.float32))
        kx = -k * torch.sin(theta_rad)
        kz = -k * torch.cos(theta_rad)
        return kx, kz

    def _incident_wave(self, kx: torch.Tensor, kz: torch.Tensor, nz: int, nx: int,
                       delta_x_a: torch.Tensor, delta_z_a: torch.Tensor) -> torch.Tensor:
        """Plane wave exp(i(kx*x + kz*z)) over an (Nz, Nx) grid, per batch
        element. kx, kz, delta_x_a, delta_z_a: (B,). Returns (B, Nz, Nx)
        complex64."""
        device = kx.device
        x = torch.arange(nx, dtype=torch.float32, device=device) * delta_x_a.to(torch.float32)[:, None]  # (B, Nx)
        z = torch.arange(nz, dtype=torch.float32, device=device) * delta_z_a.to(torch.float32)[:, None]  # (B, Nz)
        phase = kz[:, None, None] * z[:, :, None] + kx[:, None, None] * x[:, None, :]  # (B, Ny, Nx)
        return torch.complex(torch.cos(phase), torch.sin(phase))

    def forward(self, optical_constant: torch.Tensor, theta: torch.Tensor, wavelength_nm: torch.Tensor,
               delta_x_a: torch.Tensor, delta_z_a: torch.Tensor):
        """
        Args:
            optical_constant: (B, Nz, Nx, 3) complex refractive-index map.
            theta: (B,) incidence angle from the z axis, in degrees.
            wavelength_nm: (B,) vacuum wavelength.
            delta_x_a, delta_z_a: (B,) grid spacing in angstroms.

        Returns:
            (envelope, epsilon_map, incident, kz, kx):
                envelope: predicted complex envelope `a` such that the total
                    field E = incident * (1 + a), (B, Nz, Nx) for 'te' (Ey),
                    or (B, 2, Nz, Nx) for 'tm' ([Ez, Ex]) -- NOT the total
                    field itself; downstream code (the physics loss, any
                    plotting) reconstructs E = incident * (1 + envelope)
                    itself wherever it needs the total field.
                epsilon_map: matching epsilon (optical_constant**2), same
                    shape convention as envelope — pass both straight into
                    `helmholtz_residual_loss_periodic_pml`.
                incident: incident plane wave e^(i k·r), (B, Nz, Nx) complex64
                    — same for every envelope channel.
                kz, kx: incident wave's z- and x-components of the
                    wavevector, (B,) each — needed alongside `incident` to
                    anchor the z-PML residual and Bloch-correct the periodic
                    x residual (see `helmholtz_checker`).
        """
        eps = self._select_epsilon(optical_constant.to(torch.complex64))  # (B, k, Nz, Nx)
        k = eps.shape[1]
        nz, nx = eps.shape[-2:]

        # Amplify (eps - 1) for the UNet's input only; `eps` itself (used
        # below for the physics loss) stays the true, unscaled permittivity.
        contrast = (eps - 1.0) * self.contrast_scale
        x = torch.cat([contrast.real, contrast.imag], dim=1)  # (B, 2k, Nz, Nx)
        x_padded, pad_z, pad_x = self._pad_to_divisor(x)

        kx, kz = self._incident_k_vector(theta, wavelength_nm)  # (B,), (B,)
        cond = torch.stack([kx, kz], dim=1)  # (B, 2)

        out = self.model(x_padded, cond)  # (B, 2k, Nz+pad_z, Nx+pad_x)
        if pad_z:
            out = out[..., :-pad_z, :]
        if pad_x:
            out = out[..., :, :-pad_x]

        # The network predicts a slowly-varying envelope `a` of the total
        # field -- zero-init (see __init__) makes out=0 at the start of
        # training, so a=0 -> E=incident, the zeroth-order Born
        # approximation. The total field E = incident * (1 + a) is
        # reconstructed by downstream code, not here -- `a` itself (NOT
        # 1+a) is what's returned.
        envelope = torch.complex(out[:, :k], out[:, k:2 * k])
        incident = self._incident_wave(kx, kz, nz, nx, delta_x_a, delta_z_a)  # (B, Ny, Nx)

        if self.mode == 'te':
            envelope = envelope[:, 0]
            eps = eps[:, 0]

        return envelope, eps, incident, kz, kx
