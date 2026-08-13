import torch
from torch import nn
import torch.nn.functional as F

from UNet import UNet


class PeriodicMaxwellNet(nn.Module):
    """
    UNet-based Helmholtz field predictor for training against ShapeDataset.

    Unlike MaxwellNet, there is no incident-wave / scattered-field
    decomposition — ShapeDataset carries no incident-wave information
    (no energy/theta/k_vector), so the network directly predicts a complex
    field intended to satisfy the *homogeneous* Helmholtz equation for the
    given epsilon map (checked via
    `losses.helmholtz_checker.helmholtz_residual_loss_periodic_pml`, which
    is periodic along x and PML-absorbing along z).

    Grid size varies per sample (ShapeDataset has no fixed Nx/Nz), so nothing
    about the physical domain is baked into buffers at construction time. The
    UNet's pooling depth does require the spatial dims to be a multiple of
    `2**(depth-1)`, so the input is padded up to that multiple (circular in
    x, replicate in z — matching UNet.py's own internal convolution padding)
    and the output is cropped back down to the sample's native grid.
    """

    # optical_constant's last dim is 3 components ordered [Ez, Ex, Ey];
    # 'te' (s-pol) only needs Ey (index 2), 'tm' (p-pol) needs [Ez, Ex].
    _EPS_INDEX = {'te': [2], 'tm': [0, 1]}

    def __init__(self, depth=6, filter=16, norm='weight', up_mode='upconv', mode='te'):
        super().__init__()
        if mode not in ('te', 'tm'):
            raise ValueError(f"mode must be 'te' or 'tm', got {mode!r}")
        self.mode = mode
        channels = 2 if mode == 'te' else 4
        self.model = UNet(channels, channels, depth, filter, norm, up_mode)
        self._divisor = 2 ** (depth - 1)

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

    def forward(self, optical_constant: torch.Tensor):
        """
        Args:
            optical_constant: (B, Ny, Nx, 3) complex refractive-index map.

        Returns:
            (field_pred, epsilon_map):
                field_pred: predicted complex field, (B, Ny, Nx) for 'te'
                    (Ey), or (B, 2, Ny, Nx) for 'tm' ([Ez, Ex]).
                epsilon_map: matching epsilon (optical_constant**2), same
                    shape convention as field_pred — pass both straight into
                    `helmholtz_residual_loss_periodic_pml`.
        """
        eps = self._select_epsilon(optical_constant.to(torch.complex64))  # (B, k, Ny, Nx)
        k = eps.shape[1]

        x = torch.cat([eps.real, eps.imag], dim=1)  # (B, 2k, Ny, Nx)
        x_padded, pad_z, pad_x = self._pad_to_divisor(x)

        out = self.model(x_padded)  # (B, 2k, Ny+pad_z, Nx+pad_x)
        if pad_z:
            out = out[..., :-pad_z, :]
        if pad_x:
            out = out[..., :, :-pad_x]

        field = (out[:, :k] + 1j * out[:, k:2 * k]).to(torch.complex64)

        if self.mode == 'te':
            field = field[:, 0]
            eps = eps[:, 0]

        return field, eps