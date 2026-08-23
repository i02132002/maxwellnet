# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

import torch
from torch import nn
from UNet import UNet
import torch.nn.functional as F
import math
import numpy as np


class MaxwellNet(nn.Module):
    def __init__(self, depth=6, filter=16, norm='weight', up_mode='upconv',
                 wavelength=1, dpl=20, Nx=256, Nz=256, pml_thickness=16, symmetry_x=False, mode='te', high_order='fourth',
                 kx=0.0):

        super(MaxwellNet, self).__init__()
        self.mode = mode
        if mode == 'te':
            in_channels = 1
            out_channels = 2
        elif mode == 'tm':
            in_channels = 2
            out_channels = 4

        self.high_order = high_order
        self.model = UNet(in_channels, out_channels,
                          depth, filter, norm, up_mode)

        # pixel size [um / pixel]
        delta = wavelength / dpl
        # wave-number [1 / um]
        k = 2 * math.pi / wavelength
        self.register_buffer('delta', torch.tensor(
            delta, dtype=torch.float32, requires_grad=False))
        self.register_buffer('k', torch.tensor(
            k, dtype=torch.float32, requires_grad=False))

        self.symmetry_x = symmetry_x

        if self.high_order == 'second':
            pad = 2
            self.pad = pad
        elif self.high_order == 'fourth':
            pad = 4
            self.pad = pad

        self.padding_ref = nn.Sequential(nn.ReflectionPad2d(
            (0, 0, pad, 0)), nn.ZeroPad2d((pad, pad, 0, pad)))
        self.padding_zero = nn.Sequential(nn.ZeroPad2d((pad, pad, pad, pad)))

        if symmetry_x == True:
            x = np.linspace(-pad, Nx + pad - 1, Nx + 2 * pad) * delta
        else:
            x = np.linspace(-Nx // 2 - pad, Nx // 2 +
                            pad - 1, Nx + 2 * pad) * delta
        z = np.linspace(-Nz // 2 - pad, Nz // 2 +
                        pad - 1, Nz + 2 * pad) * delta

        # Coordinate set-up
        zz, xx = np.meshgrid(z, x)
        self.Nx = zz.shape[0]
        self.Nz = zz.shape[1]

        # incident electric and magnetic fields definition on the Yee grid
        fast = np.exp(1j * (k * zz))
        fast_z = np.exp(1j * (k * (zz + delta / 2)))

        self.register_buffer('fast', torch.zeros((1, 2, fast.shape[0], fast.shape[1]), dtype=torch.float32,
                                                 requires_grad=False))
        self.fast[0, 0, :, :] = torch.from_numpy(np.real(fast))
        self.fast[0, 1, :, :] = torch.from_numpy(np.imag(fast))

        self.register_buffer('fast_z', torch.zeros((1, 2, fast_z.shape[0], fast_z.shape[1]), dtype=torch.float32,
                                                   requires_grad=False))
        self.fast_z[0, 0, :, :] = torch.from_numpy(np.real(fast_z))
        self.fast_z[0, 1, :, :] = torch.from_numpy(np.imag(fast_z))

        # x is Bloch/Floquet-periodic (E(x+Lx) = E(x)*e^{i*kx*Lx}) rather
        # than PML-absorbing -- only z gets a perfectly-matched layer.
        self.kx = kx
        Lx = Nx * delta
        self.register_buffer('bloch_phase', torch.zeros(
            (1, 2, 1, 1), dtype=torch.float32, requires_grad=False))
        self.bloch_phase[0, 0, 0, 0] = math.cos(kx * Lx)
        self.bloch_phase[0, 1, 0, 0] = math.sin(kx * Lx)

        # perfectly-matched-layer set up (z only)
        m = 4
        const = 5
        rz_p = 1 + 1j * const * (zz - z[-1] + pml_thickness * delta) ** m
        rz_p[:, 0:-pml_thickness] = 0
        rz_n = 1 + 1j * const * (zz - z[0] - pml_thickness * delta) ** m
        rz_n[:, pml_thickness::] = 0
        rz = rz_p + rz_n
        rz[:, pml_thickness:-pml_thickness] = 1

        rz_inverse = 1 / rz

        self.register_buffer('rz_inverse', torch.zeros((1, 2, rz_inverse.shape[0], rz_inverse.shape[1]), dtype=torch.float32,
                                                       requires_grad=False))
        self.rz_inverse[0, 0, :, :] = torch.from_numpy(np.real(rz_inverse))
        self.rz_inverse[0, 1, :, :] = torch.from_numpy(np.imag(rz_inverse))

        # Gradient and laplacian kernels set up
        self.register_buffer('gradient_h_z', torch.zeros(
            (2, 1, 1, 3), dtype=torch.float32, requires_grad=False))
        self.gradient_h_z[:, :, 0, :] = torch.tensor(
            [-1 / delta, +1 / delta, 0])
        self.register_buffer('gradient_h_x', torch.zeros(
            (2, 1, 3, 1), dtype=torch.float32, requires_grad=False))
        self.gradient_h_x = self.gradient_h_z.permute(0, 1, 3, 2)
        self.register_buffer('gradient_h_z_ho', torch.zeros(
            (2, 1, 1, 5), dtype=torch.float32, requires_grad=False))
        self.gradient_h_z_ho[:, :, 0, :] = torch.tensor(
            [1 / 24 / delta, -9 / 8 / delta, +9 / 8 / delta, -1 / 24 / delta, 0])
        self.register_buffer('gradient_h_x_ho', torch.zeros(
            (2, 1, 5, 1), dtype=torch.float32, requires_grad=False))
        self.gradient_h_x_ho = self.gradient_h_z_ho.permute(0, 1, 3, 2)

        self.register_buffer('gradient_e_z', torch.zeros(
            (2, 1, 1, 3), dtype=torch.float32, requires_grad=False))
        self.gradient_e_z[:, :, 0, :] = torch.tensor(
            [0, -1 / delta, +1 / delta])
        self.register_buffer('gradient_e_x', torch.zeros(
            (2, 1, 3, 1), dtype=torch.float32, requires_grad=False))
        self.gradient_e_x = self.gradient_e_z.permute(0, 1, 3, 2)
        self.register_buffer('gradient_e_z_ho', torch.zeros(
            (2, 1, 1, 5), dtype=torch.float32, requires_grad=False))
        self.gradient_e_z_ho[:, :, 0, :] = torch.tensor(
            [0, 1 / 24 / delta, -9 / 8 / delta, +9 / 8 / delta, -1 / 24 / delta])
        self.register_buffer('gradient_e_x_ho', torch.zeros(
            (2, 1, 5, 1), dtype=torch.float32, requires_grad=False))
        self.gradient_e_x_ho = self.gradient_e_z_ho.permute(0, 1, 3, 2)

        self.register_buffer('dd_z_fast', torch.zeros(
            (1, 2, Nx, Nz), dtype=torch.float32, requires_grad=False))
        self.dd_z_fast = self.dd_z(self.fast)[:, :, self.pad:-self.pad:, :]
        self.register_buffer('dd_z_ho_fast', torch.zeros(
            (1, 2, Nx, Nz), dtype=torch.float32, requires_grad=False))
        self.dd_z_ho_fast = self.dd_z_ho(
            self.fast)[:, :, self.pad:-self.pad:, :]

    def forward(self, scat_pot, ri_value):
        if self.mode == 'te':
            epsillon = scat_pot * \
                (ri_value ** 2).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            epsillon = torch.where(epsillon > 1.0, epsillon, torch.tensor(
                [1], dtype=torch.float32).to(ri_value.device))

            x = self.model(scat_pot)
            total = torch.cat((x[:, 0:1, :, :] + 1, x[:, 1:2, :, :]), 1) # The output is (B,C,W,H), where C=2 for the real and complex channels

            ey = self.complex_multiplication(total[:, 0:2, :, :],
                                             self.fast[:, :, self.pad:-self.pad:, self.pad:-self.pad:])
            ey_i = self.fast
            ey_s = ey - ey_i[:, :, self.pad:-self.pad:, self.pad:-self.pad:]

            if self.symmetry_x == True:
                ey_s = self.padding_ref(ey_s)
            else:
                ey_s = self.padding_zero(ey_s)

            if self.high_order == 'second':
                diff = self.dd_x_periodic(ey) \
                    + self.dd_z_pml(ey_s)[:, :, self.pad:-self.pad, :] \
                    + self.dd_z_fast \
                    + self.k ** 2 * (epsillon * ey)

            elif self.high_order == 'fourth':
                diff = self.dd_x_ho_periodic(ey) \
                    + self.dd_z_ho_pml(ey_s)[:, :, self.pad:-self.pad, :] \
                    + self.dd_z_ho_fast \
                    + self.k ** 2 * (epsillon * ey)

        elif self.mode == 'tm':
            epsillon = scat_pot * \
                (ri_value ** 2).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            epsillon_x = torch.where(epsillon[:, 0:1, :, :] > 1.0, epsillon[:, 0:1, :, :],
                                     torch.tensor([1], dtype=torch.float32).to(ri_value.device))
            epsillon_z = torch.where(epsillon[:, 1:2, :, :] > 1.0, epsillon[:, 1:2, :, :],
                                     torch.tensor([1], dtype=torch.float32).to(ri_value.device))

            x = self.model(scat_pot)
            total = torch.cat((x[:, 0:1, :, :] + 1, x[:, 1:4, :, :]), 1)

            ex = self.complex_multiplication(
                total[:, 0:2, :, :], self.fast[:, :, self.pad:-self.pad:, self.pad:-self.pad:])
            ex_i = self.fast
            ex_s = ex - ex_i[:, :, self.pad:-self.pad:, self.pad:-self.pad:]

            ez_s = self.complex_multiplication(
                total[:, 2:4, :, :], self.fast_z[:, :, self.pad:-self.pad:, self.pad:-self.pad:])
            ez_unpadded = ez_s  # keep a handle before padding_ref/padding_zero overwrites ez_s below

            if self.symmetry_x == True:
                ex_s = self.padding_zero(ex_s)
                ez_s = self.padding_ref(ez_s)
                ex_s[:, :, 0:self.pad, :] = torch.flip(
                    ex_s[:, :, self.pad:2 * self.pad, :], [2])
                ez_s[:, :, 0:self.pad, :] = -ez_s[:, :, 0:self.pad, :]
            else:
                ex_s = self.padding_zero(ex_s)
                ez_s = self.padding_zero(ez_s)

            if self.high_order == 'second':
                diff_x = self.dd_z_pml(ex_s)[:, :, self.pad:-self.pad:, :] \
                    + self.dd_z_fast \
                    - self.dd_zx(ez_s)[:, :, self.pad // 2:-self.pad // 2:, self.pad // 2:-self.pad // 2] \
                    + self.k ** 2 * (epsillon_x * ex) \

                diff_z = self.dd_x_periodic(ez_unpadded) \
                    - self.dd_xz(ex_s)[:, :, self.pad // 2:-self.pad // 2:, self.pad // 2:-self.pad // 2] \
                    + self.k ** 2 * (epsillon_z * ez_s) \

            elif self.high_order == 'fourth':
                # dd_zx_ho_pml/dd_xz_ho_pml (the mixed x/z derivatives tm's
                # fourth-order path needs) combined the z-PML stretch with
                # an x-PML stretch that no longer exists now that x is
                # Bloch-periodic -- porting that mixed derivative to a
                # periodic-x/PML-z combination hasn't been done (and 'tm'
                # was already unsupported by Dataset.py's scat_pot/ri_value
                # adapter), so fail loudly rather than silently reuse the
                # now-nonexistent PML-x treatment.
                raise NotImplementedError(
                    "mode='tm' with high_order='fourth' needs a periodic-x/PML-z mixed "
                    "derivative (dd_zx_ho_periodic/dd_xz_ho_periodic) that hasn't been "
                    "implemented; use high_order='second' (mode='tm') or mode='te'."
                )

            diff = torch.cat((diff_x, diff_z), 1)

        return diff, total

    def complex_multiplication(self, a, b):
        r_p = torch.mul(a[:, 0:1, :, :], b[:, 0:1, :, :]) - \
            torch.mul(a[:, 1:2, :, :], b[:, 1:2, :, :])
        i_p = torch.mul(a[:, 0:1, :, :], b[:, 1:2, :, :]) + \
            torch.mul(a[:, 1:2, :, :], b[:, 0:1, :, :])
        return torch.cat((r_p, i_p), 1)

    def complex_conjugate(self, a):
        # NOTE: previously returned cat((-a[:,1:2], a[:,0:1]), 1), which is
        # actually i*a (a 90-degree rotation), not conj(a) = (re, -im) --
        # this method was defined but never called in master's own forward
        # pass, so that bug was never exercised. Fixed here since
        # _bloch_pad_x (below) is now a real caller that needs the actual
        # conjugate.
        return torch.cat((a[:, 0:1, :, :], -a[:, 1:2, :, :]), 1)

    def d_e_x(self, x):
        return F.conv2d(x, self.gradient_e_x, padding=0, groups=2)

    def d_e_x_ho(self, x):
        return F.conv2d(x, self.gradient_e_x_ho, padding=0, groups=2)

    def d_h_x(self, x):
        return F.conv2d(x, self.gradient_h_x, padding=0, groups=2)

    def d_h_x_ho(self, x):
        return F.conv2d(x, self.gradient_h_x_ho, padding=0, groups=2)

    def d_e_z(self, x):
        return F.conv2d(x, self.gradient_e_z, padding=0, groups=2)

    def d_e_z_ho(self, x):
        return F.conv2d(x, self.gradient_e_z_ho, padding=0, groups=2)

    def d_h_z(self, x):
        return F.conv2d(x, self.gradient_h_z, padding=0, groups=2)

    def d_h_z_ho(self, x):
        return F.conv2d(x, self.gradient_h_z_ho, padding=0, groups=2)

    def dd_x(self, x):
        return self.d_h_x(self.d_e_x(x))

    def dd_x_ho(self, x):
        return self.d_h_x_ho(self.d_e_x_ho(x))

    def _bloch_pad_x(self, x, n):
        """Pad n rows on each side of dim-2 ('x') with a Bloch/Floquet phase
        correction (exp(-i*kx*Lx) on the low side, exp(+i*kx*Lx) on the high
        side) instead of PML-stretching -- x is periodic only up to this
        phase factor (E(x+Lx) = E(x)*e^{i*kx*Lx}) unless kx*Lx happens to be
        an integer multiple of 2*pi, which is generically false for an
        oblique incidence angle."""
        phase = self.bloch_phase
        phase_conj = self.complex_conjugate(phase)
        left = self.complex_multiplication(x[:, :, -n:, :], phase_conj)
        right = self.complex_multiplication(x[:, :, :n, :], phase)
        return torch.cat([left, x, right], dim=2)

    def dd_x_periodic(self, x):
        """∂²/∂x² with Bloch-periodic wraparound, output dim-2 size == x's.
        Two sequential 3-tap (K=3) convs each consume 2 (1/side), so the
        Bloch pad must add self.pad(=2) on each side to land back at x's
        original size."""
        return self.d_h_x(self.d_e_x(self._bloch_pad_x(x, self.pad)))

    def dd_x_ho_periodic(self, x):
        """As dd_x_periodic, but with the 5-tap (K=5) fourth-order kernels,
        each consuming 4 (2/side) -- self.pad(=4) per side again lands back
        at x's original size."""
        return self.d_h_x_ho(self.d_e_x_ho(self._bloch_pad_x(x, self.pad)))

    def dd_z(self, x):
        return self.d_h_z(self.d_e_z(x))

    def dd_z_ho(self, x):
        return self.d_h_z_ho(self.d_e_z_ho(x))

    def dd_z_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, :, 2:-2], self.d_h_z(
            self.complex_multiplication(self.rz_inverse[:, :, :, 1:-1], self.d_e_z(x))))

    def dd_z_ho_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, :, 4:-4], self.d_h_z_ho(
            self.complex_multiplication(self.rz_inverse[:, :, :, 2:-2], self.d_e_z_ho(x))))

    def dd_zx(self, x):
        return self.d_h_z(self.d_e_x(x))

    def dd_zx_ho(self, x):
        return self.d_h_z_ho(self.d_e_x_ho(x))

    def dd_xz(self, x):
        return self.d_h_x(self.d_e_z(x))

    def dd_xz_ho(self, x):
        return self.d_h_x_ho(self.d_e_z_ho(x))
