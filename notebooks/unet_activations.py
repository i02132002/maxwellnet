"""Activation probes for PeriodicMaxwellNet's UNet encoder/decoder path.

Captures feature maps from contrast input through the bottleneck (with
incidence conditioning) and back up to the envelope head. Used by the
geometry-selectable residual-stream notebook for INIT vs FIT tours.

UNet skip links are concatenations, not an additive ViT residual stream.
Do not treat these maps as the registers-paper residual-stream analysis.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from ShapeNet import PeriodicMaxwellNet
from UNet import UNet


@dataclass(frozen=True)
class ActivationMap:
    """One named activation tensor from the UNet path.

    Attributes
    ----------
    name : str
        Stage label (e.g. ``enc_0``, ``bottleneck``, ``dec_0``, ``head``).
    tensor : torch.Tensor
        Feature map ``(B, C, H, W)`` on CPU.
    """

    name: str
    tensor: torch.Tensor

    @property
    def energy(self) -> torch.Tensor:
        """Channel-mean absolute map ``(B, H, W)``."""
        return self.tensor.float().abs().mean(dim=1)


class UNetPathProbe:
    """Record encoder → bottleneck → decoder activations on a PeriodicMaxwellNet.

    Parameters
    ----------
    net : PeriodicMaxwellNet
        Model whose ``.model`` attribute is a ``UNet``.
    """

    def __init__(self, net: PeriodicMaxwellNet):
        if not isinstance(net, PeriodicMaxwellNet):
            raise TypeError(f"expected PeriodicMaxwellNet, got {type(net).__name__}")
        if not isinstance(net.model, UNet):
            raise TypeError(
                f"expected net.model to be UNet, got {type(net.model).__name__}. "
                "SwinMaxwellNet is a different backbone."
            )
        self.net = net
        self.activations: OrderedDict[str, ActivationMap] = OrderedDict()
        self._handles: list = []

    def clear(self) -> None:
        """Drop stored activations without removing hooks."""
        self.activations.clear()

    def _store(self, name: str, tensor: torch.Tensor) -> None:
        self.activations[name] = ActivationMap(name=name, tensor=tensor.detach().cpu())

    def _hook(self, name: str):
        def _fn(_module, _inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            self._store(name, tensor)

        return _fn

    def attach(self) -> None:
        """Install forward hooks on encoder, bottleneck, decoder, and head."""
        self.detach()
        unet: UNet = self.net.model
        n_down = len(unet.down_path)
        enc_idx = 0
        for i, layer in enumerate(unet.down_path):
            if i == n_down - 1:
                self._handles.append(
                    layer.register_forward_hook(self._hook("bottleneck"))
                )
            elif i % 2 == 0:
                self._handles.append(
                    layer.register_forward_hook(self._hook(f"enc_{enc_idx}"))
                )
                enc_idx += 1
            else:
                self._handles.append(
                    layer.register_forward_hook(self._hook(f"pool_{enc_idx - 1}"))
                )
        for j, layer in enumerate(unet.up_path):
            self._handles.append(layer.register_forward_hook(self._hook(f"dec_{j}")))
        self._handles.append(unet.last_conv.register_forward_hook(self._hook("head")))

    def detach(self) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def recording(self):
        """Attach hooks, clear buffers, yield, then detach."""
        self.attach()
        self.clear()
        try:
            yield self
        finally:
            self.detach()


def contrast_and_cond(
    net: PeriodicMaxwellNet,
    optical_constant: torch.Tensor,
    theta: torch.Tensor,
    wavelength_nm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Reproduce PeriodicMaxwellNet's UNet inputs without running the head.

    Parameters
    ----------
    net : PeriodicMaxwellNet
        Provides epsilon selection, contrast scale, padding, and k-vector.
    optical_constant : torch.Tensor
        ``(B, Nz, Nx, 3)`` complex refractive index.
    theta, wavelength_nm : torch.Tensor
        ``(B,)`` incidence angle (deg) and vacuum wavelength (nm).

    Returns
    -------
    contrast_padded : torch.Tensor
        ``(B, 2k, H', W')`` scaled ``(ε−1)`` real/imag, padded to UNet divisor.
    cond : torch.Tensor
        ``(B, 2)`` incident ``(kx, kz)`` in Å⁻¹.
    contrast_unpadded : torch.Tensor
        Same contrast before padding, native grid.
    pad_z, pad_x : int
        Padding amounts applied along z and x.
    """
    eps = net._select_epsilon(optical_constant.to(torch.complex64))
    contrast = (eps - 1.0) * net.contrast_scale
    contrast_unpadded = torch.cat([contrast.real, contrast.imag], dim=1)
    contrast_padded, pad_z, pad_x = net._pad_to_divisor(contrast_unpadded)
    kx, kz = net._incident_k_vector(theta, wavelength_nm)
    cond = torch.stack([kx, kz], dim=1)
    return contrast_padded, cond, contrast_unpadded, pad_z, pad_x


def forward_with_activations(
    net: PeriodicMaxwellNet,
    probe: UNetPathProbe,
    optical_constant: torch.Tensor,
    theta: torch.Tensor,
    wavelength_nm: torch.Tensor,
    delta_x_a: torch.Tensor,
    delta_z_a: torch.Tensor,
) -> dict:
    """Run one forward pass and return physics outputs plus UNet activations.

    Parameters
    ----------
    net : PeriodicMaxwellNet
        Eval-mode model.
    probe : UNetPathProbe
        Probe attached to ``net``.
    optical_constant, theta, wavelength_nm, delta_x_a, delta_z_a : torch.Tensor
        Batched PeriodicMaxwellNet inputs.

    Returns
    -------
    dict
        Keys: ``envelope``, ``eps``, ``incident``, ``kz``, ``kx``,
        ``contrast``, ``cond``, ``activations``, ``pad_z``, ``pad_x``.
    """
    _contrast_padded, cond, contrast_unpadded, pad_z, pad_x = contrast_and_cond(
        net, optical_constant, theta, wavelength_nm
    )
    with probe.recording():
        envelope, eps, incident, kz, kx = net(
            optical_constant, theta, wavelength_nm, delta_x_a, delta_z_a
        )
        activations = OrderedDict((k, v) for k, v in probe.activations.items())

    expected = (
        [f"enc_{i}" for i in range(len(net.model.up_path))]
        + [f"pool_{i}" for i in range(len(net.model.up_path))]
        + ["bottleneck"]
        + [f"dec_{i}" for i in range(len(net.model.up_path))]
        + ["head"]
    )
    missing = [name for name in expected if name not in activations]
    if missing:
        raise RuntimeError(f"UNet probe missed activations: {missing}")

    return {
        "envelope": envelope.detach().cpu(),
        "eps": eps.detach().cpu(),
        "incident": incident.detach().cpu(),
        "kz": kz.detach().cpu(),
        "kx": kx.detach().cpu(),
        "contrast": contrast_unpadded.detach().cpu(),
        "cond": cond.detach().cpu(),
        "activations": activations,
        "pad_z": pad_z,
        "pad_x": pad_x,
    }


def path_order(activations: OrderedDict[str, ActivationMap]) -> list[str]:
    """Return activation keys in encoder → bottleneck → decoder → head order."""
    enc = sorted(
        (k for k in activations if k.startswith("enc_")),
        key=lambda s: int(s.split("_")[1]),
    )
    pools = sorted(
        (k for k in activations if k.startswith("pool_")),
        key=lambda s: int(s.split("_")[1]),
    )
    # Interleave enc_i, pool_i for a readable down-path, then bottleneck/dec/head.
    down = []
    for e, p in zip(enc, pools):
        down.extend([e, p])
    rest = ["bottleneck"] + sorted(
        (k for k in activations if k.startswith("dec_")),
        key=lambda s: int(s.split("_")[1]),
    )
    if "head" in activations:
        rest.append("head")
    return down + rest


# Human-readable labels for figure titles (depth=5 PeriodicMaxwellNet / UNet).
STAGE_LABELS: dict[str, str] = {
    "contrast Re": "input: Re[(eps-1)*scale]",
    "contrast Im": "input: Im[(eps-1)*scale]",
    "enc_0": "enc_0: first conv block (full res, skip #0)",
    "pool_0": "pool_0: AvgPool 256->128",
    "enc_1": "enc_1: conv block (skip #1)",
    "pool_1": "pool_1: AvgPool 128->64",
    "enc_2": "enc_2: conv block (skip #2)",
    "pool_2": "pool_2: AvgPool 64->32",
    "enc_3": "enc_3: conv block (skip #3)",
    "pool_3": "pool_3: AvgPool 32->16",
    "bottleneck": "bottleneck: deepest latents + (kx,kz) concat",
    "dec_0": "dec_0: upsample + concat skip enc_3",
    "dec_1": "dec_1: upsample + concat skip enc_2",
    "dec_2": "dec_2: upsample + concat skip enc_1",
    "dec_3": "dec_3: upsample + concat skip enc_0",
    "head": "head: 1x1 conv -> envelope channels (a)",
}


def stage_title(key: str, shape: tuple[int, ...] | None = None) -> str:
    """Build a figure subtitle for an activation key.

    Parameters
    ----------
    key : str
        Internal activation name (``enc_0``, ``bottleneck``, …) or contrast label.
    shape : tuple or None
        Optional tensor spatial/feature shape to append.

    Returns
    -------
    str
        Multi-line title string.
    """
    label = STAGE_LABELS.get(key, key)
    if shape is None:
        return label
    return f"{label}\n{tuple(shape)}"
