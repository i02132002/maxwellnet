"""Registers-style residual-stream probes for SwinMaxwellNet.

Implements the token-norm analysis from Darcet et al., \"Vision Transformers
Need Registers\" (ICLR 2024): after each transformer block, measure the L2
norm of every spatial token in the residual stream, map those norms back to
``(H, W)``, and flag high-norm outliers.

This analysis only applies to architectures with an *additive* residual
stream (``SwinBlock``). UNet skip concatenations in ``PeriodicMaxwellNet``
are not a residual stream and must not be treated as one.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from Swin import SwinBlock, SwinMaxwellNet


@dataclass(frozen=True)
class StreamSnapshot:
    """One residual-stream state after a residual add inside a ``SwinBlock``.

    Attributes
    ----------
    name : str
        Fully-qualified module path of the ``SwinBlock``.
    tag : str
        ``\"post_attn\"`` or ``\"post_mlp\"``.
    tokens : torch.Tensor
        Residual stream ``(B, H, W, C)`` on CPU.
    """

    name: str
    tag: str
    tokens: torch.Tensor

    @property
    def norms(self) -> torch.Tensor:
        """Per-token L2 norms with shape ``(B, H, W)``."""
        return torch.linalg.vector_norm(self.tokens.float(), dim=-1)


class ResidualStreamProbe:
    """Attach/detach hooks that record Swin residual-stream states.

    Parameters
    ----------
    root : nn.Module
        Typically a ``SwinMaxwellNet`` (or its ``.model`` backbone). Every
        nested ``SwinBlock`` is instrumented.
    """

    def __init__(self, root: nn.Module):
        self.root = root
        self.snapshots: OrderedDict[str, StreamSnapshot] = OrderedDict()
        self._blocks: list[tuple[str, SwinBlock]] = [
            (name, mod) for name, mod in root.named_modules() if isinstance(mod, SwinBlock)
        ]
        if not self._blocks:
            raise TypeError(
                f"{type(root).__name__} contains no SwinBlock modules; "
                "registers-style residual-stream analysis requires SwinMaxwellNet"
            )

    def clear(self) -> None:
        """Drop stored snapshots without removing hooks."""
        self.snapshots.clear()

    def _make_hook(self, qualname: str):
        def _hook(block: SwinBlock, tag: str, tokens: torch.Tensor, H: int, W: int) -> None:
            B, L, C = tokens.shape
            if L != H * W:
                raise RuntimeError(f"token length {L} != H*W={H * W} for {qualname}/{tag}")
            key = f"{qualname}/{tag}"
            self.snapshots[key] = StreamSnapshot(
                name=qualname,
                tag=tag,
                tokens=tokens.detach().cpu().view(B, H, W, C),
            )

        return _hook

    def attach(self) -> None:
        """Install stream hooks on every ``SwinBlock`` under ``root``."""
        for qualname, block in self._blocks:
            block.stream_hook = self._make_hook(qualname)

    def detach(self) -> None:
        """Remove stream hooks from every instrumented ``SwinBlock``."""
        for _, block in self._blocks:
            block.stream_hook = None

    @contextmanager
    def recording(self):
        """Context manager that attaches hooks, clears, yields, then detaches."""
        self.attach()
        self.clear()
        try:
            yield self
        finally:
            self.detach()


def token_norm_map(snapshot: StreamSnapshot, batch_index: int = 0) -> np.ndarray:
    """Return the ``(H, W)`` L2-norm map for one batch element.

    Parameters
    ----------
    snapshot : StreamSnapshot
        Recorded residual-stream state.
    batch_index : int
        Batch element to visualize.

    Returns
    -------
    ndarray
        Float32 array of shape ``(H, W)``.
    """
    return snapshot.norms[batch_index].numpy().astype(np.float32)


def outlier_mask(norms: np.ndarray, cutoff: float | None = None, z_score: float = 5.0) -> tuple[np.ndarray, float]:
    """Flag high-norm tokens using a cutoff or robust z-score rule.

    Matches the registers paper's qualitative criterion (tokens whose norm
    sits far above the bulk of the distribution). Default uses median/MAD
    so it adapts across layers and model scales without a hand-picked 150.

    Parameters
    ----------
    norms : ndarray
        Flat or spatial norm array.
    cutoff : float or None
        Absolute threshold. When None, use ``median + z_score * 1.4826 * MAD``.
    z_score : float
        Robust z threshold when ``cutoff`` is None.

    Returns
    -------
    mask : ndarray of bool
        True where a token is an outlier (same shape as ``norms``).
    used_cutoff : float
        Threshold that was applied.
    """
    flat = norms.reshape(-1).astype(np.float64)
    if cutoff is None:
        med = float(np.median(flat))
        mad = float(np.median(np.abs(flat - med)))
        if mad < 1e-8 * (abs(med) + 1.0):
            # Degenerate (nearly constant norms): fall back to top-1% rule so a
            # flat map does not mark every token as an outlier.
            cutoff = float(np.quantile(flat, 0.99))
        else:
            cutoff = med + z_score * 1.4826 * mad
    return norms >= cutoff, float(cutoff)


def layer_norm_summary(probe: ResidualStreamProbe, tag: str = "post_mlp", batch_index: int = 0) -> list[dict]:
    """Summarize token-norm statistics for every recorded block at one residual site.

    Parameters
    ----------
    probe : ResidualStreamProbe
        Probe that already holds snapshots from a forward pass.
    tag : str
        ``\"post_mlp\"`` (block output / paper Fig. 3) or ``\"post_attn\"``.
    batch_index : int
        Batch element to summarize.

    Returns
    -------
    list of dict
        One entry per block with keys ``key``, ``mean``, ``median``, ``max``,
        ``p99``, ``outlier_frac``, ``cutoff``, ``shape``.
    """
    rows = []
    for key, snap in probe.snapshots.items():
        if snap.tag != tag:
            continue
        norms = token_norm_map(snap, batch_index)
        mask, cutoff = outlier_mask(norms)
        rows.append(
            {
                "key": key,
                "mean": float(norms.mean()),
                "median": float(np.median(norms)),
                "max": float(norms.max()),
                "p99": float(np.quantile(norms, 0.99)),
                "outlier_frac": float(mask.mean()),
                "cutoff": cutoff,
                "shape": tuple(norms.shape),
            }
        )
    return rows


def ensure_swin(net: nn.Module) -> SwinMaxwellNet:
    """Validate that ``net`` is a ``SwinMaxwellNet`` suitable for stream probing.

    Parameters
    ----------
    net : nn.Module
        Candidate model.

    Returns
    -------
    SwinMaxwellNet
        The same object, typed for callers.

    Raises
    ------
    TypeError
        If ``net`` is not a ``SwinMaxwellNet``.
    """
    if not isinstance(net, SwinMaxwellNet):
        raise TypeError(
            f"residual-stream analysis requires SwinMaxwellNet, got {type(net).__name__}. "
            "rx9tbiai's PeriodicMaxwellNet/UNet has skip concatenations, not an additive "
            "ViT residual stream (Darcet et al. 2024)."
        )
    return net
