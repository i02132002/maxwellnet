"""Checkpoint loading helpers for residual-stream notebooks.

Infers PeriodicMaxwellNet ``NetworkSpecs`` / mode from a training
``state_dict``, builds an INIT (Born) net, and loads a FIT checkpoint
without depending on a particular wandb run id.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from ShapeNet import PeriodicMaxwellNet


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata and architecture inferred from a MaxwellNet ``.pt`` file.

    Attributes
    ----------
    path : Path
        Checkpoint file that was read.
    epoch : int or None
        Trainer epoch stored in the checkpoint, if present.
    wandb_run_id : str or None
        Wandb run id embedded by ``train_maxwellnet``, if present.
    mode : str
        Polarization mode (``te`` or ``tm``).
    network_specs : dict
        Keyword arguments for ``PeriodicMaxwellNet`` (depth, filter, …).
    """

    path: Path
    epoch: int | None
    wandb_run_id: str | None
    mode: str
    network_specs: dict


def strip_state_dict(state: dict) -> OrderedDict:
    """Return a state dict with ``module.`` prefixes removed.

    Parameters
    ----------
    state : dict
        Raw ``state_dict`` (possibly DataParallel-wrapped).

    Returns
    -------
    OrderedDict
        Keys suitable for ``PeriodicMaxwellNet.load_state_dict``.
    """
    return OrderedDict((k.removeprefix("module."), v) for k, v in state.items())


def infer_network_specs(state: dict) -> tuple[str, dict]:
    """Infer ``(mode, NetworkSpecs)`` from a PeriodicMaxwellNet state dict.

    Parameters
    ----------
    state : dict
        Preferably already stripped of ``module.`` prefixes. Keys are
        expected under ``model.*`` as saved by ``train_maxwellnet``.

    Returns
    -------
    mode : str
        ``te`` if the first conv has 2 input channels, else ``tm``.
    network_specs : dict
        ``depth``, ``filter``, ``norm``, ``up_mode`` for construction.

    Raises
    ------
    KeyError
        When required UNet weight tensors are missing.
    """
    sd = strip_state_dict(state)
    first_key = "model.down_path.0.block.1.weight_v"
    last_key = "model.last_conv.weight"
    if first_key not in sd:
        raise KeyError(f"missing {first_key} in checkpoint state_dict")
    if last_key not in sd:
        raise KeyError(f"missing {last_key} in checkpoint state_dict")

    in_ch = int(sd[first_key].shape[1])
    filter_w = int(sd[first_key].shape[0])
    mode = "te" if in_ch == 2 else "tm"

    up_idxs = sorted(
        {int(k.split("up_path.")[1].split(".")[0]) for k in sd if "model.up_path." in k}
    )
    depth = len(up_idxs) + 1 if up_idxs else 5

    return mode, {
        "depth": depth,
        "filter": filter_w,
        "norm": "weight",
        "up_mode": "upconv",
    }


def load_checkpoint(path: Path | str) -> tuple[dict, CheckpointInfo]:
    """Load a ``.pt`` file and return ``(stripped_state_dict, info)``.

    Parameters
    ----------
    path : Path or str
        Path to ``latest.pt`` / ``{epoch}_te.pt``.

    Returns
    -------
    state : dict
        Stripped ``state_dict``.
    info : CheckpointInfo
        Epoch, wandb id, and inferred architecture.

    Raises
    ------
    FileNotFoundError
        When ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    state = strip_state_dict(raw)
    mode, network_specs = infer_network_specs(state)
    epoch = None
    wandb_run_id = None
    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            epoch = int(ckpt["epoch"])
        wandb_run_id = ckpt.get("wandb_run_id")
    info = CheckpointInfo(
        path=path.resolve(),
        epoch=epoch,
        wandb_run_id=wandb_run_id,
        mode=mode,
        network_specs=network_specs,
    )
    return state, info


def build_init_net(
    mode: str, network_specs: dict, device: str | torch.device
) -> PeriodicMaxwellNet:
    """Construct an untrained PeriodicMaxwellNet (Born ``a=0`` via zero head).

    Parameters
    ----------
    mode : str
        ``te`` or ``tm``.
    network_specs : dict
        Constructor kwargs (depth, filter, norm, up_mode).
    device : str or torch.device
        Placement device.

    Returns
    -------
    PeriodicMaxwellNet
        Eval-mode model with stock zero-initialized ``last_conv``.
    """
    net = PeriodicMaxwellNet(**network_specs, mode=mode)
    return net.to(device).eval()


def build_fit_net(
    path: Path | str,
    device: str | torch.device,
) -> tuple[PeriodicMaxwellNet, CheckpointInfo]:
    """Load FIT weights from ``path`` into a PeriodicMaxwellNet.

    Parameters
    ----------
    path : Path or str
        Checkpoint path.
    device : str or torch.device
        Placement device.

    Returns
    -------
    net : PeriodicMaxwellNet
        Eval-mode model with loaded weights.
    info : CheckpointInfo
        Inferred architecture and checkpoint metadata.
    """
    state, info = load_checkpoint(path)
    net = build_init_net(info.mode, info.network_specs, device)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"missing keys when loading {path}: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading {path}: {unexpected}")
    return net, info


def list_sibling_snapshots(ckpt_path: Path | str, mode: str = "te") -> list[Path]:
    """List ``{epoch}_{mode}.pt`` files next to a checkpoint (excluding latest).

    Parameters
    ----------
    ckpt_path : Path or str
        Any file in the checkpoint directory (e.g. ``latest.pt``).
    mode : str
        Filename mode suffix.

    Returns
    -------
    list of Path
        Sorted by epoch number.
    """
    directory = Path(ckpt_path).parent
    suffix = f"_{mode}.pt"
    numbered: list[tuple[int, Path]] = []
    for path in directory.glob(f"*{suffix}"):
        stem = path.name[: -len(suffix)]
        if stem.isdigit():
            numbered.append((int(stem), path))
    return [p for _, p in sorted(numbered)]
