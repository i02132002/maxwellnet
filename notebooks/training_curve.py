"""Training-curve helpers for scrubbing MaxwellNet checkpoints.

Loads a compact wandb history cache (epoch/step loss), finds inflection-like
epochs on a smoothed log-loss curve, and resolves local checkpoint paths of
the form ``{epoch}_te.pt`` / ``latest.pt``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LossCurve:
    """Epoch-aligned training loss for one run.

    Attributes
    ----------
    epochs : ndarray
        Epoch indices (int), sorted ascending.
    loss : ndarray
        Train loss at those epochs (float).
    steps : ndarray
        Optional per-step indices (may be empty).
    step_loss : ndarray
        Optional per-step losses aligned with ``steps``.
    snapshot_frequency : int
        Trainer ``SnapshotFrequency`` (named epoch checkpoints).
    latest_every : int
        How often ``latest.pt`` was overwritten during training.
    mode : str
        Polarization mode used in checkpoint filenames (``te`` / ``tm``).
    """

    epochs: np.ndarray
    loss: np.ndarray
    steps: np.ndarray
    step_loss: np.ndarray
    snapshot_frequency: int
    latest_every: int
    mode: str
    run_id: str

    @classmethod
    def from_history_json(cls, path: Path | str) -> LossCurve:
        """Load a compact history cache written next to the analysis notebook.

        Parameters
        ----------
        path : Path or str
            JSON with ``epoch_loss``, optional ``step_loss``, and metadata.

        Returns
        -------
        LossCurve
            Parsed arrays and checkpoint cadence metadata.
        """
        data = json.loads(Path(path).read_text())
        meta: dict = {}
        if isinstance(data, list):
            epochs = np.asarray([r["epoch"] for r in data], dtype=np.int64)
            loss = np.asarray([r["loss"] for r in data], dtype=np.float64)
            steps = np.asarray([], dtype=np.int64)
            step_loss = np.asarray([], dtype=np.float64)
        elif "epoch_loss" in data:
            epoch_loss = data["epoch_loss"]
            epochs = np.asarray([r["epoch"] for r in epoch_loss], dtype=np.int64)
            loss = np.asarray([r["loss"] for r in epoch_loss], dtype=np.float64)
            step_rows = data.get("step_loss") or []
            steps = np.asarray([r["step"] for r in step_rows], dtype=np.int64)
            step_loss = np.asarray([r["loss"] for r in step_rows], dtype=np.float64)
            meta = data
        else:
            epochs = np.asarray(data["epoch"], dtype=np.int64)
            loss = np.asarray(data["loss"], dtype=np.float64)
            steps = np.asarray(data.get("step") or [], dtype=np.int64)
            step_loss = np.asarray(data.get("step_loss") or [], dtype=np.float64)
            meta = data
        return cls(
            epochs=epochs,
            loss=loss,
            steps=steps,
            step_loss=step_loss,
            snapshot_frequency=int(
                meta.get("snapshot_frequency", meta.get("snapshot_every", 100))
            ),
            latest_every=int(meta.get("latest_every", 100)),
            mode=str(meta.get("mode", "te")),
            run_id=str(meta.get("run_id", "")),
        )

    @classmethod
    def from_overfit_dir(cls, ckpt_dir: Path | str) -> LossCurve:
        """Load loss history + snapshot cadence from an overfit checkpoint directory.

        Parameters
        ----------
        ckpt_dir : Path or str
            Directory containing ``loss_history.json`` and optional ``run_meta.json``.

        Returns
        -------
        LossCurve
            Epoch losses with ``snapshot_frequency`` taken from ``run_meta.json``
            when present.
        """
        ckpt_dir = Path(ckpt_dir)
        hist_path = ckpt_dir / "loss_history.json"
        curve = cls.from_history_json(hist_path)
        meta_path = ckpt_dir / "run_meta.json"
        if not meta_path.is_file():
            return curve
        meta = json.loads(meta_path.read_text())
        return cls(
            epochs=curve.epochs,
            loss=curve.loss,
            steps=curve.steps,
            step_loss=curve.step_loss,
            snapshot_frequency=int(
                meta.get("snapshot_every", curve.snapshot_frequency)
            ),
            latest_every=int(meta.get("snapshot_every", curve.latest_every)),
            mode=str((meta.get("physical_specs") or {}).get("mode", curve.mode)),
            run_id=str(meta.get("wandb_run_id", curve.run_id)),
        )


def smooth_log_loss(loss: np.ndarray, window: int = 51) -> np.ndarray:
    """Return a centered moving average of ``log10(loss)``.

    Parameters
    ----------
    loss : ndarray
        Positive loss values.
    window : int
        Odd window length in epochs (clamped to ``len(loss)`` if needed).

    Returns
    -------
    ndarray
        Smoothed log10 loss, same length as ``loss``.
    """
    y = np.log10(np.clip(loss, 1e-30, None))
    w = int(window)
    if w < 3:
        return y
    if w % 2 == 0:
        w += 1
    if w >= len(y):
        w = len(y) if len(y) % 2 == 1 else max(len(y) - 1, 1)
    if w < 3:
        return y
    kernel = np.ones(w, dtype=np.float64) / w
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def find_inflection_epochs(
    curve: LossCurve,
    window: int = 51,
    max_points: int = 8,
    min_separation: int = 500,
) -> list[dict]:
    """Locate strong curvature / slope-change epochs on smoothed log-loss.

    Uses peaks of ``|d²/de² log10(loss)|`` after smoothing, with a minimum
    separation so nearby bumps do not all get listed. Always includes the
    first and last epoch as anchors for scrubbing.

    Parameters
    ----------
    curve : LossCurve
        Epoch loss curve.
    window : int
        Smoothing window for ``smooth_log_loss``.
    max_points : int
        Maximum interior inflection epochs to return (excluding endpoints).
    min_separation : int
        Minimum epoch gap between reported interior points.

    Returns
    -------
    list of dict
        Each entry has ``epoch``, ``loss``, ``kind`` (``start`` / ``inflection``
        / ``end``), and ``curvature`` (0 for endpoints).
    """
    epochs = curve.epochs
    loss = curve.loss
    if len(epochs) == 0:
        return []

    log_s = smooth_log_loss(loss, window=window)
    if len(log_s) >= 3:
        d2 = np.gradient(np.gradient(log_s))
        score = np.abs(d2)
        order = np.argsort(score)[::-1]
    else:
        score = np.zeros_like(loss)
        order = np.arange(len(loss))

    chosen: list[int] = []
    for idx in order:
        ep = int(epochs[idx])
        if ep == int(epochs[0]) or ep == int(epochs[-1]):
            continue
        if any(abs(ep - int(epochs[j])) < min_separation for j in chosen):
            continue
        chosen.append(int(idx))
        if len(chosen) >= max_points:
            break
    chosen.sort(key=lambda i: int(epochs[i]))

    out = [
        {
            "epoch": int(epochs[0]),
            "loss": float(loss[0]),
            "kind": "start",
            "curvature": 0.0,
        }
    ]
    for i in chosen:
        out.append(
            {
                "epoch": int(epochs[i]),
                "loss": float(loss[i]),
                "kind": "inflection",
                "curvature": float(score[i]),
            }
        )
    if int(epochs[-1]) != int(epochs[0]):
        out.append(
            {
                "epoch": int(epochs[-1]),
                "loss": float(loss[-1]),
                "kind": "end",
                "curvature": 0.0,
            }
        )
    return out


def list_local_checkpoints(ckpt_dir: Path | str, mode: str = "te") -> dict[str, Path]:
    """Index local checkpoint files by label.

    Parameters
    ----------
    ckpt_dir : Path or str
        Directory containing ``{epoch}_{mode}.pt`` and optional ``latest.pt``.
    mode : str
        Mode suffix in filenames.

    Returns
    -------
    dict
        Maps label (``\"0\"``, ``\"1000\"``, ``\"latest\"``, ...) to path.
    """
    ckpt_dir = Path(ckpt_dir)
    found: dict[str, Path] = {}
    if not ckpt_dir.is_dir():
        return found
    latest = ckpt_dir / "latest.pt"
    if latest.is_file():
        found["latest"] = latest
    suffix = f"_{mode}.pt"
    for path in sorted(ckpt_dir.glob(f"*{suffix}")):
        stem = path.name[: -len(suffix)]
        if stem.isdigit():
            found[stem] = path
    return found


def resolve_checkpoint(
    ckpt_dir: Path | str,
    epoch: int | str,
    mode: str = "te",
    snapshot_frequency: int = 1000,
) -> Path | None:
    """Resolve the best local checkpoint path for a requested epoch.

    Prefers an exact ``{epoch}_{mode}.pt``. Otherwise snaps to the nearest
    available snapshot epoch (multiples of ``snapshot_frequency``) or
    ``latest.pt`` if nothing closer exists.

    Parameters
    ----------
    ckpt_dir : Path or str
        Checkpoint directory.
    epoch : int or str
        Requested epoch, or ``\"latest\"``.
    mode : str
        Filename mode suffix.
    snapshot_frequency : int
        Expected spacing of named epoch checkpoints.

    Returns
    -------
    Path or None
        Path if a usable file exists locally.
    """
    available = list_local_checkpoints(ckpt_dir, mode=mode)
    if not available:
        return None
    if epoch == "latest" or epoch == -1:
        if "latest" in available:
            return available["latest"]
        numbered = sorted((int(k), p) for k, p in available.items() if k.isdigit())
        return numbered[-1][1] if numbered else None

    ep = int(epoch)
    key = str(ep)
    if key in available:
        return available[key]

    numbered = sorted((int(k), p) for k, p in available.items() if k.isdigit())
    if not numbered:
        return available.get("latest")

    # Nearest stored epoch by absolute distance (ties → earlier snap).
    nearest_epoch, nearest_path = min(
        numbered, key=lambda item: (abs(item[0] - ep), item[0])
    )
    return nearest_path


def nearest_weight_epoch(
    ckpt_dir: Path | str,
    epoch: int,
    mode: str = "te",
) -> tuple[int | None, Path | None]:
    """Return ``(epoch_in_filename, path)`` for the nearest local weight file.

    Parameters
    ----------
    ckpt_dir : Path or str
        Checkpoint directory.
    epoch : int
        Desired training epoch (inflection, scrub target, …).
    mode : str
        Filename mode suffix.

    Returns
    -------
    epoch_loaded : int or None
        Epoch encoded in ``{epoch}_{mode}.pt``, or ``None`` if only ``latest.pt``
        / nothing is available.
    path : Path or None
        Resolved file path.
    """
    available = list_local_checkpoints(ckpt_dir, mode=mode)
    numbered = sorted((int(k), p) for k, p in available.items() if k.isdigit())
    if numbered:
        ep_loaded, path = min(
            numbered, key=lambda item: (abs(item[0] - int(epoch)), item[0])
        )
        return ep_loaded, path
    if "latest" in available:
        return None, available["latest"]
    return None, None


def map_targets_to_available_weights(
    target_epochs: list[int],
    ckpt_dir: Path | str,
    mode: str = "te",
    include_init: bool = True,
) -> list[dict]:
    """Map requested scrub/inflection epochs onto nearest stored checkpoints.

    Parameters
    ----------
    target_epochs : list of int
        Desired epochs (e.g. inflection points). ``0`` means random init.
    ckpt_dir : Path or str
        Local checkpoint directory.
    mode : str
        Filename mode suffix.
    include_init : bool
        Always keep epoch ``0`` as a zero-init reference when True.

    Returns
    -------
    list of dict
        Each entry has ``target_epoch``, ``weight_epoch``, ``path``, ``kind``.
        ``kind`` is ``init``, ``exact``, ``nearest``, or ``missing``.
    """
    rows: list[dict] = []
    seen_paths: set[str] = set()
    for target in target_epochs:
        target = int(target)
        if target == 0 and include_init:
            rows.append(
                {
                    "target_epoch": 0,
                    "weight_epoch": 0,
                    "path": None,
                    "kind": "init",
                }
            )
            continue
        weight_epoch, path = nearest_weight_epoch(ckpt_dir, target, mode=mode)
        if path is None:
            rows.append(
                {
                    "target_epoch": target,
                    "weight_epoch": None,
                    "path": None,
                    "kind": "missing",
                }
            )
            continue
        key = str(path.resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)
        kind = "exact" if weight_epoch == target else "nearest"
        rows.append(
            {
                "target_epoch": target,
                "weight_epoch": weight_epoch,
                "path": path,
                "kind": kind,
            }
        )
    return rows


def suggested_scrub_epochs(
    curve: LossCurve, inflections: list[dict] | None = None
) -> list[int]:
    """Build a short list of epochs to scrub: start, inflections, snapshots, end.

    Parameters
    ----------
    curve : LossCurve
        Training curve metadata (uses ``snapshot_frequency``).
    inflections : list of dict or None
        Output of ``find_inflection_epochs``; computed if omitted.

    Returns
    -------
    list of int
        Unique sorted epoch numbers.
    """
    if inflections is None:
        inflections = find_inflection_epochs(curve)
    epochs = {int(p["epoch"]) for p in inflections}
    freq = max(int(curve.snapshot_frequency), 1)
    last = int(curve.epochs[-1]) if len(curve.epochs) else 0
    for ep in range(freq, last + 1, freq):
        epochs.add(ep)
    epochs.add(0)
    return sorted(epochs)
