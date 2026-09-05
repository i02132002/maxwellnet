"""Quick supervised overfit of PeriodicMaxwellNet on sample_0000 (theta 0/45).

Loads only ``training/train-00000-of-00001.parquet`` (avoids the multi-shard
HF glob), trains with ``LossSpecs.SupervisedCoef=1`` / no Helmholtz term, and
writes checkpoints under ``checkpoints/overfit_sample0000/`` for the
activation notebook. Logs metrics and ``.pt`` snapshots to wandb.

Architecture matches ``rx9tbiai``: depth=5, filter=32, TE, weight-norm UNet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Dataset import ShapeDataset
from ShapeNet import PeriodicMaxwellNet
from train_maxwellnet import _compute_loss

DEFAULT_PARQUET = (
    Path.home()
    / ".cache/huggingface/hub/datasets--als-rixs--latent-image-training"
    / "snapshots/3eb3f19dbd2b972031f99e4a38398f8c683de40e"
    / "training/train-00000-of-00001.parquet"
)

NETWORK_SPECS = {
    "norm": "weight",
    "depth": 5,
    "filter": 32,
    "up_mode": "upconv",
}


def resolve_parquet(explicit: Path | None) -> Path:
    """Return a readable parquet path for the single-shard training file.

    Parameters
    ----------
    explicit : Path or None
        User-supplied path; used when it exists.

    Returns
    -------
    Path
        Existing parquet file.

    Raises
    ------
    FileNotFoundError
        When neither ``explicit`` nor the default hub snapshot is present.
    """
    if explicit is not None and explicit.is_file():
        return explicit
    if DEFAULT_PARQUET.is_file():
        return DEFAULT_PARQUET
    raise FileNotFoundError(
        f"parquet not found (tried {explicit} and {DEFAULT_PARQUET}). "
        "Download als-rixs/latent-image-training training/train-00000-of-00001.parquet first."
    )


def load_sample0000_te(parquet: Path) -> ShapeDataset:
    """Build a ShapeDataset with only ``sample_0000`` s-pol rows from one parquet.

    Parameters
    ----------
    parquet : Path
        Local ``train-00000-of-00001.parquet``.

    Returns
    -------
    ShapeDataset
        Cropped 256x256 rows for theta in {0, 45} (whatever the shard holds).
    """
    ds = load_dataset("parquet", data_files=str(parquet), split="train")
    indices = [
        i
        for i, row in enumerate(ds)
        if row["sample_id"] == "sample_0000" and row["pol"] == "s"
    ]
    if not indices:
        raise RuntimeError(f"no sample_0000 s-pol rows in {parquet}")
    subset = ds.select(indices).select_columns(ShapeDataset._COLUMNS)
    subset.set_format(
        type="torch",
        columns=ShapeDataset._ARRAY_COLUMNS,
        output_all_columns=True,
    )
    return ShapeDataset(hf_dataset=subset)


def save_checkpoint(state: dict, out_dir: Path, stem: str) -> Path:
    """Write ``{stem}.pt`` under ``out_dir`` and return the path.

    Parameters
    ----------
    state : dict
        Torch checkpoint payload (epoch, state_dict, optimizer, …).
    out_dir : Path
        Destination directory (created if missing).
    stem : str
        Filename without ``.pt``.

    Returns
    -------
    Path
        Written checkpoint path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pt"
    torch.save(state, path)
    return path


def snapshot_epochs(epochs: int, every: int) -> list[int]:
    """Build inclusive checkpoint epochs: 0, every N, and final.

    Parameters
    ----------
    epochs : int
        Final training epoch (>=1).
    every : int
        Spacing between named snapshots (>=1).

    Returns
    -------
    list of int
        Sorted unique epochs to write ``{epoch}_te.pt`` for.
    """
    points = {0, int(epochs)}
    points.update(range(every, epochs + 1, every))
    return sorted(points)


def upload_pt(path: Path, epoch: int, *, also_latest: bool = False) -> None:
    """Log a checkpoint file (and optional latest alias) as a wandb model artifact.

    Parameters
    ----------
    path : Path
        Local ``.pt`` file to upload.
    epoch : int
        Epoch encoded in the artifact name / metadata.
    also_latest : bool
        If True, also log an alias artifact named ``latest``.
    """
    artifact = wandb.Artifact(
        name=f"overfit_sample0000_epoch_{epoch}",
        type="model",
        metadata={"epoch": epoch, "sample_id": "sample_0000"},
    )
    artifact.add_file(str(path), name=path.name)
    wandb.log_artifact(artifact, aliases=[f"epoch-{epoch}", "snapshot"])
    if also_latest:
        latest = wandb.Artifact(
            name="overfit_sample0000_latest",
            type="model",
            metadata={"epoch": epoch, "sample_id": "sample_0000"},
        )
        latest.add_file(str(path), name="latest.pt")
        wandb.log_artifact(latest, aliases=["latest", f"epoch-{epoch}"])
    wandb.save(str(path), base_path=str(path.parent), policy="now")


def train(
    epochs: int,
    out_dir: Path,
    parquet: Path | None,
    lr: float,
    snapshot_every: int,
    seed: int,
    wandb_entity: str,
    wandb_project: str,
    wandb_mode: str,
) -> tuple[Path, str]:
    """Run supervised overfit with wandb logging; return latest path and run URL.

    Parameters
    ----------
    epochs : int
        Number of training epochs (>=1).
    out_dir : Path
        Checkpoint directory.
    parquet : Path or None
        Optional parquet override.
    lr : float
        Adam learning rate.
    snapshot_every : int
        Named ``{epoch}_te.pt`` cadence (also always includes 0 and final).
    seed : int
        RNG seed for reproducibility.
    wandb_entity, wandb_project : str
        Wandb destination.
    wandb_mode : str
        Wandb mode (``online``, ``offline``, ``disabled``).

    Returns
    -------
    latest_path : Path
        Path to ``latest.pt`` after the final epoch.
    run_url : str
        Wandb run URL (empty if disabled).
    """
    torch.manual_seed(seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parquet_path = resolve_parquet(parquet)
    dataset = load_sample0000_te(parquet_path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False
    )

    mode = "te"
    model = PeriodicMaxwellNet(**NETWORK_SPECS, mode=mode).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=max(epochs // 2, 1), gamma=0.5)

    snap_epochs = set(snapshot_epochs(epochs, snapshot_every))
    meta = {
        "sample_id": "sample_0000",
        "thetas_deg": [float(dataset[i]["theta"]) for i in range(len(dataset))],
        "network_arch": "periodicmaxwellnet",
        "network_specs": NETWORK_SPECS,
        "loss_specs": {"SupervisedCoef": 1.0, "HelmholtzCoef": None},
        "physical_specs": {"mode": mode, "pml_thickness": 10},
        "epochs": epochs,
        "lr": lr,
        "snapshot_every": snapshot_every,
        "snapshot_epochs": sorted(snap_epochs),
        "parquet": str(parquet_path),
        "device": str(device),
        "seed": seed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name="supervised_overfit_sample0000",
        mode=wandb_mode,
        dir=str(out_dir),
        config=meta,
        tags=["supervised", "overfit", "sample_0000", "periodicmaxwellnet"],
    )
    run_url = getattr(run, "url", "") or ""
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")

    def build_payload(epoch: int, loss_history: list[dict]) -> dict:
        return {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss_train": torch.tensor([row["loss"] for row in loss_history]),
            "meta": meta,
            "wandb_run_id": run.id if run is not None else None,
        }

    loss_history: list[dict] = []
    batch = next(iter(loader))

    with torch.no_grad():
        init_loss, *_ = _compute_loss(
            batch,
            model,
            device,
            mode,
            pml_thickness=10,
            supervised_coef=1.0,
            helmholtz_coef=None,
        )
    init_loss_val = float(init_loss.detach().cpu())
    wandb.log(
        {
            "epoch": 0,
            "train/loss": init_loss_val,
            "train/loss_supervised": init_loss_val,
            "train/lr": optimizer.param_groups[0]["lr"],
        }
    )
    init_path = save_checkpoint(build_payload(0, []), out_dir, f"0_{mode}")
    upload_pt(init_path, 0)
    print(f"saved init {init_path}  loss={init_loss_val:.6e}")

    latest_path = out_dir / "latest.pt"
    pbar = tqdm(range(1, epochs + 1), desc="supervised overfit")
    for epoch in pbar:
        optimizer.zero_grad(set_to_none=True)
        loss, *_rest = _compute_loss(
            batch,
            model,
            device,
            mode,
            pml_thickness=10,
            supervised_coef=1.0,
            helmholtz_coef=None,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.detach().cpu())
        loss_history.append({"epoch": epoch, "loss": loss_val})
        pbar.set_postfix(loss=f"{loss_val:.3e}")
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": loss_val,
                "train/loss_supervised": loss_val,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
        )

        payload = build_payload(epoch, loss_history)
        if epoch in snap_epochs:
            snap_path = save_checkpoint(payload, out_dir, f"{epoch}_{mode}")
            upload_pt(snap_path, epoch, also_latest=(epoch == epochs))
            latest_path = save_checkpoint(payload, out_dir, "latest")
            print(f"saved snapshot {snap_path}  loss={loss_val:.6e}")
        elif epoch == epochs:
            latest_path = save_checkpoint(payload, out_dir, "latest")
            upload_pt(latest_path, epoch, also_latest=True)

    (out_dir / "loss_history.json").write_text(
        json.dumps(loss_history, indent=2) + "\n"
    )
    wandb.summary["final_loss"] = loss_history[-1]["loss"]
    wandb.summary["final_epoch"] = epochs
    wandb.summary["checkpoint_dir"] = str(out_dir)
    print(f"wrote {latest_path}  final_loss={loss_history[-1]['loss']:.6e}")
    print(f"wandb: {run_url}")
    wandb.finish()
    return latest_path, run_url


def main(argv: list[str] | None = None) -> None:
    """CLI entry: train supervised overfit and print the checkpoint / wandb URL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "overfit_sample0000",
    )
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=100,
        help="Save/upload {epoch}_te.pt every N epochs (always includes 0 and final).",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--wandb-entity", type=str, default="als-rixs")
    parser.add_argument("--wandb-project", type=str, default="maxwellnet")
    parser.add_argument("--wandb-mode", type=str, default="online")
    args = parser.parse_args(argv)
    _path, url = train(
        epochs=args.epochs,
        out_dir=args.out_dir,
        parquet=args.parquet,
        lr=args.lr,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
    )
    if url:
        print(url)


if __name__ == "__main__":
    main()
