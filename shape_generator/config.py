"""Schema + loader for shape_generator/config.json.

Each section is a dataclass; unknown keys, wrong types and out-of-range values
raise a ValueError naming the offending field.
"""
import json
import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _check(cond, msg):
    if not cond:
        raise ValueError(msg)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class DataConfig:
    dataset: str = "shapes"             # "shapes" (ShapeDataset) or "mnist" (sanity-check dataset)
    mnist_root: str = "shape_generator/data"  # where MNIST is downloaded
    num_files: Optional[int] = None     # number of general-shapes-1m parquet shards (of 4) to use, null = all (shapes only)
    component: int = 2                  # which tensor component to render: 0 = xx, 1 = yy, 2 = zz (shapes only)
    max_oc_value: Optional[float] = 0.1  # drop shapes whose delta or beta exceeds this, null = keep all (shapes only)
    n_samples: Optional[int] = None     # number of training samples to use (null = all)
    num_workers: int = 4                # DataLoader workers
    valid_fraction: float = 0.1         # fraction of unique shapes held out for validation (0 = none)
    normalization: str = "standard"     # "standard" (zero mean, unit std per channel) or "minmax" (to [-1, 1])

    def validate(self):
        _check(self.dataset in ("shapes", "mnist"), 'data.dataset must be "shapes" or "mnist"')
        _check(isinstance(self.mnist_root, str) and self.mnist_root, "data.mnist_root must be a non-empty string")
        _check(self.num_files is None or (_is_int(self.num_files) and self.num_files > 0),
               "data.num_files must be a positive integer or null")
        _check(self.component in (0, 1, 2), "data.component must be 0, 1 or 2")
        _check(self.max_oc_value is None or (_is_num(self.max_oc_value) and self.max_oc_value > 0),
               "data.max_oc_value must be a positive number or null")
        _check(self.n_samples is None or (_is_int(self.n_samples) and self.n_samples > 0),
               "data.n_samples must be a positive integer or null")
        _check(_is_int(self.num_workers) and self.num_workers >= 0, "data.num_workers must be an integer >= 0")
        _check(self.normalization in ("standard", "minmax"), 'data.normalization must be "standard" or "minmax"')
        _check(_is_num(self.valid_fraction) and 0 <= self.valid_fraction < 1,
               "data.valid_fraction must be a number in [0, 1)")


@dataclass
class ModelConfig:
    block_out_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128, 256, 256])
    attention_levels: int = 2           # self-attention at this many lowest-resolution levels (0 = none)
    layers_per_block: int = 2           # residual blocks per UNet level
    image_size: int = 256               # H = W of the generated images (shapes: 256, mnist: 28)
    timesteps: int = 1000               # DDPM training timesteps
    vae_ckpt: Optional[str] = None      # trained VAE checkpoint (vae.pt): diffusion runs in its latent space (null = pixel space)

    def validate(self):
        b = self.block_out_channels
        _check(isinstance(b, list) and len(b) >= 2 and all(_is_int(c) and c > 0 for c in b),
               "model.block_out_channels must be a list of >= 2 positive integers")
        _check(all(c % 8 == 0 for c in b), "model.block_out_channels entries must be divisible by 8 (GroupNorm)")
        _check(_is_int(self.attention_levels) and 0 <= self.attention_levels <= len(b),
               "model.attention_levels must be an integer between 0 and the number of levels")
        _check(_is_int(self.layers_per_block) and self.layers_per_block > 0,
               "model.layers_per_block must be a positive integer")
        _check(_is_int(self.image_size) and self.image_size > 0, "model.image_size must be a positive integer")
        # in latent mode the UNet sees the VAE's latent, not the image; train.py checks that size at run time
        _check(self.vae_ckpt is not None or self.image_size % (2 ** (len(b) - 1)) == 0,
               f"model.image_size ({self.image_size}) must be divisible by 2**(levels-1) = "
               f"{2 ** (len(b) - 1)} for {len(b)} block_out_channels entries")
        _check(_is_int(self.timesteps) and self.timesteps > 0, "model.timesteps must be a positive integer")
        _check(self.vae_ckpt is None or (isinstance(self.vae_ckpt, str) and self.vae_ckpt),
               "model.vae_ckpt must be a non-empty string or null")


@dataclass
class VAEConfig:
    """Stage-1 autoencoder (train_vae.py). Its input is data.* images normalized by data.normalization.
    The latent is image_size / 2**(levels-1) on a side with latent_channels channels."""
    block_out_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    latent_channels: int = 16
    layers_per_block: int = 1
    recon_loss: str = "mse"              # "l1" (sharper edges) or "mse"
    kl_weight: float = 1e-5             # weight on the per-element mean KL(q(z|x) || N(0, 1))
    epochs: int = 100
    batch_size: int = 16
    lr: float = 3e-4
    lr_schedule: str = "cosine"         # "cosine" (linear warmup, then cosine decay to min_lr) or "constant"
    min_lr: float = 1e-6                # final lr the cosine decay ends at (must be <= lr)
    warmup_steps: int = 100             # linear lr warmup steps (optimizer steps, not epochs)
    grad_clip: Optional[float] = 1.0
    ckpt_every: int = 500               # every N optimizer steps: checkpoint (vae_step_XXXXXX.pt, plus vae.pt = latest)
                                        # and run the metrics + reconstruction plots (pushed to wandb)
    seed: Optional[int] = 0
    out_dir: str = "shape_generator/runs/vae"
    resume: Optional[str] = None        # vae checkpoint to continue from (weights, pixel normalizer, step); runs `epochs` more

    def validate(self):
        b = self.block_out_channels
        _check(isinstance(b, list) and len(b) >= 2 and all(_is_int(c) and c > 0 for c in b),
               "vae.block_out_channels must be a list of >= 2 positive integers")
        _check(all(c % 8 == 0 for c in b), "vae.block_out_channels entries must be divisible by 8 (GroupNorm)")
        _check(_is_int(self.latent_channels) and self.latent_channels > 0,
               "vae.latent_channels must be a positive integer")
        _check(_is_int(self.layers_per_block) and self.layers_per_block > 0,
               "vae.layers_per_block must be a positive integer")
        _check(self.recon_loss in ("l1", "mse"), 'vae.recon_loss must be "l1" or "mse"')
        _check(_is_num(self.kl_weight) and self.kl_weight >= 0, "vae.kl_weight must be a number >= 0")
        _check(_is_int(self.epochs) and self.epochs > 0, "vae.epochs must be a positive integer")
        _check(_is_int(self.batch_size) and self.batch_size > 0, "vae.batch_size must be a positive integer")
        _check(_is_num(self.lr) and self.lr > 0, "vae.lr must be a positive number")
        _check(self.lr_schedule in ("cosine", "constant"), 'vae.lr_schedule must be "cosine" or "constant"')
        _check(_is_num(self.min_lr) and 0 <= self.min_lr <= self.lr, "vae.min_lr must be a number in [0, lr]")
        _check(_is_int(self.warmup_steps) and self.warmup_steps >= 0, "vae.warmup_steps must be an integer >= 0")
        _check(self.grad_clip is None or (_is_num(self.grad_clip) and self.grad_clip > 0),
               "vae.grad_clip must be a positive number or null")
        _check(_is_int(self.ckpt_every) and self.ckpt_every > 0, "vae.ckpt_every must be a positive integer")
        _check(self.seed is None or _is_int(self.seed), "vae.seed must be an integer or null")
        _check(isinstance(self.out_dir, str) and self.out_dir, "vae.out_dir must be a non-empty string")
        _check(self.resume is None or (isinstance(self.resume, str) and self.resume),
               "vae.resume must be a non-empty string or null")


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    lr_schedule: str = "cosine"         # "cosine" (linear warmup, then cosine decay to 0) or "constant"
    min_lr: float = 0.0                 # final lr the cosine decay ends at (must be <= lr)
    warmup_steps: int = 0               # linear lr warmup steps (optimizer steps, not epochs)
    grad_clip: Optional[float] = 1.0    # max grad norm (null = no clipping)
    ckpt_every: int = 500               # every N optimizer steps: checkpoint (ckpt_step_XXXXXX.pt, plus ckpt.pt = latest)
                                        # and run the eval (valid loss + sample grid + wandb push)
    seed: Optional[int] = 0
    out_dir: str = "shape_generator/runs/default"
    eval_diffusion_timesteps: int = 100               # DDPM sampling steps used for each eval
    eval_seed: int = 0                  # fixed noise seed so every eval starts from the same z

    def validate(self):
        _check(_is_int(self.epochs) and self.epochs > 0, "train.epochs must be a positive integer")
        _check(_is_int(self.batch_size) and self.batch_size > 0, "train.batch_size must be a positive integer")
        _check(_is_num(self.lr) and self.lr > 0, "train.lr must be a positive number")
        _check(self.lr_schedule in ("cosine", "constant"), 'train.lr_schedule must be "cosine" or "constant"')
        _check(_is_num(self.min_lr) and 0 <= self.min_lr <= self.lr, "train.min_lr must be a number in [0, lr]")
        _check(_is_int(self.warmup_steps) and self.warmup_steps >= 0, "train.warmup_steps must be an integer >= 0")
        _check(self.grad_clip is None or (_is_num(self.grad_clip) and self.grad_clip > 0),
               "train.grad_clip must be a positive number or null")
        _check(_is_int(self.ckpt_every) and self.ckpt_every > 0, "train.ckpt_every must be a positive integer")
        _check(self.seed is None or _is_int(self.seed), "train.seed must be an integer or null")
        _check(isinstance(self.out_dir, str) and self.out_dir, "train.out_dir must be a non-empty string")
        _check(_is_int(self.eval_diffusion_timesteps) and self.eval_diffusion_timesteps > 0, "train.eval_diffusion_timesteps must be a positive integer")
        _check(_is_int(self.eval_seed), "train.eval_seed must be an integer")


@dataclass
class LoggingConfig:
    wandb: bool = True                  # set false to disable wandb entirely
    entity: str = "als-rixs"
    project: str = "shape-generator"
    run_name: Optional[str] = None      # null = wandb auto-generated name

    def validate(self):
        _check(isinstance(self.wandb, bool), "logging.wandb must be a boolean")
        _check(isinstance(self.entity, str) and self.entity, "logging.entity must be a non-empty string")
        _check(isinstance(self.project, str) and self.project, "logging.project must be a non-empty string")
        _check(self.run_name is None or isinstance(self.run_name, str), "logging.run_name must be a string or null")


@dataclass
class InferenceConfig:
    ckpt: str = "shape_generator/runs/default/ckpt.pt"
    steps: Optional[int] = None         # sampling steps (null = full timesteps)
    seed: Optional[int] = None
    out: str = "shape_generator/samples.png"  # _real / _imag is inserted before the extension

    def validate(self):
        _check(isinstance(self.ckpt, str) and self.ckpt, "inference.ckpt must be a non-empty string")
        _check(self.steps is None or (_is_int(self.steps) and self.steps > 0),
               "inference.steps must be a positive integer or null")
        _check(self.seed is None or _is_int(self.seed), "inference.seed must be an integer or null")
        _check(isinstance(self.out, str) and self.out, "inference.out must be a non-empty string")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _build(cls, raw, name):
    _check(isinstance(raw, dict), f"'{name}' must be an object")
    unknown = set(raw) - {f.name for f in fields(cls)}
    _check(not unknown, f"unknown key(s) in '{name}': {sorted(unknown)}")
    section = cls(**raw)
    section.validate()
    return section


def load_config(path=DEFAULT_PATH):
    with open(path) as f:
        raw = json.load(f)
    _check(isinstance(raw, dict), "config root must be an object")
    unknown = set(raw) - {f.name for f in fields(Config)}
    _check(not unknown, f"unknown top-level key(s): {sorted(unknown)}")
    return Config(**{f.name: _build(f.type, raw.get(f.name, {}), f.name) for f in fields(Config)})
