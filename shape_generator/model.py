"""Small UNet + DDPM scheduler for generating (2, 256, 256) shapes.

Channel 0 is optical_constant_real, channel 1 is optical_constant_imag.
"""
import torch
from diffusers import DDPMScheduler, UNet2DModel

IMAGE_SIZE = 256  # default; set model.image_size in the config for other datasets
IN_CHANNELS = 2


def build_unet(block_out_channels=(32, 64, 128, 128, 256, 256), image_size=IMAGE_SIZE, layers_per_block=2,
               attention_levels=2):
    """Small UNet with one level per entry of block_out_channels (each level after
    the first halves the resolution; image_size must be divisible by 2**(levels-1)).
    Self-attention only at the `attention_levels` lowest-resolution levels (0 = none)."""
    n = len(block_out_channels)
    down = tuple("AttnDownBlock2D" if i >= n - attention_levels else "DownBlock2D" for i in range(n))
    up = tuple("AttnUpBlock2D" if i < attention_levels else "UpBlock2D" for i in range(n))
    return UNet2DModel(
        sample_size=image_size,
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS,
        layers_per_block=layers_per_block,
        block_out_channels=tuple(block_out_channels),
        down_block_types=down,
        up_block_types=up,
        norm_num_groups=8,
    )


def build_scheduler(num_train_timesteps=1000, clip_sample=True):
    """clip_sample clips each predicted clean image to [-1, 1]; only valid for min/max-normalized data."""
    return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2",
                         clip_sample=clip_sample)


class Normalizer:
    """Per-channel affine map z = (x - shift) / scale from raw optical constants to model space (and back).

    mode "minmax": maps [lo, hi] to [-1, 1] (shift = midpoint, scale = half range); decode clamps to [-1, 1].
    mode "standard": zero mean, unit std (shift = mean, scale = std); the output is unbounded.
    """

    def __init__(self, shift, scale, mode="standard"):
        assert mode in ("minmax", "standard"), mode
        self.shift = torch.as_tensor(shift, dtype=torch.float32).view(1, -1, 1, 1)
        self.scale = torch.as_tensor(scale, dtype=torch.float32).view(1, -1, 1, 1)
        self.mode = mode

    @classmethod
    def fit(cls, imgs, mode="standard"):
        """imgs: (N, C, H, W) raw images."""
        if mode == "minmax":
            lo, hi = imgs.amin(dim=(0, 2, 3)), imgs.amax(dim=(0, 2, 3))
            # Guard against a constant channel (e.g. all-zero imag part).
            hi = torch.where(hi - lo < 1e-8, lo + 1.0, hi)
            return cls((lo + hi) / 2, (hi - lo) / 2, mode)
        mean, std = imgs.mean(dim=(0, 2, 3)), imgs.std(dim=(0, 2, 3))
        return cls(mean, torch.where(std < 1e-8, torch.ones_like(std), std), mode)

    @property
    def bounded(self):
        return self.mode == "minmax"

    def encode(self, x):
        return (x - self.shift.to(x.device)) / self.scale.to(x.device)

    def decode(self, x):
        if self.bounded:
            x = x.clamp(-1, 1)
        return x * self.scale.to(x.device) + self.shift.to(x.device)

    def state_dict(self):
        return {"mode": self.mode, "shift": self.shift.flatten().tolist(), "scale": self.scale.flatten().tolist()}

    @classmethod
    def from_state_dict(cls, d):
        if "lo" in d:  # checkpoints saved before standard scaling existed
            lo, hi = torch.tensor(d["lo"]), torch.tensor(d["hi"])
            return cls((lo + hi) / 2, (hi - lo).clamp_min(1e-12) / 2, "minmax")
        return cls(d["shift"], d["scale"], d["mode"])


def save_checkpoint(path, unet, normalizer, unet_kwargs, num_train_timesteps, step):
    torch.save({
        "unet": unet.state_dict(),
        "unet_kwargs": unet_kwargs,
        "normalizer": normalizer.state_dict(),
        "num_train_timesteps": num_train_timesteps,
        "step": step,
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    unet = build_unet(**ckpt["unet_kwargs"]).to(device)
    unet.load_state_dict(ckpt["unet"])
    unet.eval()
    normalizer = Normalizer.from_state_dict(ckpt["normalizer"])
    scheduler = build_scheduler(ckpt["num_train_timesteps"], clip_sample=normalizer.bounded)
    return unet, scheduler, normalizer
