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


def build_scheduler(num_train_timesteps=1000):
    return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2")


class Normalizer:
    """Per-channel affine map from raw optical constants to [-1, 1] (and back)."""

    def __init__(self, lo, hi):
        self.lo = torch.as_tensor(lo, dtype=torch.float32).view(1, -1, 1, 1)
        self.hi = torch.as_tensor(hi, dtype=torch.float32).view(1, -1, 1, 1)

    def encode(self, x):
        lo, hi = self.lo.to(x.device), self.hi.to(x.device)
        return 2 * (x - lo) / (hi - lo).clamp_min(1e-12) - 1

    def decode(self, x):
        lo, hi = self.lo.to(x.device), self.hi.to(x.device)
        return (x.clamp(-1, 1) + 1) / 2 * (hi - lo) + lo

    def state_dict(self):
        return {"lo": self.lo.flatten().tolist(), "hi": self.hi.flatten().tolist()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["lo"], d["hi"])


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
    scheduler = build_scheduler(ckpt["num_train_timesteps"])
    return unet, scheduler, Normalizer.from_state_dict(ckpt["normalizer"])
