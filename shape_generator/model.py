"""Small UNet + DDPM scheduler for generating (2, 256, 256) shapes.

Channel 0 is optical_constant_real, channel 1 is optical_constant_imag.
"""
import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel

IMAGE_SIZE = 256  # default; set model.image_size in the config for other datasets
IN_CHANNELS = 2
STD_CLIP_RANGE = 4.5  # |x0| clip for unit-variance (standard-scaled / latent) data, see build_scheduler


def build_unet(block_out_channels=(32, 64, 128, 128, 256, 256), image_size=IMAGE_SIZE, layers_per_block=2,
               attention_levels=2, channels=IN_CHANNELS):
    """Small UNet with one level per entry of block_out_channels (each level after
    the first halves the resolution; image_size must be divisible by 2**(levels-1)).
    Self-attention only at the `attention_levels` lowest-resolution levels (0 = none).
    `channels` is the number of image channels (2 for pixels, vae.latent_channels for latents)."""
    n = len(block_out_channels)
    down = tuple("AttnDownBlock2D" if i >= n - attention_levels else "DownBlock2D" for i in range(n))
    up = tuple("AttnUpBlock2D" if i < attention_levels else "UpBlock2D" for i in range(n))
    return UNet2DModel(
        sample_size=image_size,
        in_channels=channels,
        out_channels=channels,
        layers_per_block=layers_per_block,
        block_out_channels=tuple(block_out_channels),
        down_block_types=down,
        up_block_types=up,
        norm_num_groups=8,
    )


def build_scheduler(num_train_timesteps=1000, clip_range=1.0):
    """clip_range: each predicted clean image is clipped to [-clip_range, clip_range] during sampling.
    Needed even for unbounded data: at t=T the estimate divides by sqrt(alpha_bar) ~ 5e-5, which
    amplifies tiny model errors enormously over a full 1000-step chain."""
    return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2",
                         clip_sample=True, clip_sample_range=clip_range)


def build_vae(block_out_channels=(32, 64, 128, 128), latent_channels=4, layers_per_block=1, image_size=IMAGE_SIZE):
    """KL autoencoder for (2, H, W) images. Downsamples by 2**(levels-1) to latent_channels x H/f x W/f."""
    n = len(block_out_channels)
    return AutoencoderKL(
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS,
        down_block_types=("DownEncoderBlock2D",) * n,
        up_block_types=("UpDecoderBlock2D",) * n,
        block_out_channels=tuple(block_out_channels),
        latent_channels=latent_channels,
        layers_per_block=layers_per_block,
        norm_num_groups=8,
        sample_size=image_size,
    )


def _symlog(x, s):
    """sign(x) * log1p(|x| / s): 0 -> 0, linear for |x| << s, logarithmic for |x| >> s, odd, exactly invertible."""
    return torch.sign(x) * torch.log1p(x.abs() / s)


def _symexp(y, s):
    return torch.sign(y) * s * torch.expm1(y.abs())


class Normalizer:
    """Per-channel map from raw optical constants to model space (and back).

    Optional first step (log_s given): the signed log y = sign(x) * log1p(|x| / log_s) with a per-channel
    log_s, so values spanning many orders of magnitude (delta / beta) are comparable; decode exps back.
    Then the per-channel affine map z = (y - shift) / scale:
      mode "minmax": maps [lo, hi] to [-1, 1] (shift = midpoint, scale = half range); decode clamps to [-1, 1].
      mode "standard": zero mean, unit std (shift = mean, scale = std); the output is unbounded.
    shift / scale are fit on the log-space values when the log is used. log_max bounds |y| on decode so the
    exp cannot overflow on an out-of-range model output.
    """

    def __init__(self, shift, scale, mode="standard", log_s=None, log_max=None):
        assert mode in ("minmax", "standard"), mode
        self.shift = torch.as_tensor(shift, dtype=torch.float32).view(1, -1, 1, 1)
        self.scale = torch.as_tensor(scale, dtype=torch.float32).view(1, -1, 1, 1)
        self.mode = mode
        self.log_s = None if log_s is None else torch.as_tensor(log_s, dtype=torch.float32).view(1, -1, 1, 1)
        self.log_max = None if log_max is None else torch.as_tensor(log_max, dtype=torch.float32).view(1, -1, 1, 1)

    @classmethod
    def fit(cls, imgs, mode="standard", log=False):
        """imgs: (N, C, H, W) raw images."""
        log_s = log_max = None
        if log:
            # per-channel s = 1st percentile of the nonzero magnitudes: below it the map is ~linear, above it log
            log_s = torch.ones(imgs.shape[1])
            for c in range(imgs.shape[1]):
                mag = imgs[:, c].abs().flatten()
                mag = mag[mag > 0].numpy()
                if mag.size:
                    log_s[c] = float(np.percentile(mag, 1))
            imgs = _symlog(imgs, log_s.view(1, -1, 1, 1))
            log_max = imgs.abs().amax(dim=(0, 2, 3)) * 1.25
        if mode == "minmax":
            lo, hi = imgs.amin(dim=(0, 2, 3)), imgs.amax(dim=(0, 2, 3))
            # Guard against a constant channel (e.g. all-zero imag part).
            hi = torch.where(hi - lo < 1e-8, lo + 1.0, hi)
            return cls((lo + hi) / 2, (hi - lo) / 2, mode, log_s, log_max)
        mean, std = imgs.mean(dim=(0, 2, 3)), imgs.std(dim=(0, 2, 3))
        return cls(mean, torch.where(std < 1e-8, torch.ones_like(std), std), mode, log_s, log_max)

    @property
    def log(self):
        return self.log_s is not None

    @property
    def bounded(self):
        return self.mode == "minmax"

    @property
    def clip_range(self):
        return 1.0 if self.bounded else STD_CLIP_RANGE

    def encode(self, x, sample=False):  # `sample` is for interface parity with LatentCodec (deterministic here)
        if self.log:
            x = _symlog(x, self.log_s.to(x.device))
        return (x - self.shift.to(x.device)) / self.scale.to(x.device)

    def decode(self, x):
        if self.bounded:
            x = x.clamp(-1, 1)
        x = x * self.scale.to(x.device) + self.shift.to(x.device)
        if self.log:
            m = self.log_max.to(x.device)
            x = _symexp(torch.minimum(torch.maximum(x, -m), m), self.log_s.to(x.device))
        return x

    def describe(self):
        return self.state_dict()

    def state_dict(self):
        return {"mode": self.mode, "shift": self.shift.flatten().tolist(), "scale": self.scale.flatten().tolist(),
                "log_s": None if not self.log else self.log_s.flatten().tolist(),
                "log_max": None if not self.log else self.log_max.flatten().tolist()}

    @classmethod
    def from_state_dict(cls, d):
        if "lo" in d:  # checkpoints saved before standard scaling existed
            lo, hi = torch.tensor(d["lo"]), torch.tensor(d["hi"])
            return cls((lo + hi) / 2, (hi - lo).clamp_min(1e-12) / 2, "minmax")
        return cls(d["shift"], d["scale"], d["mode"], d.get("log_s"), d.get("log_max"))  # no log keys: pre-log checkpoints


class LatentCodec:
    """Pixel <-> latent map with the Normalizer interface, so diffusion code treats latents like data.

    encode: raw image -> pixel normalizer -> VAE posterior (mean, or a sample) -> * scale_factor
    decode: latent / scale_factor -> VAE decoder -> pixel normalizer inverse -> raw image
    scale_factor (~1 / latent std, measured on the train set) puts latents at unit variance for the DDPM.
    """
    bounded = False
    clip_range = STD_CLIP_RANGE

    def __init__(self, vae, vae_kwargs, pixel_normalizer, scale_factor):
        self.vae = vae.eval().requires_grad_(False)
        self.vae_kwargs = vae_kwargs
        self.pixel_normalizer = pixel_normalizer
        self.scale_factor = float(scale_factor)

    @property
    def downsample(self):
        return 2 ** (len(self.vae_kwargs["block_out_channels"]) - 1)

    @property
    def latent_channels(self):
        return self.vae_kwargs["latent_channels"]

    def to(self, device):
        self.vae.to(device)
        return self

    @torch.no_grad()
    def encode(self, x, sample=False):
        device = next(self.vae.parameters()).device
        post = self.vae.encode(self.pixel_normalizer.encode(x).to(device)).latent_dist
        return (post.sample() if sample else post.mean) * self.scale_factor

    @torch.no_grad()
    def decode(self, z):
        device = next(self.vae.parameters()).device
        x = self.vae.decode(z.to(device) / self.scale_factor).sample
        return self.pixel_normalizer.decode(x)

    def describe(self):
        return {"vae_kwargs": self.vae_kwargs, "scale_factor": self.scale_factor,
                "pixel_normalizer": self.pixel_normalizer.state_dict()}

    def state_dict(self):
        return {"type": "latent", "vae_kwargs": self.vae_kwargs,
                "vae": {k: v.detach().cpu() for k, v in self.vae.state_dict().items()},
                "pixel_normalizer": self.pixel_normalizer.state_dict(), "scale_factor": self.scale_factor}

    @classmethod
    def from_state_dict(cls, d):
        vae = build_vae(**d["vae_kwargs"])
        vae.load_state_dict(d["vae"])
        return cls(vae, d["vae_kwargs"], Normalizer.from_state_dict(d["pixel_normalizer"]), d["scale_factor"])

    @classmethod
    def from_checkpoint(cls, path, device):
        return cls.from_state_dict(torch.load(path, map_location="cpu")).to(device)


def save_vae_checkpoint(path, vae, vae_kwargs, pixel_normalizer, scale_factor, step):
    """A VAE checkpoint is a LatentCodec state dict (+ step), so it loads straight into LatentCodec.
    Built by hand: constructing a LatentCodec would freeze the VAE that is still being trained."""
    torch.save({"type": "latent", "vae_kwargs": vae_kwargs,
                "vae": {k: v.detach().cpu() for k, v in vae.state_dict().items()},
                "pixel_normalizer": pixel_normalizer.state_dict(), "scale_factor": float(scale_factor),
                "step": step}, path)


def load_normalizer(state, device):
    """Rebuild whatever a diffusion checkpoint stored as its 'normalizer' (a Normalizer or a LatentCodec)."""
    if state.get("type") == "latent":
        return LatentCodec.from_state_dict(state).to(device)
    return Normalizer.from_state_dict(state)


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
    normalizer = load_normalizer(ckpt["normalizer"], device)
    scheduler = build_scheduler(ckpt["num_train_timesteps"], normalizer.clip_range)
    return unet, scheduler, normalizer
