"""
Swin-Transformer counterpart of ShapeNet.PeriodicMaxwellNet.

SwinMaxwellNet subclasses PeriodicMaxwellNet and overrides only the
image-to-image backbone (`self.model`): everywhere PeriodicMaxwellNet builds
`self.model` from UNet.py's convolutional UNet, SwinMaxwellNet builds it from
SwinUNetBackbone below instead. Every other piece of PeriodicMaxwellNet --
epsilon-map selection, contrast scaling, the incident-wave-vector/plane-wave
construction, the padding-to-a-window-multiple bookkeeping, and forward()'s
reconstruction of the predicted field -- is inherited unchanged, so
SwinMaxwellNet takes exactly the same inputs and returns exactly the same
outputs as PeriodicMaxwellNet (see that class's docstring).

SwinUNetBackbone mirrors UNet.py's own down_path/up_path structure (patch
merging in place of AvgPool2d + channel-doubling conv block; patch expanding
in place of ConvTranspose2d/upsample + concatenated skip connection), but
operates on window-attention tokens instead of convolutional feature maps.
The incident wave-vector (kx, kz) is injected at the bottleneck -- added,
after a linear projection, to every token at the coarsest resolution -- the
same location UNet.py's `cond_channels` mechanism injects it at (see
PeriodicMaxwellNet._COND_CHANNELS).
"""

import torch
from torch import nn

from ShapeNet import PeriodicMaxwellNet


def window_partition(x, window_size):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)."""
    B = windows.shape[0] // (H // window_size * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def shifted_window_attn_mask(H, W, window_size, shift_size, device):
    """Additive attention mask (0 / -100) that stops a shifted window's
    attention from mixing tokens that wrapped around from opposite edges of
    the image -- standard Swin bookkeeping, since torch.roll doesn't know
    the shifted-out region is discontiguous from where it wrapped back in."""
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class WindowAttention(nn.Module):
    """Multi-head self-attention restricted to non-overlapping windows, with
    a learned relative-position bias (standard Swin windowed attention)."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        # x: (num_windows*B, N, C), N = window_size**2
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """One pre-norm Swin transformer block: (shifted) window attention +
    residual, then an MLP + residual."""

    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, H, W):
        # x: (B, H*W, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = shifted_window_attn_mask(H, W, self.window_size, self.shift_size, x.device)
        else:
            mask = None

        windows = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        attn_windows = self.attn(windows, mask=mask)
        x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x.view(B, L, C)
        x = x + self.mlp(self.norm2(x))
        return x


class SwinStage(nn.Module):
    """`blocks` Swin blocks at a fixed resolution/channel count, alternating
    plain and shifted windows (as in the original Swin Transformer)."""

    def __init__(self, dim, blocks, num_heads, window_size):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size, shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(blocks)
        ])

    def forward(self, x, H, W):
        B, _, _, C = x.shape
        x = x.view(B, H * W, C)
        for layer in self.layers:
            x = layer(x, H, W)
        return x.view(B, H, W, C)


class PatchMerging(nn.Module):
    """2x spatial downsample / 2x channel-count increase: concatenate each
    2x2 neighborhood along channels, then project 4*dim -> 2*dim. The
    token-sequence analogue of UNet.py's AvgPool2d + channel-doubling conv."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        # x: (B, H, W, C) -> (B, H/2, W/2, 2C)
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                       x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], dim=-1)
        return self.reduction(self.norm(x))


class PatchExpanding(nn.Module):
    """2x spatial upsample / 2x channel-count decrease: project dim -> 2*dim
    then pixel-shuffle that into (2x spatial, dim/2 channels). The
    token-sequence analogue of UNet.py's ConvTranspose2d/upsample."""

    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        # x: (B, H, W, C) -> (B, 2H, 2W, C/2)
        B, H, W, C = x.shape
        x = self.expand(x).view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * 2, W * 2, C // 2)
        return self.norm(x)


class SwinUNetBackbone(nn.Module):
    """Encoder-decoder Swin transformer with UNet-style skip connections.
    Interface-compatible with UNet.UNet: forward(x, cond) where x is
    (B, in_channels, H, W) and cond is (B, cond_channels) or None, returning
    (B, out_channels, H, W)."""

    def __init__(self, in_channels, out_channels, depth=4, embed_dim=16, window_size=4,
                num_heads=4, blocks_per_stage=2, cond_channels=0):
        super().__init__()
        assert depth >= 2, "SwinUNetBackbone needs at least one down-stage plus a bottleneck"
        assert embed_dim % num_heads == 0, "embed_dim (filter) must be divisible by num_heads"
        self.cond_channels = cond_channels

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        dims = [embed_dim * (2 ** i) for i in range(depth)]

        self.down_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(depth - 1):
            self.down_stages.append(SwinStage(dims[i], blocks_per_stage, num_heads, window_size))
            self.downsamples.append(PatchMerging(dims[i]))

        if cond_channels > 0:
            self.cond_proj = nn.Linear(cond_channels, dims[-1])
        self.bottleneck = SwinStage(dims[-1], blocks_per_stage, num_heads, window_size)

        self.upsamples = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.up_stages = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.upsamples.append(PatchExpanding(dims[i + 1]))
            self.skip_proj.append(nn.Linear(2 * dims[i], dims[i]))
            self.up_stages.append(SwinStage(dims[i], blocks_per_stage, num_heads, window_size))

        self.norm_out = nn.LayerNorm(dims[0])
        self.head = nn.Linear(dims[0], out_channels, bias=False)

    def forward(self, x, cond=None):
        B, _, H, W = x.shape
        x = self.patch_embed(x).permute(0, 2, 3, 1)  # (B, H, W, embed_dim)

        skips = []
        for stage, down in zip(self.down_stages, self.downsamples):
            x = stage(x, x.shape[1], x.shape[2])
            skips.append(x)
            x = down(x)

        if cond is not None and self.cond_channels > 0:
            x = x + self.cond_proj(cond)[:, None, None, :]

        x = self.bottleneck(x, x.shape[1], x.shape[2])

        for up, proj, stage, skip in zip(self.upsamples, self.skip_proj, self.up_stages, reversed(skips)):
            x = up(x)
            x = proj(torch.cat([x, skip], dim=-1))
            x = stage(x, x.shape[1], x.shape[2])

        x = self.head(self.norm_out(x))
        return x.permute(0, 3, 1, 2)  # (B, out_channels, H, W)


class SwinMaxwellNet(PeriodicMaxwellNet):
    """Drop-in Swin-Transformer replacement for PeriodicMaxwellNet -- see
    module docstring. Selected via NetworkArch: "swin" in specs_maxwell.json.

    `norm` and `up_mode` are accepted (and ignored) purely so the same
    NetworkSpecs dict in specs_maxwell.json can be reused verbatim across
    both NetworkArch options without pruning UNet-specific keys.
    """

    def __init__(self, depth=4, filter=16, norm='weight', up_mode='upconv', mode='te',
                contrast_scale=100.0, window_size=4, num_heads=4, blocks_per_stage=2):
        nn.Module.__init__(self)
        if mode not in ('te', 'tm'):
            raise ValueError(f"mode must be 'te' or 'tm', got {mode!r}")
        self.mode = mode
        channels = 2 if mode == 'te' else 4
        self.model = SwinUNetBackbone(channels, channels, depth=depth, embed_dim=filter,
                                      window_size=window_size, num_heads=num_heads,
                                      blocks_per_stage=blocks_per_stage,
                                      cond_channels=self._COND_CHANNELS)
        # Zero-init the last layer for the same reason as PeriodicMaxwellNet:
        # start at the zeroth-order Born approximation (envelope a=0).
        nn.init.zeros_(self.model.head.weight)
        # Window attention needs H, W divisible by window_size at every
        # stage down to the bottleneck; padding to a multiple of
        # window_size * 2**(depth-1) (rather than PeriodicMaxwellNet's plain
        # 2**(depth-1)) guarantees that at every resolution in between too,
        # since each is a power-of-two multiple of the bottleneck size.
        self._divisor = window_size * 2 ** (depth - 1)
        self.contrast_scale = contrast_scale
