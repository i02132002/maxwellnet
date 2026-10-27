"""Loader for als-rixs/general-shapes-1m: rasterizes its geometry + optical-constant parameters into images.

The dataset stores no images, only a polygon (vertices), a substrate rectangle and a domain rectangle per
sample, plus delta / beta for the shape and the substrate (scalar, per-component tensor, or a polynomial
gradient along a direction). Each sample is rendered on the fly to {"optical_constant_image": (2, S, S)}:
channel 0 = delta, channel 1 = beta, of one tensor component (xx / yy / zz). The domain rectangle fills the
whole image (row 0 = top), vacuum is 0, the shape wins over the substrate where they overlap. Splines are
rendered as plain polygons through their vertices.

Samples with any delta or beta above `max_value` are dropped (see keep_mask and the pixel check in _read_shard).
"""
import json
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download

REPO = "als-rixs/general-shapes-1m"
COMPONENTS = ("xx", "yy", "zz")
MAX_V = 20
CHECK_SIZE = 64  # resolution of the pixel-level max_value check
COLUMNS = ["vertices", "vertex_mask", "domain_rect", "substrate_present", "substrate_rect",
           "shape_delta", "shape_beta", "shape_td", "shape_tb", "shape_grad_json",
           "sub_delta", "sub_beta", "sub_td", "sub_tb", "sub_grad_json"]


def list_shards():
    files = [f.path for f in HfApi().list_repo_tree(REPO, repo_type="dataset", path_in_repo="data")
             if re.fullmatch(r"data/train-\d+-of-\d+\.parquet", f.path)]
    return sorted(files)


# --- rasterization ---------------------------------------------------------------------------------------

def _eval_poly(poly, X, Y):
    """Polynomial of the normalized distance along poly's direction (gradient optical constants)."""
    t = ((X - poly["originX"]) * poly["dirX"] + (Y - poly["originY"]) * poly["dirY"]) / max(poly["length"], 1e-12)
    v, pw = np.zeros_like(X), np.ones_like(X)
    for c in poly["coeffs"]:
        v += c * pw
        pw = pw * t
    return v


def _in_polygon(X, Y, pts):
    """Even-odd ray casting, vectorized over the pixel grid."""
    inside = np.zeros(X.shape, dtype=bool)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        inside ^= ((yi > Y) != (yj > Y)) & (X < (xj - xi) * (Y - yi) / (yj - yi + 1e-18) + xi)
        j = i
    return inside


def render(row, size, component):
    """row: one sample (dict of the COLUMNS). Returns (2, size, size) float32: delta, beta."""
    dx, dy, dw, dh = (float(v) for v in row["domain_rect"])
    X, Y = np.meshgrid(np.linspace(dx, dx + dw, size), np.linspace(dy + dh, dy, size))  # row 0 = top
    pts = np.asarray(row["vertices"], dtype=np.float64)[np.asarray(row["vertex_mask"], dtype=bool)]
    in_shape = _in_polygon(X, Y, pts)
    regions = [("shape", in_shape)]
    if row["substrate_present"]:
        sx, sy, sw, sh = (float(v) for v in row["substrate_rect"])
        in_sub = (sx <= X) & (X <= sx + sw) & (sy <= Y) & (Y <= sy + sh) & ~in_shape
        regions.append(("sub", in_sub))
    out = np.zeros((2, size, size), dtype=np.float32)
    for prefix, region in regions:
        if not region.any():
            continue
        grad = json.loads(row[f"{prefix}_grad_json"]) if row[f"{prefix}_grad_json"] else None
        for k, (name, tensor) in enumerate((("delta", "td"), ("beta", "tb"))):
            if grad:
                out[k][region] = _eval_poly(grad[name][COMPONENTS[component]], X, Y)[region]
            else:
                out[k][region] = row[f"{prefix}_{tensor}"][component]
    return out


# --- filtering -------------------------------------------------------------------------------------------

def keep_mask(cols, max_value):
    """True for samples whose delta and beta parameters are all <= max_value: the scalar and per-component
    values of the shape (and of the substrate, when present), and the mean coefficient of every gradient
    polynomial. Gradients can still rise above max_value away from their mean."""
    ok = np.ones(len(cols["shape_delta"]), dtype=bool)
    has_sub = cols["substrate_present"]
    for prefix, applies in (("shape", np.ones_like(ok)), ("sub", has_sub)):
        worst = np.maximum.reduce([cols[f"{prefix}_delta"], cols[f"{prefix}_beta"],
                                   cols[f"{prefix}_td"].max(axis=1), cols[f"{prefix}_tb"].max(axis=1)])
        ok &= ~applies | (worst <= max_value)
        for i in np.flatnonzero(ok & applies):  # gradient means; parse only the rows still in the running
            if not cols[f"{prefix}_grad_json"][i]:
                continue
            g = json.loads(cols[f"{prefix}_grad_json"][i])
            if max(p["coeffs"][0] for name in ("delta", "beta") for p in g[name].values()) > max_value:
                ok[i] = False
    return ok


def _take(cols, idx):
    return {k: ([v[i] for i in idx] if isinstance(v, list) else v[idx]) for k, v in cols.items()}


def _read_shard(path, max_value, component):
    tab = pq.read_table(path, columns=COLUMNS)

    def arr(name, width=None):
        a = tab.column(name).combine_chunks()
        if width is None:
            return a.to_numpy(zero_copy_only=False)
        while pa.types.is_list(a.type):
            a = a.flatten()
        return a.to_numpy(zero_copy_only=False).reshape(len(tab), *width)

    cols = {"vertices": arr("vertices", (MAX_V, 2)).astype(np.float32),
            "vertex_mask": arr("vertex_mask", (MAX_V,)),
            "substrate_present": arr("substrate_present"),
            "shape_grad_json": tab.column("shape_grad_json").to_pylist(),
            "sub_grad_json": tab.column("sub_grad_json").to_pylist()}
    for name in ("domain_rect", "substrate_rect"):
        cols[name] = arr(name, (4,)).astype(np.float64)
    for name in ("shape_td", "shape_tb", "sub_td", "sub_tb"):
        cols[name] = arr(name, (3,)).astype(np.float64)
    for name in ("shape_delta", "shape_beta", "sub_delta", "sub_beta"):
        cols[name] = arr(name).astype(np.float64)
    if max_value is not None:
        keep = np.flatnonzero(keep_mask(cols, max_value))
        cols = _take(cols, keep)
        # a gradient can exceed max_value away from its mean, so also check the rendered pixels
        ok = [render({k: v[i] for k, v in cols.items()}, CHECK_SIZE, component).max() <= max_value
              for i in range(len(keep))]
        cols = _take(cols, np.flatnonzero(ok))
    return cols, len(tab)


# --- dataset ---------------------------------------------------------------------------------------------

class GeneralShapeDataset(torch.utils.data.Dataset):
    def __init__(self, cols, image_size, component):
        self.cols, self.image_size, self.component = cols, image_size, component

    def __len__(self):
        return len(self.cols["shape_delta"])

    def __getitem__(self, idx):
        row = {k: v[idx] for k, v in self.cols.items()}
        return {"optical_constant_image": torch.from_numpy(render(row, self.image_size, self.component))}


def load_general_shapes(num_files=None, component=2, n_samples=None, valid_fraction=0.0, seed=0, image_size=256,
                        max_value=0.1):
    """Returns (train, valid). Uses the first `num_files` parquet shards (None = all, 4 x 250k samples),
    drops samples with delta or beta > `max_value` (None = keep all; checked on the parameters and on pixels at 64x64), shuffles with `seed`, and gives the last
    `valid_fraction` to valid. `n_samples` caps the train split. `component` picks xx / yy / zz (0 / 1 / 2).
    Images are rendered lazily at `image_size`."""
    shards = list_shards()[:num_files]
    print(f"loading {len(shards)} shards: {shards[0]} ... {shards[-1]}")
    parts, total = [], 0
    for shard in shards:
        cols, n = _read_shard(hf_hub_download(REPO, shard, repo_type="dataset"), max_value, component)
        parts.append(cols)
        total += n
    cols = {k: ([x for p in parts for x in p[k]] if isinstance(parts[0][k], list)
                else np.concatenate([p[k] for p in parts])) for k in parts[0]}
    n_kept = len(cols["shape_delta"])
    print(f"kept {n_kept} of {total} samples (delta, beta <= {max_value})")
    order = np.random.default_rng(seed).permutation(n_kept)
    n_valid = round(n_kept * valid_fraction)
    train_idx, valid_idx = order[:n_kept - n_valid], order[n_kept - n_valid:]
    if n_samples is not None:
        train_idx = train_idx[:n_samples]
    return (GeneralShapeDataset(_take(cols, train_idx), image_size, component),
            GeneralShapeDataset(_take(cols, valid_idx), image_size, component))
