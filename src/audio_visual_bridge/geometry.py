"""Geometry helpers that put audio (u, v) and SigLIP image patches in one frame.

Coordinate convention (must match ``src/audio/projection.doa_to_uv``):

    u = x_pixel / W   0 → image left,  1 → image right
    v = y_pixel / H   0 → image TOP,   1 → image bottom   (OpenCV, Y down)

PaliGemma's vision tower emits a flat sequence of ``N = G * G`` patch tokens in
row-major order, so patch index ``p = row * G + col`` sits at normalised centre
``((col + 0.5) / G, (row + 0.5) / G)``. Everything here is built on that single
correspondence; nothing else in the bridge needs to know about image sizes.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def patch_grid_size(n_tokens: int) -> int:
    """Side length ``G`` of the square patch grid holding ``n_tokens`` tokens.

    PaliGemma/SigLIP at 224x224 with patch 14 gives ``N = 256 -> G = 16``.
    """
    g = int(round(math.isqrt(int(n_tokens))))
    if g * g != int(n_tokens):
        raise ValueError(
            f"expected a square patch grid, got n_tokens={n_tokens} "
            "(is a CLS/register token still attached?)"
        )
    return g


def patch_centers(g: int, *, device=None, dtype=torch.float32) -> Tensor:
    """Normalised (u, v) centre of every patch. Returns ``[G*G, 2]``, row-major."""
    idx = torch.arange(g, device=device, dtype=dtype)
    centres = (idx + 0.5) / g
    v, u = torch.meshgrid(centres, centres, indexing="ij")  # row = v, col = u
    return torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)


def sample_patch_features(tokens: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Bilinearly read the patch-feature grid at continuous (u, v).

    Parameters
    ----------
    tokens : ``[B, N, W]`` image patch tokens, row-major, ``N = G*G``.
    u, v   : ``[B, K]`` normalised coordinates in ``[0, 1]``.

    Returns ``[B, K, W]`` — "what the camera sees where the sound is".

    Out-of-range coordinates clamp to the border rather than returning zeros, so
    a slot sitting just off the edge still yields the nearest real feature; the
    caller is responsible for masking slots that are absent entirely.
    """
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B, N, W], got {tuple(tokens.shape)}")
    b, n, w = tokens.shape
    g = patch_grid_size(n)

    # grid_sample needs float32/float16 and NCHW; bf16 is not supported on all
    # backends, so compute in float32 and cast back at the end.
    out_dtype = tokens.dtype
    grid_feats = (
        tokens.to(torch.float32).reshape(b, g, g, w).permute(0, 3, 1, 2).contiguous()
    )  # [B, W, G(row=v), G(col=u)]

    # align_corners=False maps a continuous [0,1] image coordinate to [-1,1].
    gx = (2.0 * u.to(torch.float32) - 1.0).clamp(-1.0, 1.0)
    gy = (2.0 * v.to(torch.float32) - 1.0).clamp(-1.0, 1.0)
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [B, K, 1, 2] = (x, y)

    sampled = F.grid_sample(
        grid_feats, grid, mode="bilinear", padding_mode="border", align_corners=False
    )  # [B, W, K, 1]
    return sampled.squeeze(-1).permute(0, 2, 1).to(out_dtype)  # [B, K, W]


def gaussian_patch_weights(
    u: Tensor, v: Tensor, g: int, sigma: Tensor | float
) -> Tensor:
    """Soft spatial mask over patches, peaked at (u, v).

    Parameters
    ----------
    u, v  : ``[B, K]`` normalised coordinates.
    g     : patch grid side length.
    sigma : scalar or ``[]``/``[1]`` tensor, in normalised image units
            (0.1 ≈ 10 % of the image width).

    Returns ``[B, K, G*G]`` with a peak value of exactly 1 at the centre, so the
    magnitude of whatever it multiplies is set by a separate learned gate rather
    than by the number of patches.
    """
    centres = patch_centers(g, device=u.device, dtype=torch.float32)  # [N, 2]
    du = u.to(torch.float32).unsqueeze(-1) - centres[:, 0]            # [B, K, N]
    dv = v.to(torch.float32).unsqueeze(-1) - centres[:, 1]
    d2 = du * du + dv * dv

    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=u.device, dtype=torch.float32)
    s = sigma.to(torch.float32).clamp_min(1e-3)
    return torch.exp(-0.5 * d2 / (s * s))


def uv_after_resize_with_pad(
    u: Tensor, v: Tensor, src_wh: tuple[int, int], dst_wh: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Re-normalise (u, v) through openpi's ``resize_with_pad``.

    The projection in ``src/audio/projection.py`` is computed against the raw
    render, while the vision tower sees a letterboxed copy. When the render is
    already square and matches the model resolution (the current VLABench eval
    path renders 224x224 directly) this is the identity — but do not rely on
    that silently if a camera is ever re-configured to a non-square resolution.
    """
    sw, sh = float(src_wh[0]), float(src_wh[1])
    dw, dh = float(dst_wh[0]), float(dst_wh[1])
    scale = min(dw / sw, dh / sh)
    new_w, new_h = sw * scale, sh * scale
    pad_x = (dw - new_w) / 2.0
    pad_y = (dh - new_h) / 2.0
    return ((u * new_w + pad_x) / dw, (v * new_h + pad_y) / dh)
