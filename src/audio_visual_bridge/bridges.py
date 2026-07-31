"""Explicit spatial bridges between the audio slot (u, v) and image patches.

Why this exists
---------------
Today the projected sound position reaches pi0 as ``SlotEncoder(u, v, e, c)`` —
one extra prefix token carrying Fourier features of two numbers, appended after
the image tokens. Nothing tells the model that ``u = 0.31`` refers to the same
place in the scene as image patch ``(row 7, col 5)``. The correspondence has to
be discovered implicitly from reward-free imitation data, and measurements on
the find_hidden task show it only ever gets discovered for the coarse axis
(azimuth separates cleanly, elevation does not).

Each bridge below makes some part of that correspondence *architectural*:

  B1 ``PatchAnchoredSlot``       audio token ← visual feature sampled at (u, v)
  B2 ``SoundMarkerOnPatches``    image patches ← a marker painted at (u, v)
  B3 ``SpatialAttentionBias``    audio query → patch keys, prior peaked at (u, v)
  B4 ``SharedGridPositionalCode`` one learned 2D code indexed by both sides

They are independent and composable. All of them are gated by ``tanh(a)`` with
``a`` initialised so the gate starts small (or exactly zero), mirroring the
Flamingo-style gating already used by ``src/audio/slot_encoder.SlotEncoder``, so
adding them to a trained checkpoint changes nothing until the gate learns.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import BridgeConfig
from .geometry import (
    gaussian_patch_weights,
    patch_centers,
    patch_grid_size,
    sample_patch_features,
)


def _gate_param(gate_init: float) -> nn.Parameter:
    """``tanh``-gate parameter starting at ``gate_init`` (0 → exact no-op)."""
    g = float(gate_init)
    if not -1.0 < g < 1.0:
        raise ValueError(f"gate_init must be in (-1, 1), got {g}")
    return nn.Parameter(torch.tensor(math.atanh(g), dtype=torch.float32))


class PatchAnchoredSlot(nn.Module):
    """B1 — pull the image feature at (u, v) into the audio slot token.

    The audio token stops being "a sound at some abstract coordinate" and starts
    being "a sound *at this thing the camera can see*". This is the shortest
    possible path from the audio cue to visual evidence: one bilinear read.

    Returns a residual to be **added** to the existing slot embedding.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.b1_hidden),
            nn.GELU(),
            nn.Linear(cfg.b1_hidden, cfg.d_model),
        )
        self.out_ln = nn.LayerNorm(cfg.d_model)
        self.a = _gate_param(cfg.b1_gate_init)

    def forward(
        self, patch_tokens: Tensor, u: Tensor, v: Tensor, present: Tensor
    ) -> Tensor:
        """``patch_tokens`` ``[B, N, W]``; ``u``/``v``/``present`` ``[B, K]``.

        Returns ``[B, K, W]``, zero wherever ``present`` is False.
        """
        sampled = sample_patch_features(patch_tokens, u, v)          # [B, K, W]
        residual = self.out_ln(self.proj(sampled.to(torch.float32)))
        residual = torch.tanh(self.a) * residual
        residual = residual * present.unsqueeze(-1).to(residual.dtype)
        return residual.to(patch_tokens.dtype)

    @property
    def gate(self) -> float:
        return float(torch.tanh(self.a).item())


class SoundMarkerOnPatches(nn.Module):
    """B2 — paint the sound onto the visual tokens.

    A learned marker embedding (optionally class-conditioned, so "a radio here"
    and "a chime here" mark differently) is added to every patch token, scaled
    by a Gaussian centred on (u, v). The model no longer has to *find* the sound
    location in the image: the location arrives already highlighted, in the same
    tensor as the pixels.

    This is the bridge that most directly targets a weak cue — an elevation
    difference too small to separate as two numbers is still two distinguishable
    highlight positions once painted onto a 16x16 grid.

    Returns a residual to be **added** to the image patch tokens.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.use_class = bool(cfg.b2_use_class)
        if self.use_class:
            # +1 row: sentinel for silence/absent slots (never actually used
            # because absent slots are masked, but keeps indexing total).
            self.marker = nn.Embedding(cfg.n_classes + 1, cfg.d_model)
            nn.init.normal_(self.marker.weight, std=0.02)
        else:
            self.marker = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.energy_scale = nn.Linear(1, 1)
        nn.init.zeros_(self.energy_scale.weight)
        nn.init.ones_(self.energy_scale.bias)

        sigma = torch.tensor(float(cfg.b2_sigma_init), dtype=torch.float32)
        self.log_sigma = (
            nn.Parameter(sigma.log()) if cfg.b2_learn_sigma
            else nn.Parameter(sigma.log(), requires_grad=False)
        )
        self.a = _gate_param(cfg.b2_gate_init)

    def forward(
        self,
        patch_tokens: Tensor,
        u: Tensor,
        v: Tensor,
        present: Tensor,
        class_id: Tensor | None = None,
        energy: Tensor | None = None,
    ) -> Tensor:
        """``patch_tokens`` ``[B, N, W]``; slot tensors ``[B, K]``.

        Returns ``[B, N, W]`` — the summed contribution of all present slots.
        """
        b, n, w = patch_tokens.shape
        g = patch_grid_size(n)
        weights = gaussian_patch_weights(u, v, g, self.log_sigma.exp())  # [B,K,N]
        weights = weights * present.unsqueeze(-1).to(weights.dtype)

        if energy is not None:
            scale = self.energy_scale(energy.to(torch.float32).unsqueeze(-1))
            weights = weights * scale.clamp(min=0.0)                     # [B,K,N]

        if self.use_class:
            if class_id is None:
                raise ValueError("b2_use_class=True requires class_id")
            sentinel = self.marker.num_embeddings - 1
            safe = torch.where(present, class_id.clamp(min=0),
                               torch.full_like(class_id, sentinel))
            vecs = self.marker(safe).to(torch.float32)                   # [B,K,W]
        else:
            vecs = self.marker.to(torch.float32).expand(b, u.shape[1], w)

        # [B,K,N] x [B,K,W] -> [B,N,W]
        residual = torch.einsum("bkn,bkw->bnw", weights, vecs)
        residual = torch.tanh(self.a) * residual
        return residual.to(patch_tokens.dtype)

    @property
    def sigma(self) -> float:
        return float(self.log_sigma.exp().item())

    @property
    def gate(self) -> float:
        return float(torch.tanh(self.a).item())


class SpatialAttentionBias(nn.Module):
    """B3 — an attention prior from the audio token toward its own patches.

    Produces an additive, pre-softmax bias ``[B, K, N]`` that the caller scatters
    into the model's attention mask (see ``integration.scatter_attention_bias``).
    Unlike B1/B2 this adds no signal to the residual stream; it only makes the
    audio token *look* at the right patches from the first step, which is what
    an implicit-only design has to spend training data discovering.

    The bias is bounded by ``b3_max_logit_bias`` so it can nudge attention but
    can never act as a hard mask (a wrong (u, v) from a noisy DOA estimate must
    stay recoverable).
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        sigma = torch.tensor(float(cfg.b3_sigma_init), dtype=torch.float32)
        self.log_sigma = nn.Parameter(sigma.log())
        self.max_bias = float(cfg.b3_max_logit_bias)
        self.a = _gate_param(cfg.b3_gate_init)

    def forward(
        self, n_patches: int, u: Tensor, v: Tensor, present: Tensor
    ) -> Tensor:
        """Returns ``[B, K, N]`` additive logit bias, zero for absent slots."""
        g = patch_grid_size(n_patches)
        weights = gaussian_patch_weights(u, v, g, self.log_sigma.exp())
        bias = torch.tanh(self.a) * self.max_bias * weights
        return bias * present.unsqueeze(-1).to(bias.dtype)

    @property
    def gate(self) -> float:
        return float(torch.tanh(self.a).item())


class SharedGridPositionalCode(nn.Module):
    """B4 — one learned 2D code, read by both sides.

    A ``G x G x W`` table of learned position codes. Image patch ``p`` gets row
    ``p`` exactly; the audio slot gets the *bilinear interpolation* of the same
    table at its continuous (u, v). Both streams then carry position in a single
    shared basis instead of two unrelated ones (patch order vs. Fourier(u, v)).

    Note on silence — **the patch-side code is added unconditionally**, whether
    or not any slot is present, while the slot-side code is masked by
    ``present``. This is deliberate: the code is a positional basis for the
    image stream, so it must be identical in silent and non-silent frames. Were
    it gated on audio presence, the same image would encode differently
    depending on whether a sound happened to be audible, which is a worse
    distribution shift than the (gated, initially tiny) constant it adds.

    Consequence: with B4 enabled the bridge is *not* a no-op on silent frames.
    It is still an exact no-op at ``gate = 0``. Disable B4 if strict
    silence-invariance of the visual stream matters more than a shared basis.

    Returns ``(patch_residual [B, N, W], slot_residual [B, K, W])``.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.g = int(cfg.b4_grid)
        self.table = nn.Parameter(torch.randn(self.g * self.g, cfg.d_model) * 0.02)
        self.a = _gate_param(cfg.b4_gate_init)

    def forward(
        self, patch_tokens: Tensor, u: Tensor, v: Tensor, present: Tensor
    ) -> tuple[Tensor, Tensor]:
        b, n, w = patch_tokens.shape
        if n != self.g * self.g:
            raise ValueError(
                f"b4_grid={self.g} implies {self.g * self.g} patches, got {n}. "
                "Set BridgeConfig.b4_grid to the vision tower's grid."
            )
        gate = torch.tanh(self.a)

        patch_res = gate * self.table.unsqueeze(0).expand(b, n, w)

        table_b = self.table.unsqueeze(0).expand(b, n, w)
        slot_res = sample_patch_features(table_b, u, v)                  # [B,K,W]
        slot_res = gate * slot_res * present.unsqueeze(-1).to(slot_res.dtype)

        return patch_res.to(patch_tokens.dtype), slot_res.to(patch_tokens.dtype)

    @property
    def gate(self) -> float:
        return float(torch.tanh(self.a).item())


class AudioVisualBridge(nn.Module):
    """Composite of the enabled bridges, with one call per forward pass.

    Usage inside a host model's prefix builder::

        out = bridge(patch_tokens=img_emb, slot_emb=audio_emb,
                     u=u, v=v, present=present, class_id=cid, energy=energy)
        img_emb   = img_emb   + out.patch_residual
        audio_emb = audio_emb + out.slot_residual
        # out.attention_bias -> scatter into the attention mask if using B3

    ``patch_tokens`` must be the tokens of the camera the projection was
    computed against (``cam_2`` / ``base_0_rgb`` for this repo — it is both the
    policy's main view and the microphone). Feeding a different camera's tokens
    silently grounds the sound in the wrong image.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.b1 = PatchAnchoredSlot(cfg) if cfg.patch_anchored_slot else None
        self.b2 = SoundMarkerOnPatches(cfg) if cfg.sound_marker_on_patches else None
        self.b3 = SpatialAttentionBias(cfg) if cfg.spatial_attention_bias else None
        self.b4 = SharedGridPositionalCode(cfg) if cfg.shared_grid_code else None

    def forward(
        self,
        patch_tokens: Tensor,
        slot_emb: Tensor,
        u: Tensor,
        v: Tensor,
        present: Tensor,
        class_id: Tensor | None = None,
        energy: Tensor | None = None,
    ) -> "BridgeOutput":
        if patch_tokens.ndim != 3:
            raise ValueError(f"patch_tokens must be [B, N, W], got {tuple(patch_tokens.shape)}")
        if slot_emb.ndim != 3:
            raise ValueError(f"slot_emb must be [B, K, W], got {tuple(slot_emb.shape)}")
        if patch_tokens.shape[0] != slot_emb.shape[0]:
            raise ValueError("patch_tokens and slot_emb disagree on batch size")
        if patch_tokens.shape[-1] != slot_emb.shape[-1]:
            raise ValueError("patch_tokens and slot_emb disagree on width")
        present = present.to(torch.bool)

        patch_res = torch.zeros_like(patch_tokens)
        slot_res = torch.zeros_like(slot_emb)
        att_bias = None

        if self.b1 is not None:
            slot_res = slot_res + self.b1(patch_tokens, u, v, present)
        if self.b2 is not None:
            patch_res = patch_res + self.b2(
                patch_tokens, u, v, present, class_id=class_id, energy=energy
            )
        if self.b4 is not None:
            p4, s4 = self.b4(patch_tokens, u, v, present)
            patch_res = patch_res + p4
            slot_res = slot_res + s4
        if self.b3 is not None:
            att_bias = self.b3(patch_tokens.shape[1], u, v, present)

        return BridgeOutput(patch_res, slot_res, att_bias)

    def gates(self) -> dict[str, float]:
        """Current gate values — log these; a gate stuck at its init means the
        bridge is not being used and the run is not testing what you think."""
        out: dict[str, float] = {}
        for name in ("b1", "b2", "b3", "b4"):
            mod = getattr(self, name)
            if mod is not None:
                out[name] = mod.gate
        if self.b2 is not None:
            out["b2_sigma"] = self.b2.sigma
        return out


class BridgeOutput:
    """Residuals produced by :class:`AudioVisualBridge`."""

    __slots__ = ("patch_residual", "slot_residual", "attention_bias")

    def __init__(
        self, patch_residual: Tensor, slot_residual: Tensor, attention_bias: Tensor | None
    ):
        self.patch_residual = patch_residual
        self.slot_residual = slot_residual
        self.attention_bias = attention_bias
