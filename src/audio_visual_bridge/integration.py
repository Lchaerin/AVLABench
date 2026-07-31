"""Wiring helpers — keep the bridge out of the existing model files.

Two routes:

1. ``attach_to_pi0(model, cfg)`` — runtime attachment. Wraps three small pi0
   methods (``embed_prefix``, ``embed_image``, ``_embed_audio_slots``) so no file
   under ``src/models`` or ``third_party/openpi`` is edited. Good for
   experiments and ablations.

2. The explicit edit shown in ``README.md``. Preferred once a configuration is
   settled, because it is visible in a diff and does not depend on pi0's
   internal call order.

Route 1 relies on pi0 calling ``embed_image`` for every camera *before*
``_embed_audio_slots`` inside ``embed_prefix`` (true as of the vendored openpi
revision). ``attach_to_pi0`` asserts this at attach time and raises rather than
silently grounding the sound in nothing.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from .bridges import AudioVisualBridge
from .config import BridgeConfig

logger = logging.getLogger(__name__)


def unpack_audio_slots(audio_slots: Tensor) -> dict[str, Tensor]:
    """Split pi0's ``audio_slots`` ``[B, K, 6]`` into named fields.

    Layout (matches ``pi0_pytorch._embed_audio_slots`` slots_uv path):
    ``(u, v, energy, conf, class_id, present)``.
    """
    if audio_slots.ndim != 3 or audio_slots.shape[-1] < 6:
        raise ValueError(
            "expected slots_uv audio_slots of shape [B, K, 6], got "
            f"{tuple(audio_slots.shape)} — the legacy az/el layout has no (u, v) "
            "and cannot be bridged to image space."
        )
    cid = audio_slots[..., 4].round().to(torch.long)
    return {
        "u": audio_slots[..., 0].to(torch.float32),
        "v": audio_slots[..., 1].to(torch.float32),
        "energy": audio_slots[..., 2].to(torch.float32),
        "conf": audio_slots[..., 3].to(torch.float32),
        "class_id": cid,
        "present": (cid >= 0) & (audio_slots[..., 5] > 0.5),
    }


def scatter_attention_bias(
    att_bias_2d: Tensor,
    slot_patch_bias: Tensor,
    *,
    patch_start: int,
    slot_start: int,
) -> Tensor:
    """Write B3's ``[B, K, N]`` bias into a full ``[B, L, L]`` additive mask.

    ``patch_start`` / ``slot_start`` are the offsets of the first image-patch
    token and the first audio-slot token in the prefix. Rows are queries, so the
    bias lands at ``[:, slot_start + k, patch_start + n]``.

    The mask must already be additive log-space (0 = attend, large negative =
    blocked). Adding to a boolean mask is a silent no-op, so this raises.
    """
    if att_bias_2d.dtype == torch.bool:
        raise TypeError(
            "att_bias_2d is boolean; convert to an additive float mask before "
            "adding a spatial prior"
        )
    b, k, n = slot_patch_bias.shape
    out = att_bias_2d.clone()
    out[:, slot_start : slot_start + k, patch_start : patch_start + n] += (
        slot_patch_bias.to(out.dtype)
    )
    return out


def attach_to_pi0(
    model: Any,
    cfg: BridgeConfig | None = None,
    *,
    image_slot: int = 0,
    device=None,
    dtype=None,
) -> AudioVisualBridge:
    """Attach the bridge to a live pi0 PyTorch model without editing its source.

    Parameters
    ----------
    model      : ``PI0Pytorch`` instance with ``audio_conditioning`` enabled and
                 ``audio_slot_uv=True`` (the ``slots_uv`` path).
    cfg        : defaults to :class:`BridgeConfig` with ``d_model`` taken from
                 the model's audio class embedding.
    image_slot : index of the camera the projection was computed against. For
                 this repo that is ``cam_2`` = the first image = ``0``. It is
                 also the microphone and the (u, v) reference; pointing the
                 bridge at any other camera grounds the sound in the wrong view.

    Returns the attached :class:`AudioVisualBridge` (also set as
    ``model.audio_visual_bridge`` so it is registered and saved with the model).
    """
    for attr in ("embed_prefix", "embed_image", "_embed_audio_slots"):
        target = model if attr != "embed_image" else getattr(
            model, "paligemma_with_expert", None
        )
        if target is None or not hasattr(target, attr):
            raise AttributeError(
                f"model does not expose {attr!r}; the vendored openpi revision "
                "changed and attach_to_pi0 needs updating"
            )
    if not getattr(model, "audio_conditioning", False):
        raise ValueError("model has audio_conditioning disabled — nothing to bridge")
    if not getattr(model, "audio_slot_uv", False):
        raise ValueError(
            "model is on the legacy az/el audio path; the bridge needs slots_uv"
        )
    if getattr(model, "audio_visual_bridge", None) is not None:
        raise RuntimeError("a bridge is already attached to this model")

    if cfg is None:
        cfg = BridgeConfig(d_model=model.audio_class_embedding.embedding_dim,
                           n_classes=model.audio_class_embedding.num_embeddings - 1)
    bridge = AudioVisualBridge(cfg)
    ref = model.audio_class_embedding.weight
    bridge = bridge.to(device or ref.device, dtype=dtype or torch.float32)
    model.audio_visual_bridge = bridge  # registered submodule -> saved in ckpt

    state: dict[str, Any] = {"slots": None, "img_idx": 0, "slot_residual": None,
                             "attention_bias": None}
    model._av_bridge_state = state

    orig_prefix = model.embed_prefix
    orig_image = model.paligemma_with_expert.embed_image
    orig_slots = model._embed_audio_slots

    def wrapped_prefix(images, img_masks, lang_tokens, lang_masks, audio_slots=None):
        state["slots"] = audio_slots
        state["img_idx"] = 0
        state["slot_residual"] = None
        state["attention_bias"] = None
        try:
            return orig_prefix(images, img_masks, lang_tokens, lang_masks, audio_slots)
        finally:
            state["slots"] = None

    def wrapped_image(img):
        emb = orig_image(img)
        idx = state["img_idx"]
        state["img_idx"] = idx + 1
        slots = state["slots"]
        if idx != image_slot or slots is None:
            return emb
        f = unpack_audio_slots(slots)
        out = bridge(
            patch_tokens=emb,
            slot_emb=emb.new_zeros(emb.shape[0], f["u"].shape[1], emb.shape[-1]),
            u=f["u"], v=f["v"], present=f["present"],
            class_id=f["class_id"], energy=f["energy"],
        )
        state["slot_residual"] = out.slot_residual
        state["attention_bias"] = out.attention_bias
        return emb + out.patch_residual

    def wrapped_audio_slots(audio_slots, ref_emb):
        emb, present = orig_slots(audio_slots, ref_emb)
        res = state["slot_residual"]
        if res is None:
            raise RuntimeError(
                "audio slots were embedded before any image — pi0's call order "
                "changed; use the explicit edit route instead (see README.md)"
            )
        state["slot_residual"] = None
        return emb + res.to(emb.dtype), present

    model.embed_prefix = wrapped_prefix
    model.paligemma_with_expert.embed_image = wrapped_image
    model._embed_audio_slots = wrapped_audio_slots

    logger.info(
        "[av-bridge] attached (image_slot=%d, bridges=%s)",
        image_slot,
        ",".join(k for k in ("b1", "b2", "b3", "b4") if getattr(bridge, k) is not None),
    )
    if cfg.spatial_attention_bias:
        logger.warning(
            "[av-bridge] B3 produces an attention bias but attach_to_pi0 cannot "
            "reach pi0's attention mask; read model._av_bridge_state"
            "['attention_bias'] and apply it yourself, or disable B3."
        )
    return bridge
