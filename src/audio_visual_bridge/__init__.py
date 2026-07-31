"""Explicit audio↔visual spatial bridges for the SELD-VLA / pi0 audio path.

Self-contained: nothing under ``src/models``, ``src/audio`` or
``third_party/openpi`` imports this package, and this package imports nothing
from them. Wire it in via :func:`integration.attach_to_pi0` or the explicit edit
documented in ``README.md``.
"""
from .bridges import (
    AudioVisualBridge,
    BridgeOutput,
    PatchAnchoredSlot,
    SharedGridPositionalCode,
    SoundMarkerOnPatches,
    SpatialAttentionBias,
)
from .config import BridgeConfig
from .geometry import (
    gaussian_patch_weights,
    patch_centers,
    patch_grid_size,
    sample_patch_features,
    uv_after_resize_with_pad,
)
from .integration import attach_to_pi0, scatter_attention_bias, unpack_audio_slots

__all__ = [
    "AudioVisualBridge",
    "BridgeConfig",
    "BridgeOutput",
    "PatchAnchoredSlot",
    "SharedGridPositionalCode",
    "SoundMarkerOnPatches",
    "SpatialAttentionBias",
    "attach_to_pi0",
    "gaussian_patch_weights",
    "patch_centers",
    "patch_grid_size",
    "sample_patch_features",
    "scatter_attention_bias",
    "unpack_audio_slots",
    "uv_after_resize_with_pad",
]
