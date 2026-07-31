"""Configuration for the audio↔visual spatial bridge."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BridgeConfig:
    """Which bridges to enable and how strongly.

    Every bridge is gated so that at ``gate_init = 0`` the whole module is a
    mathematical no-op. That matters here: these modules are meant to be added
    to an already fine-tuned pi0 checkpoint, and a bridge that perturbs the
    prefix at step 0 would destroy the behaviour we are trying to improve.
    Start at 0.0 to resume from a checkpoint, or ~0.1 to train from the base
    weights.
    """

    # --- which bridges ------------------------------------------------------
    patch_anchored_slot: bool = True   # B1  audio token ← visual feature at (u,v)
    sound_marker_on_patches: bool = True   # B2  image patches ← sound marker at (u,v)
    spatial_attention_bias: bool = False   # B3  audio→patch attention prior
    shared_grid_code: bool = True   # B4  one 2D positional code for both

    # --- shapes -------------------------------------------------------------
    d_model: int = 2048          # PaliGemma width; must match the host model
    n_classes: int = 38          # audio taxonomy size (excluding the sentinel)

    # --- B1 -----------------------------------------------------------------
    b1_hidden: int = 512
    b1_gate_init: float = 0.1

    # --- B2 -----------------------------------------------------------------
    b2_sigma_init: float = 0.10  # normalised image units (~1.6 patches at G=16)
    b2_learn_sigma: bool = True
    b2_gate_init: float = 0.1
    b2_use_class: bool = True    # marker carries which sound it is, not just where

    # --- B3 -----------------------------------------------------------------
    b3_sigma_init: float = 0.15
    b3_max_logit_bias: float = 4.0   # cap so the prior can never hard-mask
    b3_gate_init: float = 0.1

    # --- B4 -----------------------------------------------------------------
    b4_grid: int = 16            # must equal the vision tower's patch grid
    b4_gate_init: float = 0.1

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.b4_grid <= 0:
            raise ValueError("b4_grid must be positive")
        if not (
            self.patch_anchored_slot
            or self.sound_marker_on_patches
            or self.spatial_attention_bias
            or self.shared_grid_code
        ):
            raise ValueError("at least one bridge must be enabled")
