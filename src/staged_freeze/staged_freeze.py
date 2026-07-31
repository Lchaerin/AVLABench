"""Mid-run freeze of the PaliGemma tower, with optimizer state carried over."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# Everything under this prefix is the vision tower + LLM ("the PaliGemma half").
# Injected LoRA adapters live *inside* it (openpi's own freeze block tests
# `name.startswith(PALIGEMMA_PREFIX)` and `".lora_A" in name` separately), so
# freezing by prefix freezes the adapters too — which is the intent here.
PALIGEMMA_PREFIX = "paligemma_with_expert.paligemma."

# The action expert. Kept trainable in phase 2 by definition.
ACTION_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert."


@dataclass
class StagedFreezeConfig:
    """When to switch and what to freeze."""

    switch_step: int
    """Global step at which phase 2 begins (inclusive)."""

    freeze_audio_heads: bool = False
    """Also freeze the audio class embedding / slot encoder. Off by default:
    openpi's own ``OPENPI_FREEZE_PALIGEMMA`` leaves every non-PaliGemma module
    trainable, and matching that keeps phase 2 comparable to existing runs."""

    reset_optimizer_state: bool = False
    """Rebuild AdamW moments from scratch instead of transplanting the state of
    the surviving parameters. Off by default — dropping the action expert's
    momentum at the switch causes an avoidable loss spike."""

    def validate(self, num_train_steps: int) -> None:
        if self.switch_step <= 0:
            raise ValueError(f"switch_step must be > 0, got {self.switch_step}")
        if self.switch_step >= num_train_steps:
            raise ValueError(
                f"switch_step={self.switch_step} is at/after the end of training "
                f"({num_train_steps}); phase 2 would never run"
            )


def config_from_env(num_train_steps: int) -> StagedFreezeConfig | None:
    """Read the staged-freeze request from the environment.

    ``OPENPI_FREEZE_LLM_AT_FRAC=0.6``  → switch at 60 % of training, or
    ``OPENPI_FREEZE_LLM_AT_STEP=6000`` → switch at an absolute step.
    ``OPENPI_FREEZE_LLM_AUDIO_HEADS=1`` also freezes the audio modules.

    Returns ``None`` when no staged freeze was requested, so the call site is a
    no-op for every existing run.
    """
    frac = os.environ.get("OPENPI_FREEZE_LLM_AT_FRAC", "").strip()
    step = os.environ.get("OPENPI_FREEZE_LLM_AT_STEP", "").strip()
    if not frac and not step:
        return None
    if frac and step:
        raise ValueError(
            "set only one of OPENPI_FREEZE_LLM_AT_FRAC / OPENPI_FREEZE_LLM_AT_STEP"
        )

    if frac:
        f = float(frac)
        if not 0.0 < f < 1.0:
            raise ValueError(f"OPENPI_FREEZE_LLM_AT_FRAC must be in (0, 1), got {f}")
        # The epsilon is not cosmetic: 0.7 is not representable in binary, so
        # 0.7 * 11000 == 7699.999999999999 and a bare floor() switches at 7699
        # instead of 7700 -- which also means a SAVE_INTERVAL chosen to land a
        # checkpoint exactly on the switch lands one step after it instead.
        switch = int(math.floor(f * num_train_steps + 1e-9))
    else:
        switch = int(step)

    cfg = StagedFreezeConfig(
        switch_step=switch,
        freeze_audio_heads=os.environ.get("OPENPI_FREEZE_LLM_AUDIO_HEADS", "0") == "1",
    )
    cfg.validate(num_train_steps)
    return cfg


class StagedFreeze:
    """Applies the phase-2 freeze exactly once, at or after ``switch_step``.

    Usage inside the training loop, before the forward pass::

        optim = staged.maybe_apply(model, optim, global_step)

    Safe to call every step; it does nothing until the switch and nothing again
    afterwards. On resume past the switch it applies immediately, so a job that
    dies in phase 2 comes back in phase 2.
    """

    def __init__(self, cfg: StagedFreezeConfig):
        self.cfg = cfg
        self.applied = False

    # ------------------------------------------------------------------
    def maybe_apply(self, model, optim, global_step: int):
        if self.applied or global_step < self.cfg.switch_step:
            return optim
        return self.apply(model, optim, global_step)

    def apply(self, model, optim, global_step: int):
        raw = model.module if hasattr(model, "module") else model

        n_frozen = n_trainable = 0
        newly_frozen = 0
        for name, p in raw.named_parameters():
            freeze = name.startswith(PALIGEMMA_PREFIX)
            if self.cfg.freeze_audio_heads and name.startswith("audio_"):
                freeze = True
            if freeze:
                if p.requires_grad:
                    newly_frozen += p.numel()
                p.requires_grad = False
                n_frozen += p.numel()
            else:
                n_trainable += p.numel()

        if n_trainable == 0:
            raise RuntimeError(
                "staged freeze would leave nothing trainable — check that this "
                f"model actually has modules outside {PALIGEMMA_PREFIX!r}"
            )
        if newly_frozen == 0:
            logger.warning(
                "[staged-freeze] nothing new was frozen at step %d; the LLM was "
                "already frozen (was this run already using OPENPI_FREEZE_PALIGEMMA?)",
                global_step,
            )

        new_optim = self._rebuild_optimizer(raw, optim)

        logger.info(
            "[staged-freeze] step %d: PaliGemma frozen (%.1fM params, %.1fM newly "
            "frozen); trainable now %.1fM (action expert%s). optimizer groups: "
            "%d -> %d params",
            global_step,
            n_frozen / 1e6,
            newly_frozen / 1e6,
            n_trainable / 1e6,
            "" if self.cfg.freeze_audio_heads else " + heads",
            sum(len(g["params"]) for g in optim.param_groups),
            sum(len(g["params"]) for g in new_optim.param_groups),
        )
        self.applied = True
        return new_optim

    # ------------------------------------------------------------------
    def _rebuild_optimizer(self, raw_model, optim):
        """New AdamW over the surviving params, keeping their moments.

        Rebuilding (rather than leaving the frozen params in the optimizer) is
        what actually releases the AdamW state — for pi0 that is the difference
        between ~24GB and ~2-3GB of optimizer memory.
        """
        survivors = [p for p in raw_model.parameters() if p.requires_grad]
        if not survivors:
            raise RuntimeError("no trainable parameters left after the freeze")

        group_defaults = {
            k: v for k, v in optim.param_groups[0].items() if k != "params"
        }
        new_optim = torch.optim.AdamW([{"params": survivors, **group_defaults}])

        if not self.cfg.reset_optimizer_state:
            kept = 0
            for p in survivors:
                st = optim.state.get(p)
                if st:
                    # same tensor objects -> state transplants directly
                    new_optim.state[p] = st
                    kept += 1
            logger.info(
                "[staged-freeze] carried AdamW state for %d/%d surviving tensors",
                kept,
                len(survivors),
            )

        # Drop references to the old state so the frozen params' moments are
        # actually freed rather than kept alive by the old optimizer object.
        optim.state.clear()
        for g in optim.param_groups:
            g["params"] = []

        return new_optim
