"""Two-phase fine-tuning: train everything, then freeze the LLM half.

Phase 1 (steps ``0 .. switch-1``)  — whatever the run was already configured to
do (full fine-tune, or PaliGemma LoRA, …). Nothing here interferes.

Phase 2 (steps ``switch .. end``)  — the whole PaliGemma tower is frozen,
including any injected LoRA adapters, leaving the action expert (and by default
the projection / audio heads) as the only trainable parameters.

Why a module instead of two sbatch jobs
---------------------------------------
openpi decides freezing *once*, before the optimizer exists, and
``OPENPI_PALIGEMMA_LORA`` takes precedence over ``OPENPI_FREEZE_PALIGEMMA``
(they are an ``elif`` chain in ``scripts/train_pytorch.py``). So a phase-1 LoRA
run cannot be resumed "with freeze on": the second job would either skip LoRA
injection — and then fail to load a checkpoint that contains adapters — or
re-inject them and train them again. Doing the switch in-process also keeps the
LR schedule and the AdamW moments of the surviving parameters continuous, which
a restart would throw away.

Wiring — one call at the top of the training loop, see ``README.md``.
"""
from .staged_freeze import (
    StagedFreeze,
    StagedFreezeConfig,
    config_from_env,
)

__all__ = ["StagedFreeze", "StagedFreezeConfig", "config_from_env"]
