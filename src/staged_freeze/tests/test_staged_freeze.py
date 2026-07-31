"""Tests for the mid-run PaliGemma freeze."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.staged_freeze import StagedFreeze, StagedFreezeConfig, config_from_env
from src.staged_freeze.staged_freeze import ACTION_EXPERT_PREFIX, PALIGEMMA_PREFIX


class _FakePaliGemmaWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = nn.ModuleDict(
            {
                "vision": nn.Linear(4, 4),
                "llm": nn.Linear(4, 4),
                # a LoRA adapter, injected *inside* paligemma like openpi does
                "llm_lora_A": nn.Linear(4, 2, bias=False),
            }
        )
        self.gemma_expert = nn.Linear(4, 4)


class _FakePi0(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _FakePaliGemmaWithExpert()
        self.action_out_proj = nn.Linear(4, 4)
        self.audio_class_embedding = nn.Embedding(6, 4)


def _model_and_optim():
    m = _FakePi0()
    for p in m.parameters():
        p.requires_grad = True
    o = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01
    )
    return m, o


def _step(model, optim):
    """One real optimizer step so AdamW builds state."""
    x = torch.randn(2, 4)
    loss = (
        model.paligemma_with_expert.gemma_expert(x).sum()
        + model.paligemma_with_expert.paligemma["llm"](x).sum()
        + model.action_out_proj(x).sum()
        + model.audio_class_embedding(torch.tensor([0, 1])).sum()
    )
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)


def _names_trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def test_config_from_env_frac(monkeypatch):
    monkeypatch.setenv("OPENPI_FREEZE_LLM_AT_FRAC", "0.6")
    cfg = config_from_env(10000)
    assert cfg is not None and cfg.switch_step == 6000


def test_config_from_env_absolute_step(monkeypatch):
    monkeypatch.setenv("OPENPI_FREEZE_LLM_AT_STEP", "1234")
    assert config_from_env(10000).switch_step == 1234


def test_config_absent_is_none():
    assert config_from_env(10000) is None


def test_config_rejects_both(monkeypatch):
    monkeypatch.setenv("OPENPI_FREEZE_LLM_AT_FRAC", "0.6")
    monkeypatch.setenv("OPENPI_FREEZE_LLM_AT_STEP", "6000")
    with pytest.raises(ValueError, match="only one"):
        config_from_env(10000)


def test_config_rejects_out_of_range_frac(monkeypatch):
    monkeypatch.setenv("OPENPI_FREEZE_LLM_AT_FRAC", "1.5")
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        config_from_env(10000)


def test_config_rejects_switch_past_end():
    with pytest.raises(ValueError, match="never run"):
        StagedFreezeConfig(switch_step=10000).validate(10000)


# --------------------------------------------------------------------------
# freezing behaviour
# --------------------------------------------------------------------------
def test_nothing_happens_before_the_switch():
    m, o = _model_and_optim()
    s = StagedFreeze(StagedFreezeConfig(switch_step=100))
    before = _names_trainable(m)
    o2 = s.maybe_apply(m, o, 99)
    assert o2 is o and not s.applied
    assert _names_trainable(m) == before


def test_switch_freezes_paligemma_and_keeps_action_expert():
    m, o = _model_and_optim()
    s = StagedFreeze(StagedFreezeConfig(switch_step=10))
    o = s.maybe_apply(m, o, 10)
    trainable = _names_trainable(m)
    assert all(not n.startswith(PALIGEMMA_PREFIX) for n in trainable)
    assert any(n.startswith(ACTION_EXPERT_PREFIX) for n in trainable)
    assert any(n.startswith("action_out_proj") for n in trainable)


def test_switch_also_freezes_lora_adapters():
    """The whole point: adapters live inside paligemma and must freeze too."""
    m, o = _model_and_optim()
    s = StagedFreeze(StagedFreezeConfig(switch_step=0 + 1))
    o = s.maybe_apply(m, o, 1)
    lora = [n for n, p in m.named_parameters() if "lora" in n]
    assert lora, "fixture lost its LoRA param"
    assert all(not dict(m.named_parameters())[n].requires_grad for n in lora)


def test_audio_heads_trainable_by_default_and_freezable_on_request():
    m, o = _model_and_optim()
    StagedFreeze(StagedFreezeConfig(switch_step=1)).maybe_apply(m, o, 1)
    assert "audio_class_embedding.weight" in _names_trainable(m)

    m2, o2 = _model_and_optim()
    StagedFreeze(
        StagedFreezeConfig(switch_step=1, freeze_audio_heads=True)
    ).maybe_apply(m2, o2, 1)
    assert "audio_class_embedding.weight" not in _names_trainable(m2)


def test_applies_only_once():
    m, o = _model_and_optim()
    s = StagedFreeze(StagedFreezeConfig(switch_step=5))
    o1 = s.maybe_apply(m, o, 5)
    o2 = s.maybe_apply(m, o1, 6)
    assert o2 is o1


def test_resume_past_switch_applies_immediately():
    m, o = _model_and_optim()
    s = StagedFreeze(StagedFreezeConfig(switch_step=6000))
    o = s.maybe_apply(m, o, 7500)          # resumed mid phase 2
    assert s.applied
    assert all(not n.startswith(PALIGEMMA_PREFIX) for n in _names_trainable(m))


# --------------------------------------------------------------------------
# optimizer surgery
# --------------------------------------------------------------------------
def test_optimizer_drops_frozen_params():
    m, o = _model_and_optim()
    _step(m, o)
    n_before = sum(len(g["params"]) for g in o.param_groups)
    o = StagedFreeze(StagedFreezeConfig(switch_step=1)).maybe_apply(m, o, 1)
    n_after = sum(len(g["params"]) for g in o.param_groups)
    assert n_after < n_before
    assert all(p.requires_grad for g in o.param_groups for p in g["params"])


def test_optimizer_state_is_carried_over():
    """Momentum for the action expert must survive the switch."""
    m, o = _model_and_optim()
    _step(m, o)
    expert_w = m.paligemma_with_expert.gemma_expert.weight
    old_exp_avg = o.state[expert_w]["exp_avg"].clone()

    o = StagedFreeze(StagedFreezeConfig(switch_step=1)).maybe_apply(m, o, 1)
    assert expert_w in o.state
    assert torch.allclose(o.state[expert_w]["exp_avg"], old_exp_avg)


def test_reset_optimizer_state_option_drops_moments():
    m, o = _model_and_optim()
    _step(m, o)
    o = StagedFreeze(
        StagedFreezeConfig(switch_step=1, reset_optimizer_state=True)
    ).maybe_apply(m, o, 1)
    assert not o.state


def test_hyperparameters_are_preserved():
    m, o = _model_and_optim()
    o.param_groups[0]["lr"] = 3.21e-5
    o = StagedFreeze(StagedFreezeConfig(switch_step=1)).maybe_apply(m, o, 1)
    g = o.param_groups[0]
    assert g["lr"] == pytest.approx(3.21e-5)
    assert g["weight_decay"] == pytest.approx(0.01)


def test_training_still_steps_after_the_switch():
    """End-to-end: frozen weights stay put, trainable ones move."""
    m, o = _model_and_optim()
    _step(m, o)
    o = StagedFreeze(StagedFreezeConfig(switch_step=1)).maybe_apply(m, o, 1)

    llm_w = m.paligemma_with_expert.paligemma["llm"].weight.detach().clone()
    exp_w = m.paligemma_with_expert.gemma_expert.weight.detach().clone()
    _step(m, o)

    assert torch.allclose(m.paligemma_with_expert.paligemma["llm"].weight, llm_w), \
        "frozen LLM weight moved"
    assert not torch.allclose(m.paligemma_with_expert.gemma_expert.weight, exp_w), \
        "action expert did not train"


def test_lr_schedule_can_still_drive_the_new_optimizer():
    m, o = _model_and_optim()
    o = StagedFreeze(StagedFreezeConfig(switch_step=1)).maybe_apply(m, o, 1)
    for pg in o.param_groups:      # exactly what the training loop does
        pg["lr"] = 1e-6
    assert all(pg["lr"] == 1e-6 for pg in o.param_groups)


def test_raises_when_nothing_would_stay_trainable():
    class _AllPaliGemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_with_expert = nn.ModuleDict({"paligemma": nn.Linear(4, 4)})

    m = _AllPaliGemma()
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    with pytest.raises(RuntimeError, match="nothing trainable"):
        StagedFreeze(StagedFreezeConfig(switch_step=1)).maybe_apply(m, o, 1)
