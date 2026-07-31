"""attach_to_pi0 wrapping logic, exercised against a stand-in for pi0.

The real model is 3.3B parameters and needs the GPU, so this mimics only the
three methods the attachment wraps and pi0's call order inside embed_prefix.
If the vendored openpi revision changes that order, `test_attach_rejects_wrong_call_order`
is the one that should start failing.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.audio_visual_bridge import BridgeConfig, attach_to_pi0

G, N, W = 4, 16, 8
B, K = 2, 2


class _FakeVisionTower(nn.Module):
    def embed_image(self, img):
        b = img.shape[0]
        # deterministic, position-dependent patch features
        return torch.arange(b * N * W, dtype=torch.float32).reshape(b, N, W) * 0.01


class _FakePi0(nn.Module):
    """Mimics pi0's audio path: images embedded first, then audio slots."""

    def __init__(self, n_images: int = 3):
        super().__init__()
        self.paligemma_with_expert = _FakeVisionTower()
        self.audio_conditioning = True
        self.audio_slot_uv = True
        self.audio_class_embedding = nn.Embedding(6, W)
        self.audio_visual_bridge = None
        self.n_images = n_images
        self.last_img_embs: list[torch.Tensor] = []

    def _embed_audio_slots(self, audio_slots, ref_emb):
        cid = audio_slots[..., 4].round().to(torch.long)
        present = (cid >= 0) & (audio_slots[..., 5] > 0.5)
        safe = torch.where(present, cid.clamp(min=0), torch.full_like(cid, 5))
        return self.audio_class_embedding(safe).to(ref_emb.dtype), present

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, audio_slots=None):
        self.last_img_embs = [self.paligemma_with_expert.embed_image(i) for i in images]
        audio_emb, present = self._embed_audio_slots(
            audio_slots, self.last_img_embs[0]
        )
        return self.last_img_embs, audio_emb, present


def _cfg(**kw):
    base = dict(d_model=W, n_classes=5, b4_grid=G, b1_hidden=16)
    base.update(kw)
    return BridgeConfig(**base)


def _inputs():
    images = [torch.zeros(B, 3, 8, 8) for _ in range(3)]
    slots = torch.zeros(B, K, 6)
    slots[:, 0] = torch.tensor([0.3, 0.7, 0.8, 1.0, 2.0, 1.0])
    slots[:, 1] = torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    return images, slots


def test_attach_modifies_first_image_and_audio_token():
    torch.manual_seed(0)
    model = _FakePi0()
    images, slots = _inputs()
    base_imgs, base_audio, _ = model.embed_prefix(images, None, None, None, slots)
    base_img0 = base_imgs[0].clone()
    base_audio = base_audio.clone()

    attach_to_pi0(model, _cfg(), image_slot=0)
    new_imgs, new_audio, _ = model.embed_prefix(images, None, None, None, slots)

    assert not torch.allclose(new_imgs[0], base_img0), "B2 did not reach the patches"
    assert not torch.allclose(new_audio, base_audio), "B1/B4 did not reach the slot token"
    # non-target cameras must be untouched
    assert torch.allclose(new_imgs[1], base_imgs[1])
    assert torch.allclose(new_imgs[2], base_imgs[2])


def test_attach_with_zero_gates_changes_nothing():
    model = _FakePi0()
    images, slots = _inputs()
    base_imgs, base_audio, _ = model.embed_prefix(images, None, None, None, slots)
    base_img0, base_audio = base_imgs[0].clone(), base_audio.clone()

    attach_to_pi0(model, _cfg(b1_gate_init=0.0, b2_gate_init=0.0, b4_gate_init=0.0))
    new_imgs, new_audio, _ = model.embed_prefix(images, None, None, None, slots)

    assert torch.allclose(new_imgs[0], base_img0, atol=1e-6)
    assert torch.allclose(new_audio, base_audio, atol=1e-6)


def test_attach_targets_the_requested_camera():
    model = _FakePi0()
    images, slots = _inputs()
    base_imgs, _, _ = model.embed_prefix(images, None, None, None, slots)
    base = [e.clone() for e in base_imgs]

    attach_to_pi0(model, _cfg(), image_slot=1)
    new_imgs, _, _ = model.embed_prefix(images, None, None, None, slots)

    assert torch.allclose(new_imgs[0], base[0]), "camera 0 should be untouched"
    assert not torch.allclose(new_imgs[1], base[1]), "camera 1 should be marked"


def test_bridge_is_registered_so_it_is_saved():
    model = _FakePi0()
    attach_to_pi0(model, _cfg())
    keys = [k for k in model.state_dict() if k.startswith("audio_visual_bridge.")]
    assert keys, "bridge params missing from state_dict — they would not be checkpointed"


def test_double_attach_is_refused():
    model = _FakePi0()
    attach_to_pi0(model, _cfg())
    with pytest.raises(RuntimeError, match="already attached"):
        attach_to_pi0(model, _cfg())


def test_attach_requires_slots_uv_path():
    model = _FakePi0()
    model.audio_slot_uv = False
    with pytest.raises(ValueError, match="slots_uv"):
        attach_to_pi0(model, _cfg())


def test_attach_requires_audio_conditioning():
    model = _FakePi0()
    model.audio_conditioning = False
    with pytest.raises(ValueError, match="audio_conditioning"):
        attach_to_pi0(model, _cfg())


def test_attach_rejects_wrong_call_order():
    """If pi0 ever embeds audio before images, fail loudly instead of silently."""
    model = _FakePi0()

    def bad_prefix(images, img_masks, lang_tokens, lang_masks, audio_slots=None):
        audio_emb, present = model._embed_audio_slots(
            audio_slots, torch.zeros(B, N, W)
        )
        return [], audio_emb, present

    attach_to_pi0(model, _cfg())
    inner = model.embed_prefix          # wrapped version
    model.embed_prefix = lambda *a, **k: (
        model._av_bridge_state.__setitem__("slots", a[4] if len(a) > 4 else k.get("audio_slots")),
        bad_prefix(*a, **k),
    )[1]
    images, slots = _inputs()
    with pytest.raises(RuntimeError, match="call order"):
        model.embed_prefix(images, None, None, None, slots)
    assert inner is not None


def _silence(slots):
    slots = slots.clone()
    slots[:, :, 4] = -1.0
    slots[:, :, 5] = 0.0
    return slots


def test_absent_slots_do_not_mark_the_image():
    """Silence must leave the visual stream untouched — B4 off (see its docstring)."""
    model = _FakePi0()
    images, slots = _inputs()
    slots = _silence(slots)
    base_imgs, _, _ = model.embed_prefix(images, None, None, None, slots)
    base0 = base_imgs[0].clone()

    attach_to_pi0(model, _cfg(shared_grid_code=False))
    new_imgs, _, _ = model.embed_prefix(images, None, None, None, slots)
    assert torch.allclose(new_imgs[0], base0, atol=1e-6)


def test_b4_patch_code_is_intentionally_silence_independent():
    """B4's patch code is a positional basis: identical with and without sound.

    Documents the deliberate exception to silence-invariance. If this ever
    starts failing, the basis has become audio-dependent and B4's premise
    (both streams share ONE coordinate system) no longer holds.
    """
    model = _FakePi0()
    images, slots = _inputs()
    attach_to_pi0(model, _cfg(patch_anchored_slot=False,
                              sound_marker_on_patches=False))  # B4 only
    loud_imgs, _, _ = model.embed_prefix(images, None, None, None, slots)
    loud = loud_imgs[0].clone()
    quiet_imgs, _, _ = model.embed_prefix(images, None, None, None, _silence(slots))
    assert torch.allclose(quiet_imgs[0], loud, atol=1e-6)

    base = _FakePi0().embed_prefix(images, None, None, None, slots)[0][0]
    assert not torch.allclose(loud, base), "B4 patch code was never applied"


def test_b3_attach_warns_and_exposes_bias(caplog):
    model = _FakePi0()
    with caplog.at_level("WARNING"):
        attach_to_pi0(model, _cfg(spatial_attention_bias=True))
    assert any("B3" in r.message for r in caplog.records)
    images, slots = _inputs()
    model.embed_prefix(images, None, None, None, slots)
    bias = model._av_bridge_state["attention_bias"]
    assert bias is not None and bias.shape == (B, K, N)
    assert math.isfinite(float(bias.abs().max()))
