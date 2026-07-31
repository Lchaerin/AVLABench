"""Tests for the audio↔visual bridge.

Run:  python -m pytest src/audio_visual_bridge/tests/test_bridges.py -q
"""
from __future__ import annotations

import math

import pytest
import torch

from src.audio_visual_bridge import (
    AudioVisualBridge,
    BridgeConfig,
    PatchAnchoredSlot,
    SharedGridPositionalCode,
    SoundMarkerOnPatches,
    SpatialAttentionBias,
    gaussian_patch_weights,
    patch_centers,
    patch_grid_size,
    sample_patch_features,
    scatter_attention_bias,
    unpack_audio_slots,
    uv_after_resize_with_pad,
)

G = 4                 # small grid keeps the tests fast and readable
N = G * G
W = 8
B, K = 2, 3


def _cfg(**kw) -> BridgeConfig:
    base = dict(d_model=W, n_classes=5, b4_grid=G, b1_hidden=16)
    base.update(kw)
    return BridgeConfig(**base)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_patch_grid_size_rejects_non_square():
    assert patch_grid_size(256) == 16
    with pytest.raises(ValueError):
        patch_grid_size(257)


def test_patch_centers_layout_is_row_major_with_v_down():
    c = patch_centers(G)
    assert c.shape == (N, 2)
    # index 0 = top-left patch
    assert c[0].tolist() == pytest.approx([0.5 / G, 0.5 / G])
    # index G-1 = top-right: u large, v still small
    assert c[G - 1][0] > c[0][0] and c[G - 1][1] == pytest.approx(c[0][1])
    # index (G-1)*G = bottom-left: v large, u small
    assert c[(G - 1) * G][1] > c[0][1] and c[(G - 1) * G][0] == pytest.approx(c[0][0])


def test_sample_patch_features_reads_the_addressed_patch():
    """A one-hot feature grid must be recovered at the matching (u, v)."""
    tokens = torch.zeros(1, N, W)
    row, col = 3, 1
    tokens[0, row * G + col, :] = 1.0
    u = torch.tensor([[(col + 0.5) / G]])
    v = torch.tensor([[(row + 0.5) / G]])
    out = sample_patch_features(tokens, u, v)
    assert out.shape == (1, 1, W)
    assert out[0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)


def test_sample_patch_features_uv_axes_are_not_swapped():
    """Regression guard: u must index the column, v the row."""
    tokens = torch.zeros(1, N, W)
    tokens[0, 0 * G + (G - 1), :] = 1.0          # top-RIGHT patch
    u = torch.tensor([[(G - 0.5) / G]])          # right
    v = torch.tensor([[0.5 / G]])                # top
    assert sample_patch_features(tokens, u, v)[0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)
    # swapping the arguments must NOT find it
    swapped = sample_patch_features(tokens, v, u)[0, 0, 0].item()
    assert swapped < 0.5


def test_gaussian_weights_peak_at_uv_and_decay():
    u = torch.tensor([[0.5]])
    v = torch.tensor([[0.5]])
    w = gaussian_patch_weights(u, v, G, 0.1)[0, 0]
    centres = patch_centers(G)
    d = ((centres[:, 0] - 0.5) ** 2 + (centres[:, 1] - 0.5) ** 2).sqrt()
    assert torch.argmax(w).item() == torch.argmin(d).item()
    assert w.max() <= 1.0 + 1e-6
    assert w[torch.argmax(d)] < w.max()


def test_uv_after_resize_with_pad_identity_for_square():
    u = torch.tensor([[0.25]])
    v = torch.tensor([[0.75]])
    nu, nv = uv_after_resize_with_pad(u, v, (224, 224), (224, 224))
    assert nu.item() == pytest.approx(0.25)
    assert nv.item() == pytest.approx(0.75)


def test_uv_after_resize_with_pad_letterboxes_wide_image():
    """A 448x224 render into a 224x224 square gets vertical padding."""
    u = torch.tensor([[0.5]])
    v = torch.tensor([[0.0]])
    nu, nv = uv_after_resize_with_pad(u, v, (448, 224), (224, 224))
    assert nu.item() == pytest.approx(0.5)      # horizontal fills, centre stays
    assert nv.item() == pytest.approx(0.25)     # top edge pushed down by the bar


# --------------------------------------------------------------------------
# individual bridges
# --------------------------------------------------------------------------
def _slots():
    u = torch.rand(B, K)
    v = torch.rand(B, K)
    present = torch.tensor([[True, True, False], [True, False, False]])
    cid = torch.tensor([[1, 2, -1], [3, -1, -1]])
    energy = torch.rand(B, K)
    return u, v, present, cid, energy


def test_b1_masks_absent_slots():
    u, v, present, _, _ = _slots()
    b1 = PatchAnchoredSlot(_cfg())
    out = b1(torch.randn(B, N, W), u, v, present)
    assert out.shape == (B, K, W)
    assert torch.all(out[~present] == 0)
    assert torch.any(out[present] != 0)


def test_b1_output_depends_on_uv():
    """The whole point: move the sound, get a different visual anchor."""
    torch.manual_seed(0)
    b1 = PatchAnchoredSlot(_cfg())
    tokens = torch.randn(1, N, W)
    present = torch.tensor([[True]])
    a = b1(tokens, torch.tensor([[0.1]]), torch.tensor([[0.1]]), present)
    b = b1(tokens, torch.tensor([[0.9]]), torch.tensor([[0.9]]), present)
    assert not torch.allclose(a, b, atol=1e-4)


def test_b2_paints_energy_where_the_sound_is():
    torch.manual_seed(0)
    b2 = SoundMarkerOnPatches(_cfg())
    tokens = torch.zeros(1, N, W)
    u = torch.tensor([[0.5 / G]])          # top-left patch
    v = torch.tensor([[0.5 / G]])
    res = b2(tokens, u, v, torch.tensor([[True]]),
             class_id=torch.tensor([[1]]), energy=torch.ones(1, 1))
    mag = res[0].norm(dim=-1)              # [N]
    assert torch.argmax(mag).item() == 0   # top-left patch is the hottest
    assert mag[N - 1] < mag[0]             # bottom-right barely touched


def test_b2_absent_slot_contributes_nothing():
    b2 = SoundMarkerOnPatches(_cfg())
    tokens = torch.zeros(1, N, W)
    res = b2(tokens, torch.tensor([[0.5]]), torch.tensor([[0.5]]),
             torch.tensor([[False]]), class_id=torch.tensor([[-1]]),
             energy=torch.ones(1, 1))
    assert torch.all(res == 0)


def test_b3_bias_is_bounded_and_masked():
    cfg = _cfg(spatial_attention_bias=True, b3_gate_init=0.9)
    b3 = SpatialAttentionBias(cfg)
    u, v, present, _, _ = _slots()
    bias = b3(N, u, v, present)
    assert bias.shape == (B, K, N)
    assert bias.abs().max().item() <= cfg.b3_max_logit_bias + 1e-5
    assert torch.all(bias[~present] == 0)


def test_b4_shares_one_table_between_both_streams():
    """The slot code at a patch centre must equal that patch's own code."""
    b4 = SharedGridPositionalCode(_cfg(b4_gate_init=0.5))
    tokens = torch.zeros(B, N, W)
    row, col = 2, 3
    u = torch.full((B, 1), (col + 0.5) / G)
    v = torch.full((B, 1), (row + 0.5) / G)
    patch_res, slot_res = b4(tokens, u, v, torch.ones(B, 1, dtype=torch.bool))
    assert torch.allclose(slot_res[:, 0], patch_res[:, row * G + col], atol=1e-5)


# --------------------------------------------------------------------------
# composite + no-op guarantee
# --------------------------------------------------------------------------
def test_zero_gate_is_an_exact_no_op():
    """Critical: attaching to a trained checkpoint must change nothing."""
    cfg = _cfg(spatial_attention_bias=True, b1_gate_init=0.0, b2_gate_init=0.0,
               b3_gate_init=0.0, b4_gate_init=0.0)
    bridge = AudioVisualBridge(cfg)
    u, v, present, cid, energy = _slots()
    out = bridge(torch.randn(B, N, W), torch.randn(B, K, W), u, v, present,
                 class_id=cid, energy=energy)
    assert torch.all(out.patch_residual == 0)
    assert torch.all(out.slot_residual == 0)
    assert torch.all(out.attention_bias == 0)


def test_composite_shapes_and_gradients():
    cfg = _cfg(spatial_attention_bias=True)
    bridge = AudioVisualBridge(cfg)
    u, v, present, cid, energy = _slots()
    patches = torch.randn(B, N, W, requires_grad=True)
    slots = torch.randn(B, K, W, requires_grad=True)
    out = bridge(patches, slots, u, v, present, class_id=cid, energy=energy)
    assert out.patch_residual.shape == (B, N, W)
    assert out.slot_residual.shape == (B, K, W)
    assert out.attention_bias.shape == (B, K, N)
    (out.patch_residual.sum() + out.slot_residual.sum()).backward()
    assert patches.grad is not None
    for name, p in bridge.named_parameters():
        if p.requires_grad and "b3" not in name:
            assert p.grad is not None, f"no gradient reached {name}"


def test_composite_rejects_mismatched_widths():
    bridge = AudioVisualBridge(_cfg())
    u, v, present, cid, energy = _slots()
    with pytest.raises(ValueError, match="width"):
        bridge(torch.randn(B, N, W), torch.randn(B, K, W + 1), u, v, present,
               class_id=cid, energy=energy)


def test_gates_report_is_populated():
    bridge = AudioVisualBridge(_cfg(spatial_attention_bias=True))
    g = bridge.gates()
    assert set(g) >= {"b1", "b2", "b3", "b4", "b2_sigma"}
    assert all(math.isfinite(x) for x in g.values())


# --------------------------------------------------------------------------
# integration helpers
# --------------------------------------------------------------------------
def test_unpack_audio_slots_matches_pi0_layout():
    slots = torch.zeros(1, 2, 6)
    slots[0, 0] = torch.tensor([0.3, 0.7, 0.5, 0.9, 4.0, 1.0])
    slots[0, 1] = torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    f = unpack_audio_slots(slots)
    assert f["u"][0, 0].item() == pytest.approx(0.3)
    assert f["v"][0, 0].item() == pytest.approx(0.7)
    assert f["class_id"][0, 0].item() == 4
    assert f["present"].tolist() == [[True, False]]


def test_unpack_audio_slots_rejects_legacy_azel_layout():
    with pytest.raises(ValueError, match="slots_uv"):
        unpack_audio_slots(torch.zeros(1, 2, 4))


def test_scatter_attention_bias_lands_on_the_right_block():
    L = 40
    mask = torch.zeros(1, L, L)
    bias = torch.full((1, 2, N), 1.5)
    out = scatter_attention_bias(mask, bias, patch_start=0, slot_start=30)
    assert torch.all(out[0, 30:32, 0:N] == 1.5)
    assert out.sum().item() == pytest.approx(1.5 * 2 * N)


def test_scatter_attention_bias_refuses_boolean_masks():
    with pytest.raises(TypeError):
        scatter_attention_bias(torch.zeros(1, 40, 40, dtype=torch.bool),
                               torch.zeros(1, 2, N), patch_start=0, slot_start=30)
