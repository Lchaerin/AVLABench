"""Unit tests for M4 SlotEncoder (seld_vla_implementation_spec.md §6)."""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio.slot_encoder import SlotEncoder, fourier_features


def test_fourier_shape():
    p = torch.rand(4, 3)
    f = fourier_features(p, L=6)
    assert f.shape == (4, 3, 12)  # 2L


def test_output_shape():
    enc = SlotEncoder(d_model=2048, L=6, hidden=256)
    B, K = 5, 3
    u = torch.rand(B, K)
    v = torch.rand(B, K)
    e = torch.rand(B, K)
    c = torch.rand(B, K)
    out = enc(u, v, e, c)
    assert out.shape == (B, K, 2048)


def test_gate_init():
    enc = SlotEncoder(d_model=64, gate_init=0.1)
    assert abs(enc.gate - 0.1) < 1e-5


def test_gate_scales_output():
    # With the gate forced to zero the slot embedding is exactly zero.
    enc = SlotEncoder(d_model=64, gate_init=0.1)
    with torch.no_grad():
        enc.a.fill_(0.0)  # tanh(0) = 0
    out = enc(torch.rand(2, 3), torch.rand(2, 3), torch.rand(2, 3), torch.rand(2, 3))
    assert torch.allclose(out, torch.zeros_like(out))


if __name__ == "__main__":
    test_fourier_shape()
    test_output_shape()
    test_gate_init()
    test_gate_scales_output()
    print("slot_encoder OK")
