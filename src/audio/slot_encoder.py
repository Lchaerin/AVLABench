"""M4 — SlotEncoder (seld_vla_implementation_spec.md §6).

Encodes an audio event's *grounded* position and loudness into a single
continuous token embedding that replaces the ``<ASLOT>`` placeholder next to the
event's class word.

Input features per slot:
  * fourier(u, L) ⊕ fourier(v, L)   — positional encoding of the projected
    image coordinate (u, v) ∈ [0, 1]², 2L dims each.
  * [u, v, energy, confidence]       — raw scalars (energy = loudness in [0,1]).

  in_dim = 2*(2L) + 4 = 4L + 4   (L=6 → 28)

The MLP output passes through LayerNorm and a tanh gate initialised near zero
(Flamingo-style) so the untrained slot embedding barely perturbs the pretrained
attention, then ramps up as the gate learns.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def fourier_features(p: torch.Tensor, L: int) -> torch.Tensor:
    """Fourier features of a scalar in [0,1]. Returns 2L features on the last dim.

    ``concat_k [sin(2^k · π · p), cos(2^k · π · p)]`` for k in 0..L-1.
    """
    freqs = (2.0 ** torch.arange(L, device=p.device, dtype=p.dtype)) * math.pi
    ang = p.unsqueeze(-1) * freqs  # [..., L]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [..., 2L]


class SlotEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        L: int = 6,
        hidden: int = 256,
        gate_init: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.L = int(L)
        in_dim = 4 * self.L + 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
        # gate = tanh(a); a = atanh(gate_init) so gate starts at gate_init.
        self.a = nn.Parameter(torch.tensor(math.atanh(float(gate_init))))

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        energy: torch.Tensor,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        """Encode slots. Inputs broadcast to a common shape [...]; returns
        embeddings of shape [..., d_model]."""
        u = u.float()
        v = v.float()
        energy = energy.float()
        conf = conf.float()
        feats = torch.cat(
            [
                fourier_features(u, self.L),
                fourier_features(v, self.L),
                torch.stack([u, v, energy, conf], dim=-1),
            ],
            dim=-1,
        )
        out = self.ln(self.mlp(feats))
        return torch.tanh(self.a) * out

    @property
    def gate(self) -> float:
        return float(torch.tanh(self.a).item())
