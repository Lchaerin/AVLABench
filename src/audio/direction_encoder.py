"""Encoders for (azimuth, elevation) direction features."""
from __future__ import annotations

import math

import torch
from torch import nn


class DirectionEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int,
        internal_dim: int = 128,
        dropout: float = 0.0,
        encoder_type: str = "mlp",
    ):
        super().__init__()
        if encoder_type not in {"mlp", "fixed_fourier"}:
            raise ValueError(
                "encoder_type must be 'mlp' or 'fixed_fourier', "
                f"got {encoder_type!r}"
            )
        self.out_dim = int(out_dim)
        self.encoder_type = encoder_type

        if encoder_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(4, internal_dim),
                nn.GELU(),
                nn.Linear(internal_dim, internal_dim),
                nn.GELU(),
                nn.Linear(internal_dim, out_dim),
            )
            nn.init.normal_(self.mlp[-1].weight, std=0.01)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            n_freq = max(1, math.ceil(out_dim / 4))
            # Log-spaced frequencies give a transformer-style deterministic
            # Fourier basis without the extreme magnitudes of raw 2**i.
            freq = torch.logspace(0.0, math.log10(64.0), n_freq)
            self.register_buffer("frequencies", freq, persistent=False)

        # Post-MLP dropout. Has no state, so state_dict shape is unchanged
        # whether dropout is enabled or not — existing checkpoints load fine.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, az_deg: torch.Tensor, el_deg: torch.Tensor) -> torch.Tensor:
        az = torch.deg2rad(az_deg)
        el = torch.deg2rad(el_deg)
        if self.encoder_type == "fixed_fourier":
            freq = self.frequencies.to(device=az.device, dtype=az.dtype)
            azf = az.unsqueeze(-1) * freq
            elf = el.unsqueeze(-1) * freq
            feats = torch.cat(
                [torch.sin(azf), torch.cos(azf), torch.sin(elf), torch.cos(elf)],
                dim=-1,
            )
            feats = feats[..., : self.out_dim]
            return self.dropout(feats)

        feats = torch.stack(
            [torch.sin(az), torch.cos(az), torch.sin(el), torch.cos(el)], dim=-1
        )
        return self.dropout(self.mlp(feats))
