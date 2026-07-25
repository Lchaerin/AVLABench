"""Plain-text audio prompt builder for text-conditioned VLA backbones.

This path is used by pi0/pi0.5-style openpi policies where the practical
extension point is the language prompt rather than an internal token embedding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.audio.audio_token_builder import _azimuth_words, _class_to_words, load_class_names


def _to_np_1d(x) -> np.ndarray:
    """Flatten any array-like (torch tensor on cpu/gpu, numpy, list) to a 1-D
    numpy array. Lets the same prompt builder run in the torch eval loop and in
    openpi's numpy data pipeline so train/eval prompts are byte-identical."""
    if hasattr(x, "detach"):          # torch.Tensor (possibly on GPU)
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


@dataclass
class AudioPromptConfig:
    taxonomy_path: str
    top_k: int = 3
    n_classes: int = 38
    include_confidence: bool = True
    include_elevation: bool = True
    include_empty_slots: bool = False


class AudioPromptBuilder:
    """Serialize top-K SELD outputs into compact natural language."""

    def __init__(self, config: AudioPromptConfig):
        self.config = config
        self.class_names = load_class_names(config.taxonomy_path, n_classes=config.n_classes)

    def build_one(
        self,
        class_id: torch.Tensor,
        azimuth_deg: torch.Tensor,
        elevation_deg: torch.Tensor,
        confidence: torch.Tensor,
    ) -> str:
        cid = _to_np_1d(class_id).astype(np.int64)
        az = _to_np_1d(azimuth_deg).astype(np.float64)
        el = _to_np_1d(elevation_deg).astype(np.float64)
        conf = _to_np_1d(confidence).astype(np.float64)

        present = []
        limit = min(self.config.top_k, cid.size, az.size, el.size, conf.size)
        for k in range(limit):
            cls_id = int(cid[k])
            cf = float(conf[k])
            if cls_id < 0 or cls_id >= len(self.class_names) or cf <= 0:
                if self.config.include_empty_slots:
                    present.append(f"source {k + 1}: no sound detected")
                continue

            cls_name = _class_to_words(self.class_names[cls_id])
            az_val = float(az[k])
            el_val = float(el[k])
            parts = [
                f"source {k + 1}: {cls_name}",
                f"{_azimuth_words(az_val)}",
                f"azimuth {int(round(az_val))} degrees",
            ]
            if self.config.include_elevation:
                parts.append(f"elevation {int(round(el_val))} degrees")
            if self.config.include_confidence:
                parts.append(f"confidence {cf:.2f}")
            present.append(", ".join(parts))

        count = sum(1 for k in range(limit) if int(cid[k]) >= 0 and float(conf[k]) > 0)
        if not present:
            return "Audio scene: no sound source detected."
        noun = "source" if count == 1 else "sources"
        return f"Audio scene: {count} sound {noun} detected. " + "; ".join(present) + "."

    def build_batch(
        self,
        class_id: torch.Tensor,
        azimuth_deg: torch.Tensor,
        elevation_deg: torch.Tensor,
        confidence: torch.Tensor,
    ) -> list[str]:
        return [
            self.build_one(class_id[b], azimuth_deg[b], elevation_deg[b], confidence[b])
            for b in range(class_id.shape[0])
        ]

