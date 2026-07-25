"""Build audio token sequences (Path A) for SmolVLA's language stream.

For each frame the dataset stores up to K SLED events as parallel arrays:
    azimuth_deg [B, K], elevation_deg [B, K], confidence [B, K], class_id [B, K]
    (class_id == -1 marks an empty slot.)

We turn these into a *fixed-length* token sequence so it can be batched cleanly:
    "[AUDIO] <CLASS> dir conf 0.NN ; <CLASS> dir conf 0.NN ; ... [/AUDIO]"

or, for VLM-first fusion:

    "[AUDIO] sound 1: <CLASS> is to the left at azimuth -54 degrees,
     confidence 0.NN. sound 2: <CLASS> is to the right ... [/AUDIO]"

The literal "<CLASS>" for empty slots is replaced with "silence" and confidence with 0.
The single token immediately after each class name is treated as the
*direction token* whose embedding is overwritten by DirectionEncoder(az, el).

This keeps the model architecture simple: the LLM sees a normal token sequence
plus a learned per-direction perturbation in a fixed slot.
"""
from __future__ import annotations

from typing import Tuple

import torch
import yaml


# ---------- Class id / name helpers ----------------------------------------
def load_class_names(taxonomy_path: str, n_classes: int = 38) -> list[str]:
    """Return list[str] of length n_classes; missing slots are 'unknown'."""
    with open(taxonomy_path) as f:
        data = yaml.safe_load(f)
    n_classes = int(data.get("n_classes", n_classes))
    classes = data.get("classes", {}) or {}
    out = ["unknown"] * n_classes
    for k, v in classes.items():
        idx = int(k)
        if 0 <= idx < n_classes:
            out[idx] = v["name"]
    return out


def _class_to_words(name: str) -> str:
    """`Footsteps_Walk_Run` -> `footsteps walk run` (vocab-friendly)."""
    return name.replace("_", " ").replace("-", " ").lower()


def _azimuth_words(azimuth_deg: float) -> str:
    """Map stored audio azimuth to coarse task-space language.

    Empirically, the select-radio task layout and eval metadata use positive
    stored azimuth for the left radio and negative stored azimuth for the right
    radio. The words below follow that task-space convention so the VLM text
    agrees with the button layout the policy must act on.
    """
    az = float(azimuth_deg)
    mag = abs(az)
    # The three-radio layout is visually/audio-spatially narrow: the side
    # radios are usually only about +/-12..18 degrees from the listener while
    # the middle radio is near 0.  A wider "straight ahead" bucket erases the
    # side labels for the one-radio task and leaves only the learned continuous
    # direction token to distinguish left/middle/right.
    if mag < 6:
        return "straight ahead"
    if mag < 25:
        side = "left" if az > 0 else "right"
        return f"slightly to the {side}"
    side = "left" if az > 0 else "right"
    return f"to the {side}"


# ---------- Token builder ---------------------------------------------------
class AudioTokenBuilder:
    """Builds a fixed-length token sequence per sample.

    Format (per slot k, K=top_k slots total):
        "<class_words> @ ; conf <conf> "
    where '@' is the *direction placeholder* — a single token whose embedding
    is later replaced by DirectionEncoder(az, el).
    """

    def __init__(
        self,
        tokenizer,
        class_names: list[str],
        top_k: int = 3,
        dir_placeholder: str = "@",
    ):
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.top_k = top_k
        self.dir_placeholder = dir_placeholder

        # ─── Pre-compute every token id that decodes to (some variant of)
        # the direction placeholder. BPE tokenisers like SmolLM2 produce
        # different ids for "@", " @", "@ ", etc. — we need to recognise
        # all of them in build_batch, otherwise the direction embedding
        # never lands on the right slot.
        self.dir_token_ids = self._collect_placeholder_token_ids(dir_placeholder)
        if len(self.dir_token_ids) == 0:
            raise RuntimeError(
                f"could not locate any token id for the direction placeholder "
                f"{dir_placeholder!r} in the tokenizer's vocab"
            )
        # Kept for backward-compatibility / debugging.
        self.dir_token_id = self.dir_token_ids[0]

        # Sentinel words used to pad empty audio slots.
        self.silence_word = "silence"

    # --- helpers ----------------------------------------------------------
    def _collect_placeholder_token_ids(self, ch: str) -> list[int]:
        """Return all token ids that decode to a string containing only `ch`.

        We probe a few common surface forms (bare, space-prefixed, both ends)
        and union the resulting single-token encodings. We then verify each
        candidate by re-decoding to make sure it's still purely the placeholder.
        """
        candidates = set()
        for surface in (ch, " " + ch, ch + " ", " " + ch + " "):
            try:
                ids = self.tokenizer.encode(surface, add_special_tokens=False)
            except Exception:
                continue
            for tid in ids:
                # Only single-token surface forms count — anything that
                # required ≥2 ids to encode wouldn't appear in a clean slot.
                decoded = self.tokenizer.decode([tid]).strip()
                if decoded == ch:
                    candidates.add(int(tid))
        return sorted(candidates)

    def _slot_text(self, class_name: str, conf: float) -> str:
        words = _class_to_words(class_name) if class_name else self.silence_word
        return f"{words} {self.dir_placeholder} conf {conf:.2f}"

    def _slot_text_natural(
        self,
        slot_index: int,
        class_name: str,
        azimuth_deg: float,
        confidence: float,
    ) -> str:
        if not class_name:
            return f"sound {slot_index + 1}: no sound detected."
        words = _class_to_words(class_name)
        direction = _azimuth_words(float(azimuth_deg))
        az_int = int(round(float(azimuth_deg)))
        return (
            f"sound {slot_index + 1}: {words} is {direction} {self.dir_placeholder}, "
            f"azimuth {az_int} degrees, confidence {confidence:.2f}."
        )

    # --- main API ---------------------------------------------------------
    @torch.no_grad()
    def build_batch(
        self,
        class_id: torch.Tensor,        # [B, K] long
        azimuth_deg: torch.Tensor,     # [B, K] float
        elevation_deg: torch.Tensor,   # [B, K] float
        confidence: torch.Tensor,      # [B, K] float
        max_length: int,
        natural_language: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (token_ids[B,L], attention_mask[B,L],
                   dir_slot_mask[B,L], slot_index[B,L]).

        dir_slot_mask is True at the K positions of `@`; slot_index gives
        the slot index k in [0, K) so the direction encoder output for slot k
        can be added at the correct position. Non-slot positions get slot_index = -1.
        """
        B, K = class_id.shape
        assert K == self.top_k, f"expected K={self.top_k}, got {K}"

        # Build raw text per sample
        texts = []
        for b in range(B):
            parts = ["[AUDIO]"]
            for k in range(K):
                cid = int(class_id[b, k].item())
                if cid < 0 or cid >= len(self.class_names):
                    if natural_language:
                        parts.append(self._slot_text_natural(k, "", 0.0, 0.0))
                    else:
                        parts.append(self._slot_text("", 0.0))
                else:
                    conf = float(confidence[b, k].item())
                    if natural_language:
                        parts.append(
                            self._slot_text_natural(
                                k,
                                self.class_names[cid],
                                float(azimuth_deg[b, k].item()),
                                conf,
                            )
                        )
                    else:
                        parts.append(self._slot_text(self.class_names[cid], conf))
                if (not natural_language) and k < K - 1:
                    parts.append(";")
            parts.append("[/AUDIO]")
            texts.append(" ".join(parts))

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"]               # [B, L]
        mask = enc["attention_mask"].bool()  # [B, L]

        # Locate the K direction-placeholder tokens in each row.
        dir_slot_mask = torch.zeros_like(ids, dtype=torch.bool)
        slot_index = torch.full_like(ids, fill_value=-1, dtype=torch.long)
        # Match any of the placeholder's surface-form ids ("@", " @", …).
        is_dir = torch.zeros_like(ids, dtype=torch.bool)
        for tid in self.dir_token_ids:
            is_dir |= ids.eq(tid)
        is_dir &= mask
        for b in range(B):
            positions = is_dir[b].nonzero(as_tuple=False).flatten()
            # Take only the first top_k matches (in case the placeholder also
            # appears inside class names — extremely unlikely but defensive).
            positions = positions[: self.top_k]
            for k, pos in enumerate(positions.tolist()):
                dir_slot_mask[b, pos] = True
                slot_index[b, pos] = k

        return ids, mask, dir_slot_mask, slot_index
