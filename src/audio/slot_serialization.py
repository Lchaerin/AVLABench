"""M3 — audio-block serialization with ``<ASLOT>`` placeholders
(seld_vla_implementation_spec.md §5).

Builds the text prefix ``{audio_block} {instruction}`` where the audio block is
always present for structural consistency:

  events present : ``<audio> radio loud <ASLOT> ; radio quiet <ASLOT> </audio>``
  no events      : ``<audio> silence </audio>``           (no slots)
  sensor absent  : ``<audio> unavailable </audio>``       (modality dropout)

Design choices (from the user's scope decision "encoder scalar + text"):
  * The **class word** stays as a real vocabulary token so the LLM's prior
    knowledge links it to the instruction ("press the radio ...").
  * A coarse **loudness word** (loud/moderate/quiet) is emitted from energy so
    the text channel also carries loudness, complementing the continuous energy
    scalar fed to the SlotEncoder at the ``<ASLOT>`` position.
  * ``<ASLOT>`` is the only new token; its embedding is overwritten in-model by
    the SlotEncoder output. Events are sorted by **energy descending** and
    truncated to ``k_max``.

This module is pure-Python (no torch) so it runs identically in the openpi
numpy data pipeline and in the eval loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

SLOT_TOKEN = "<ASLOT>"
BLOCK_OPEN = "<audio>"
BLOCK_CLOSE = "</audio>"
SILENCE_TEXT = "silence"
UNAVAILABLE_TEXT = "unavailable"
SEP = " ; "


@dataclass
class SlotEvent:
    """One serialisable audio event."""
    class_name: str
    energy: float            # [0, 1]
    confidence: float = 1.0  # [0, 1]
    # (u, v) fed to the SlotEncoder; None means off-screen (dropped by default).
    uv: Optional[tuple[float, float]] = None


def loudness_word(energy: float) -> str:
    """Coarse loudness bin for the text channel."""
    e = float(energy)
    if e >= 0.66:
        return "loud"
    if e >= 0.33:
        return "moderate"
    return "quiet"


def _order_events(
    events: Sequence[SlotEvent],
    k_max: int,
    offscreen_policy: str,
) -> list[SlotEvent]:
    kept: list[SlotEvent] = []
    for ev in events:
        if ev.uv is None and offscreen_policy == "drop":
            continue
        kept.append(ev)
    kept.sort(key=lambda e: e.energy, reverse=True)
    return kept[:k_max]


def build_audio_block(
    events: Sequence[SlotEvent],
    k_max: int = 4,
    unavailable: bool = False,
    include_loudness: bool = True,
    include_slot_token: bool = True,
    offscreen_policy: str = "drop",
) -> tuple[str, int]:
    """Return ``(audio_block_text, n_slots)``.

    ``n_slots`` is the number of present events (== number of ``<ASLOT>``
    placeholders when ``include_slot_token``); the injection step asserts it
    equals the number of slot embeddings it has.

    ``include_slot_token`` controls whether the literal ``<ASLOT>`` marker is
    written into the text. Keep it True for the inline embedding-replacement
    mechanism (spec M5). Set it False for backbones that instead *append* a
    continuous slot token (pi0 slots_uv), where an un-replaced ``<ASLOT>`` would
    just tokenise to noise subwords.
    """
    if unavailable:
        return f"{BLOCK_OPEN} {UNAVAILABLE_TEXT} {BLOCK_CLOSE}", 0

    ordered = _order_events(events, k_max, offscreen_policy)
    if not ordered:
        return f"{BLOCK_OPEN} {SILENCE_TEXT} {BLOCK_CLOSE}", 0

    parts = []
    for ev in ordered:
        toks = [ev.class_name]
        if include_loudness:
            toks.append(loudness_word(ev.energy))
        if include_slot_token:
            toks.append(SLOT_TOKEN)
        parts.append(" ".join(toks))
    return f"{BLOCK_OPEN} {SEP.join(parts)} {BLOCK_CLOSE}", len(ordered)


def build_prompt(
    events: Sequence[SlotEvent],
    instruction: str,
    k_max: int = 4,
    unavailable: bool = False,
    include_loudness: bool = True,
    include_slot_token: bool = True,
    offscreen_policy: str = "drop",
) -> tuple[str, int]:
    """Compose ``{audio_block} {instruction}`` per spec §5. Returns
    ``(prompt, n_slots)``."""
    block, n = build_audio_block(
        events,
        k_max=k_max,
        unavailable=unavailable,
        include_loudness=include_loudness,
        include_slot_token=include_slot_token,
        offscreen_policy=offscreen_policy,
    )
    instruction = (instruction or "").strip()
    prompt = f"{block} {instruction}".strip() if instruction else block
    return prompt, n
