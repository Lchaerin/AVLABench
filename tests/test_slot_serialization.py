"""Unit tests for M3 audio-block serialization (spec §5)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio.slot_serialization import (
    SLOT_TOKEN,
    SlotEvent,
    build_audio_block,
    build_prompt,
    loudness_word,
)


def test_silence_block_has_no_slots():
    block, n = build_audio_block([])
    assert n == 0
    assert "silence" in block
    assert SLOT_TOKEN not in block


def test_unavailable_block():
    block, n = build_audio_block([SlotEvent("radio", 0.9, uv=(0.5, 0.5))], unavailable=True)
    assert n == 0
    assert "unavailable" in block


def test_energy_descending_and_kmax():
    evs = [
        SlotEvent("radio", 0.2, uv=(0.4, 0.5)),
        SlotEvent("alarm", 0.9, uv=(0.6, 0.5)),
        SlotEvent("phone", 0.5, uv=(0.5, 0.5)),
    ]
    block, n = build_audio_block(evs, k_max=2)
    assert n == 2
    # alarm (0.9) must come before phone (0.5); radio (0.2) truncated.
    assert block.index("alarm") < block.index("phone")
    assert "radio" not in block
    assert block.count(SLOT_TOKEN) == 2


def test_slot_count_matches():
    evs = [SlotEvent("radio", 0.8, uv=(0.3, 0.5)), SlotEvent("radio", 0.7, uv=(0.7, 0.5))]
    _, n = build_audio_block(evs)
    assert n == 2


def test_offscreen_dropped():
    evs = [SlotEvent("radio", 0.8, uv=None), SlotEvent("alarm", 0.7, uv=(0.5, 0.5))]
    block, n = build_audio_block(evs, offscreen_policy="drop")
    assert n == 1
    assert "radio" not in block and "alarm" in block


def test_loudness_words():
    assert loudness_word(0.9) == "loud"
    assert loudness_word(0.5) == "moderate"
    assert loudness_word(0.1) == "quiet"


def test_build_prompt_order():
    evs = [SlotEvent("radio", 0.8, uv=(0.5, 0.5))]
    prompt, n = build_prompt(evs, "press the radio that is making sound")
    assert n == 1
    assert prompt.startswith("<audio>")
    assert prompt.endswith("making sound")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("slot_serialization OK")
