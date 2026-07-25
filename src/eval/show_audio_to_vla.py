"""Print one concrete example of how a SLED output flows into the VLA prefix.

Usage:
    python src/eval/show_audio_to_vla.py [--ckpt path/to/ckpt_step*.pt]

Steps printed (with actual values):
    1) Hypothetical SLED output  →  per-class top-K events
    2) AudioTokenBuilder text    →  the literal string the LLM sees
    3) Tokenizer output          →  token IDs + decoded tokens (one per line)
    4) Direction encoder         →  hidden vector at each `@` slot
    5) Final audio embedding     →  (L, hidden) tensor with stats
    6) Where it sits in the prefix: [image] + [language] + [AUDIO] + [state]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio.audio_token_builder import AudioTokenBuilder, load_class_names  # noqa: E402
from src.audio.direction_encoder import DirectionEncoder                        # noqa: E402
from src.models.smolvla_audio import (                                          # noqa: E402
    AudioAwareSmolVLAPolicy, AudioConfig,
)


def _print_block(title: str, body: str) -> None:
    bar = "─" * (len(title) + 2)
    print(f"\n┌{bar}┐\n│ {title} │\n└{bar}┘")
    print(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="optional ckpt; if omitted, uses base lerobot/smolvla_vlabench")
    ap.add_argument("--pretrained", default="lerobot/smolvla_vlabench")
    ap.add_argument("--taxonomy", default=str(REPO_ROOT / "class_taxonomy.yaml"))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--audio-max-len", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ──────────────────────────────────────────────────────────────────────
    # Hypothetical SLED output (per-class top-K events for one timestep).
    # Values picked to mimic a real select_radio episode where SLED detected
    # a strong "Wind_Brass" source roughly straight ahead, plus one weaker
    # secondary "Speech" detection slightly right.
    # ──────────────────────────────────────────────────────────────────────
    sled_demo = {
        "class_id":      np.array([11,  0, -1], dtype=np.int64),  # 11=Wind_Brass, 0=Speech, -1=empty
        "azimuth_deg":   np.array([1.5, 25.0, 0.0], dtype=np.float32),
        "elevation_deg": np.array([-12.0, 5.0, 0.0], dtype=np.float32),
        "confidence":    np.array([0.92, 0.31, 0.0], dtype=np.float32),
    }
    classes = load_class_names(args.taxonomy)
    pretty = [
        (int(sled_demo["class_id"][k]),
         classes[int(sled_demo["class_id"][k])] if int(sled_demo["class_id"][k]) >= 0 else "—",
         float(sled_demo["azimuth_deg"][k]),
         float(sled_demo["elevation_deg"][k]),
         float(sled_demo["confidence"][k]))
        for k in range(args.top_k)
    ]
    _print_block("1) SLED top-K snapshot fed to the policy at this step",
        "\n".join(
            f"  slot {k}: cid={cid:>2}  class={name:<24}  az={az:+6.1f}°  el={el:+6.1f}°  conf={c:.2f}"
            for k, (cid, name, az, el, c) in enumerate(pretty)
        ))

    # ──────────────────────────────────────────────────────────────────────
    # Build the policy so we have the *real* tokenizer + direction encoder
    # ──────────────────────────────────────────────────────────────────────
    print("\n[load] building policy …")
    audio_cfg = AudioConfig(
        taxonomy_path=args.taxonomy,
        top_k=args.top_k,
        audio_max_len=args.audio_max_len,
    )
    policy = AudioAwareSmolVLAPolicy.from_pretrained_with_audio(
        args.pretrained, audio_config=audio_cfg
    )
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[load] fine-tuned weights from {args.ckpt}")
    policy.to(args.device).eval()

    builder: AudioTokenBuilder = policy.model.audio_token_builder
    dir_enc:  DirectionEncoder = policy.model.direction_encoder
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    hidden_size = policy.model.vlm_with_expert.config.text_config.hidden_size

    # ──────────────────────────────────────────────────────────────────────
    # 2) The literal text the LLM sees (before tokenisation)
    # ──────────────────────────────────────────────────────────────────────
    parts = ["[AUDIO]"]
    for k in range(args.top_k):
        cid = int(sled_demo["class_id"][k])
        if cid < 0:
            text = builder._slot_text("", 0.0)
        else:
            text = builder._slot_text(classes[cid], float(sled_demo["confidence"][k]))
        parts.append(text)
        if k < args.top_k - 1:
            parts.append(";")
    parts.append("[/AUDIO]")
    audio_text = " ".join(parts)
    _print_block("2) Audio block as plain text (what the LLM 'reads')",
                 f"  {audio_text}")

    # ──────────────────────────────────────────────────────────────────────
    # 3) Tokenisation: token IDs + decoded pieces, with `@` slots marked
    # ──────────────────────────────────────────────────────────────────────
    cid_b = torch.from_numpy(sled_demo["class_id"]).long().unsqueeze(0)
    az_b  = torch.from_numpy(sled_demo["azimuth_deg"]).float().unsqueeze(0)
    el_b  = torch.from_numpy(sled_demo["elevation_deg"]).float().unsqueeze(0)
    cf_b  = torch.from_numpy(sled_demo["confidence"]).float().unsqueeze(0)
    ids, mask, dir_slot_mask, slot_idx = builder.build_batch(
        cid_b, az_b, el_b, cf_b, max_length=args.audio_max_len
    )
    print()
    _print_block("3) Token IDs (kept positions only) — `@` rows = direction slots",
        "  " + "  ".join(f"id={int(t):>5d}" for t in ids[0][mask[0]][:18]) + " ...")
    decoded = tokenizer.convert_ids_to_tokens(ids[0][mask[0]].tolist())
    rows = []
    pos_in_kept = 0
    for i, tid in enumerate(ids[0].tolist()):
        if not bool(mask[0, i].item()):
            continue
        marker = "← DIR slot " if bool(dir_slot_mask[0, i].item()) else ""
        if marker:
            marker += f"k={int(slot_idx[0, i].item())}"
        rows.append(f"  [pos {pos_in_kept:>2}]  id={tid:>5d}  tok={decoded[pos_in_kept]!r:<14} {marker}")
        pos_in_kept += 1
    print("\n".join(rows))

    # ──────────────────────────────────────────────────────────────────────
    # 4) Direction encoder output at each `@` slot
    # ──────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        dir_emb = dir_enc(az_b.to(args.device), el_b.to(args.device))   # [1, K, H]
    dir_emb_cpu = dir_emb[0].cpu()
    print()
    rows = []
    for k in range(args.top_k):
        v = dir_emb_cpu[k]
        rows.append(
            f"  slot {k}: ‖v‖₂={v.norm():.4f}   v[:6]={v[:6].tolist()}"
        )
    _print_block(
        f"4) Direction encoder output (sin/cos-MLP, hidden={hidden_size})",
        "\n".join(rows))

    # ──────────────────────────────────────────────────────────────────────
    # 5) Full audio embedding block (text-emb + dir-emb at slot positions)
    # ──────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        audio_emb, audio_mask = policy.model._embed_audio_block(
            cid_b, az_b, el_b, cf_b
        )
    audio_emb = audio_emb[0].detach().cpu()         # [L, H]
    audio_mask = audio_mask[0].detach().cpu()       # [L]
    L_kept = int(audio_mask.sum().item())
    # Re-run the text-only embedding to compare and isolate the dir contribution
    import math
    text_only_ids = ids.to(args.device)
    with torch.no_grad():
        text_only_emb = policy.model.vlm_with_expert.embed_language_tokens(text_only_ids)
    text_only_emb = (text_only_emb * math.sqrt(text_only_emb.shape[-1])).detach().cpu()
    delta_at_slots = []
    for k in range(args.top_k):
        positions = torch.where(slot_idx[0].cpu() == k)[0]
        if positions.numel() == 0:
            continue
        pos = int(positions[0].item())
        delta = (audio_emb[pos] - text_only_emb[0, pos]).float().norm().item()
        delta_at_slots.append((pos, k, delta))
    rows = [
        f"  audio_emb shape: {tuple(audio_emb.shape)}, kept length L={L_kept}",
        f"  ‖audio_emb‖_F = {audio_emb.norm().item():.2f}   (over kept positions)",
        "  per-slot ‖dir contribution‖ (audio_emb - text_only_emb at slot):",
    ]
    for pos, k, d in delta_at_slots:
        rows.append(f"      slot k={k} at pos={pos}:  Δ = {d:.4f}")
    _print_block("5) Final audio embedding block", "\n".join(rows))

    # ──────────────────────────────────────────────────────────────────────
    # 6) Where this lives in the SmolVLA prefix
    # ──────────────────────────────────────────────────────────────────────
    explanation = """
  prefix layout (each row = one section concatenated along sequence axis):
      [ image embs   ]  ←  3 cameras × 64 SigLIP tokens = 192 vectors (image/second/wrist)
      [ language emb ]  ←  tokenizer("primitive: Press the button … \\n")
      [ AUDIO embs   ]  ←  this is the block we just built (≈ {L} vectors)
      [ state emb    ]  ←  state_proj([x, y, z, rx, ry, rz, gripper])

  Then the action expert cross-attends to the cached prefix and predicts a
  50-step action chunk via flow matching."""
    _print_block("6) Where the audio block sits in the SmolVLA prefix",
                 explanation.format(L=L_kept))

    # ──────────────────────────────────────────────────────────────────────
    # 7) Sync / latency commentary
    # ──────────────────────────────────────────────────────────────────────
    sled_cls_name = type(policy).__name__
    msg = """
  Audio sample-rate    : 44 100 Hz (sounddevice OutputStream callback)
  SLED window (v5)     : 11 520 samples @ 24 kHz ≈ 480 ms (~21 168 native samples)
  SLED inference rate  : 5 Hz  (1/infer_hz = 200 ms cycle)
  VLA inference rate   : ~10–20 Hz on RTX 5090 (50–100 ms / chunk)
  Sync model           : decoupled / asynchronous.
                         • the audio engine fills its recording buffer at 44.1 kHz
                         • SLED's daemon thread reads the *last 480 ms* of that
                           buffer every 200 ms and updates a shared snapshot
                         • VLA grabs whichever snapshot is current at the
                           instant it builds its batch (μs cost — just a lock+copy)
                         • when SLED hasn't produced anything yet (first ~1 s) or
                           rejected the most recent run, VLA sees the empty snapshot
                           (`class_id=-1` for every slot) — same pattern that
                           appeared in training, so the network has been exposed
                           to it via the audio dropout / silence path."""
    _print_block(
        f"7) How VLA & SLED stay in sync (real-time mode, {sled_cls_name})",
        msg)


if __name__ == "__main__":
    main()
