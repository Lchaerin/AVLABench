"""Run one episode and dump every text the VLA actually sees, step by step.

Output (single markdown-ish text file):

    outputs/vla_inputs/episode_<N>.txt

Each VLA step records:
  • step index, wall-clock dt since previous step
  • the constant per-episode instruction (shown once at top + reminded each step)
  • the SLED top-K snapshot at this step (cid / az / el / conf)
  • the audio block text built from that snapshot
  • the *combined* text that ends up in the language stream of the model
    (instruction + audio block, in the order they're concatenated)
  • the tokenised form (ids + decoded pieces, with `@` direction slots marked)

Usage
-----
    python src/eval/dump_vla_inputs.py --episode-idx 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

from VLABench.envs import load_env                              # noqa: E402
from VLABench.tasks import *                                    # noqa: E402, F401, F403
from VLABench.robots import *                                   # noqa: E402, F401, F403

from src.models.smolvla_audio import (                          # noqa: E402
    AudioAwareSmolVLAPolicy, AudioConfig,
)
from src.audio.audio_token_builder import load_class_names      # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from trajectory_generation import (                             # noqa: E402
    _load_class_taxonomy, _pick_random_sound,
)
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
from eval_smolvla_audio import (                                # noqa: E402
    _build_episode_audio_cfg, _live_snapshot, _empty_audio,
    _build_policy_batch, _apply_action,
    DEFAULT_INSTRUCTION,
)


def _build_audio_text(builder, sled_snap: dict, classes: list[str], top_k: int) -> str:
    """Reproduce AudioTokenBuilder's text exactly (no tokenisation)."""
    parts = ["[AUDIO]"]
    for k in range(top_k):
        cid = int(sled_snap["class_id"][k])
        if cid < 0 or cid >= len(classes):
            text = builder._slot_text("", 0.0)
        else:
            text = builder._slot_text(classes[cid], float(sled_snap["confidence"][k]))
        parts.append(text)
        if k < top_k - 1:
            parts.append(";")
    parts.append("[/AUDIO]")
    return " ".join(parts)


def _format_snapshot(snap: dict, classes: list[str]) -> str:
    rows = []
    for k in range(len(snap["class_id"])):
        cid = int(snap["class_id"][k])
        name = classes[cid] if cid >= 0 else "—"
        rows.append(
            f"    slot {k}: cid={cid:>3}  class={name:<24}  "
            f"az={float(snap['azimuth_deg'][k]):+7.2f}°  "
            f"el={float(snap['elevation_deg'][k]):+7.2f}°  "
            f"conf={float(snap['confidence'][k]):.3f}"
        )
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pretrained", default="lerobot/smolvla_vlabench")
    ap.add_argument("--taxonomy", default=str(REPO_ROOT / "class_taxonomy.yaml"))
    ap.add_argument("--audio-config",
                    default=str(REPO_ROOT / "audio_generation" / "scene_audio_config.json"))
    ap.add_argument("--sled-ckpt", default=
        "/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt")
    ap.add_argument("--task-name", default="select_radio")
    ap.add_argument("--robot", default="franka")
    ap.add_argument("--episode-idx", type=int, default=0,
                    help="seed offset → which episode to record")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--max-substeps", type=int, default=4)
    ap.add_argument("--warmup-seconds", type=float, default=2.0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--out", default=None,
                    help="output path; default outputs/vla_inputs/episode_<N>.txt")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "outputs" / "vla_inputs" / f"episode_{args.episode_idx:03d}.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # ── 1. Build policy + load fine-tuned weights ──────────────────────────
    print(f"[load] base = {args.pretrained}")
    audio_cfg = AudioConfig(taxonomy_path=args.taxonomy, top_k=args.top_k)
    policy = AudioAwareSmolVLAPolicy.from_pretrained_with_audio(
        args.pretrained, audio_config=audio_cfg)

    from lerobot.configs.types import FeatureType, PolicyFeature
    policy.config.input_features = {
        "observation.images.image":        PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.second_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.wrist_image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state":               PolicyFeature(type=FeatureType.STATE,  shape=(7,)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    print(f"[load] ckpt = {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"], strict=True)
    policy.to(device).eval()

    builder   = policy.model.audio_token_builder
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    classes   = load_class_names(args.taxonomy)

    # ── 2. SLED + env setup, mirroring eval_smolvla_audio ──────────────────
    from audio_generation.audio_sim_manager import AudioSimManager
    from audio_generation.sled_overlay import SLEDOverlay

    sled_overlay = SLEDOverlay(
        ckpt_path=args.sled_ckpt, torch_device=str(device), realtime=True,
    )

    audio_classes, audio_label_map = _load_class_taxonomy()

    np.random.seed(args.seed + args.episode_idx)
    import random as _rd
    _rd.seed(args.seed + args.episode_idx)

    env = load_env(args.task_name, robot=args.robot, eval=False, run_mode="eval")
    env.reset()

    cm = env.task.config_manager
    active_idx = getattr(cm, "active_radio_idx", None)
    active_radio = f"radio_{active_idx}" if active_idx is not None else "?"
    position_label = getattr(cm, "active_position_label", "?")
    sound_meta = _pick_random_sound(audio_label_map, audio_classes)

    cfg_dict = _build_episode_audio_cfg(args.audio_config, active_radio, sound_meta)
    audio_mgr = AudioSimManager(
        config_path=args.audio_config,
        task_name=args.task_name,
        config_dict=cfg_dict,
    )
    audio_mgr.attach_to_env(env)
    audio_mgr.start()
    sled_overlay.set_audio_engine(audio_mgr.engine)
    env.step()
    time.sleep(args.warmup_seconds)

    robot_pos = np.asarray(env.get_robot_frame_position(), dtype=np.float32)

    # ── 3. Run episode + dump text ─────────────────────────────────────────
    instruction_with_newline = args.instruction if args.instruction.endswith("\n") \
                              else args.instruction + "\n"

    with open(out_path, "w") as f:
        # Header
        f.write("═══════════════════════════════════════════════════════════════\n")
        f.write(f" VLA text-input dump — Episode {args.episode_idx}\n")
        f.write("═══════════════════════════════════════════════════════════════\n\n")
        f.write(f"  ckpt              : {args.ckpt}\n")
        f.write(f"  task              : {args.task_name}\n")
        f.write(f"  active_radio      : {active_radio}  (position={position_label})\n")
        if sound_meta is not None:
            f.write(f"  source sound      : {sound_meta['folder']}/"
                    f"{Path(sound_meta['sound_file']).name}\n")
            f.write(f"  GT class_id/name  : {sound_meta['class_id']} → "
                    f"{sound_meta['class_name']}\n")
        f.write(f"  policy chunk_size : {policy.config.chunk_size}\n")
        f.write(f"  horizon (steps/chunk): {args.horizon}\n")
        f.write(f"  top_k (audio slots): {args.top_k}\n")
        f.write(f"  audio_max_len      : {policy.model.audio_config.audio_max_len} tokens\n")
        f.write(f"  tokenizer_max_len  : {policy.config.tokenizer_max_length} tokens\n\n")

        # Instruction (constant for the whole episode)
        f.write("───────────────────────────────────────────────────────────────\n")
        f.write(" CONSTANT INSTRUCTION (every step uses this verbatim)\n")
        f.write("───────────────────────────────────────────────────────────────\n\n")
        f.write(f"  raw text  : {instruction_with_newline!r}\n")
        instr_enc = tokenizer(
            [instruction_with_newline],
            padding="max_length", truncation=True,
            max_length=policy.config.tokenizer_max_length,
            return_tensors="pt",
        )
        instr_ids = instr_enc["input_ids"][0]
        instr_mask = instr_enc["attention_mask"][0].bool()
        kept_ids = instr_ids[instr_mask]
        kept_toks = tokenizer.convert_ids_to_tokens(kept_ids.tolist())
        f.write(f"  token IDs : {kept_ids.tolist()}\n")
        f.write(f"  pieces    : {kept_toks}\n\n")

        # Per-step audio
        f.write("───────────────────────────────────────────────────────────────\n")
        f.write(" PER-STEP AUDIO BLOCKS (these change each step)\n")
        f.write("───────────────────────────────────────────────────────────────\n")

        success = False
        step = 0
        chunk_count = 0
        while step < args.max_steps:
            t0 = time.time()
            audio_snap = _live_snapshot(sled_overlay, args.top_k)

            f.write(f"\n══════════════ STEP {step:3d}  (chunk #{chunk_count}) ══════════════\n")
            f.write(f"\n  SLED snapshot (top-{args.top_k}):\n")
            f.write(_format_snapshot(audio_snap, classes) + "\n")

            audio_text = _build_audio_text(builder, audio_snap, classes, args.top_k)
            f.write(f"\n  Audio block text (what AudioTokenBuilder emits):\n")
            f.write(f"    {audio_text}\n")

            # Tokenise the audio block as the model sees it (max_length padding,
            # then mark direction `@` slots).
            cid_b = torch.from_numpy(audio_snap["class_id"]).long().unsqueeze(0)
            az_b  = torch.from_numpy(audio_snap["azimuth_deg"]).float().unsqueeze(0)
            el_b  = torch.from_numpy(audio_snap["elevation_deg"]).float().unsqueeze(0)
            cf_b  = torch.from_numpy(audio_snap["confidence"]).float().unsqueeze(0)
            ids, mask, dir_slot_mask, slot_idx = builder.build_batch(
                cid_b, az_b, el_b, cf_b,
                max_length=policy.model.audio_config.audio_max_len,
            )
            kept = mask[0].bool()
            audio_ids = ids[0][kept].tolist()
            audio_pieces = tokenizer.convert_ids_to_tokens(audio_ids)
            f.write(f"\n  Tokenised audio ({len(audio_ids)} tokens, ★ = direction slot):\n")
            kept_dir = dir_slot_mask[0][kept]
            kept_slot = slot_idx[0][kept]
            for i, (tid, tok) in enumerate(zip(audio_ids, audio_pieces)):
                marker = f" ★ slot k={int(kept_slot[i])}" if bool(kept_dir[i]) else ""
                f.write(f"    [pos {i:>2}]  id={tid:>5d}  tok={tok!r}{marker}\n")

            # Combined text (instruction first, then audio — same order they're
            # concatenated in the prefix embedding sequence).
            f.write(f"\n  Combined LLM-stream text (instruction → audio):\n")
            combined = instruction_with_newline.rstrip("\n") + "\n  " + audio_text
            f.write(f"    {combined}\n")

            # Now actually take the policy step (so the next snapshot evolves)
            batch = _build_policy_batch(
                env, audio_snap, args.instruction, tokenizer, device,
                policy.config.tokenizer_max_length,
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                with torch.no_grad():
                    actions = policy.predict_action_chunk(batch)
            actions = actions[0].float().cpu().numpy()
            chunk_count += 1
            took = time.time() - t0
            f.write(f"\n  [VLA timing] {took*1000:.1f} ms for predict_action_chunk\n")

            for h in range(min(args.horizon, actions.shape[0])):
                if step >= args.max_steps:
                    break
                finished = _apply_action(
                    env, actions[h], robot_pos,
                    max_substeps=args.max_substeps, tolerance=1e-2,
                )
                step += 1
                if finished:
                    success = True
                    break
            if success:
                break

        f.write(f"\n═══════════════════════════════════════════════════════════════\n")
        f.write(f" episode finished: success={success}  steps={step}  chunks={chunk_count}\n")
        f.write(f"═══════════════════════════════════════════════════════════════\n")

    # cleanup
    sled_overlay.set_audio_engine(None)
    audio_mgr.detach_from_env(env)
    audio_mgr.engine.stop()
    env.close()

    print(f"\n[done] wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"       success={success}  steps={step}  chunks={chunk_count}")


if __name__ == "__main__":
    main()
