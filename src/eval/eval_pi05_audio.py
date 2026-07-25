"""Evaluate audio-conditioned pi0.5/openpi on AVLABench radio tasks.

By default this loads pi0.5 locally through openpi. A websocket server mode is
also available for dependency-isolated evaluation. Audio class, source count,
confidence, and direction are appended to the pi0.5 prompt as plain text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

from src.audio.oracle_sled import OracleNoiseConfig, TopKClassSmoother  # noqa: E402
from src.models.pi05_audio import AudioAwarePi05Policy, Pi05AudioConfig  # noqa: E402
from dm_control.rl.control import PhysicsError  # noqa: E402

DEFAULT_INSTRUCTION = (
    "primitive: Press the button in front of the radio that is making sound."
)
DEFAULT_ONE_RADIO_INSTRUCTION = DEFAULT_INSTRUCTION
DEFAULT_TWO_RADIO_INSTRUCTION_TEMPLATE = (
    "primitive: Press the button in front of the radio playing the {class_name} sound."
)
DEFAULT_SILENT_RADIO_INSTRUCTION = (
    "primitive: Press the button in front of the radio that is not making sound."
)
REVISED_ONE_RADIO_INSTRUCTION = (
    "primitive: Tap the button located before the radio that is producing audio."
)
REVISED_TWO_RADIO_INSTRUCTION_TEMPLATE = (
    "primitive: Tap the button positioned before the radio that is emitting "
    "the {class_name} sound."
)
REVISED_SILENT_RADIO_INSTRUCTION = (
    "primitive: Tap the button placed before the radio that is not making any sound."
)


def _eval_instruction_metadata(args) -> dict:
    meta = {
        "arg_instruction": args.instruction,
        "revised_instruction": args.revised_instruction,
        "one_radio_instruction": args.one_radio_instruction,
        "two_radio_instruction_template": args.two_radio_instruction_template,
        "silent_radio_instruction": args.silent_radio_instruction,
        "effective_instruction_source": "arg_instruction",
        "effective_instruction_static": args.instruction,
    }
    if args.task_name == "select_radio":
        meta.update({
            "effective_instruction_source": "task_fixed",
            "effective_instruction_static": args.one_radio_instruction,
        })
    elif args.task_name == "select_radio_silent":
        meta.update({
            "effective_instruction_source": "task_fixed",
            "effective_instruction_static": args.silent_radio_instruction,
        })
    elif args.task_name == "select_radio_two":
        meta.update({
            "effective_instruction_source": "per_episode_target_sound_class",
            "effective_instruction_template": args.two_radio_instruction_template,
            "effective_instruction_static": None,
        })
    return meta


class _DummyTokenizer:
    """Shape-compatible tokenizer for the shared eval loop.

    pi0.5 consumes `raw_instruction`; token ids are only produced so the reused
    `_build_policy_batch` function can populate its standard keys.
    """

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        batch = len(texts)
        return {
            "input_ids": torch.zeros(batch, max_length, dtype=torch.long),
            "attention_mask": torch.ones(batch, max_length, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", dest="local", action="store_true", default=True,
                    help="Load openpi/pi0.5 in this process. This is the default.")
    ap.add_argument("--server", dest="local", action="store_false",
                    help="Use websocket server mode instead of local loading.")
    ap.add_argument("--policy-config", default="pi05_vlabench_primitive_lora",
                    help="openpi TrainConfig name for local loading.")
    ap.add_argument("--policy-dir", default=None,
                    help="openpi checkpoint dir for local loading. Required with --local.")
    ap.add_argument("--openpi-root", default=str(REPO_ROOT / "third_party" / "openpi"),
                    help="Path to an openpi checkout. Its src/ is added to sys.path for local loading.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--taxonomy", default=str(REPO_ROOT / "class_taxonomy.yaml"))
    ap.add_argument("--audio-config", default=str(
        REPO_ROOT / "audio_generation" / "scene_audio_config.json"))
    ap.add_argument("--sled-ckpt", default=
        "/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt")
    ap.add_argument("--task-name", default="select_radio")
    ap.add_argument("--robot", default="franka")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--max-episode-length", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--max-substeps", type=int, default=4)
    ap.add_argument("--intention-threshold", type=float, default=0.1)
    ap.add_argument("--intention-mode", choices=["exclusive", "legacy"],
                    default="exclusive")
    ap.add_argument("--exclusive-intention-margin", type=float, default=0.03)
    ap.add_argument("--warmup-seconds", type=float, default=3.0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="metadata only; actual action chunk comes from openpi")
    ap.add_argument("--tokenizer-max-length", type=int, default=256)
    ap.add_argument("--save-dir", default=str(REPO_ROOT / "outputs/eval_pi05_audio"))
    ap.add_argument("--save-video", action="store_true", default=False)
    ap.add_argument("--save-audio", action="store_true", default=False)
    ap.add_argument("--save-audio-log", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--revised-instruction", action="store_true", default=False)
    ap.add_argument("--one-radio-instruction", default=DEFAULT_ONE_RADIO_INSTRUCTION)
    ap.add_argument("--two-radio-instruction-template",
                    default=DEFAULT_TWO_RADIO_INSTRUCTION_TEMPLATE)
    ap.add_argument("--silent-radio-instruction",
                    default=DEFAULT_SILENT_RADIO_INSTRUCTION)
    ap.add_argument("--oracle-mode", action="store_true", default=False)
    ap.add_argument("--noise-az-std", type=float, default=3.0)
    ap.add_argument("--noise-el-std", type=float, default=5.0)
    ap.add_argument("--noise-conf-min", type=float, default=0.85)
    ap.add_argument("--noise-conf-max", type=float, default=0.98)
    ap.add_argument("--noise-class-flip-prob", type=float, default=0.02)
    ap.add_argument("--noise-energy-std", type=float, default=0.05,
                    help="Gaussian std on the [0,1] loudness/energy scalar "
                         "(slots_uv path). Set 0 for clean parity with clean "
                         "pi0 training data.")
    ap.add_argument("--noise-n-classes", type=int, default=38)
    ap.add_argument("--noise-source-drop-prob", type=float, default=0.0)
    ap.add_argument("--noise-distractor-prob", type=float, default=0.0)
    ap.add_argument("--noise-distractor-conf-max", type=float, default=0.15)
    ap.add_argument("--audio-mode", choices=["text", "slots", "slots_uv"], default="text",
                    help="How audio reaches pi0. 'text': serialise into the "
                         "prompt. 'slots': send raw SELD arrays for the model's "
                         "continuous audio tokens. 'slots_uv': SELD-VLA "
                         "SlotEncoder path (send energy + projected (u,v)). Must "
                         "match the training config's data.audio_mode.")
    ap.add_argument("--no-instruction", action="store_true", default=False)
    ap.add_argument("--class-smoothing-window", type=int, default=0)
    ap.add_argument("--shuffle-audio-slots", type=str, default="off",
                    choices=["on", "off"])
    ap.add_argument("--target-first-audio-slots", type=str, default="off",
                    choices=["on", "off"])
    ap.add_argument("--canonicalize-audio-slots", type=str, default="none",
                    choices=["none", "azimuth"])
    ap.add_argument("--motion-threshold", type=float, default=0.02,
                    help="Primary end-effector displacement (metres) from the "
                         "rest pose above which the robot counts as 'moving'. "
                         "Reported as the top-level motion_timing fields.")
    ap.add_argument("--motion-thresholds", type=str,
                    default="0.005,0.01,0.02,0.05,0.1",
                    help="Comma-separated extra displacement thresholds (m) "
                         "evaluated in the same rollout, so the 'did it move?' "
                         "bar can be swept without re-running. The primary "
                         "--motion-threshold is always included.")
    ap.add_argument("--save-motion-trace", action="store_true", default=False,
                    help="Dump per-step EE displacement-from-rest traces to "
                         "<save_dir>/motion_traces/ so any threshold can be "
                         "re-derived and plotted offline without re-running.")
    args = ap.parse_args()

    if args.revised_instruction:
        args.one_radio_instruction = REVISED_ONE_RADIO_INSTRUCTION
        args.two_radio_instruction_template = REVISED_TWO_RADIO_INSTRUCTION_TEMPLATE
        args.silent_radio_instruction = REVISED_SILENT_RADIO_INSTRUCTION
        if args.task_name == "select_radio":
            args.instruction = args.one_radio_instruction
        elif args.task_name == "select_radio_two":
            args.instruction = args.two_radio_instruction_template
        elif args.task_name == "select_radio_silent":
            args.instruction = args.silent_radio_instruction

    from src.eval.eval_smolvla_audio import (  # noqa: E402
        _load_audio_modules,
        evaluate_episode,
        _summarize_motion,
        BadInitialPlacement,
    )
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from trajectory_generation import _load_class_taxonomy  # noqa: E402

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "eval_args.json", "w") as f:
        eval_args = dict(vars(args))
        eval_args["instruction_metadata"] = _eval_instruction_metadata(args)
        json.dump(eval_args, f, indent=2)

    device = torch.device(args.device)
    policy = AudioAwarePi05Policy(
        audio_config=Pi05AudioConfig(
            taxonomy_path=args.taxonomy,
            top_k=args.top_k,
            tokenizer_max_length=args.tokenizer_max_length,
            chunk_size=args.chunk_size,
            n_classes=args.noise_n_classes,
            audio_mode=args.audio_mode,
        ),
        host=args.host,
        port=args.port,
        replan_steps=args.horizon,
        local=args.local,
        policy_config_name=args.policy_config,
        checkpoint_dir=args.policy_dir,
        openpi_root=args.openpi_root,
    )
    tokenizer = _DummyTokenizer()

    oracle_noise_cfg = None
    AudioSimManagerCls = None
    sled_overlay = None
    if args.oracle_mode:
        oracle_noise_cfg = OracleNoiseConfig(
            az_std_deg=args.noise_az_std,
            el_std_deg=args.noise_el_std,
            conf_min=args.noise_conf_min,
            conf_max=args.noise_conf_max,
            class_flip_prob=args.noise_class_flip_prob,
            n_classes=args.noise_n_classes,
            source_drop_prob=args.noise_source_drop_prob,
            distractor_prob=args.noise_distractor_prob,
            distractor_conf_max=args.noise_distractor_conf_max,
            energy_std=args.noise_energy_std,
        )
        print(f"[oracle] eval noise ON {asdict(oracle_noise_cfg)}")
    else:
        AudioSimManagerCls, SLEDOverlayCls = _load_audio_modules(
            args.audio_config, args.sled_ckpt, args.task_name)
        if SLEDOverlayCls is not None:
            sled_overlay = SLEDOverlayCls(
                ckpt_path=args.sled_ckpt,
                torch_device=str(device),
                realtime=True,
            )

    audio_classes, audio_label_map = _load_class_taxonomy()
    class_smoother = None
    if args.class_smoothing_window and args.class_smoothing_window > 1:
        class_smoother = TopKClassSmoother(
            top_k=args.top_k,
            window=args.class_smoothing_window,
        )

    infos = []
    max_reset_retries = 3
    for i in tqdm(range(args.n_episodes), desc="eval-pi05"):
        info = None
        last_err: Exception | None = None
        # Retry PhysicsError (intermittent MuJoCo mjWARN_BADQACC at reset)
        # and BadInitialPlacement (food spawned outside the microwave or
        # the door settled open). Any other exception is a real bug and is
        # recorded as a failed episode on the first hit.
        for retry in range(max_reset_retries):
            try:
                policy.reset()
                info = evaluate_episode(
                    args, i, policy, tokenizer, device,
                    audio_classes, audio_label_map,
                    AudioSimManagerCls, sled_overlay,
                    instruction=args.instruction, top_k=args.top_k,
                    oracle_noise_cfg=oracle_noise_cfg,
                    class_smoother=class_smoother,
                    no_instruction=args.no_instruction,
                    seed_offset=retry * 9973,
                )
                break
            except PhysicsError as e:
                last_err = e
                print(f"[warn] episode {i} PhysicsError (retry {retry+1}/{max_reset_retries}): {e}")
            except BadInitialPlacement as e:
                last_err = e
                print(f"[warn] episode {i} BadInitialPlacement (retry {retry+1}/{max_reset_retries}): {e}")
            except Exception as e:
                last_err = e
                print(f"[err] episode {i}: {e}")
                traceback.print_exc()
                break
        if info is not None:
            infos.append(info)
        else:
            infos.append({"episode_idx": i, "success": False,
                          "intention_score": 0.0,
                          "intention_success": False,
                          "legacy_intention_score": 0.0,
                          "legacy_intention_success": False,
                          "exclusive_intention": None,
                          "error": str(last_err), "consumed_steps": 0,
                          "progress": 0.0})

    n = len(infos)
    # Subtask completion rates for take_out_microwave_food (no-op for
    # other tasks because they never set `microwave_subtasks`).
    mw_eps = [x.get("microwave_subtasks") for x in infos
              if x.get("microwave_subtasks") is not None]
    mw_n = len(mw_eps)

    def _mw_rate(key):
        return sum(int(bool(d.get(key))) for d in mw_eps) / max(1, mw_n)

    # Motion-vs-sound aggregation: of the episodes where the robot moved at
    # all, what fraction started moving only after the sound onset, and how
    # long after on average. `by_threshold` repeats this at each sensitivity.
    motion_summary = _summarize_motion(infos)

    # Per-position (left/middle/right) success breakdown. `position_label` is
    # the target radio's task-frame position, already recorded per episode by
    # evaluate_episode; success rate varies strongly by position so we report
    # it separately (mirrors eval_smolvla_audio.py's by_position block). The
    # label set is built from the episodes actually seen so it works for both
    # the radio tasks (left/middle/right) and find_hidden_object (left_top/
    # right_top/left_bottom/right_bottom).
    _pos_labels = sorted({
        x.get("position_label") for x in infos
        if x.get("position_label") is not None
    })
    by_position = {
        p: {"n": 0, "succ": 0, "intention_succ": 0}
        for p in _pos_labels
    }
    for x in infos:
        p = x.get("position_label")
        if p in by_position:
            by_position[p]["n"]    += 1
            by_position[p]["succ"] += int(x.get("success", False))
            by_position[p]["intention_succ"] += int(
                x.get("intention_success", False)
            )

    summary = {
        "n": n,
        "success_rate": sum(1 for x in infos if x.get("success")) / max(1, n),
        "intention_success_rate": sum(1 for x in infos if x.get("intention_success")) / max(1, n),
        "mean_intention_score": float(np.mean([x.get("intention_score", 0.0) for x in infos])) if n else 0.0,
        "mean_progress": float(np.mean([x.get("progress", 0.0) for x in infos])) if n else 0.0,
        "microwave_subtask_rates": {
            "n_microwave_episodes":  mw_n,
            "door_opened_rate":      _mw_rate("door_opened"),
            "food_grasped_rate":     _mw_rate("food_grasped"),
            "food_on_tray_rate":     _mw_rate("food_on_tray"),
        },
        "motion_vs_sound": motion_summary,
        "by_position": {
            p: {
                **v,
                "rate": v["succ"] / max(1, v["n"]),
                "intention_rate": v["intention_succ"] / max(1, v["n"]),
            }
            for p, v in by_position.items()
        },
    }
    with open(save_dir / "episodes.json", "w") as f:
        json.dump(infos, f, indent=2)
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Readable one-line-per-position log so success by left/middle/right is
    # visible without parsing the JSON.
    print("\n=== success rate by target position ===")
    for p, v in summary["by_position"].items():
        print(f"  {p:<12} : {v['succ']:>3}/{v['n']:<3} = {v['rate']:.3f}"
              f"   (intention {v['intention_rate']:.3f})")


if __name__ == "__main__":
    main()
