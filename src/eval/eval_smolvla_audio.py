"""Evaluate a trained AudioAwareSmolVLA checkpoint on VLABench select_radio.

For each episode:
  1. reset env (seeded), pick a random radio + sound (same logic as
     trajectory_generation.py)
  2. start the binaural audio sim, run a brief warm-up so SLED has audio
  3. run SLED on the warm-up buffer to obtain a top-K event snapshot
     (camera is static for select_radio, so a single snapshot suffices)
  4. closed-loop control:
       observation → policy.predict_action_chunk → take first H actions
       (default H = `--horizon`), apply each one through IK + env.step
  5. record success / progress, save a video if requested
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

# --- AVLABench imports (defer until env is ready) ---------------------------
import VLABench.robots                                              # noqa: E402,F401
import VLABench.tasks                                               # noqa: E402,F401
from VLABench.envs import load_env                                  # noqa: E402
from VLABench.tasks import *                                        # noqa: E402, F401, F403
from VLABench.robots import *                                       # noqa: E402, F401, F403
from VLABench.utils.utils import (euler_to_quaternion,  # noqa: E402
                                  fold_roll_to_negative_branch,
                                  quaternion_to_euler)
from dm_control.rl.control import PhysicsError                      # noqa: E402

# --- Our project imports ----------------------------------------------------
from src.models.smolvla_audio import (                              # noqa: E402
    AudioAwareSmolVLAPolicy, AudioConfig,
    AUDIO_AZ_KEY, AUDIO_EL_KEY, AUDIO_CONF_KEY, AUDIO_CLASS_KEY,
    AUDIO_ENERGY_KEY, AUDIO_UV_KEY,
)
from src.audio.oracle_sled import (                                 # noqa: E402
    OracleNoiseConfig, extract_episode_gt, build_oracle_topk,
    TopKClassSmoother, resolve_mic_cam_id,
)

# Reuse the trajectory-generation helpers verbatim
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from trajectory_generation import (                                 # noqa: E402
    _load_class_taxonomy, _pick_random_sound,
    _build_two_radio_instruction, _build_silent_radio_instruction,
    _build_one_radio_instruction, _natural_class_name,
)


# --------------------------------------------------------------------------- #
DEFAULT_INSTRUCTION = (
    "primitive: Press the button in front of the radio that is making sound."
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
# find_hidden_object_open — must match the config-manager instruction and the
# --instruction passed to convert_hdf5_to_lerobot.py so train/eval prompts agree.
FIND_HIDDEN_INSTRUCTION = (
    "Open the drawer of the cabinet that contains the object making the sound."
)
# Cameras the policy sees. These must match convert_hdf5_to_lerobot's
# DEFAULT_CAM_MAP ({"image": 2, "second_image": 0, "wrist_image": 3}).
FRONT_CAM = 2
SECOND_CAM = 0
WRIST_CAM = 3
# The binaural listener ("mic") is a separate camera from the policy image — for
# find_hidden it is camera 1, re-posed low and centred so the top/bottom drawer
# elevation cue separates. See src/audio/oracle_sled.resolve_mic_cam_id.
# Resolved per task at rollout time, never hardcoded here.
GRIPPER_OPEN = np.full(2, 0.04)
GRIPPER_CLOSED = np.zeros(2)


def _eval_instruction_metadata(args) -> dict:
    """Describe the instruction text actually fed to the policy.

    `args.instruction` is only the argparse fallback/default. For task variants
    whose language is episode-dependent, evaluate_episode rebuilds the effective
    instruction after sampling the target sound class.
    """
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
    elif args.task_name == "take_out_microwave_food":
        # The microwave task's language is owned by the task itself —
        # evaluate_episode pulls `env.task.get_instruction()` per episode
        # (matching the converter's `--use-episode-instruction` behaviour
        # at training time), so `args.instruction` is never actually fed
        # to the policy. Recording this here keeps eval_args.json from
        # misleadingly suggesting the radio prompt was used.
        meta.update({
            "effective_instruction_source": "task_per_episode",
            "effective_instruction_static": None,
        })
    return meta


def _format_two_radio_instruction(template: str, class_name: str) -> str:
    natural_class = _natural_class_name(class_name)
    try:
        return template.format(class_name=natural_class, class_name_raw=class_name)
    except KeyError as exc:
        raise ValueError(
            "--two-radio-instruction-template only supports "
            "{class_name} and {class_name_raw} placeholders"
        ) from exc


# --------------------------------------------------------------------------- #
def _load_audio_modules(audio_config_path: str, sled_ckpt: str, task_name: str):
    """Build AudioSimManager + SLEDOverlay for a single episode.

    Returns (audio_mgr, sled_overlay). Both are None if the inputs are missing.
    """
    if audio_config_path is None or sled_ckpt is None:
        return None, None
    from audio_generation.audio_sim_manager import AudioSimManager
    from audio_generation.sled_overlay import SLEDOverlay
    return AudioSimManager, SLEDOverlay


def _build_episode_audio_cfg(audio_config_path: str, active_radio: str,
                             sound_meta: dict, task_name: str = "select_radio",
                             extra_sources: list[dict] | None = None) -> dict:
    """Mirror trajectory_generation.py's per-episode audio config.

    `extra_sources` is a list of additional {radio_name, sound_meta} dicts
    used by select_radio_two so a second radio also emits audio.
    """
    with open(audio_config_path) as f:
        base = json.load(f)
    sources = [{
        "object_name": active_radio,
        "geom_type":   "body",
        "sound_file":  sound_meta["sound_file"],
        "gain":        1.0,
    }]
    for extra in extra_sources or []:
        sources.append({
            "object_name": extra["radio_name"],
            "geom_type":   "body",
            "sound_file":  extra["sound_meta"]["sound_file"],
            "gain":        1.0,
        })
    return {
        "cam_id":    base.get("cam_id", resolve_mic_cam_id(task_name)),
        "hrtf_path": base.get("hrtf_path", ""),
        "tasks": {
            task_name: {"sources": sources},
        },
    }


def _live_snapshot(sled_overlay, top_k: int) -> dict:
    """Pull the most recent prediction from SLED's background thread and
    convert it to the (cid, az, el, conf) layout the policy expects.

    Thread-safe: `SLEDOverlay.get_latest_prediction()` returns a copy under
    its internal lock. Returns the empty snapshot if SLED hasn't produced a
    prediction yet (first ~1 s of an episode) or if its threshold rejected
    the most recent inference.
    """
    if sled_overlay is None:
        return _empty_audio(top_k)
    pred = sled_overlay.get_latest_prediction()
    if pred is None:
        return _empty_audio(top_k)
    doa, conf, cls = pred
    cid_arr = np.full(top_k, -1, dtype=np.int32)
    az_arr  = np.zeros(top_k, dtype=np.float32)
    el_arr  = np.zeros(top_k, dtype=np.float32)
    cf_arr  = np.zeros(top_k, dtype=np.float32)
    # SLED outputs S sources sorted by confidence; keep top_k.
    order = np.argsort(-conf)
    for k, s in enumerate(order[:top_k]):
        dx, dy, dz = float(doa[s, 0]), float(doa[s, 1]), float(doa[s, 2])
        cid_arr[k] = int(cls[s])
        az_arr[k]  = float(np.degrees(np.arctan2(dy, dx)))
        el_arr[k]  = float(np.degrees(np.arctan2(dz, np.sqrt(dx * dx + dy * dy))))
        cf_arr[k]  = float(conf[s])
    return {"class_id": cid_arr, "azimuth_deg": az_arr,
            "elevation_deg": el_arr, "confidence": cf_arr}


def _empty_audio(top_k: int) -> dict:
    return {
        "class_id":      np.full(top_k, -1, dtype=np.int32),
        "azimuth_deg":   np.zeros(top_k, dtype=np.float32),
        "elevation_deg": np.zeros(top_k, dtype=np.float32),
        "confidence":    np.zeros(top_k, dtype=np.float32),
    }


# Diagnostic audio ablation mode, read once at import from the environment.
_AUDIO_ABLATE = os.environ.get("VLABENCH_EVAL_AUDIO_ABLATE", "none").lower()


def _apply_audio_ablation(audio_snap: dict, mode: str) -> dict:
    """Mutate an oracle audio snapshot for the audio-usage probe.

    zero     -> remove all audio (class_id=-1, conf=0, az/el=0): the policy
                sees silence, so any residual left/right choice is NOT audio.
    flipaz   -> negate azimuth (swap left<->right cabinet cue).
    flipel   -> negate elevation (swap top<->bottom drawer cue).
    flipboth -> negate both.
    Only present slots (class_id>=0, conf>0) are touched.
    """
    out = {k: np.asarray(v).copy() for k, v in audio_snap.items()}
    present = (np.asarray(out["class_id"]) >= 0) & (np.asarray(out["confidence"]) > 0)
    if mode == "zero":
        out["class_id"][:] = -1
        out["confidence"][:] = 0.0
        out["azimuth_deg"][:] = 0.0
        out["elevation_deg"][:] = 0.0
        if "uv" in out:
            out["uv"][:] = 0.0
        if "energy" in out:
            out["energy"][:] = 0.0
        return out
    if mode in ("flipaz", "flipboth"):
        out["azimuth_deg"][present] = -out["azimuth_deg"][present]
    if mode in ("flipel", "flipboth"):
        out["elevation_deg"][present] = -out["elevation_deg"][present]
    return out


def _shuffle_present_audio_slots(audio_snap: dict, rng: np.random.Generator) -> dict:
    """Randomize source order when two or more confident audio slots exist.

    This mirrors the unordered nature of multi-source SELD reports. Single-source
    tasks are unaffected because fewer than two present slots are a no-op.
    """
    present = np.flatnonzero(
        (np.asarray(audio_snap["class_id"]) >= 0)
        & (np.asarray(audio_snap["confidence"]) > 0)
    )
    if present.shape[0] < 2:
        return audio_snap

    out = {k: np.asarray(v).copy() for k, v in audio_snap.items()}
    shuffled = present[rng.permutation(present.shape[0])]
    for key in ("class_id", "azimuth_deg", "elevation_deg", "confidence"):
        out[key][present] = out[key][shuffled]
    return out


def _canonicalize_audio_slots_by_azimuth(audio_snap: dict) -> dict:
    """Order present SELD detections from task-left to task-right."""
    present = np.flatnonzero(
        (np.asarray(audio_snap["class_id"]) >= 0)
        & (np.asarray(audio_snap["confidence"]) > 0)
    )
    if present.shape[0] < 2:
        return audio_snap

    az = np.asarray(audio_snap["azimuth_deg"])
    ordered = present[np.argsort(-az[present])]
    if np.array_equal(present, ordered):
        return audio_snap

    out = {k: np.asarray(v).copy() for k, v in audio_snap.items()}
    for key in ("class_id", "azimuth_deg", "elevation_deg", "confidence"):
        out[key][present] = out[key][ordered]
    return out


def _target_first_audio_slots(audio_snap: dict, target_class_id: int) -> dict:
    """Optional ablation: move the target class slot to index 0."""
    present = np.flatnonzero(
        (np.asarray(audio_snap["class_id"]) >= 0)
        & (np.asarray(audio_snap["confidence"]) > 0)
    )
    if present.shape[0] < 2 or target_class_id < 0:
        return audio_snap

    cid = np.asarray(audio_snap["class_id"])
    conf = np.asarray(audio_snap["confidence"])
    matches = present[cid[present] == int(target_class_id)]
    if matches.shape[0] == 0:
        return audio_snap

    best = int(matches[np.argmax(conf[matches])])
    if best == 0:
        return audio_snap

    out = {k: np.asarray(v).copy() for k, v in audio_snap.items()}
    for key in ("class_id", "azimuth_deg", "elevation_deg", "confidence"):
        out[key][[0, best]] = out[key][[best, 0]]
    return out


def _oracle_snapshot(
    env,
    active_sources: list[dict],   # [{"name", "class_id", "class_name"}]
    cam_id: int,
    top_k: int,
    noise_cfg: OracleNoiseConfig,
    rng: np.random.Generator,
) -> dict:
    """Oracle equivalent of `_live_snapshot`: compute per-frame GT from the
    current env state and inject noise. Assumes static scene (camera/source
    positions don't meaningfully change within one policy step)."""
    odict = extract_episode_gt(env, active_sources, cam_id=cam_id, n_frames=1)
    srcs = [{
        "class_id": s["class_id"],
        "az_deg":   s["az_deg"],
        "el_deg":   s["el_deg"],
        "energy":   s.get("energy", 1.0),
    } for s in odict["active_sources"]]
    snap = build_oracle_topk(srcs, top_k=top_k, noise_cfg=noise_cfg, rng=rng)
    # --- diagnostic audio ablation (env VLABENCH_EVAL_AUDIO_ABLATE) -----------
    # Applied HERE (before uv projection) so uv is recomputed from the mutated
    # az/el and the slots_uv policy actually sees the ablation. Probes whether
    # the policy uses audio direction at all.
    if _AUDIO_ABLATE != "none":
        snap = _apply_audio_ablation(snap, _AUDIO_ABLATE)
    # Project each slot's (possibly noised) DoA to the fixed camera image plane
    # for the SlotEncoder path (spec M2). Off-screen / silent slots → (-1,-1).
    from src.audio.projection import CameraIntrinsics, doa_to_uv
    fovy = odict.get("cam_fovy_deg")
    intr = CameraIntrinsics.from_fovy(fovy, 224, 224) if fovy is not None else None
    uv = np.full((top_k, 2), -1.0, dtype=np.float32)
    for k in range(top_k):
        if int(snap["class_id"][k]) < 0 or intr is None:
            continue
        p = doa_to_uv(float(snap["azimuth_deg"][k]), float(snap["elevation_deg"][k]),
                      intr, (224, 224), convention="sled")
        if p is not None:
            uv[k] = p
    snap["uv"] = uv
    return snap


# --------------------------------------------------------------------------- #
def _build_policy_batch(env, audio_snap: dict, instruction: str,
                        tokenizer, device: torch.device,
                        tokenizer_max_length: int,
                        no_instruction: bool = False) -> dict:
    """Convert a live VLABench observation into the dict the policy expects."""
    try:
        from lerobot.utils.constants import (
            OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE,
        )
    except ImportError:
        # lerobot 0.1.0 (openpi venv) ships these constants under
        # lerobot.common.constants but doesn't expose the OBS_LANGUAGE_*
        # keys at all. Match the lerobot 0.4.x string values verbatim so
        # downstream batch keys stay consistent with the training-time dict.
        OBS_LANGUAGE_TOKENS = "observation.language.tokens"
        OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
        OBS_STATE = "observation.state"

    # 1. Cameras (mujoco renders may have negative strides → copy first)
    rgb_front  = np.ascontiguousarray(env.physics.render(camera_id=FRONT_CAM,  height=224, width=224))
    rgb_second = np.ascontiguousarray(env.physics.render(camera_id=SECOND_CAM, height=224, width=224))
    rgb_wrist  = np.ascontiguousarray(env.physics.render(camera_id=WRIST_CAM,  height=224, width=224))
    def _to_chw(rgb_uint8):
        x = torch.from_numpy(rgb_uint8).float() / 255.0
        return x.permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
    out = {
        "observation.images.image":        _to_chw(rgb_front),
        "observation.images.second_image": _to_chw(rgb_second),
        "observation.images.wrist_image":  _to_chw(rgb_wrist),
    }

    # 2. State (matches converter: [x, y, z, rx, ry, rz, gripper])
    ee_state = env.robot.get_ee_state(env.physics)            # 8-dim world frame
    robot_pos = env.get_robot_frame_position()
    pos_world = np.asarray(ee_state[:3], dtype=np.float32)
    pos_base  = pos_world - np.asarray(robot_pos, dtype=np.float32)
    quat = np.asarray(ee_state[3:7], dtype=np.float32)
    euler = np.asarray(quaternion_to_euler(quat), dtype=np.float32)
    # The top-down home pose has roll ~= +-pi, exactly on scipy's branch cut, so
    # the same physical wrist reads as +3.141 or -3.132 frame to frame. Fold it
    # onto one branch — byte-identical to what the converter writes into
    # observation.state (convert_hdf5_to_lerobot._compute_real_state); without
    # this the policy sees a 5.4-sigma state jump for an unchanged wrist.
    euler[0] = fold_roll_to_negative_branch(euler[0])
    # NOTE: `franka.get_ee_open_state` is named wrong — it returns True when
    # the fingers are CLOSED (qpos < 0.035), not open. The converter writes
    # state[6] = 1.0 for *open* fingers (matching the gripper command
    # convention `_binarize_gripper`), so we invert the call here. Without
    # this, the policy sees its gripper state flipped vs training and
    # outputs wildly wrong actions from the very first step.
    gripper_open = float(not bool(env.robot.get_ee_open_state(env.physics)))
    state_7 = np.concatenate([pos_base, euler, [gripper_open]]).astype(np.float32)
    out[OBS_STATE] = torch.from_numpy(state_7).unsqueeze(0).to(device)

    # 3. Audio
    out[AUDIO_CLASS_KEY] = torch.from_numpy(audio_snap["class_id"]).long().unsqueeze(0).to(device)
    out[AUDIO_AZ_KEY]    = torch.from_numpy(audio_snap["azimuth_deg"]).float().unsqueeze(0).to(device)
    out[AUDIO_EL_KEY]    = torch.from_numpy(audio_snap["elevation_deg"]).float().unsqueeze(0).to(device)
    out[AUDIO_CONF_KEY]  = torch.from_numpy(audio_snap["confidence"]).float().unsqueeze(0).to(device)
    # Optional SlotEncoder-path fields (present when the snapshot was built with
    # energy/uv; harmless for the az/el SmolVLA path which ignores them).
    if "energy" in audio_snap:
        out[AUDIO_ENERGY_KEY] = torch.from_numpy(
            np.asarray(audio_snap["energy"], dtype=np.float32)).unsqueeze(0).to(device)
    if "uv" in audio_snap:
        out[AUDIO_UV_KEY] = torch.from_numpy(
            np.asarray(audio_snap["uv"], dtype=np.float32)).unsqueeze(0).to(device)

    # 4. Language
    if no_instruction:
        # Zero-out language conditioning to probe text-instruction bias.
        # Tokenize an empty string so the tensor shapes are correct, then force
        # the attention mask to all-zeros so no language token attends.
        enc = tokenizer(["\n"], padding="max_length", truncation=True,
                        max_length=tokenizer_max_length, return_tensors="pt")
        out[OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(device)
        out[OBS_LANGUAGE_ATTENTION_MASK] = torch.zeros_like(enc["attention_mask"]).to(device).bool()
    else:
        text = instruction if instruction.endswith("\n") else instruction + "\n"
        enc = tokenizer([text], padding="max_length", truncation=True,
                        max_length=tokenizer_max_length, return_tensors="pt")
        out[OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(device)
        out[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].to(device).bool()
    # Non-SmolVLA backbones (pi0.5/openpi) consume the raw text prompt rather
    # than token ids. Keeping it here lets them reuse the same env/audio loop.
    out["raw_instruction"] = [instruction]
    out["no_instruction"] = bool(no_instruction)
    return out


def _binarize_to_fingers(g: float) -> np.ndarray:
    return GRIPPER_OPEN if g > 0.5 else GRIPPER_CLOSED


class BadInitialPlacement(RuntimeError):
    """Raised when env.reset() returns a state that violates task invariants
    (e.g. take_out_microwave_food without the food sitting inside the closed
    microwave cavity). Caught by the outer retry loop so the next attempt
    rerolls layout randomness."""


def _validate_microwave_food_placement(env) -> tuple[bool, str | None]:
    """Sanity-check the initial state of take_out_microwave_food.

    The task spec requires the cooked food to start *inside* the *closed*
    microwave. Occasionally the spawn lands the food on the table next to
    the appliance, or physics settles the door slightly open. Both make
    the rollout impossible to complete from the policy's perspective, so
    we detect them here and force a layout reroll.
    """
    task = env.task
    physics = env.physics
    target_entity = getattr(task, "target_entity", None)
    cm = getattr(task, "config_manager", None)
    target_container = getattr(cm, "target_container", None) if cm is not None else None
    if not target_entity or not target_container:
        return False, "missing_target_metadata"

    food = task.entities.get(target_entity)
    mw = task.entities.get(target_container)
    if food is None or mw is None:
        return False, "missing_entities"

    food_pos = np.asarray(food.get_xpos(physics), dtype=np.float64)
    mw_pos = np.asarray(mw.get_xpos(physics), dtype=np.float64)
    xy_dist = float(np.linalg.norm(food_pos[:2] - mw_pos[:2]))
    z_diff = float(food_pos[2] - mw_pos[2])

    # Microwave cavity in body-local frame: roughly x∈[-0.26, 0.16],
    # y∈[-0.16, 0.22], z∈[-0.12, 0.14]. A small slack handles rotation
    # variants. Anything beyond these bounds means the food is outside.
    if xy_dist > 0.28:
        return False, f"food_outside_microwave_xy_dist={xy_dist:.3f}"
    if z_diff < -0.18 or z_diff > 0.22:
        return False, f"food_outside_microwave_z_diff={z_diff:.3f}"

    # Door must start closed — the task's `door_opened_before_chime`
    # failure flag would otherwise trip almost immediately.
    is_closed_fn = getattr(mw, "is_closed", None)
    if is_closed_fn is not None and not is_closed_fn(physics):
        return False, "microwave_door_open_at_start"

    return True, None


def _collect_entity_geom_ids(env, entity) -> set[int]:
    """Best-effort enumeration of MuJoCo geom IDs belonging to `entity`."""
    ids: set[int] = set()
    if entity is None:
        return ids
    mjcf_model = getattr(entity, "mjcf_model", None)
    if mjcf_model is None:
        return ids
    try:
        geoms = mjcf_model.find_all("geom")
    except Exception:
        return ids
    for geom in geoms:
        try:
            gid = env.physics.bind(geom).element_id
            if gid is not None and gid >= 0:
                ids.add(int(gid))
        except Exception:
            continue
    return ids


def _robot_microwave_in_contact(env, robot_ids: set[int], mw_ids: set[int]) -> bool:
    """Return True iff any active MuJoCo contact pair touches one robot geom
    and one microwave geom."""
    if not robot_ids or not mw_ids:
        return False
    data = env.physics.data
    ncon = int(getattr(data, "ncon", 0))
    if ncon <= 0:
        return False
    contacts = data.contact
    for i in range(ncon):
        c = contacts[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in robot_ids and g2 in mw_ids) or (g2 in robot_ids and g1 in mw_ids):
            return True
    return False


def _synthesize_microwave_chime_wav(out_path: Path, chime_sec: float,
                                    total_sec: float,
                                    sample_rate: int = 44100) -> bool:
    """Write a `total_sec` silent stereo wav with the canonical microwave
    chime spliced in starting at `chime_sec`. Returns True on success.

    Used so that videos saved during oracle-mode runs (where no real
    binaural audio engine is attached) still carry an audible cue at the
    same sim-time second the policy was told the chime fired.
    """
    try:
        import soundfile as sf  # local — soundfile is already a dep elsewhere
    except Exception as e:
        print(f"  [warn] soundfile unavailable, skipping synth chime: {e}")
        return False
    ring_path = REPO_ROOT / "audio_generation" / "sound" / "ring.wav"
    if not ring_path.exists():
        print(f"  [warn] {ring_path} not found, skipping synth chime")
        return False
    try:
        chime, sr = sf.read(str(ring_path), dtype="float32")
    except Exception as e:
        print(f"  [warn] could not read ring.wav: {e}")
        return False
    if chime.ndim == 1:
        chime = np.stack([chime, chime], axis=1)
    if sr != sample_rate:
        new_n = int(round(chime.shape[0] * sample_rate / sr))
        idx_new = np.linspace(0, chime.shape[0] - 1, new_n)
        idx_old = np.arange(chime.shape[0])
        chime = np.stack([
            np.interp(idx_new, idx_old, chime[:, 0]),
            np.interp(idx_new, idx_old, chime[:, 1]),
        ], axis=1).astype(np.float32)
    n_total = max(1, int(round(total_sec * sample_rate)))
    out = np.zeros((n_total, 2), dtype=np.float32)
    start = max(0, int(round(chime_sec * sample_rate)))
    end = min(n_total, start + chime.shape[0])
    if end > start:
        out[start:end] = chime[: end - start]
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), out, sample_rate)
        return True
    except Exception as e:
        print(f"  [warn] writing synth chime failed: {e}")
        return False


def _mux_audio_into_video(video_path: Path, audio_path: Path,
                          output_path: Path, audio_offset_sec: float = 0.0) -> bool:
    """Mux `audio_path` onto `video_path` with ffmpeg. `audio_offset_sec`
    trims the start of the audio (positive seconds skip warmup). Returns
    True on success; logs a warning and returns False otherwise (e.g.
    ffmpeg not installed)."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        print("  [warn] ffmpeg not found; skipping audio+video mux.")
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(video_path)]
    if audio_offset_sec and audio_offset_sec > 0:
        cmd += ["-ss", f"{audio_offset_sec:.3f}"]
    cmd += ["-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac",
            "-shortest", str(output_path)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [warn] ffmpeg mux failed (rc={e.returncode}): {e}")
        return False
    except Exception as e:
        print(f"  [warn] ffmpeg mux failed: {e}")
        return False


def _apply_action(env, action_7d: np.ndarray, robot_pos: np.ndarray,
                  max_substeps: int, tolerance: float) -> tuple[bool, bool, float]:
    """Apply one 7-dim policy action via IK + env.step.

    Returns (finished, ik_success, reach_error_m). ik_success is the raw
    convergence flag from dm_control's qpos_from_site_pose for this step's
    IK solve — it is NOT used to gate anything (the (possibly non-converged)
    qpos is applied either way, matching the existing behavior). reach_error_m
    is the actual post-step Cartesian gap between the requested EE target and
    where the end effector physically ended up (after the substep/tolerance
    settling loop) — the true "how far off did we land" number, as opposed to
    the IK solver's internal (pre-simulation) err_norm.
    """
    pos_base = action_7d[:3]
    euler    = action_7d[3:6]
    gripper  = float(action_7d[6])

    pos_world = pos_base + robot_pos
    quat = euler_to_quaternion(*euler)
    ik_success, qpos = env.robot.get_qpos_from_ee_pos(
        physics=env.physics, pos=pos_world, quat=quat
    )
    qaction = np.concatenate([qpos, _binarize_to_fingers(gripper)])
    finished = False
    for _ in range(max_substeps):
        ts = env.step(qaction)
        if ts.last():
            finished = True
            break
        cur = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
        if np.max(cur - qaction[:7]) < tolerance and np.min(cur - qaction[:7]) > -tolerance:
            break
    achieved_pos = np.asarray(env.robot.get_end_effector_pos(env.physics), dtype=np.float64)
    reach_error_m = float(np.linalg.norm(achieved_pos - pos_world))
    return finished, ik_success, reach_error_m


# --------------------------------------------------------------------------- #
def _parse_motion_thresholds(spec: str | None, primary: float) -> list[float]:
    """Parse a comma-separated metre list and always include `primary`.

    Returned sorted ascending so the most relaxed (smallest) threshold comes
    first. Lets one rollout report motion onset at several sensitivities, so
    the "did the robot move?" bar isn't a single hard-coded guess.
    """
    vals = {float(primary)}
    if spec:
        for tok in str(spec).split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.add(float(tok))
                except ValueError:
                    pass
    return sorted(v for v in vals if v > 0)


def _summarize_motion(infos: list[dict]) -> dict:
    """Aggregate per-episode `motion_timing` into moved / after-sound rates,
    including a per-threshold breakdown (`by_threshold`)."""
    mt_eps = [x.get("motion_timing") for x in infos
              if x.get("motion_timing") is not None]

    def _rates(entries):
        moved = [e for e in entries if e and e.get("moved")]
        after = [e for e in moved if e.get("moved_after_sound")]
        lags = [e["motion_lag_sec"] for e in moved
                if e.get("motion_lag_sec") is not None]
        return {
            "moved_rate":             len(moved) / max(1, len(entries)),
            "moved_after_sound_rate": len(after) / max(1, len(moved)),
            "mean_motion_lag_sec":    float(np.mean(lags)) if lags else None,
        }

    thr_keys = sorted(
        {k for m in mt_eps for k in (m.get("by_threshold") or {})},
        key=float,
    )
    return {
        "n_episodes": len(mt_eps),
        **_rates(mt_eps),
        "by_threshold": {
            k: _rates([m["by_threshold"][k] for m in mt_eps
                       if (m.get("by_threshold") or {}).get(k) is not None])
            for k in thr_keys
        },
    }


def evaluate_episode(args, episode_idx: int, policy, tokenizer, device,
                     audio_classes, audio_label_map,
                     AudioSimManagerCls, SLEDOverlayInst,
                     instruction: str, top_k: int,
                     oracle_noise_cfg: OracleNoiseConfig | None = None,
                     class_smoother: TopKClassSmoother | None = None,
                     no_instruction: bool = False,
                     seed_offset: int = 0) -> dict:
    # `seed_offset` lets the outer retry loop reroll randomness when a
    # MuJoCo PhysicsError fires during env.reset(); without it the next
    # attempt would replay the exact unstable initial state.
    effective_seed = args.seed + episode_idx + seed_offset
    np.random.seed(effective_seed)
    import random as _rd
    _rd.seed(effective_seed)
    oracle_rng = np.random.default_rng(args.seed * 1000 + episode_idx + 7 + seed_offset * 131)

    # dm_control's composer.Environment ignores np.random.seed() and pulls its
    # own RandomState from OS entropy when none is provided, so we feed it an
    # explicit one keyed off effective_seed. That way retries with a different
    # seed_offset actually produce a different initial layout.
    env_random_state = np.random.RandomState(effective_seed & 0x7FFFFFFF)
    env = load_env(args.task_name, robot=args.robot, eval=False, run_mode="eval",
                   random_state=env_random_state)
    env.reset()

    is_microwave_task = (args.task_name == "take_out_microwave_food")
    is_hidden_task = args.task_name in ("find_hidden_object_open", "find_hidden_object")
    cm = env.task.config_manager

    # ------------------------------------------------------------------
    # Per-task source setup. The radio family populates `active_radio`
    # from the config manager; `take_out_microwave_food` instead has a
    # single delayed source (the microwave chime) whose trigger time is
    # owned by the task itself.
    # ------------------------------------------------------------------
    other_radio = None
    other_position_label = None
    other_sound_meta = None
    sounding_sources = []
    microwave_trigger_step = None

    if is_microwave_task:
        # Reject obviously broken resets before we pay for the full rollout.
        # The outer retry loop will reroll the seed and try again.
        ok, reason = _validate_microwave_food_placement(env)
        if not ok:
            env.close()
            raise BadInitialPlacement(f"episode {episode_idx}: {reason}")

        active_radio = cm.target_container        # e.g. "microwave_seen"
        position_label = None
        # Manufactured sound_meta keeps the rest of the eval / info dict
        # path (which expects {class_id, class_name, folder, sound_file})
        # working unchanged. Class 34 = Domestic_Appliance in
        # class_taxonomy.yaml, matching the oracle GT recorded at
        # dataset time and the chime's training-time class label.
        sound_meta = {
            "class_id":   34,
            "class_name": "Domestic_Appliance",
            "folder":     "microwave_chime",
            "sound_file": "ring.wav",
        }
        episode_instruction = env.task.get_instruction() or instruction
        microwave_trigger_step = int(cm.trigger_step)
    elif is_hidden_task:
        # The sound emanates from the hidden OBJECT itself (a top-level entity
        # whose world position encodes azimuth = which cabinet and elevation =
        # which drawer). Mirrors the trajectory_generation.py hidden-task branch.
        active_radio = cm.target_entity
        position_label = getattr(cm, "position_label", None)
        sound_meta = _pick_random_sound(audio_label_map, audio_classes)
        if sound_meta is None:
            env.close()
            raise RuntimeError("find_hidden_object: failed to pick a sound")
        episode_instruction = FIND_HIDDEN_INSTRUCTION
    else:
        # Pick the active radio + sound exactly like trajectory_generation.py
        active_idx = getattr(cm, "active_radio_idx", None)
        if active_idx is None:
            env.close()
            raise RuntimeError("config_manager has no active_radio_idx")
        active_radio = f"radio_{active_idx}"
        position_label = getattr(cm, "active_position_label", None)
        sound_meta = None if args.task_name == "select_radio_silent" else _pick_random_sound(audio_label_map, audio_classes)
        episode_instruction = instruction

    # select_radio_two: a second radio also plays from a different class. The
    # instruction is rebuilt per episode to name the *target* sound class so
    # the policy has the language signal it was trained with.
    if args.task_name == "select_radio":
        episode_instruction = args.one_radio_instruction
    if args.task_name == "select_radio_two":
        other_idx = getattr(cm, "other_radio_idx", None)
        if other_idx is None:
            env.close()
            raise RuntimeError(
                "select_radio_two: config_manager missing other_radio_idx"
            )
        other_radio = f"radio_{other_idx}"
        other_position_label = getattr(cm, "other_position_label", None)
        if sound_meta is None:
            env.close()
            raise RuntimeError("select_radio_two: failed to pick target sound")
        other_sound_meta = _pick_random_sound(
            audio_label_map, audio_classes,
            exclude_class_ids=[sound_meta["class_id"]],
        )
        if other_sound_meta is None:
            env.close()
            raise RuntimeError(
                "select_radio_two: could not find a second sound from a "
                "different class than the target."
            )
        episode_instruction = _format_two_radio_instruction(
            args.two_radio_instruction_template,
            sound_meta["class_name"],
        )
    elif args.task_name == "select_radio_silent":
        sounding_indices = getattr(cm, "sounding_radio_indices", None)
        if sounding_indices is None or len(sounding_indices) != 2:
            env.close()
            raise RuntimeError(
                "select_radio_silent: config_manager missing two sounding_radio_indices"
            )
        used_class_ids = []
        for sounding_idx in sounding_indices:
            sm = _pick_random_sound(
                audio_label_map, audio_classes,
                exclude_class_ids=used_class_ids,
            )
            if sm is None:
                env.close()
                raise RuntimeError(
                    "select_radio_silent: could not pick two distinct sounding classes"
                )
            used_class_ids.append(sm["class_id"])
            sounding_sources.append({
                "radio": f"radio_{sounding_idx}",
                "position": ["left", "middle", "right"][sounding_idx],
                "sound_meta": sm,
            })
        episode_instruction = args.silent_radio_instruction

    oracle_mode = oracle_noise_cfg is not None

    # Build the audio manager — SLED's background thread will read its
    # recording buffer continuously. Skipped entirely in oracle mode.
    audio_mgr = None
    oracle_sources = None
    if oracle_mode:
        if sound_meta is None and not sounding_sources:
            raise RuntimeError("oracle mode needs a sound_meta for class_id")
        # Include all active sources so the network sees the full audio scene:
        # target in slot 0, distractor (if any) in slot 1. The VLM inside
        # SmolVLA is expected to match the instruction's class name against
        # the audio class tokens and route the correct direction to the action
        # expert — that matching happens inside the network, not here.
        oracle_sources = []
        if sound_meta is not None:
            oracle_sources.append({
                "name":       active_radio,
                "class_id":   sound_meta["class_id"],
                "class_name": sound_meta["class_name"],
            })
        if other_sound_meta is not None and other_radio is not None:
            oracle_sources.append({
                "name":       other_radio,
                "class_id":   other_sound_meta["class_id"],
                "class_name": other_sound_meta["class_name"],
            })
        for src in sounding_sources:
            sm = src["sound_meta"]
            oracle_sources.append({
                "name":       src["radio"],
                "class_id":   sm["class_id"],
                "class_name": sm["class_name"],
            })
        env.step()   # one step to seed entity positions (no audio needed)
    elif AudioSimManagerCls is not None and (
        sound_meta is not None or sounding_sources
    ):
        if is_microwave_task:
            # One-shot chime gated by `start_step` (= task.trigger_step),
            # built inline to mirror the dynamic config that
            # trajectory_generation.py creates for this task.
            with open(args.audio_config) as _f:
                _base_cfg = json.load(_f)
            cfg_dict = {
                "cam_id":    _base_cfg.get("cam_id", resolve_mic_cam_id(args.task_name)),
                "hrtf_path": _base_cfg.get("hrtf_path", ""),
                "tasks": {
                    args.task_name: {"sources": [{
                        "object_name": active_radio,
                        "geom_type":   "body",
                        "sound_file":  sound_meta["sound_file"],
                        "gain":        1.0,
                        "loop":        False,
                        "start_step":  microwave_trigger_step,
                    }]},
                },
            }
        else:
            extra = None
            primary_radio = active_radio
            primary_sound_meta = sound_meta
            if sounding_sources:
                primary_radio = sounding_sources[0]["radio"]
                primary_sound_meta = sounding_sources[0]["sound_meta"]
                extra = [
                    {"radio_name": src["radio"], "sound_meta": src["sound_meta"]}
                    for src in sounding_sources[1:]
                ]
            elif other_sound_meta is not None and other_radio is not None:
                extra = [{"radio_name": other_radio, "sound_meta": other_sound_meta}]
            cfg_dict = _build_episode_audio_cfg(
                args.audio_config, primary_radio, primary_sound_meta,
                task_name=args.task_name, extra_sources=extra,
            )
        audio_mgr = AudioSimManagerCls(
            config_path=args.audio_config,
            task_name=args.task_name,
            config_dict=cfg_dict,
        )
        audio_mgr.attach_to_env(env)
        audio_mgr.start()
        if SLEDOverlayInst is not None:
            SLEDOverlayInst.set_audio_engine(audio_mgr.engine)

        # Warm-up: let the engine fill at least one SLED window (~480 ms for
        # v5; ~960 ms for v3/v4) before the policy starts querying. Without
        # this, the first few steps see _empty_audio.
        env.step()                                # seed source positions
        import time
        time.sleep(args.warmup_seconds)

    robot_pos = np.asarray(env.get_robot_frame_position(), dtype=np.float32)
    # Camera that acts as the binaural listener. Separate from FRONT_CAM (the
    # policy image) because find_hidden puts the mic on a low, centred camera 1
    # so the top/bottom drawer elevation cue separates; the audio labels in the
    # training set were computed from this same camera, so a mismatch here is a
    # silent train/eval divergence.
    mic_cam_id = resolve_mic_cam_id(args.task_name)
    success = False
    frames = []
    audio_log = []                                # one entry per VLA step
    ik_step_count = 0
    ik_fail_count = 0
    reach_errors_m: list[float] = []

    # For take_out_microwave_food, track the first moment the robot
    # touches the appliance (door, handle, body, …) so we can report
    # the reaction lag against the chime onset, plus the first step each
    # subtask (door open → food grasped → food on tray) completes so we
    # can score the rollout's partial progress.
    microwave_geom_ids: set[int] = set()
    robot_geom_ids: set[int] = set()
    first_contact_step: int | None = None
    door_opened_step: int | None = None
    food_grasped_step: int | None = None
    food_on_tray_step: int | None = None
    mw_entity_obj = None
    food_entity_obj = None
    if is_microwave_task:
        mw_entity_obj = env.task.entities.get(cm.target_container)
        food_entity_obj = env.task.entities.get(getattr(env.task, "target_entity", None))
        microwave_geom_ids = _collect_entity_geom_ids(env, mw_entity_obj)
        robot_geom_ids = _collect_entity_geom_ids(env, env.task.robot)

    # Per-episode reset: the smoother carries no history across episodes.
    if class_smoother is not None:
        class_smoother.reset()

    # Robot-motion vs sound-onset tracking. Capture the resting end-effector
    # position before the policy acts; `first_motion_step` is the first env
    # step where the EE has moved more than `motion_threshold` metres from
    # that rest pose. Comparing it against the sound onset tells us whether
    # the robot actually waits for the cue before reacting.
    motion_threshold = float(getattr(args, "motion_threshold", 0.02))
    motion_thresholds = _parse_motion_thresholds(
        getattr(args, "motion_thresholds", ""), motion_threshold)
    try:
        initial_ee_pos = np.asarray(env.get_ee_pos(), dtype=np.float32).copy()
    except Exception:
        initial_ee_pos = None
    first_motion_step_by_thr: dict[float, int | None] = {
        t: None for t in motion_thresholds}
    max_ee_displacement: float = 0.0
    # Per-step EE displacement-from-rest trace (step, metres). Optional dump
    # lets any motion threshold be re-derived offline without re-running.
    ee_disp_trace: list[list] = []

    step = 0
    while step < args.max_episode_length:
        if oracle_mode:
            # `take_out_microwave_food` has a *delayed* source: report
            # silence until the chime's trigger_step is reached, so the
            # policy actually has to wait for the cue exactly as during
            # training (where pre-trigger frames are gated by
            # `active_from_frame` in the converter).
            if is_microwave_task and (
                env.task.step_count < env.task.audio_trigger_step
            ):
                audio_snap = _empty_audio(top_k)
            else:
                audio_snap = _oracle_snapshot(
                    env, oracle_sources, cam_id=mic_cam_id, top_k=top_k,
                    noise_cfg=oracle_noise_cfg, rng=oracle_rng,
                )
        else:
            audio_snap = _live_snapshot(SLEDOverlayInst, top_k)

        # Non-ML clean-up: temporal majority vote on class IDs to suppress
        # single-frame flips. Do this before optional slot shuffling: the
        # smoother is per slot, while multi-source SELD slot order is
        # intentionally unordered. Smoothing after shuffling can pair a class
        # from one source with another source's direction.
        raw_class_ids = audio_snap["class_id"].tolist()
        if class_smoother is not None:
            audio_snap = dict(audio_snap)
            audio_snap["class_id"] = class_smoother.smooth(audio_snap["class_id"])

        if args.shuffle_audio_slots == "on":
            audio_snap = _shuffle_present_audio_slots(audio_snap, oracle_rng)
        if args.canonicalize_audio_slots == "azimuth":
            audio_snap = _canonicalize_audio_slots_by_azimuth(audio_snap)
        if args.target_first_audio_slots == "on":
            target_cid = int(sound_meta["class_id"]) if sound_meta is not None else -1
            audio_snap = _target_first_audio_slots(audio_snap, target_cid)

        audio_log.append({
            "step": step,
            "class_id":         audio_snap["class_id"].tolist(),
            "class_id_raw":     raw_class_ids,
            "azimuth_deg":      [round(float(x), 2) for x in audio_snap["azimuth_deg"]],
            "elevation_deg":    [round(float(x), 2) for x in audio_snap["elevation_deg"]],
            "confidence":       [round(float(x), 3) for x in audio_snap["confidence"]],
        })
        batch = _build_policy_batch(
            env, audio_snap, episode_instruction, tokenizer, device,
            policy.config.tokenizer_max_length,
            no_instruction=no_instruction,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            with torch.no_grad():
                actions = policy.predict_action_chunk(batch)
        actions = actions[0].float().cpu().numpy()           # (chunk_size, 7)
        for h in range(min(args.horizon, actions.shape[0])):
            if step >= args.max_episode_length:
                break
            if args.save_video:
                frames.append(env.get_observation(require_pcd=False)["rgb"])
            finished, ik_success, reach_error_m = _apply_action(
                env, actions[h], robot_pos,
                max_substeps=args.max_substeps, tolerance=1e-2,
            )
            ik_step_count += 1
            if not ik_success:
                ik_fail_count += 1
            reach_errors_m.append(reach_error_m)
            step += 1
            # Track how far the EE has drifted from its rest pose so we can
            # flag the first step the robot "really" moves (vs sound onset).
            if initial_ee_pos is not None:
                try:
                    ee_now = np.asarray(env.get_ee_pos(), dtype=np.float32)
                    disp = float(np.linalg.norm(ee_now - initial_ee_pos))
                    ee_disp_trace.append([step, round(disp, 4)])
                    if disp > max_ee_displacement:
                        max_ee_displacement = disp
                    for thr in motion_thresholds:
                        if first_motion_step_by_thr[thr] is None and disp > thr:
                            first_motion_step_by_thr[thr] = step
                except Exception:
                    pass
            if is_microwave_task:
                if (first_contact_step is None and
                        _robot_microwave_in_contact(env, robot_geom_ids, microwave_geom_ids)):
                    first_contact_step = step
                # Subtask 1: door opened (only counts after the chime — opening
                # before the cue is an explicit task failure mode).
                if (door_opened_step is None and mw_entity_obj is not None and
                        microwave_trigger_step is not None and
                        step >= int(microwave_trigger_step)):
                    try:
                        if mw_entity_obj.is_open(env.physics):
                            door_opened_step = step
                    except Exception:
                        pass
                # Subtask 2: food grasped.
                if food_grasped_step is None and food_entity_obj is not None:
                    try:
                        if food_entity_obj.is_grasped(env.physics, env.task.robot):
                            food_grasped_step = step
                    except Exception:
                        pass
                # Subtask 3: food sitting in the tray (the contain condition).
                if food_on_tray_step is None and env.task.conditions is not None:
                    try:
                        if env.task.conditions.is_met(env.physics):
                            food_on_tray_step = step
                    except Exception:
                        pass
            if finished:
                success = True
                break
        if success:
            break

    # Final EE position (world). Used by the audio-usage probe to measure
    # whether the policy's endpoint shifts left/right with the audio cue.
    try:
        final_ee_pos = np.asarray(env.get_ee_pos(), dtype=np.float32).tolist()
    except Exception:
        final_ee_pos = None

    progress = float(env.get_task_progress())
    legacy_intention_score = float(
        env.get_intention_score(threshold=args.intention_threshold)
    )
    legacy_intention_success = bool(success) or bool(legacy_intention_score)

    exclusive_intention = None
    get_exclusive_intention_info = getattr(env, "get_exclusive_intention_info", None)
    if get_exclusive_intention_info is not None:
        exclusive_intention = get_exclusive_intention_info(
            threshold=args.intention_threshold,
            margin=args.exclusive_intention_margin,
        )

    if args.intention_mode == "exclusive" and exclusive_intention is not None:
        intention_score = float(exclusive_intention["exclusive_score"])
        intention_success = bool(exclusive_intention["exclusive_success"])
    else:
        intention_score = legacy_intention_score
        intention_success = legacy_intention_success

    # Tear the audio sim down cleanly, optionally dumping the recorded WAV
    # so users can audit what SLED was listening to.
    if audio_mgr is not None:
        if SLEDOverlayInst is not None:
            SLEDOverlayInst.set_audio_engine(None)
        audio_mgr.detach_from_env(env)
        if args.save_audio:
            wav_dir = Path(args.save_dir) / "audio"
            wav_dir.mkdir(parents=True, exist_ok=True)
            audio_mgr.stop_and_save(str(wav_dir / f"audio_eval_{episode_idx}.wav"))
        else:
            audio_mgr.engine.stop()

    # Per-step SLED stats: confident = top-1 class != -1 AND conf >= conf_thresh
    conf_thresh = float(getattr(SLEDOverlayInst, "_conf_thresh", 0.30)) \
                  if SLEDOverlayInst is not None else 0.0
    n_steps = max(1, len(audio_log))
    n_confident = sum(
        1 for x in audio_log
        if x["class_id"][0] != -1 and x["confidence"][0] >= conf_thresh
    )
    sled_confident_rate = n_confident / n_steps

    _reach_arr = np.asarray(reach_errors_m, dtype=np.float64) if reach_errors_m else np.zeros(1)
    info = {
        "episode_idx":         episode_idx,
        "success":             bool(success),
        "ik_step_count":       ik_step_count,
        "ik_fail_count":       ik_fail_count,
        "ik_fail_rate":        (ik_fail_count / ik_step_count) if ik_step_count else 0.0,
        "reach_error_mean_m":  float(_reach_arr.mean()),
        "reach_error_p90_m":   float(np.percentile(_reach_arr, 90)),
        "reach_error_max_m":   float(_reach_arr.max()),
        "reach_error_final_m": float(_reach_arr[-1]) if reach_errors_m else 0.0,
        "final_ee_pos":        final_ee_pos,
        "audio_ablate":        _AUDIO_ABLATE,
        "instruction":         episode_instruction,
        "intention_score":     intention_score,
        "intention_success":   intention_success,
        "legacy_intention_score": legacy_intention_score,
        "legacy_intention_success": legacy_intention_success,
        "exclusive_intention":  exclusive_intention,
        "consumed_steps":      int(step),
        "progress":            progress,
        "active_radio":        active_radio,
        "position_label":      position_label,
        "sled_confident_rate": round(sled_confident_rate, 3),
        "sled_n_steps":        n_steps,
        "sled_conf_thresh":    conf_thresh,
        "final_audio_snapshot": {
            "class_id":      audio_snap["class_id"].tolist(),
            "azimuth_deg":   audio_snap["azimuth_deg"].tolist(),
            "elevation_deg": audio_snap["elevation_deg"].tolist(),
            "confidence":    audio_snap["confidence"].tolist(),
        },
    }

    # Robot motion vs. sound onset. For take_out_microwave_food the cue is the
    # delayed chime (microwave_trigger_step); for the radio tasks the sound is
    # present from the first step, so onset_step = 0. `moved_after_sound` tells
    # us whether the robot stayed put until the sound and only then moved.
    motion_step_dt = float(getattr(cm, "step_dt_sec", 0.1))
    if is_microwave_task and microwave_trigger_step is not None:
        sound_onset_step_m = int(microwave_trigger_step)
    else:
        sound_onset_step_m = 0
    sound_onset_sec_m = sound_onset_step_m * motion_step_dt

    def _motion_entry(fms):
        # Motion-onset timing relative to the sound, for one threshold.
        if fms is None:
            return {
                "moved":             False,
                "first_motion_step": None,
                "first_motion_sec":  None,
                "moved_after_sound": None,
                "motion_lag_sec":    None,
            }
        fsec = fms * motion_step_dt
        return {
            "moved":             True,
            "first_motion_step": fms,
            "first_motion_sec":  round(fsec, 3),
            "moved_after_sound": bool(fms >= sound_onset_step_m),
            "motion_lag_sec":    round(fsec - sound_onset_sec_m, 3),
        }

    by_threshold = {
        f"{thr:g}": _motion_entry(first_motion_step_by_thr[thr])
        for thr in motion_thresholds
    }
    primary_entry = _motion_entry(first_motion_step_by_thr.get(motion_threshold))
    info["motion_timing"] = {
        "motion_threshold_m":    motion_threshold,
        "max_ee_displacement_m": round(max_ee_displacement, 4),
        "sound_onset_step":      sound_onset_step_m,
        "sound_onset_sec":       round(sound_onset_sec_m, 3),
        **primary_entry,
        "by_threshold":          by_threshold,
    }

    if getattr(args, "save_motion_trace", False):
        trace_dir = Path(args.save_dir) / "motion_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        with open(trace_dir / f"motion_trace_ep{episode_idx:03d}.json", "w") as f:
            json.dump({
                "episode_idx":     episode_idx,
                "step_dt_sec":     motion_step_dt,
                "sound_onset_step": sound_onset_step_m,
                "sound_onset_sec":  round(sound_onset_sec_m, 3),
                "trace":           ee_disp_trace,   # [[step, disp_m], ...]
            }, f)

    if is_microwave_task:
        step_dt = float(getattr(cm, "step_dt_sec", 0.1))
        trigger_step = int(microwave_trigger_step) if microwave_trigger_step is not None else None
        sound_onset_sec = (trigger_step * step_dt) if trigger_step is not None else None
        contact_sec = (first_contact_step * step_dt) if first_contact_step is not None else None
        if contact_sec is not None and sound_onset_sec is not None:
            react_lag_sec = contact_sec - sound_onset_sec
        else:
            react_lag_sec = None
        info["microwave_timing"] = {
            "step_dt_sec":               step_dt,
            "sound_onset_step":          trigger_step,
            "sound_onset_sec":           sound_onset_sec,
            "first_microwave_contact_step": first_contact_step,
            "first_microwave_contact_sec":  contact_sec,
            "react_lag_sec":             react_lag_sec,
        }

        def _step_to_sec(s):
            return (s * step_dt) if s is not None else None

        info["microwave_subtasks"] = {
            "door_opened":          door_opened_step is not None,
            "door_opened_step":     door_opened_step,
            "door_opened_sec":      _step_to_sec(door_opened_step),
            "food_grasped":         food_grasped_step is not None,
            "food_grasped_step":    food_grasped_step,
            "food_grasped_sec":     _step_to_sec(food_grasped_step),
            "food_on_tray":         food_on_tray_step is not None,
            "food_on_tray_step":    food_on_tray_step,
            "food_on_tray_sec":     _step_to_sec(food_on_tray_step),
        }
        # Count of subtasks completed (0–3), handy for quick triage.
        info["microwave_subtasks_completed"] = sum(
            int(v) for v in (
                door_opened_step is not None,
                food_grasped_step is not None,
                food_on_tray_step is not None,
            )
        )
    if sound_meta is not None:
        info["sound"] = {
            "folder":     sound_meta["folder"],
            "class_id":   sound_meta["class_id"],
            "class_name": sound_meta["class_name"],
        }
    if other_sound_meta is not None:
        info["other_sound"] = {
            "folder":     other_sound_meta["folder"],
            "class_id":   other_sound_meta["class_id"],
            "class_name": other_sound_meta["class_name"],
            "radio":      other_radio,
            "position":   other_position_label,
        }
    if sounding_sources:
        info["silent_target"] = True
        info["sounding_sources"] = [
            {
                "radio": src["radio"],
                "position": src["position"],
                "folder": src["sound_meta"]["folder"],
                "class_id": src["sound_meta"]["class_id"],
                "class_name": src["sound_meta"]["class_name"],
            }
            for src in sounding_sources
        ]
    if args.save_audio_log:
        # Dump the per-step audio_log to a sidecar file (it can be huge).
        log_dir = Path(args.save_dir) / "audio_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"audio_log_ep{episode_idx:03d}.json", "w") as f:
            json.dump(audio_log, f, indent=2)

    if args.save_video and frames:
        import mediapy
        vid_dir = Path(args.save_dir) / "videos"
        vid_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for f in frames:
            f = np.asarray(f)
            out.append(np.vstack([np.hstack(f[:2]), np.hstack(f[2:4])]))
        # tag the filename with the ground-truth target position (e.g. left_top)
        _pos_tag = str(position_label) if position_label else "na"
        vid = vid_dir / f"ep_{episode_idx:03d}_pos_{_pos_tag}_success_{success}.mp4"
        try:
            # qp=None + explicit bitrate avoids mediapy's default `-qp` flag,
            # which the ffmpeg shipped in some envs (e.g. the openpi uv venv's
            # anaconda ffmpeg) rejects with "Unrecognized option 'qp'",
            # silently dropping every video. Targeting a bitrate instead works
            # across both that build and the imageio-ffmpeg bundled binary.
            mediapy.write_video(str(vid), out, fps=10, qp=None, bps=int(8e6))
        except Exception as e:
            print(f"  [warn] video save failed: {e}", flush=True)

        # Mux an audio track onto the silent mp4.
        #   - take_out_microwave_food: always synthesize a sim-time-aligned
        #     chime so oracle-mode runs (no binaural engine) still produce
        #     an audible cue at the right second.
        #   - other tasks: only mux when --save-audio captured a real
        #     binaural recording; trim the warmup prefix so the chime
        #     lines up with roughly the right env step.
        audio_for_mux = None
        audio_offset = 0.0
        if is_microwave_task and vid.exists() and microwave_trigger_step is not None:
            step_dt = float(getattr(cm, "step_dt_sec", 0.1))
            video_total_sec = max(0.1, float(step) * step_dt)
            chime_sec = float(microwave_trigger_step) * step_dt
            synth_wav = Path(args.save_dir) / "audio" / f"audio_eval_{episode_idx}_synth.wav"
            if _synthesize_microwave_chime_wav(
                synth_wav, chime_sec=chime_sec, total_sec=video_total_sec
            ):
                audio_for_mux = synth_wav
        elif args.save_audio and vid.exists():
            wav_path = Path(args.save_dir) / "audio" / f"audio_eval_{episode_idx}.wav"
            if wav_path.exists():
                audio_for_mux = wav_path
                audio_offset = float(getattr(args, "warmup_seconds", 0.0) or 0.0)
        if audio_for_mux is not None:
            muxed = vid_dir / f"ep_{episode_idx:03d}_pos_{_pos_tag}_success_{success}_with_audio.mp4"
            _mux_audio_into_video(
                vid, audio_for_mux, muxed, audio_offset_sec=audio_offset,
            )
    env.close()
    return info


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ckpt_step*.pt")
    ap.add_argument("--pretrained", default="lerobot/smolvla_vlabench",
                    help="base SmolVLA repo to instantiate the wrapper from "
                         "(weights are then overwritten by --ckpt)")
    ap.add_argument("--taxonomy", default=str(REPO_ROOT / "class_taxonomy.yaml"))
    ap.add_argument("--audio-config", default=str(
        REPO_ROOT / "audio_generation" / "scene_audio_config.json"))
    ap.add_argument("--sled-ckpt", default=
        "/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt")
    ap.add_argument("--task-name", default="select_radio")
    ap.add_argument("--robot", default="franka")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--max-episode-length", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=5,
                    help="how many actions to execute open-loop per chunk")
    ap.add_argument("--max-substeps", type=int, default=4)
    ap.add_argument("--intention-threshold", type=float, default=0.1,
                    help="Distance threshold for env.get_intention_score(). "
                         "In exclusive mode, the target button must be "
                         "approached within this threshold and distinctly "
                         "closer than non-target buttons.")
    ap.add_argument("--intention-mode", choices=["exclusive", "legacy"],
                    default="exclusive",
                    help="How to compute intention_success. 'exclusive' "
                         "requires the target button to be reached and at "
                         "least --exclusive-intention-margin closer than any "
                         "other button when the task supports it. 'legacy' "
                         "only checks whether the target button was reached.")
    ap.add_argument("--exclusive-intention-margin", type=float, default=0.03,
                    help="Minimum distance gap in meters required for "
                         "exclusive button intention: target_min_dist + "
                         "margin must be less than every non-target button's "
                         "minimum distance.")
    ap.add_argument("--warmup-seconds", type=float, default=3.0,
                    help="wall-clock seconds to record audio before SLED")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--save-dir", default=str(REPO_ROOT / "outputs/eval_smolvla_audio"))
    ap.add_argument("--save-video", action="store_true", default=False)
    ap.add_argument("--save-audio", action="store_true", default=False,
                    help="dump the per-episode binaural recording to outputs/<dir>/audio/")
    ap.add_argument("--save-audio-log", action="store_true", default=False,
                    help="dump per-step SLED snapshots to outputs/<dir>/audio_logs/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--revised-instruction", action="store_true", default=False,
                    help="Use the paraphrased radio-task instructions. Default "
                         "keeps the original training instructions.")
    ap.add_argument("--one-radio-instruction", default=_build_one_radio_instruction(),
                    help="Effective instruction for select_radio episodes.")
    ap.add_argument("--two-radio-instruction-template",
                    default=(
                        "primitive: Press the button in front of the radio "
                        "playing the {class_name} sound."
                    ),
                    help="Effective select_radio_two instruction template. "
                         "Supports {class_name} and {class_name_raw}.")
    ap.add_argument("--silent-radio-instruction",
                    default=_build_silent_radio_instruction(),
                    help="Effective instruction for select_radio_silent episodes.")
    # ---- Oracle mode ------------------------------------------------------
    ap.add_argument("--oracle-mode", action="store_true", default=False,
                    help="Skip SLED; compute top-K directly from ground-truth "
                         "source/camera positions in the env and add Gaussian "
                         "noise. Use this to isolate policy learning from SLED "
                         "perception quality.")
    ap.add_argument("--noise-az-std", type=float, default=3.0)
    ap.add_argument("--noise-el-std", type=float, default=5.0)
    ap.add_argument("--noise-conf-min", type=float, default=0.85)
    ap.add_argument("--noise-conf-max", type=float, default=0.98)
    ap.add_argument("--noise-class-flip-prob", type=float, default=0.02)
    ap.add_argument("--noise-n-classes", type=int, default=38)
    ap.add_argument("--noise-source-drop-prob", type=float, default=0.0,
                    help="Drop each real oracle source with this probability, "
                         "simulating too few detected sources.")
    ap.add_argument("--noise-distractor-prob", type=float, default=0.0,
                    help="Fill each empty top-K slot with a phantom source with "
                         "this probability, simulating too many detections.")
    ap.add_argument("--noise-distractor-conf-max", type=float, default=0.15)
    # ---- Instruction ablation ---------------------------------------------
    ap.add_argument("--no-instruction", action="store_true", default=False,
                    help="Zero out the language-token attention mask so the "
                         "policy sees only image + audio + state. Used to "
                         "measure text-instruction bias.")
    # ---- Non-ML post-processing -------------------------------------------
    ap.add_argument("--class-smoothing-window", type=int, default=0,
                    help="If >0, run a per-slot majority-vote temporal "
                         "smoother on the top-K class IDs over the last N "
                         "frames before feeding them to the policy. Cheap "
                         "and structure-external way to suppress single-frame "
                         "class flips that survive the dataset-time noise. "
                         "Recommended: 5 for oracle eval (~2% flip), 7 for "
                         "real-SLED eval. 0 = disabled (raw passthrough).")
    ap.add_argument("--shuffle-audio-slots", type=str, default="off",
                    choices=["on", "off"],
                    help="Randomize the order of present audio slots before "
                         "feeding them to the policy. This matches unordered "
                         "multi-source SELD output and is a no-op for "
                         "single-source snapshots.")
    ap.add_argument("--target-first-audio-slots", type=str, default="off",
                    choices=["on", "off"],
                    help="Ablation/backward-compatibility path: after optional "
                         "slot shuffling, move the instruction target class to "
                         "slot 0 before feeding SmolVLA. Keep off for the "
                         "standard two-radio benchmark.")
    ap.add_argument("--canonicalize-audio-slots", type=str, default="none",
                    choices=["none", "azimuth"],
                    help="Target-agnostic SELD slot ordering. 'azimuth' sorts "
                         "present detections from task-left to task-right.")
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

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "eval_args.json", "w") as f:
        eval_args = dict(vars(args))
        eval_args["instruction_metadata"] = _eval_instruction_metadata(args)
        json.dump(eval_args, f, indent=2)

    device = torch.device(args.device)

    print(f"[load] ckpt = {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # 1. Build policy from pretrained config, then load fine-tuned weights.
    # Prefer the checkpoint's saved audio_config so new audio-fusion variants
    # remain evaluable while older checkpoints still fall back to CLI defaults.
    audio_cfg_dict = {
        "taxonomy_path": args.taxonomy,
        "top_k": args.top_k,
    }
    audio_cfg_dict.update(ckpt.get("audio_config", {}))
    audio_cfg = AudioConfig(**audio_cfg_dict)
    print(f"[load] base = {args.pretrained}")
    print(f"[load] audio_config = {asdict(audio_cfg)}")
    policy = AudioAwareSmolVLAPolicy.from_pretrained_with_audio(
        args.pretrained, audio_config=audio_cfg
    )

    # The training script overrode these; mirror it so prepare_state /
    # tokenizer_max_length / chunk_size are consistent.
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

    policy.load_state_dict(ckpt["model_state_dict"], strict=True)

    policy.to(device)
    policy.eval()

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    # 2. Audio + SLED helpers (or oracle-mode noise config)
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
        )
        print(f"[oracle] eval noise ON  "
              f"az_std={oracle_noise_cfg.az_std_deg}° "
              f"el_std={oracle_noise_cfg.el_std_deg}° "
              f"conf~U[{oracle_noise_cfg.conf_min},{oracle_noise_cfg.conf_max}] "
              f"flip={oracle_noise_cfg.class_flip_prob} "
              f"drop={oracle_noise_cfg.source_drop_prob} "
              f"distractor={oracle_noise_cfg.distractor_prob}")
    else:
        AudioSimManagerCls, SLEDOverlayCls = _load_audio_modules(
            args.audio_config, args.sled_ckpt, args.task_name)
        if SLEDOverlayCls is not None:
            # realtime=True spawns a daemon thread that runs SLED at infer_hz Hz
            # on the most recent ~480 ms of audio from whichever engine is
            # currently `set_audio_engine`d to it. The thread keeps running
            # across episodes — we just rebind its audio engine each reset.
            sled_overlay = SLEDOverlayCls(
                ckpt_path=args.sled_ckpt,
                torch_device=str(device),
                realtime=True,
            )

    audio_classes, audio_label_map = _load_class_taxonomy()

    # Optional non-ML class smoother. Reused across episodes (its history
    # is reset at the start of each episode by `evaluate_episode`).
    class_smoother = None
    if args.class_smoothing_window and args.class_smoothing_window > 1:
        class_smoother = TopKClassSmoother(
            top_k=args.top_k,
            window=args.class_smoothing_window,
        )
        print(f"[smoother] class temporal smoothing ON "
              f"(window={args.class_smoothing_window}, top_k={args.top_k})")

    # 3. Evaluate episodes
    infos = []
    max_reset_retries = 3
    for i in tqdm(range(args.n_episodes), desc="eval"):
        info = None
        last_err: Exception | None = None
        # Retry PhysicsError (intermittent MuJoCo mjWARN_BADQACC at reset)
        # and BadInitialPlacement (food spawned outside the microwave or
        # the door settled open). Any other exception is a real bug and is
        # recorded as a failed episode on the first hit.
        for retry in range(max_reset_retries):
            try:
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

    # 4. Aggregate metrics
    n = len(infos)
    n_success = sum(1 for x in infos if x.get("success"))
    n_intention_success = sum(1 for x in infos if x.get("intention_success"))
    n_legacy_intention_success = sum(
        1 for x in infos if x.get("legacy_intention_success")
    )
    n_exclusive_intention_success = sum(
        1 for x in infos
        if (x.get("exclusive_intention") or {}).get("exclusive_success")
    )
    avg_progress = float(np.mean([x.get("progress", 0.0) for x in infos]))
    avg_intention_score = float(
        np.mean([x.get("intention_score", 0.0) for x in infos])
    )
    avg_legacy_intention_score = float(
        np.mean([x.get("legacy_intention_score", 0.0) for x in infos])
    )
    by_position = {
        p: {
            "n": 0,
            "succ": 0,
            "intention_succ": 0,
            "legacy_intention_succ": 0,
            "exclusive_intention_succ": 0,
        }
        for p in ["left", "middle", "right"]
    }
    for x in infos:
        p = x.get("position_label")
        if p in by_position:
            by_position[p]["n"]    += 1
            by_position[p]["succ"] += int(x.get("success", False))
            by_position[p]["intention_succ"] += int(
                x.get("intention_success", False)
            )
            by_position[p]["legacy_intention_succ"] += int(
                x.get("legacy_intention_success", False)
            )
            by_position[p]["exclusive_intention_succ"] += int(
                (x.get("exclusive_intention") or {}).get(
                    "exclusive_success", False
                )
            )

    # Subtask completion rates for take_out_microwave_food. Episodes from
    # other tasks won't carry the `microwave_subtasks` key, so these stay
    # at zero (and `microwave_n` reflects only the episodes that ran the
    # microwave task) — safe to leave in the summary unconditionally.
    mw_eps = [x.get("microwave_subtasks") for x in infos
              if x.get("microwave_subtasks") is not None]
    mw_n = len(mw_eps)

    def _mw_rate(key):
        return sum(int(bool(d.get(key))) for d in mw_eps) / max(1, mw_n)

    # Motion-vs-sound aggregation: of the episodes where the robot moved at
    # all, what fraction started moving only after the sound onset, and how
    # long after on average. `by_threshold` repeats this at each sensitivity.
    motion_summary = _summarize_motion(infos)

    summary = {
        "ckpt":            args.ckpt,
        "n_episodes":      n,
        "success_rate":    n_success / max(1, n),
        "intention_success_rate": n_intention_success / max(1, n),
        "avg_intention_score": avg_intention_score,
        "intention_mode": args.intention_mode,
        "intention_threshold": args.intention_threshold,
        "exclusive_intention_margin": args.exclusive_intention_margin,
        "legacy_intention_success_rate": (
            n_legacy_intention_success / max(1, n)
        ),
        "avg_legacy_intention_score": avg_legacy_intention_score,
        "exclusive_intention_success_rate": (
            n_exclusive_intention_success / max(1, n)
        ),
        "avg_progress":    avg_progress,
        "microwave_subtask_rates": {
            "n_microwave_episodes":  mw_n,
            "door_opened_rate":      _mw_rate("door_opened"),
            "food_grasped_rate":     _mw_rate("food_grasped"),
            "food_on_tray_rate":     _mw_rate("food_on_tray"),
        },
        "motion_vs_sound": motion_summary,
        "by_position":     {
            p: {
                **v,
                "rate": v["succ"] / max(1, v["n"]),
                "intention_rate": v["intention_succ"] / max(1, v["n"]),
                "legacy_intention_rate": (
                    v["legacy_intention_succ"] / max(1, v["n"])
                ),
                "exclusive_intention_rate": (
                    v["exclusive_intention_succ"] / max(1, v["n"])
                ),
            }
            for p, v in by_position.items()
        },
        "no_instruction":  args.no_instruction,
    }
    print("\n=== eval summary ===")
    print(json.dumps(summary, indent=2))

    with open(save_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(save_dir / "eval_infos.json", "w") as f:
        json.dump(infos, f, indent=2)
    print(f"\n[saved] {save_dir}/eval_summary.json")


if __name__ == "__main__":
    main()
