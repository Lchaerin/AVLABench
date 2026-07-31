"""Convert AVLABench select_radio HDF5 dataset → LeRobot dataset format.

Output format mirrors `lerobot/vlabench_unified` (the dataset that
`lerobot/smolvla_vlabench` was trained on) so the pretrained
state/action projections, camera embeddings and normalisation stats
remain meaningful at fine-tune time:

    observation.images.image         (224, 224, 3) video — front cam (cam_2)
    observation.images.second_image  (224, 224, 3) video — right cam (cam_0)
    observation.images.wrist_image   (224, 224, 3) video — wrist cam (cam_3)
    observation.state                (7,)          float32 [x, y, z, rx, ry, rz, gripper]
    action                           (7,)          float32 [x, y, z, rx, ry, rz, gripper_cmd]

Plus our audio additions (Path-A SLED features per timestep):

    observation.audio.azimuth_deg    (top_k,)      float32
    observation.audio.elevation_deg  (top_k,)      float32
    observation.audio.confidence     (top_k,)      float32
    observation.audio.class_id       (top_k,)      int32

The instruction is overridden to a *fixed* English prompt with the
"primitive: " prefix that vlabench_unified uses, so the audio modality
is the only signal disambiguating which radio to press:

    "primitive: Press the button in front of the radio that is making sound."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio.sled_features import parse_sled_json, temporal_smooth_topk  # noqa: E402
from src.audio.oracle_sled import (  # noqa: E402
    OracleNoiseConfig, build_oracle_topk,
)
from src.audio.projection import CameraIntrinsics, doa_to_uv  # noqa: E402


def _project_slot_uv(az_deg: float, el_deg: float,
                     intrinsics: CameraIntrinsics | None,
                     wh: tuple[int, int]) -> tuple[float, float]:
    """DoA→(u,v) for one slot. Returns (-1,-1) when off-screen/unavailable so
    the sentinel survives as a plain float column (the SlotEncoder path masks
    those slots out)."""
    if intrinsics is None:
        return (-1.0, -1.0)
    uv = doa_to_uv(az_deg, el_deg, intrinsics, wh, convention="sled")
    return uv if uv is not None else (-1.0, -1.0)


def _sort_slots_by_azimuth(cid: np.ndarray, az: np.ndarray, el: np.ndarray,
                           cf: np.ndarray, en: np.ndarray, uv: np.ndarray) -> None:
    """In-place per-frame reorder of the top_k audio slots so *present*
    sources are ordered task-left to task-right by azimuth (descending
    az_deg — positive/left first), matching the eval-time
    canonicalize_audio_slots="azimuth" convention. Padding slots (cid < 0)
    are left wherever they already are.

    Without this, upstream generation stores select_radio_two sources as
    active_sources[0]=target, [1]=distractor "by dataset convention" (see
    scripts/trajectory_generation.py), which would otherwise leak straight
    into observation.audio.* and let the model learn a "always attend slot
    0" positional shortcut instead of matching the instruction's class name.
    Sorting by azimuth here — a property any real SELD system reports
    regardless of correctness — removes that shortcut.
    """
    n_frames, top_k = cid.shape
    if top_k < 2:
        return
    for t in range(n_frames):
        present = np.flatnonzero(cid[t] >= 0)
        if present.shape[0] < 2:
            continue
        ordered = present[np.argsort(-az[t, present])]
        if np.array_equal(present, ordered):
            continue
        cid[t, present] = cid[t, ordered]
        az[t, present]  = az[t, ordered]
        el[t, present]  = el[t, ordered]
        cf[t, present]  = cf[t, ordered]
        en[t, present]  = en[t, ordered]
        uv[t, present]  = uv[t, ordered]


# vlabench_unified uses this exact prefix; keep it identical so the pretrained
# tokenizer + LLM see the same surface form they were fine-tuned on.
DEFAULT_INSTRUCTION = (
    "primitive: Press the button in front of the radio that is making sound."
)
POSITIONAL_SELECT_RADIO_INSTRUCTIONS = {
    "Please press the button in front of the left radio.",
    "Please press the button in front of the middle radio.",
    "Please press the button in front of the right radio.",
}

# Camera index mapping inside our HDF5's `observation.rgb[T, ncam, H, W, 3]`.
# VLABench enumerates cameras in XML order:
#   cam_0 = "right"   (table-side view)
#   cam_1 = "left"    (table-side view)
#   cam_2 = "forward" (front-of-table view, used as the audio-pipeline listener)
#   cam_3 = "wrist_cam" (mounted on the Franka end-effector)
DEFAULT_CAM_MAP = {"image": 2, "second_image": 0, "wrist_image": 3}

GRIPPER_OPEN_THRESHOLD = 0.02   # finger position above this → "open" command


def _resize_uint8(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize an HWC uint8 image to the target (H, W). No-op if same size."""
    h, w = hw
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# Origin of the robot base frame that *eval* uses, i.e. the value returned by
# ``env.get_robot_frame_position()``. Read from the live sim for
# find_hidden_object_open / select_radio; it is a scene constant. Both the state
# and the action written here must be expressed relative to it, because
# ``eval_smolvla_audio._apply_action`` adds it back to recover a world target.
ROBOT_FRAME_POS = np.array([0.0, -0.7, 0.7], dtype=np.float32)


def _binarize_gripper(g: float) -> float:
    return 1.0 if g > GRIPPER_OPEN_THRESHOLD else 0.0


def _compute_state_action(action_8d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert AVLABench's 8-dim action to vlabench_unified's 7-dim state/action.

    Our 8-dim action layout is `[x, y, z, rx, ry, rz, g_left, g_right]` (the
    last two finger positions are always equal). vlabench_unified collapses
    this to `[x, y, z, rx, ry, rz, gripper_cmd]` with gripper binarised.

    Following vlabench_unified's convention (state ≈ executed pose, action =
    target), we use the same 6-D pose for state and action; this is a small
    simplification because we lack a separately recorded ee_pose feedback
    stream — at 10 fps the policy's commanded pose is a close stand-in for
    where the robot actually is.
    """
    # Imported lazily: VLABench.utils.utils pulls in open3d/sklearn, which the
    # conversion venv does not necessarily have loaded at import time.
    from VLABench.utils.utils import fold_roll_to_negative_branch

    T = action_8d.shape[0]
    out_action = np.zeros((T, 7), dtype=np.float32)
    out_action[:, :6] = action_8d[:, :6]
    out_action[:, 3] = fold_roll_to_negative_branch(out_action[:, 3])
    for t in range(T):
        out_action[t, 6] = _binarize_gripper(float(action_8d[t, 6]))
    out_state = out_action.copy()
    return out_state, out_action


def _compute_real_state(action_8d: np.ndarray,
                        ee_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the 7-D state from the *measured* end-effector stream, mirroring
    the eval-time state construction exactly, while keeping the 7-D action as
    the commanded target.

    Rationale
    ---------
    The default `_compute_state_action` sets ``state == action`` (the commanded
    target pose). At eval time `observation.state` is instead the *achieved* EE
    pose (`env.robot.get_ee_state` → base frame, see
    `eval_smolvla_audio._build_policy_batch`). Training on ``state == action``
    lets the policy shortcut ``action[0] = state``; that then misfires at eval
    because the achieved pose lags the command. Using the recorded
    ``observation/ee_state`` removes this train/eval distribution gap.

    Conventions (must match eval byte-for-byte)
    -------------------------------------------
    * ``ee_state`` layout (VLABench ``franka.get_ee_state``):
      ``[x, y, z (world), quat_wxyz (4), open_flag (1)]`` where ``open_flag`` is
      ``get_ee_open_state`` which — due to a known upstream sign bug — returns
      ``1.0`` when the gripper is **closed**. Eval inverts it
      (``not get_ee_open_state``) so ``state[6] == 1.0`` means **open**; we
      mirror that with ``1.0 - open_flag``.
    * Position is world-frame; eval subtracts ``get_robot_frame_position()``.
      That base offset is a scene constant we don't have offline, so we
      estimate it per episode as ``median(ee_world - commanded_base_pos)``.
      Empirically this is stable to ~mm across episodes and self-cancels,
      leaving ``state == action + (real tracking error)`` — exactly the
      eval-time signal.
    * Euler uses the same ``quaternion_to_euler`` eval calls, so the state's
      orientation encoding is identical in train and eval regardless of how the
      action euler is stored.
    """
    from VLABench.utils.utils import (fold_roll_to_negative_branch,
                                      quaternion_to_euler)

    T = action_8d.shape[0]
    out_action = np.zeros((T, 7), dtype=np.float32)
    out_action[:, :6] = action_8d[:, :6]
    # Roll lives on the (-pi, pi] branch cut for this scene's top-down home
    # pose; fold it to a single branch before anything downstream differences
    # it. Must match _build_policy_batch in src/eval/eval_smolvla_audio.py.
    out_action[:, 3] = fold_roll_to_negative_branch(out_action[:, 3])
    for t in range(T):
        out_action[t, 6] = _binarize_gripper(float(action_8d[t, 6]))

    n = min(T, int(ee_state.shape[0]))
    ee = np.asarray(ee_state[:n], dtype=np.float32)
    ee_pos_world = ee[:, :3]
    # Origin of the frame the *recorded actions* live in, estimated from the
    # data. This is NOT the frame eval uses (see ROBOT_FRAME_POS below).
    action_frame_origin = np.median(ee_pos_world - out_action[:n, :3], axis=0)

    # Re-express BOTH state and action in eval's base frame. Measured on
    # find_hidden_object_open: the recorded actions sit at world origin
    # (0.002, -0.412, 0.774) while eval's env.get_robot_frame_position() is
    # (0, -0.700, 0.700) — a 28.8 cm / 7.4 cm gap in y / z. Training on the
    # un-shifted data hands the policy a state that claims the arm is already
    # 80 % through the reach, so it barely advances and stalls. See
    # eval_smolvla_audio._apply_action, which reconstructs pos_world as
    # pos_base + get_robot_frame_position().
    shift = (action_frame_origin - ROBOT_FRAME_POS).astype(np.float32)
    out_action[:, :3] += shift

    out_state = out_action.copy()  # tail frames (if ee shorter) fall back to cmd
    for t in range(n):
        pos_base = ee_pos_world[t] - ROBOT_FRAME_POS
        euler = np.asarray(quaternion_to_euler(ee[t, 3:7]), dtype=np.float32)
        euler[0] = fold_roll_to_negative_branch(euler[0])
        gripper_open = 1.0 - float(ee[t, 7])   # invert buggy closed-flag → open=1
        out_state[t] = np.concatenate([pos_base, euler, [gripper_open]]).astype(np.float32)
    return out_state, out_action


def _build_features(top_k: int, image_hw: tuple[int, int], use_video: bool,
                    cam_keys: list[str]) -> dict:
    h, w = image_hw
    img_dtype = "video" if use_video else "image"
    features = {}
    for key in cam_keys:
        features[f"observation.images.{key}"] = {
            "dtype": img_dtype,
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    features.update({
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_cmd"],
        },
        "observation.audio.azimuth_deg": {
            "dtype": "float32",
            "shape": (top_k,),
            "names": [f"slot_{k}" for k in range(top_k)],
        },
        "observation.audio.elevation_deg": {
            "dtype": "float32",
            "shape": (top_k,),
            "names": [f"slot_{k}" for k in range(top_k)],
        },
        "observation.audio.confidence": {
            "dtype": "float32",
            "shape": (top_k,),
            "names": [f"slot_{k}" for k in range(top_k)],
        },
        "observation.audio.class_id": {
            "dtype": "int32",
            "shape": (top_k,),
            "names": [f"slot_{k}" for k in range(top_k)],
        },
        # SLED now also reports a per-source loudness/energy in [0,1]; the
        # SlotEncoder consumes it alongside the projected image coordinate.
        "observation.audio.energy": {
            "dtype": "float32",
            "shape": (top_k,),
            "names": [f"slot_{k}" for k in range(top_k)],
        },
        # Projected image coordinate (u, v) ∈ [0,1]² of each slot's DoA on the
        # fixed listener camera (spec M2). (-1, -1) = off-screen / unavailable.
        "observation.audio.uv": {
            "dtype": "float32",
            "shape": (top_k, 2),
            "names": [f"slot_{k}" for k in range(top_k)],
        },
    })
    return features


def _open_episode(h5_path: Path):
    f = h5py.File(h5_path, "r")
    grp = f["data"]
    ts_keys = list(grp.keys())
    if not ts_keys:
        f.close()
        raise RuntimeError(f"empty data group in {h5_path}")
    return f, grp[ts_keys[0]]


def _normalize_episode_instruction(text: str) -> str:
    """Prevent old one-radio positional prompts from leaking into training."""
    if text in POSITIONAL_SELECT_RADIO_INSTRUCTIONS:
        return DEFAULT_INSTRUCTION
    return text


def convert(
    src_dir: Path,
    out_dir: Path,
    repo_id: str,
    fps: int = 10,
    top_k: int = 3,
    image_hw: tuple[int, int] = (224, 224),
    use_video: bool = True,
    vcodec: str = "h264",
    instruction: str = DEFAULT_INSTRUCTION,
    smooth_lookback_frames: int = 25,
    cam_map: dict[str, int] = None,
    oracle_mode: bool = False,
    use_episode_instruction: bool = False,
    use_real_state: bool = False,
):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    cam_map = dict(cam_map or DEFAULT_CAM_MAP)
    cam_keys = list(cam_map.keys())

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(
            f"out-dir {out_dir} already exists; LeRobotDataset.create requires a fresh path"
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(src_dir.glob("data_*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"no data_*.hdf5 in {src_dir}")

    features = _build_features(top_k, image_hw, use_video, cam_keys)

    import inspect as _inspect
    _create_kwargs = dict(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=str(out_dir),
        robot_type="franka",
        use_videos=use_video,
        image_writer_threads=4,
    )
    if "vcodec" in _inspect.signature(LeRobotDataset.create).parameters:
        _create_kwargs["vcodec"] = vcodec
    ds = LeRobotDataset.create(**_create_kwargs)

    n_eps = 0
    cam_intrinsics_meta: dict | None = None
    for h5_path in tqdm(h5_files, desc="episodes"):
        try:
            f, ep = _open_episode(h5_path)
        except Exception as exc:
            print(f"  skip {h5_path.name}: {exc}")
            continue

        try:
            rgb = ep["observation/rgb"][...]               # [T, ncam, H, W, 3] uint8
            act = ep["action"][...]                        # [T, 8]
            n_frames = int(rgb.shape[0])

            # Camera index sanity
            for key, idx in cam_map.items():
                if idx >= rgb.shape[1]:
                    raise ValueError(
                        f"cam_map['{key}'] = {idx} but only {rgb.shape[1]} cams in HDF5"
                    )

            # Convert to 7-dim state + action. When --use-real-state is set we
            # build the state from the recorded measured EE pose so it matches
            # the eval-time state (achieved pose) instead of the commanded
            # target; otherwise keep the legacy state == action convention.
            if use_real_state:
                if "observation/ee_state" not in ep:
                    raise RuntimeError(
                        f"--use-real-state but {h5_path.name} has no "
                        f"observation/ee_state stream"
                    )
                ee_state = ep["observation/ee_state"][...]   # [T, 8]
                state_7d, action_7d = _compute_real_state(act, ee_state)
            else:
                state_7d, action_7d = _compute_state_action(act)

            # ---- Audio features (top-K events per frame) -----------------
            # Two sources depending on how the episode was recorded:
            #   * oracle_mode: meta_info/oracle_audio holds clean GT (fast
            #     path; no real audio was ever synthesised). We fill every
            #     frame with the same GT values and confidence=1.0 so that
            #     training-time noise injection can draw noise freely.
            #   * else: meta_info/sled_predictions holds real per-frame
            #     SLED outputs — parse + temporal-smooth as before.
            if oracle_mode:
                if "meta_info/oracle_audio" not in ep:
                    raise RuntimeError(
                        f"--oracle-mode but {h5_path.name} has no "
                        f"meta_info/oracle_audio group. Re-run "
                        f"trajectory_generation.py with --oracle-mode."
                    )
                import json as _json
                raw = ep["meta_info/oracle_audio"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                odict = _json.loads(raw)
                if not odict.get("static", True):
                    raise NotImplementedError(
                        "oracle_audio with static=false is not yet supported "
                        "by the converter (needs per-frame cam/source xpos)"
                    )
                # active_sources[0] = target, [1] = distractor (raw generation-time
                # convention — see trajectory_generation.py). Store up to top_k so
                # the network sees both direction slots, but this target-first
                # order is NOT what gets written below: _sort_slots_by_azimuth()
                # re-orders the populated arrays by azimuth right after this
                # block, so the network can only learn to match the instruction's
                # class name to the correct direction, not shortcut on slot index.
                srcs = odict.get("active_sources", [])[:top_k]
                n_real = len(srcs)
                cid = np.full((n_frames, top_k), -1, dtype=np.int32)
                az  = np.zeros((n_frames, top_k), dtype=np.float32)
                el  = np.zeros((n_frames, top_k), dtype=np.float32)
                cf  = np.zeros((n_frames, top_k), dtype=np.float32)
                en  = np.zeros((n_frames, top_k), dtype=np.float32)
                uv  = np.full((n_frames, top_k, 2), -1.0, dtype=np.float32)
                # Camera intrinsics for the DoA→(u,v) projection: built from the
                # listener camera's vertical FOV (stored by extract_episode_gt).
                fovy = odict.get("cam_fovy_deg")
                intr = (CameraIntrinsics.from_fovy(fovy, image_hw[1], image_hw[0])
                        if fovy is not None else None)
                if cam_intrinsics_meta is None and intr is not None:
                    cam_intrinsics_meta = {
                        "cam_fovy_deg": float(fovy),
                        "image_wh": [int(image_hw[1]), int(image_hw[0])],
                        "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
                        "seld_convention": "sled",  # stored az is right-positive
                    }
                for k, src in enumerate(srcs):
                    # active_from_frame gates *temporal* activation per slot.
                    # 0 = active for the entire episode (radio tasks, default).
                    # >0 = silent in [0, afe), active in [afe, n_frames) — used
                    # by delayed-cue tasks like take_out_microwave_food.
                    afe = int(src.get("active_from_frame", 0))
                    afe = max(0, min(afe, n_frames))
                    az_k = float(src["az_deg"])
                    el_k = float(src["el_deg"])
                    cid[afe:, k] = int(src["class_id"])
                    az[afe:, k]  = az_k
                    el[afe:, k]  = el_k
                    cf[afe:, k]  = 1.0   # "GT is certain"; noise applied at train time
                    en[afe:, k]  = float(src.get("energy", 1.0))
                    u_v = _project_slot_uv(az_k, el_k, intr, (image_hw[1], image_hw[0]))
                    uv[afe:, k, 0] = u_v[0]
                    uv[afe:, k, 1] = u_v[1]
                # No temporal smoothing needed — GT is already dense and stable.
            else:
                sled_json = ep["meta_info/sled_predictions"][()]
                if isinstance(sled_json, bytes):
                    sled_json = sled_json.decode("utf-8")
                cid, az, el, cf = parse_sled_json(sled_json, n_frames=n_frames, top_k=top_k)
                cid, az, el, cf = temporal_smooth_topk(
                    cid, az, el, cf, lookback_frames=smooth_lookback_frames
                )
                # Real-SLED energy/uv are not yet emitted by the parser; use
                # confidence as a loudness proxy and mark uv unavailable. The
                # oracle path (used for current training) fills both properly.
                en = cf.astype(np.float32)
                uv = np.full((n_frames, top_k, 2), -1.0, dtype=np.float32)

            # Remove any correctness-linked slot order (e.g. oracle's
            # target-always-slot-0 convention) before it reaches the model.
            _sort_slots_by_azimuth(cid, az, el, cf, en, uv)

            # Per-episode instruction override: read the string actually saved
            # alongside the trajectory (e.g. select_radio_two stores a
            # class-specific prompt). Falls back to the static `instruction`
            # arg when the episode has no instruction record.
            ep_instruction = instruction
            if use_episode_instruction:
                try:
                    raw_instr = ep["instruction"][()]
                except KeyError:
                    raw_instr = None
                if raw_instr is not None:
                    if isinstance(raw_instr, np.ndarray):
                        raw_instr = raw_instr.tolist()
                    if isinstance(raw_instr, list):
                        raw_instr = raw_instr[0] if raw_instr else None
                    if isinstance(raw_instr, bytes):
                        raw_instr = raw_instr.decode("utf-8")
                    if raw_instr:
                        ep_instruction = _normalize_episode_instruction(str(raw_instr))

            for t in range(n_frames):
                frame = {
                    "observation.state": state_7d[t],
                    "action":            action_7d[t],
                    "observation.audio.azimuth_deg":  az[t].astype(np.float32),
                    "observation.audio.elevation_deg":el[t].astype(np.float32),
                    "observation.audio.confidence":   cf[t].astype(np.float32),
                    "observation.audio.class_id":     cid[t].astype(np.int32),
                    "observation.audio.energy":       en[t].astype(np.float32),
                    "observation.audio.uv":           uv[t].astype(np.float32),
                    "task":                           ep_instruction,
                }
                for key, cam_idx in cam_map.items():
                    img = np.ascontiguousarray(rgb[t, cam_idx])
                    img = _resize_uint8(img, image_hw)
                    frame[f"observation.images.{key}"] = img
                ds.add_frame(frame)

            ds.save_episode()
            n_eps += 1
        except Exception as exc:
            print(f"  failed {h5_path.name}: {exc}")
        finally:
            f.close()

    # Persist a tiny marker so downstream tools (training, eval) can
    # auto-detect oracle datasets without an extra CLI flag.
    if oracle_mode:
        import json as _json
        with open(out_dir / "oracle_mode.json", "w") as f:
            _json.dump({
                "oracle_mode":       True,
                "top_k":             top_k,
                "n_episodes":        n_eps,
                "instruction":       instruction,
                "camera_intrinsics": cam_intrinsics_meta,
            }, f, indent=2)

    print(f"Wrote {n_eps} episodes to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="dataset_v5/select_radio")
    ap.add_argument("--out-dir", required=True, help="LeRobot dataset root")
    ap.add_argument("--repo-id", default="local/avla_select_radio")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--image-h", type=int, default=224)
    ap.add_argument("--image-w", type=int, default=224)
    ap.add_argument("--use-video", action="store_true", default=True)
    ap.add_argument("--no-video", dest="use_video", action="store_false")
    ap.add_argument("--vcodec", default="h264",
                    help="ffmpeg vcodec; libsvtav1 if installed, else h264")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--cam-image",        type=int, default=DEFAULT_CAM_MAP["image"])
    ap.add_argument("--cam-second-image", type=int, default=DEFAULT_CAM_MAP["second_image"])
    ap.add_argument("--cam-wrist-image",  type=int, default=DEFAULT_CAM_MAP["wrist_image"])
    ap.add_argument("--oracle-mode", action="store_true", default=False,
                    help="Read meta_info/oracle_audio instead of sled_predictions "
                         "and store clean GT into observation.audio.*. Noise is "
                         "injected at training time (see --oracle-noise in trainer).")
    ap.add_argument("--use-episode-instruction", action="store_true", default=False,
                    help="Read each episode's saved `instruction` field from HDF5 "
                         "instead of writing the static --instruction. Required for "
                         "tasks like select_radio_two where the instruction varies "
                         "per episode (it names the target sound class).")
    ap.add_argument("--use-real-state", action="store_true", default=False,
                    help="Build observation.state from the recorded measured EE "
                         "pose (observation/ee_state) instead of the commanded "
                         "target. This removes the train/eval state distribution "
                         "gap (eval state = achieved pose, not command). Requires "
                         "re-training. Default off for backward compatibility; "
                         "smoke-test one episode against eval state before scaling.")
    args = ap.parse_args()

    convert(
        src_dir=Path(args.src_dir),
        out_dir=Path(args.out_dir),
        repo_id=args.repo_id,
        fps=args.fps,
        top_k=args.top_k,
        image_hw=(args.image_h, args.image_w),
        use_video=args.use_video,
        vcodec=args.vcodec,
        instruction=args.instruction,
        cam_map={
            "image":        args.cam_image,
            "second_image": args.cam_second_image,
            "wrist_image":  args.cam_wrist_image,
        },
        oracle_mode=args.oracle_mode,
        use_episode_instruction=args.use_episode_instruction,
        use_real_state=args.use_real_state,
    )


if __name__ == "__main__":
    main()
