"""
The scripts to launch auto scene load and key-point based trajectory generation.
"""
import numpy as np
import os
import random
import json
import open3d as o3d
import mediapy
import argparse
import traceback
import yaml
from dm_control import viewer
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from VLABench.robots import Franka
from VLABench.tasks import *
from VLABench.utils.data_utils import save_single_data, process_observations
from VLABench.utils.utils import find_key_by_value, get_logger
from VLABench.envs import load_env
from VLABench.utils.skill_lib import SkillLib
from VLABench.configs import name2config

_SOUD_EFFECTS_DIR  = "/home/rllab/Desktop/crossCorr/soud_effects"
_CLASS_TAXONOMY    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "class_taxonomy.yaml"
)
POSITION_LABELS = ["left", "middle", "right"]


def _load_class_taxonomy():
    """Return (classes: {id→name}, label_map: {fsd_label→id}) or empty dicts."""
    if not os.path.exists(_CLASS_TAXONOMY):
        return {}, {}
    with open(_CLASS_TAXONOMY) as f:
        data = yaml.safe_load(f)
    classes   = {int(k): v["name"] for k, v in data.get("classes", {}).items()}
    label_map = data.get("label_map", {})
    return classes, label_map


def _folder_to_class(folder, label_map, classes):
    """Map a soud_effects folder name to (class_id, class_name)."""
    if folder in label_map:
        cid = label_map[folder]
        return cid, classes.get(cid, "Unknown")
    s = folder.replace("_and_", ", ").replace("_", " ")
    if s in label_map:
        cid = label_map[s]
        return cid, classes.get(cid, "Unknown")
    s_lower = s.lower()
    for k, v in label_map.items():
        if k.lower() == s_lower:
            return v, classes.get(v, "Unknown")
    return -1, "Unknown"


def _pick_random_sound(label_map, classes, exclude_class_ids=None, force_class_id=None):
    """
    Randomly pick a WAV from SOUD_EFFECTS_DIR whose folder maps to a known
    SLED class (class_id >= 0).  Returns dict with keys:
    sound_file, folder, class_id, class_name.

    If `exclude_class_ids` is provided, candidates whose class_id is in
    that iterable are skipped — used by the 2-radio task to guarantee the
    two radios play sounds from *different* classes.

    If `force_class_id` is provided, only candidates from that exact class
    are considered (the file within the class is still picked at random) —
    used to balance the per-class distribution when topping up a dataset.
    """
    excluded = set(exclude_class_ids or [])
    candidates = []
    for root, _, files in os.walk(_SOUD_EFFECTS_DIR):
        folder = os.path.basename(root)
        cid, cname = _folder_to_class(folder, label_map, classes)
        if cid < 0 or cid in excluded:
            continue  # skip folders not in the taxonomy or already-used class
        if force_class_id is not None and cid != force_class_id:
            continue
        for f in files:
            if f.lower().endswith(".wav"):
                candidates.append((os.path.join(root, f), folder, cid, cname))
    if not candidates:
        return None
    chosen_path, folder, cid, cname = random.choice(candidates)
    return {"sound_file": chosen_path, "folder": folder, "class_id": cid, "class_name": cname}


def _forced_class_id(classes):
    """Optional target-sound-class override (VLABENCH_TARGET_CLASS_NAME) so
    the generator can balance the per-class distribution, mirroring
    VLABENCH_TARGET_POSITION_LABEL for direction balancing."""
    forced_name = os.environ.get("VLABENCH_TARGET_CLASS_NAME")
    if not forced_name:
        return None
    for cid, cname in classes.items():
        if cname == forced_name:
            return cid
    raise ValueError(
        f"VLABENCH_TARGET_CLASS_NAME={forced_name!r} not found in class_taxonomy.yaml"
    )


def _natural_class_name(class_name: str) -> str:
    """Convert taxonomy class names like 'Car_Automobile' to 'car automobile'
    so the LLM sees a more natural instruction."""
    if not class_name:
        return ""
    return class_name.replace("_", " ").lower()


def _build_two_radio_instruction(class_name: str) -> str:
    return (
        "primitive: Press the button in front of the radio playing the "
        f"{_natural_class_name(class_name)} sound."
    )


def _build_silent_radio_instruction() -> str:
    return "primitive: Press the button in front of the radio that is silent."


def _build_one_radio_instruction() -> str:
    return "primitive: Press the button in front of the radio that is making sound."


def _compute_radio_az_el(env, radio_name, cam_id):
    """
    Compute true (azimuth_deg, elevation_deg, distance_m) of radio_name
    relative to camera cam_id using the listener-relative direction helper.
    """
    try:
        import sys
        _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _proj not in sys.path:
            sys.path.insert(0, _proj)
        from audio_generation.binaural_engine import compute_listener_relative_direction
        cam_pos  = env.physics.data.cam_xpos[cam_id].copy()
        cam_xmat = env.physics.data.cam_xmat[cam_id].copy()
        entity = env.task.entities.get(radio_name)
        if entity is None:
            return None, None, None
        obj_pos = entity.get_xpos(env.physics).copy()
        az, el, dist = compute_listener_relative_direction(cam_pos, cam_xmat, obj_pos)
        return float(np.degrees(az)), float(np.degrees(el)), float(dist)
    except Exception:
        return None, None, None


def _build_sled_dict(sled_preds, audio_duration, actual_fps, classes):
    """Build a JSON-serialisable dict from per-frame SLED predictions."""
    frame_entries = []
    for i, snap in enumerate(sled_preds):
        t_sec = round(i / actual_fps, 5)
        preds = []
        if snap is not None:
            doa, conf, cls = snap
            for s in range(doa.shape[0]):
                dx, dy, dz = float(doa[s, 0]), float(doa[s, 1]), float(doa[s, 2])
                az = float(np.degrees(np.arctan2(dy, dx)))
                el = float(np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2))))
                cid = int(cls[s])
                preds.append({
                    "azimuth_deg":   round(az, 3),
                    "elevation_deg": round(el, 3),
                    "doa_vec":       [round(dx, 5), round(dy, 5), round(dz, 5)],
                    "class_id":      cid,
                    "class_name":    classes.get(cid, f"cls{cid}"),
                    "confidence":    round(float(conf[s]), 5),
                })
        frame_entries.append({"frame_idx": i, "time_sec": t_sec, "predictions": preds})
    return {
        "n_frames":           len(sled_preds),
        "audio_duration_sec": round(audio_duration, 4),
        "video_fps":          round(actual_fps, 4),
        "frames":             frame_entries,
    }


def _save_sled_json(sled_preds, audio_duration, actual_fps, index, out_dir, classes):
    """Save per-frame SLED predictions as JSON and return (path, dict)."""
    os.makedirs(out_dir, exist_ok=True)
    out = _build_sled_dict(sled_preds, audio_duration, actual_fps, classes)
    path = os.path.join(out_dir, f"sled_preds_{index}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path, out


def _save_gt_json(env, active_radio, sound_meta, cam_id, index, gt_dir, position_label):
    """Save ground-truth JSON for the select_radio task."""
    os.makedirs(gt_dir, exist_ok=True)
    az, el, dist = _compute_radio_az_el(env, active_radio, cam_id)
    out = {
        "active_radio":    active_radio,
        "position_label":  position_label,
        "sound_file":      sound_meta["sound_file"],
        "sound_folder":    sound_meta["folder"],
        "class_id":        sound_meta["class_id"],
        "class_name":      sound_meta["class_name"],
        "camera_id":       cam_id,
        "azimuth_deg":     round(az, 3)  if az   is not None else None,
        "elevation_deg":   round(el, 3)  if el   is not None else None,
        "distance_m":      round(dist, 4) if dist is not None else None,
    }
    path = os.path.join(gt_dir, f"gt_{index}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path

os.environ["MUJOCO_GL"] = "egl"

def get_args():
    parser = argparse.ArgumentParser(description='Generate trajectory for a task')
    parser.add_argument('--task-name', default="select_poker", type=str, help='task name')
    parser.add_argument('--record-video', default=True, help='record video')
    parser.add_argument('--save-dir', default="/media/shiduo/LENOVO_USB_HDD/dataset/VLABench")
    parser.add_argument('--n-sample', default=1, type=int, help='number of samples to generate')
    parser.add_argument('--start-id', default=0, type=int, help='start index for data storage')
    parser.add_argument('--robot', default="franka", type=str, help='robot name')
    parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
    parser.add_argument('--early-stop', action="store_true", default=False, help='whether use early stop when skill failed to carry out')
    parser.add_argument('--max-episode', default=100, type=int, help='max episode number in the directory')
    parser.add_argument('--eval-unseen', default=False, action="store_true", help='evaluate unseen object categories')
    parser.add_argument(
        '--audio-config',
        default=None,
        type=str,
        help='path to scene_audio_config.json; if provided, binaural audio is '
             'synthesised in real-time and saved alongside the trajectory data',
    )
    parser.add_argument(
        '--sled-ckpt',
        default=None,
        type=str,
        help='path to SLEDv3 checkpoint (.pt); if provided, real-time DOA '
             'predictions are overlaid on the forward camera (cam 2) in the '
             'saved video. Requires --audio-config.',
    )
    parser.add_argument(
        '--sled-min-confident-rate',
        default=0.4, type=float,
        help='Drop the episode (do not save HDF5/wav/videos/GT) if fewer than '
             'this fraction of SLED frames produced a confident prediction '
             '(top-1 conf >= --sled-conf-thresh). Default 0.4 means we discard '
             'episodes where 60%% or more of SLED runs failed to detect anything.',
    )
    parser.add_argument(
        '--sled-conf-thresh',
        default=0.30, type=float,
        help='SLED top-1 confidence threshold for "confident" frames.',
    )
    parser.add_argument(
        '--oracle-mode',
        action='store_true', default=False,
        help='Skip real audio synthesis + SLED; record ground-truth source '
             'geometry directly into HDF5 as meta_info/oracle_audio. Much '
             'faster than the full pipeline and avoids the SLED quality gate. '
             'Downstream, --oracle-mode in convert_hdf5_to_lerobot.py reads '
             'this field and feeds clean GT into observation.audio.* while '
             'training applies fresh noise each batch.',
    )
    parser.add_argument(
        '--start-idle-seconds',
        default=2.0, type=float,
        help='Hold the robot still for this long at episode start before any '
             'expert motion runs. The frames are recorded into the dataset, '
             'so the trained policy learns to wait for audio info to settle '
             'before moving — symmetric to the audio warm-up used in the '
             'real-SLED eval path. Set to 0 to disable.',
    )
    parser.add_argument(
        '--dataset-fps',
        default=10, type=int,
        help='Effective video FPS used to convert --start-idle-seconds into '
             'a frame count. Must match the FPS used by the LeRobot '
             'converter (default 10).',
    )
    parser.add_argument(
        '--target-position-label',
        choices=['left', 'middle', 'right',
                 'left_top', 'right_top', 'left_bottom', 'right_bottom'],
        default=None,
        help='Force the target slot for balanced generation. '
             'select_radio/select_radio_two: left/middle/right. '
             'find_hidden_object(_open): <azimuth>_<elevation>, e.g. left_top.',
    )
    args = parser.parse_args()
    return args


def _record_idle_frames(env, n_frames):
    """Hold the robot still for n_frames and record observations + waypoints.

    Each frame:
      * env.step(None) → MuJoCo applies the current qpos as the action and
        keeps the gripper open (see LM4ManipDMEnv.step), so the robot does
        not move under control. Tiny passive drift is recorded faithfully.
      * One observation is appended; the waypoint is the current ee pose.

    Used at episode start to give the audio pipeline (real-SLED warm-up,
    oracle latency simulation) a window before the expert begins to act,
    and to teach the policy a "don't move yet" prefix that mirrors the
    warm-up wait used at inference time.
    """
    if n_frames <= 0:
        return [], []
    from VLABench.utils.utils import quaternion_to_euler as _q2e
    ee_pos  = np.asarray(env.robot.get_end_effector_pos(env.physics)).reshape(3)
    ee_quat = np.asarray(env.robot.get_end_effector_quat(env.physics)).reshape(4)
    ee_euler = np.asarray(_q2e(ee_quat)).reshape(3)
    # Mirror SkillLib: gripper open → 0.04, closed → 0
    is_open = bool(env.robot.get_ee_open_state(env.physics))
    gripper_state = np.full(2, 0.04, dtype=np.float64) if is_open else np.zeros(2)
    waypoint = np.concatenate([ee_pos, ee_euler, gripper_state])

    observations, waypoints = [], []
    for _ in range(int(n_frames)):
        env.step()                      # action=None → static
        observations.append(env.get_observation())
        waypoints.append(waypoint.copy())
    return observations, waypoints

def get_all_hdf5_files(directory):
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files

def _build_audio_manager(args, logger, config_dict=None):
    """
    Try to construct an AudioSimManager for the current task.

    Returns the manager on success, or None if:
      - --audio-config was not given (and no config_dict)
      - the task has no entry in the config file
      - any other initialisation error

    config_dict : optional in-memory config that overrides config_path loading.
    """
    if args.audio_config is None and config_dict is None:
        return None
    try:
        import sys
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from audio_generation.audio_sim_manager import AudioSimManager
        mgr = AudioSimManager(
            config_path=args.audio_config,
            task_name=args.task_name,
            config_dict=config_dict,
        )
        return mgr
    except ValueError as exc:
        logger.warning(f"[audio] Skipping binaural audio: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"[audio] Failed to initialise AudioSimManager: {exc}")
        return None


def _build_sled_overlay(args, logger):
    """
    Try to construct a SLEDOverlay for post-hoc inference.

    Returns the overlay on success, or None if:
      - --sled-ckpt was not given
      - --audio-config was not given (no audio to run inference on)
      - any initialisation error
    """
    if args.sled_ckpt is None:
        return None
    try:
        import sys
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from audio_generation.sled_overlay import SLEDOverlay
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        overlay = SLEDOverlay(
            ckpt_path    = args.sled_ckpt,
            torch_device = device,
            realtime     = False,   # post-hoc: no background thread needed
        )
        return overlay
    except Exception as exc:
        logger.warning(f"[SLED] Failed to initialise SLEDOverlay: {exc}")
        return None


def generate_trajectory(args, index, logger):
    if args.target_position_label is not None:
        os.environ["VLABENCH_TARGET_POSITION_LABEL"] = args.target_position_label
        # find_hidden_object(_open) forces its slot via a separate env var
        # (label form "<azimuth>_<elevation>", e.g. left_top); drive it from the
        # same CLI flag so balanced generation works for both task families.
        os.environ["VLABENCH_HIDDEN_SLOT_LABEL"] = args.target_position_label
    else:
        os.environ.pop("VLABENCH_TARGET_POSITION_LABEL", None)
        os.environ.pop("VLABENCH_HIDDEN_SLOT_LABEL", None)
    env = load_env(args.task_name, robot=args.robot, eval=args.eval_unseen)
    env.reset()
    episode_config = env.save()

    # load key prior information and task specific variables
    target_entity = env.task.config_manager.target_entity
    instruction = env.task.get_instruction()
    meta_info = dict(
        target_entity=[target_entity],
        entities=list(env.task.entities.keys()),
        instruction=[instruction],
    )

    # register the expert sequence
    skill_seq = env.get_expert_skill_sequence()

    # ------------------------------------------------------------------
    # select_radio / select_radio_two:
    #   - select_radio:     1 active radio, instruction = "...the sound."
    #   - select_radio_two: 2 active radios with *different* sound classes,
    #                       instruction names the target class explicitly.
    # In both cases the *target* (instruction-referenced) radio is what
    # downstream code treats as `_active_radio`; `_other_*` only exists for
    # the 2-radio variant and provides the second source.
    # ------------------------------------------------------------------
    _radio_sound_meta = None
    _active_radio     = None
    _position_label   = None
    _other_sound_meta = None
    _other_radio      = None
    _other_position_label = None
    _sounding_sources_meta = []
    _taxonomy_classes = {}

    _is_radio_task = args.task_name in ("select_radio", "select_radio_two", "select_radio_silent")
    if _is_radio_task and (args.audio_config is not None or args.oracle_mode):
        cm = env.task.config_manager
        active_idx = getattr(cm, "active_radio_idx", None)
        if active_idx is not None:
            _active_radio   = f"radio_{active_idx}"
            _position_label = getattr(cm, "active_position_label", None)
            _taxonomy_classes, _label_map = _load_class_taxonomy()
            if args.task_name == "select_radio_silent":
                sounding_indices = getattr(cm, "sounding_radio_indices", None)
                if sounding_indices is None or len(sounding_indices) != 2:
                    raise RuntimeError(
                        "select_radio_silent: config_manager missing two sounding_radio_indices"
                    )
                used_class_ids = []
                for sounding_idx in sounding_indices:
                    sound_meta = _pick_random_sound(
                        _label_map, _taxonomy_classes,
                        exclude_class_ids=used_class_ids,
                    )
                    if sound_meta is None:
                        raise RuntimeError(
                            "select_radio_silent: could not pick two distinct sounding classes"
                        )
                    used_class_ids.append(sound_meta["class_id"])
                    _sounding_sources_meta.append({
                        "radio": f"radio_{sounding_idx}",
                        "position": POSITION_LABELS[sounding_idx],
                        "sound_meta": sound_meta,
                    })
                instruction = _build_silent_radio_instruction()
                meta_info["instruction"] = [instruction]
                cm.get_instruction()
                env.task.instructions = cm.config["task"]["instructions"]
            else:
                _radio_sound_meta = _pick_random_sound(
                    _label_map, _taxonomy_classes,
                    force_class_id=_forced_class_id(_taxonomy_classes),
                )
                if args.task_name == "select_radio":
                    instruction = _build_one_radio_instruction()
                    meta_info["instruction"] = [instruction]
                    cm.get_instruction()
                    env.task.instructions = cm.config["task"]["instructions"]

            if args.task_name == "select_radio_two":
                other_idx = getattr(cm, "other_radio_idx", None)
                if other_idx is None:
                    raise RuntimeError(
                        "select_radio_two: config_manager missing other_radio_idx"
                    )
                _other_radio = f"radio_{other_idx}"
                _other_position_label = getattr(cm, "other_position_label", None)
                if _radio_sound_meta is None:
                    raise RuntimeError("select_radio_two: failed to pick target sound")
                _other_sound_meta = _pick_random_sound(
                    _label_map, _taxonomy_classes,
                    exclude_class_ids=[_radio_sound_meta["class_id"]],
                )
                if _other_sound_meta is None:
                    raise RuntimeError(
                        "select_radio_two: could not find a second sound from a "
                        "different class than the target."
                    )
                # Update the instruction now that we know the target class.
                instruction = _build_two_radio_instruction(_radio_sound_meta["class_name"])
                meta_info["instruction"] = [instruction]
                cm.get_instruction(sound_class_name=_natural_class_name(
                    _radio_sound_meta["class_name"]))
                env.task.instructions = cm.config["task"]["instructions"]

    # ------------------------------------------------------------------
    # find_hidden_object(_open):
    #   A single sound source that emanates from the HIDDEN OBJECT itself.
    #   The object sits in the target drawer, so its world position encodes
    #   both the azimuth (which cabinet, left/right) and the elevation (top/
    #   bottom drawer) that the policy must localize. This is the same
    #   single-source spatial-localization setup as select_radio, so we reuse
    #   the radio variables (_active_radio / _radio_sound_meta / _position_label)
    #   and all the downstream source/GT plumbing keyed off them. The generic
    #   "open the drawer the sound is coming from" instruction (set by the
    #   config manager) is kept as-is.
    # ------------------------------------------------------------------
    _is_hidden_task = args.task_name in ("find_hidden_object_open", "find_hidden_object")
    if _is_hidden_task and (args.audio_config is not None or args.oracle_mode):
        cm = env.task.config_manager
        _taxonomy_classes, _label_map = _load_class_taxonomy()
        _radio_sound_meta = _pick_random_sound(
            _label_map, _taxonomy_classes,
            force_class_id=_forced_class_id(_taxonomy_classes),
        )
        # the hidden object is the sound source; task.entities resolves it to a
        # body whose xpos gives the target-slot (azimuth x elevation) position.
        _active_radio   = cm.target_entity
        _position_label = getattr(cm, "position_label", None)
        logger.info(
            f"[hidden-audio] target object {_active_radio} ({_position_label}) → "
            f"{_radio_sound_meta['folder']} (class {_radio_sound_meta['class_id']}: "
            f"{_radio_sound_meta['class_name']})"
        )

    # ------------------------------------------------------------------
    # take_out_microwave_food:
    #   One-shot microwave chime that fires at a random per-episode time
    #   sampled by the task's config_manager. We build a dynamic audio
    #   config here so the same `trigger_step` drives both the simulated
    #   sound and the task's failure check.
    # ------------------------------------------------------------------
    _microwave_audio_meta = None
    if args.task_name == "take_out_microwave_food":
        cm = env.task.config_manager
        # Captured for both real-audio and oracle paths so the same trigger
        # step is logged, sent to AudioSimManager (real), and stamped onto
        # the oracle GT below (oracle).
        _microwave_audio_meta = {
            "object_name":       cm.target_container,
            "sound_file":        "ring.wav",
            "trigger_step":      cm.trigger_step,
            "trigger_delay_sec": cm.trigger_delay_sec,
            "react_window_sec":  cm.react_window_sec,
            # Taxonomy: "Microwave oven" maps to class 34 (Domestic_Appliance).
            "class_id":          34,
            "class_name":        "Domestic_Appliance",
        }
        logger.info(
            f"[microwave-audio] chime → {cm.target_container} at step "
            f"{cm.trigger_step} ({cm.trigger_delay_sec:.2f}s), "
            f"react window {cm.react_window_sec:.1f}s"
        )
    # Real-audio path uses the dynamic config a few lines below; the oracle
    # path uses the same metadata in the oracle-GT block further down.
    _do_microwave_real_audio = (
        _microwave_audio_meta is not None
        and args.audio_config is not None
        and not args.oracle_mode
    )

    # ------------------------------------------------------------------
    # Binaural audio – set up and start before the skill loop
    # ------------------------------------------------------------------
    _audio_cfg_dict = None
    if (_radio_sound_meta is not None or _sounding_sources_meta) and not args.oracle_mode:
        with open(args.audio_config) as _f:
            _base_cfg = json.load(_f)
        _sources = []
        if _radio_sound_meta is not None:
            _sources.append({
                "object_name": _active_radio,
                "geom_type":   "body",
                "sound_file":  _radio_sound_meta["sound_file"],
                "gain":        1.0,
            })
        if _other_sound_meta is not None and _other_radio is not None:
            _sources.append({
                "object_name": _other_radio,
                "geom_type":   "body",
                "sound_file":  _other_sound_meta["sound_file"],
                "gain":        1.0,
            })
        for src in _sounding_sources_meta:
            _sources.append({
                "object_name": src["radio"],
                "geom_type":   "body",
                "sound_file":  src["sound_meta"]["sound_file"],
                "gain":        1.0,
            })
        _audio_cfg_dict = {
            "cam_id":   _base_cfg.get("cam_id", 2),
            "hrtf_path": _base_cfg.get("hrtf_path", ""),
            "tasks": {
                args.task_name: {"sources": _sources}
            },
        }
        if _radio_sound_meta is not None:
            logger.info(
                f"[radio-audio] target {_active_radio} ({_position_label}) → "
                f"{_radio_sound_meta['folder']} (class {_radio_sound_meta['class_id']}: "
                f"{_radio_sound_meta['class_name']})"
            )
        else:
            logger.info(f"[radio-audio] silent target {_active_radio} ({_position_label})")
        if _other_sound_meta is not None:
            logger.info(
                f"[radio-audio] other  {_other_radio} ({_other_position_label}) → "
                f"{_other_sound_meta['folder']} (class {_other_sound_meta['class_id']}: "
                f"{_other_sound_meta['class_name']})"
            )
        for src in _sounding_sources_meta:
            sm = src["sound_meta"]
            logger.info(
                f"[radio-audio] sound  {src['radio']} ({src['position']}) → "
                f"{sm['folder']} (class {sm['class_id']}: {sm['class_name']})"
            )

    if _do_microwave_real_audio:
        with open(args.audio_config) as _f:
            _base_cfg = json.load(_f)
        _audio_cfg_dict = {
            "cam_id":   _base_cfg.get("cam_id", 2),
            "hrtf_path": _base_cfg.get("hrtf_path", ""),
            "tasks": {
                args.task_name: {
                    "sources": [{
                        "object_name": _microwave_audio_meta["object_name"],
                        "geom_type":   "body",
                        "sound_file":  _microwave_audio_meta["sound_file"],
                        "gain":        1.0,
                        "loop":        False,
                        "start_step":  _microwave_audio_meta["trigger_step"],
                    }],
                },
            },
        }

    if args.oracle_mode:
        audio_mgr = None    # no real audio in oracle mode
        logger.info("[oracle] --oracle-mode: skipping AudioSimManager + SLED")
    else:
        audio_mgr = _build_audio_manager(args, logger, config_dict=_audio_cfg_dict)
        if audio_mgr is not None:
            audio_mgr.attach_to_env(env)   # patches env.step for position updates
            audio_mgr.start()

    # ------------------------------------------------------------------
    # Start-of-episode idle warm-up. Records `n_idle` "stay-still" frames
    # before the expert skill runs so:
    #   * Real-SLED pipeline: audio buffer fills before the robot moves,
    #     matching the inference-time warm-up.
    #   * Oracle pipeline: trains the policy to expect a brief "wait"
    #     prefix at the start of every rollout, which is more robust when
    #     the runtime smoother (see TopKClassSmoother) needs a few frames
    #     of class history to stabilise.
    # ------------------------------------------------------------------
    n_idle = int(round(getattr(args, "start_idle_seconds", 0.0) * args.dataset_fps))
    # Capture the task step counter at the moment recording starts. env.reset()
    # leaves step_count at a non-zero settling value, so recorded-frame index
    # F corresponds to task.step_count = F + _record_start_step. The microwave
    # chime time is sampled in step_count units (cm.trigger_step), so its
    # *recorded-frame* index — what `active_from_frame` must be — is
    # trigger_step - _record_start_step. Without this subtraction the audio
    # label lands ~_record_start_step frames after the robot actually reacts,
    # teaching the policy to move while the cue is still labelled silent.
    _record_start_step = int(getattr(getattr(env, "task", None), "step_count", 0) or 0)
    if n_idle > 0:
        logger.info(f"[idle] holding robot still for {n_idle} frames "
                    f"(~{args.start_idle_seconds:.2f}s @ {args.dataset_fps}fps)")
    idle_obs, idle_wps = _record_idle_frames(env, n_idle)

    # start auto trajectory generation
    observations, waypoints = list(idle_obs), list(idle_wps)
    if skill_seq is not None: # normal case
        for skill in skill_seq:
            obs, waypoint, stage_success, task_success = skill(env)
            if args.debug:
                for o in obs: observations.append(dict(rgb=o["rgb"]))
            else:
                observations.extend(obs)
                waypoints.extend(waypoint)
            if args.early_stop and not stage_success:
                logger.warning(f"{skill} failed, early quit...")
                break
            if task_success:
                break
    else: # TODO: some special tasks should be handled based on the feedback
        raise NotImplementedError("No expert skill sequence found")

    # ------------------------------------------------------------------
    # Stop audio stream regardless of task success / failure
    # ------------------------------------------------------------------
    _TARGET_VIDEO_FPS = 10.0   # dataset video fps (matches mediapy.write_video fps=10)

    audio_arr         = None   # wall-clock audio  → used for SLED inference
    audio_arr_resampled = None # simulation-time audio → saved to WAV/HDF5/video
    _sled_fps         = None   # fps aligned to wall-clock audio (for SLED)
    task_dir  = os.path.join(args.save_dir, args.task_name)
    if audio_mgr is not None:
        audio_mgr.detach_from_env(env)
        if task_success:
            os.makedirs(task_dir, exist_ok=True)
            wav_path  = os.path.join(task_dir, f"audio_{index}.wav")
            audio_arr = audio_mgr.stop_and_save(wav_path)  # wall-clock recording

            # ----------------------------------------------------------
            # The simulation runs faster than real-time (EGL headless),
            # so the wall-clock recording is much longer than the episode.
            #
            # We keep TWO versions:
            #   audio_arr          — original wall-clock audio, used by
            #                        SLED with matching wall-clock fps so
            #                        audio windows map to correct frames.
            #   audio_arr_resampled — compressed to n_frames/10fps seconds,
            #                        saved as the WAV file and muxed into
            #                        the viz video for natural-speed playback.
            # ----------------------------------------------------------
            if audio_arr is not None and audio_arr.shape[0] > 0 and len(observations) > 0:
                sr = audio_mgr.engine.sr
                _sled_fps = len(observations) / (audio_arr.shape[0] / sr)

                target_samples = int(round(len(observations) / _TARGET_VIDEO_FPS * sr))
                import soundfile as sf
                from scipy.signal import resample as _sp_resample
                audio_arr_resampled = _sp_resample(
                    audio_arr, target_samples, axis=0
                ).astype(np.float32)
                audio_arr_resampled = np.clip(audio_arr_resampled, -1.0, 1.0)
                sf.write(wav_path, audio_arr_resampled, sr)
                logger.info(
                    f"[audio] WAV resampled: {audio_arr.shape[0]/sr:.1f}s "
                    f"→ {target_samples/sr:.1f}s  "
                    f"(SLED fps={_sled_fps:.2f}, video fps={_TARGET_VIDEO_FPS})"
                )
        else:
            audio_mgr.engine.stop()   # stop cleanly even on failure

    # ------------------------------------------------------------------
    # Build SLED overlay (post-hoc, requires saved audio)
    # ------------------------------------------------------------------
    sled_overlay = None if args.oracle_mode else _build_sled_overlay(args, logger)
    _sled_json_str = None   # set if SLED runs successfully

    if args.record_video:
        import cv2
        import subprocess

        # Base frames (no overlay) – used for dataset video
        base_frames = []
        for o in observations:
            rgb = np.array(o["rgb"])   # [ncam, H, W, 3] uint8 RGB
            base_frames.append(np.vstack([np.hstack(rgb[:2]), np.hstack(rgb[2:4])]))

        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        # ── Dataset video: no markers, no audio ─────────────────────────────
        dataset_video_path = os.path.join(
            task_dir, f"demo_{index}_success_{task_success}.mp4"
        )
        try:
            mediapy.write_video(dataset_video_path, base_frames, fps=10)
            logger.info(f"Dataset video saved → {dataset_video_path}")
        except Exception as e:
            logger.warning(f"Dataset video recording failed: {e}")

        # ── Verification video: SLED markers + audio ────────────────────────
        if task_success and sled_overlay is not None and audio_arr is not None and audio_arr.shape[0] > 0:
            # Per-frame cam dimensions (all frames are the same size)
            first_rgb = np.array(observations[0]["rgb"])
            cam_h, cam_w = first_rgb.shape[1], first_rgb.shape[2]

            actual_fps = _TARGET_VIDEO_FPS
            sr = audio_mgr.engine.sr if audio_mgr is not None else 44100
            _sled_fps_str = f"{_sled_fps:.2f}" if _sled_fps is not None else "N/A"
            logger.info(f"Viz video fps: {actual_fps:.2f} "
                        f"({len(observations)} frames, SLED fps={_sled_fps_str})")

            # Post-hoc SLED inference on the ORIGINAL wall-clock audio
            # with matching wall-clock fps so each video frame maps to
            # the correct audio window (pitch/tempo are natural speed).
            sled_preds = sled_overlay.run_post_hoc(
                audio_arr, len(observations),
                video_fps=_sled_fps if _sled_fps is not None else actual_fps,
            )

            # ── SLED quality gate ───────────────────────────────────────────
            # Drop episodes where SLED almost never detected anything — they
            # carry no useful audio signal for the policy and just dilute the
            # training distribution.
            n_total = len(sled_preds)
            n_confident = 0
            for snap in sled_preds:
                if snap is None:
                    continue
                _, conf, cls = snap
                if conf is None or len(conf) == 0:
                    continue
                # "confident" = top-1 class id valid AND conf >= threshold
                top = int(np.argmax(conf))
                if int(cls[top]) >= 0 and float(conf[top]) >= args.sled_conf_thresh:
                    n_confident += 1
            confident_rate = n_confident / max(1, n_total)
            logger.info(
                f"[SLED-gate] {n_confident}/{n_total} confident frames "
                f"({confident_rate:.0%}) — threshold {args.sled_min_confident_rate:.0%}"
            )
            if confident_rate < args.sled_min_confident_rate:
                logger.warning(
                    f"[SLED-gate] dropping episode {index}: "
                    f"only {confident_rate:.0%} confident frames "
                    f"(< {args.sled_min_confident_rate:.0%})"
                )
                # Clean up files that were written before we knew this would
                # be discarded (dataset video, resampled WAV).
                for _p in (dataset_video_path, wav_path):
                    try:
                        if _p and os.path.exists(_p):
                            os.remove(_p)
                    except Exception as _e:
                        logger.warning(f"  cleanup failed for {_p}: {_e}")
                env.close()
                return

            # Build visualization frames with overlay on cam 2 (forward)
            viz_frames = []
            for i_frame, o in enumerate(observations):
                rgb = np.array(o["rgb"])
                snap = sled_preds[i_frame] if i_frame < len(sled_preds) else None
                if snap is not None:
                    cam2_bgr = cv2.cvtColor(rgb[2], cv2.COLOR_RGB2BGR)
                    sled_overlay.draw_with_pred(cam2_bgr, snap, cam_w, cam_h)
                    rgb[2] = cv2.cvtColor(cam2_bgr, cv2.COLOR_BGR2RGB)
                viz_frames.append(np.vstack([np.hstack(rgb[:2]), np.hstack(rgb[2:4])]))

            # Save verification video to a separate folder
            viz_dir = task_dir + "_viz"
            os.makedirs(viz_dir, exist_ok=True)
            viz_video_name = f"demo_{index}_success_{task_success}.mp4"
            viz_raw_path   = os.path.join(viz_dir, viz_video_name.replace(".mp4", "_raw.mp4"))
            viz_video_path = os.path.join(viz_dir, viz_video_name)
            viz_written = False

            try:
                mediapy.write_video(viz_raw_path, viz_frames, fps=actual_fps)
                viz_written = True
            except Exception as e:
                logger.warning(f"Verification video recording failed: {e}")

            if viz_written:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", viz_raw_path, "-i", wav_path,
                     "-c:v", "copy", "-c:a", "aac", viz_video_path],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    os.remove(viz_raw_path)
                    logger.info(f"Verification video (markers+audio) → {viz_video_path}")
                else:
                    os.rename(viz_raw_path, viz_video_path)
                    logger.warning(f"Audio muxing failed: {result.stderr[-300:]}")

            # ── Save SLED predictions JSON (all tasks) ───────────────────────
            _json_audio_dur = (audio_arr_resampled.shape[0] / sr
                               if audio_arr_resampled is not None else
                               audio_arr.shape[0] / sr)
            sled_json_path, _sled_dict = _save_sled_json(
                sled_preds, _json_audio_dur, actual_fps, index, viz_dir, _taxonomy_classes
            )
            logger.info(f"SLED predictions JSON → {sled_json_path}")
            # expose for HDF5 saving below
            _sled_json_str = json.dumps(_sled_dict)

            # ── Save ground-truth JSON (select_radio only) ───────────────────
            if args.task_name == "select_radio" and _radio_sound_meta is not None:
                _cam_id = audio_mgr.cam_id if audio_mgr is not None else 2
                gt_dir  = task_dir + "_gt"
                gt_path = _save_gt_json(
                    env, _active_radio, _radio_sound_meta,
                    _cam_id, index, gt_dir, _position_label
                )
                logger.info(f"Ground-truth JSON → {gt_path}")

                # Quick-reference audio metadata log in main task folder
                meta_path = os.path.join(task_dir, f"audio_meta_{index}.json")
                with open(meta_path, "w") as _mf:
                    json.dump({
                        "active_radio":   _active_radio,
                        "position_label": _position_label,
                        "sound_file":     _radio_sound_meta["sound_file"],
                        "sound_folder":   _radio_sound_meta["folder"],
                        "class_id":       _radio_sound_meta["class_id"],
                        "class_name":     _radio_sound_meta["class_name"],
                    }, _mf, indent=2)
                logger.info(f"Audio meta → {meta_path}")
    if _microwave_audio_meta is not None:
        # task_success has been resolved by here; log the chime metadata + any
        # task-side failure reason so we can audit per-episode drops.
        failure_reason = getattr(env.task, "failure_reason", None)
        if failure_reason is not None:
            logger.warning(f"[microwave-audio] failure_reason={failure_reason}")
        if task_success:
            os.makedirs(task_dir, exist_ok=True)
            meta_path = os.path.join(task_dir, f"audio_meta_{index}.json")
            with open(meta_path, "w") as _mf:
                json.dump({
                    "object_name":       _microwave_audio_meta["object_name"],
                    "sound_file":        _microwave_audio_meta["sound_file"],
                    "trigger_step":      _microwave_audio_meta["trigger_step"],
                    "trigger_delay_sec": _microwave_audio_meta["trigger_delay_sec"],
                    "react_window_sec":  _microwave_audio_meta["react_window_sec"],
                    "class_id":          _microwave_audio_meta["class_id"],
                    "class_name":        _microwave_audio_meta["class_name"],
                    "oracle_mode":       bool(args.oracle_mode),
                    "failure_reason":    failure_reason,
                }, _mf, indent=2)
            logger.info(f"Audio meta → {meta_path}")

    if not task_success:
        logger.warning("Task failed, skip saving data")
        return
    else:
        logger.info("Task success, saving data")

    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_to_save = process_observations(observations)

    robot_position = env.robot.robot_config["position"]
    robot_frame_waypoints = [np.array(waypoint) - np.concatenate([robot_position, np.zeros(5)]) for waypoint in waypoints]
    data_to_save["trajectory"] = robot_frame_waypoints   # keep for backward compat
    data_to_save["action"] = robot_frame_waypoints        # LeRobot-compatible action key
    data_to_save["entities"] = meta_info["entities"]
    data_to_save["target_entity"] = meta_info["target_entity"]
    data_to_save["episode_config"] = json.dumps(episode_config)
    data_to_save["instruction"] = meta_info["instruction"]

    # Save SLED prediction JSON string into HDF5 (if SLED was run)
    if _sled_json_str is not None:
        data_to_save["sled_predictions"] = _sled_json_str

    # ------------------------------------------------------------------
    # Oracle-mode: capture ground-truth audio geometry (no SLED needed)
    # ------------------------------------------------------------------
    if args.oracle_mode and _active_radio is not None and (
        _radio_sound_meta is not None or _sounding_sources_meta
    ):
        import sys as _sys
        _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _proj not in _sys.path:
            _sys.path.insert(0, _proj)
        from src.audio.oracle_sled import extract_episode_gt
        _oracle_cam_id = 2   # front cam; static for select_radio
        # For select_radio/select_radio_two the instruction-targeted radio is
        # listed first. For select_radio_silent, the target is deliberately
        # absent from active_sources because it emits no sound.
        #
        # NOTE: this target-first order is raw-storage bookkeeping only (it
        # also drives audio_meta.json / instruction text below) and must NOT
        # reach the model as-is — it would let a policy learn "always attend
        # slot 0" instead of matching the instruction's class name.
        # convert_hdf5_to_lerobot.py re-sorts active_sources by azimuth
        # (task-left to task-right) before writing observation.audio.* /
        # audio_slots, so this order never leaks into training data.
        _active_sources_meta = []
        if _radio_sound_meta is not None:
            _active_sources_meta.append({
                "name":       _active_radio,
                "class_id":   _radio_sound_meta["class_id"],
                "class_name": _radio_sound_meta["class_name"],
            })
        if _other_sound_meta is not None and _other_radio is not None:
            _active_sources_meta.append({
                "name":       _other_radio,
                "class_id":   _other_sound_meta["class_id"],
                "class_name": _other_sound_meta["class_name"],
            })
        for src in _sounding_sources_meta:
            sm = src["sound_meta"]
            _active_sources_meta.append({
                "name":       src["radio"],
                "class_id":   sm["class_id"],
                "class_name": sm["class_name"],
            })
        oracle_dict = extract_episode_gt(
            env,
            active_sources=_active_sources_meta,
            cam_id=_oracle_cam_id,
            n_frames=len(observations),
        )
        data_to_save["oracle_audio"] = json.dumps(oracle_dict)

        # Bookkeeping sidecar (consistent with the non-oracle flow)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        meta_path = os.path.join(task_dir, f"audio_meta_{index}.json")
        _meta_payload = {
            "active_radio":   _active_radio,
            "position_label": _position_label,
            "oracle_mode":    True,
        }
        if _radio_sound_meta is not None:
            _meta_payload.update({
                "sound_file":     _radio_sound_meta["sound_file"],
                "sound_folder":   _radio_sound_meta["folder"],
                "class_id":       _radio_sound_meta["class_id"],
                "class_name":     _radio_sound_meta["class_name"],
            })
        if _other_sound_meta is not None:
            _meta_payload["other_radio"]    = _other_radio
            _meta_payload["other_position"] = _other_position_label
            _meta_payload["other_sound_file"]   = _other_sound_meta["sound_file"]
            _meta_payload["other_sound_folder"] = _other_sound_meta["folder"]
            _meta_payload["other_class_id"]     = _other_sound_meta["class_id"]
            _meta_payload["other_class_name"]   = _other_sound_meta["class_name"]
            _meta_payload["instruction"] = meta_info["instruction"][0]
        if _sounding_sources_meta:
            _meta_payload["silent_target"] = True
            _meta_payload["instruction"] = meta_info["instruction"][0]
            _meta_payload["sounding_sources"] = [
                {
                    "radio": src["radio"],
                    "position": src["position"],
                    "sound_file": src["sound_meta"]["sound_file"],
                    "sound_folder": src["sound_meta"]["folder"],
                    "class_id": src["sound_meta"]["class_id"],
                    "class_name": src["sound_meta"]["class_name"],
                }
                for src in _sounding_sources_meta
            ]
        with open(meta_path, "w") as _mf:
            json.dump(_meta_payload, _mf, indent=2)
        for _src in oracle_dict["active_sources"]:
            logger.info(
                f"[oracle] GT: {_src['name']} (cls {_src['class_id']} "
                f"{_src['class_name']})  az={_src['az_deg']:+.2f}°  "
                f"el={_src['el_deg']:+.2f}°  dist={_src['distance_m']:.3f}m"
            )

    # ------------------------------------------------------------------
    # Oracle-mode GT for take_out_microwave_food:
    #   Single time-gated source (the microwave chime). active_from_frame
    #   = trigger_step so the converter fills frames in [0, trigger_step)
    #   with silence in this slot and the remainder with the GT class +
    #   direction. The trainer can then teach the policy to act on the
    #   silence→active transition.
    # ------------------------------------------------------------------
    if args.oracle_mode and _microwave_audio_meta is not None:
        import sys as _sys
        _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _proj not in _sys.path:
            _sys.path.insert(0, _proj)
        from src.audio.oracle_sled import extract_episode_gt
        _oracle_cam_id = 2   # front cam
        # Convert the chime time from step_count units to recorded-frame units
        # by subtracting the recording-start offset (see _record_start_step).
        _chime_frame = max(0, int(_microwave_audio_meta["trigger_step"]) - _record_start_step)
        logger.info(
            f"[oracle] chime active_from_frame={_chime_frame} "
            f"(trigger_step={_microwave_audio_meta['trigger_step']} − "
            f"record_start_step={_record_start_step})"
        )
        _active_sources_meta = [{
            "name":              _microwave_audio_meta["object_name"],
            "class_id":          _microwave_audio_meta["class_id"],
            "class_name":        _microwave_audio_meta["class_name"],
            "active_from_frame": _chime_frame,
        }]
        oracle_dict = extract_episode_gt(
            env,
            active_sources=_active_sources_meta,
            cam_id=_oracle_cam_id,
            n_frames=len(observations),
        )
        data_to_save["oracle_audio"] = json.dumps(oracle_dict)
        # Sidecar audio_meta_{index}.json is already written by the
        # _microwave_audio_meta block above (with oracle_mode=True), so we
        # only log the GT here.
        for _src in oracle_dict["active_sources"]:
            logger.info(
                f"[oracle] GT: {_src['name']} (cls {_src['class_id']} "
                f"{_src['class_name']})  az={_src['az_deg']:+.2f}°  "
                f"el={_src['el_deg']:+.2f}°  dist={_src['distance_m']:.3f}m  "
                f"active_from_frame={_src.get('active_from_frame', 0)}"
            )

    save_single_data(data_to_save,
                     save_dir=task_dir,
                     filename=f"data_{index}.hdf5",
                     )
    env.close()
    
        
if __name__ == "__main__":
    args = get_args()
    logger = get_logger()
    # Optional per-episode target-class rotation (VLABENCH_TARGET_CLASS_NAMES,
    # comma-separated), used to balance the per-class sound distribution
    # while topping up a dataset. Indexed by the number of episodes already
    # saved in this run, so failed attempts retry the same class instead of
    # skipping ahead, and (when len(class_list) == args.max_episode) each
    # class in the list is consumed exactly once.
    _class_list_env = os.environ.get("VLABENCH_TARGET_CLASS_NAMES")
    class_list = [c.strip() for c in _class_list_env.split(",") if c.strip()] if _class_list_env else []
    for i in tqdm(range(args.n_sample)):
        i += args.start_id
        try:
            h5_files = get_all_hdf5_files(os.path.join(args.save_dir, args.task_name))
            if len(h5_files) >= args.max_episode:
                logger.info(f"Task {args.task_name} has reached the maximum episode number, skip")
                break
            if class_list:
                os.environ["VLABENCH_TARGET_CLASS_NAME"] = class_list[len(h5_files) % len(class_list)]
            generate_trajectory(args, i, logger)
        except Exception as e:
            err = traceback.TracebackException.from_exception(e)
            print("".join(err.format()))
            continue
