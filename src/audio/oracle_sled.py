"""Oracle SLED: synthesise fake top-K audio events from ground-truth source
positions. Used by --oracle-mode pipelines to bypass the real SLED model and
train/evaluate SmolVLA under a "perfect perception + controlled noise" regime.

Design
------
The oracle is intentionally factored into two stages so the same code path can
be shared between:

  1. **Dataset-time GT extraction** — run once per episode; records clean
     (class_id, xpos, az, el) for every active sound source in a task-agnostic
     format that future multi-source / moving-source tasks can extend.

  2. **Runtime noise injection** — cheap per-batch operation that converts GT
     into noisy top-K events (the representation the model actually consumes).
     Keeping noise out of the stored dataset lets training do fresh draws each
     epoch and lets eval re-use the exact same noise distribution.

Sign conventions
----------------
* `compute_listener_relative_direction` returns azimuth in **HRTF** convention
  (0=front, +90=LEFT, −90=RIGHT).
* SmolVLA was trained on SLED outputs, which use `az_sled = arctan2(dy, dx)`
  where dy>0 means RIGHT — i.e. `az_sled = -az_hrtf`.
  We therefore **negate** the HRTF azimuth before storing it, so GT and SLED
  predictions share the same sign convention (right-positive).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Noise configuration
# ---------------------------------------------------------------------------
@dataclass
class OracleNoiseConfig:
    """Controls how `build_oracle_topk` perturbs ground-truth events.

    Defaults are tuned to loosely match SLED v5's typical error profile so
    the policy sees a realistic distribution. Set everything to 0 for a
    "perfect oracle" sanity check.
    """
    az_std_deg: float = 3.0        # Gaussian std on azimuth
    el_std_deg: float = 5.0        # Gaussian std on elevation (harder)
    conf_min: float = 0.85         # confidence uniformly sampled in [min, max]
    conf_max: float = 0.98
    class_flip_prob: float = 0.02  # rare misclassification to a random class
    n_classes: int = 38            # taxonomy size; used when flipping class
    # Drop real detections before filling top-K, simulating a source-count
    # underestimate from SLED. Extra detections are simulated by distractors.
    source_drop_prob: float = 0.0
    # "Distractors" are phantom detections filling empty slots (like SLED
    # hallucinations). Disabled by default — enable for extra robustness.
    distractor_prob: float = 0.0
    distractor_conf_max: float = 0.15
    # Gaussian jitter on the [0,1] energy/loudness scalar (spec §9). 0 disables.
    energy_std: float = 0.05
    # ─── Multi-source-task augmentations ──────────────────────────────────
    # `shuffle_slots`: permute the K slot positions per sample at training
    #   time. The dataset stores `[target, distractor, silence]` in slots
    #   `[0, 1, 2]` by convention; without shuffling the policy learns to
    #   shortcut on slot index ("always read slot 0's @ direction") and
    #   never has to use the instruction-↔-class matching that the
    #   two-radio task needs at inference.
    # `target_slot_protect`: freeze the target slot's class (slot index 0
    #   in the dataset, BEFORE shuffling) so class flips only land on
    #   distractor slots. The target's class is the only stable signal that
    #   ties the instruction to the right direction; corrupting it makes
    #   the matching task ambiguous and the gradients noisier.
    # `target_first_slots`: optional ablation/backward-compatibility path that
    #   moves the instruction-target class back to slot 0 after shuffling. Keep
    #   this off for the normal two-radio benchmark because it uses target
    #   knowledge to impose a slot order before the policy sees the scene.
    # `canonicalize_slots`: target-agnostic SELD post-processing. "azimuth"
    #   orders present detections from task-left to task-right using only
    #   their reported direction, stabilising unordered multi-source SELD
    #   outputs without peeking at the instruction.
    shuffle_slots: bool = False
    target_slot_protect: bool = False
    target_first_slots: bool = False
    canonicalize_slots: str = "none"


# ---------------------------------------------------------------------------
# Which camera is the microphone
# ---------------------------------------------------------------------------
# The listener ("binaural mic") pose is a camera pose, and it does not have to
# be the camera whose image the policy sees. For the radio tasks both roles sit
# on camera 2 (the front view). For find_hidden they are deliberately split:
#
#   camera 2 stays the policy image (unchanged, over-the-shoulder at z=1.75)
#   camera 1 becomes the mic, re-posed low and centred in camera_config.json
#
# Why: the elevation cue that distinguishes the top from the bottom drawer is
# the mic's *vertical* offset from the drawers. From z=1.75 both drawers are
# well below the mic and their elevations bunch up; from z=1.20 they straddle
# it. Measured over the 880-episode v2 set, on the projected (u, v) features the
# SlotEncoder actually consumes:
#
#   mic pose                      u d' (left/right)   v d' (top/bottom)  offscreen
#   cam2  (0, -1.05, 1.75)              12.35               5.54            0/880
#   cam1  (-0.775, -0.856, 1.209)        9.89               4.87           12/880
#   cam1' (0, -1.05, 1.20)              12.96               7.72            0/880
#
# Camera 1's stock pose is off-centre to the left, which squeezes the left
# cabinet against the edge of its FOV and pushes 12 episodes off-screen
# entirely (the audio slot then degrades to the (-1,-1) masked-out sentinel), so
# camera 1 is re-posed rather than used as shipped. Camera 1's image is not in
# DEFAULT_CAM_MAP, so re-posing it changes no policy input.
TASK_MIC_CAM = {
    "find_hidden_object_open": 1,
    "find_hidden_object": 1,
}
DEFAULT_MIC_CAM = 2


def resolve_mic_cam_id(task_name: str | None) -> int:
    """Camera index that acts as the binaural listener for `task_name`.

    `VLABENCH_MIC_CAM` overrides everything — needed to evaluate a checkpoint
    trained before this split, whose audio labels were computed from camera 2.
    """
    override = os.environ.get("VLABENCH_MIC_CAM")
    if override:
        return int(override)
    return TASK_MIC_CAM.get(task_name or "", DEFAULT_MIC_CAM)


# ---------------------------------------------------------------------------
# GT extraction from env state
# ---------------------------------------------------------------------------
def _ensure_project_on_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


# Oracle loudness model (mirrors SLED's per-source energy estimate). SLED now
# reports a loudness/energy value derived from the mic signal, which depends on
# source level and distance. For the oracle we synthesise it from the GT
# distance with a simple inverse-distance SPL law, normalised to [0,1] over the
# spec's energy_db_range = [30, 90] dB.
ENERGY_REF_DIST_M = 1.0        # distance at which a source hits ENERGY_BASE_DB
ENERGY_BASE_DB = 75.0          # SPL of a nominal source at ENERGY_REF_DIST_M
ENERGY_DB_MIN = 30.0           # → energy 0
ENERGY_DB_MAX = 90.0           # → energy 1
ENERGY_MIN_DIST_M = 0.2        # clamp so near sources don't blow up


def energy_from_distance(dist_m: float, level_db: float = 0.0) -> float:
    """Synthetic oracle loudness in [0,1] from GT distance.

    ``level_db`` is an optional per-source gain offset (0 = nominal). Uses a
    20·log10 inverse-distance law and the spec's [30,90] dB → [0,1] mapping.
    """
    d = max(float(dist_m), ENERGY_MIN_DIST_M)
    spl = ENERGY_BASE_DB + float(level_db) - 20.0 * np.log10(d / ENERGY_REF_DIST_M)
    e = (spl - ENERGY_DB_MIN) / (ENERGY_DB_MAX - ENERGY_DB_MIN)
    return float(np.clip(e, 0.0, 1.0))


def gt_az_el(cam_pos: np.ndarray,
             cam_xmat: np.ndarray,
             source_xpos: np.ndarray) -> tuple[float, float, float]:
    """Return (az_deg, el_deg, distance_m) in **SLED sign convention**
    (az > 0 means source is on the listener's RIGHT)."""
    _ensure_project_on_path()
    from audio_generation.binaural_engine import compute_listener_relative_direction
    az_hrtf, el_deg, dist = compute_listener_relative_direction(
        np.asarray(cam_pos, dtype=float).reshape(3),
        np.asarray(cam_xmat, dtype=float).reshape(-1),
        np.asarray(source_xpos, dtype=float).reshape(3),
    )
    az_sled = -float(az_hrtf)
    return az_sled, float(el_deg), float(dist)


def extract_episode_gt(env, active_sources: Sequence[dict], cam_id: int,
                       n_frames: int) -> dict:
    """Snapshot ground-truth audio geometry for an episode.

    Parameters
    ----------
    env : VLABench env (post-reset, pre-teardown)
    active_sources : list of {"name": str, "class_id": int, "class_name": str}
        Sound sources that are actively emitting this episode.
    cam_id : listener camera index (usually 2 = front)
    n_frames : number of recorded observation frames

    Returns
    -------
    dict that can be JSON-serialised and stored under
    `meta_info/oracle_audio` in the HDF5 file. The `static=true` form is the
    only one currently supported — per-frame variants can be added later for
    tasks with moving cameras/sources without breaking the reader contract.
    """
    cam_pos = env.physics.data.cam_xpos[cam_id].copy().tolist()
    cam_xmat = env.physics.data.cam_xmat[cam_id].copy().tolist()
    # Vertical FOV of the listener camera → lets the converter build the pinhole
    # intrinsics for the DoA→(u,v) projection (spec M2). MuJoCo stores fovy in
    # degrees per camera.
    cam_fovy = float(env.physics.model.cam_fovy[cam_id])

    src_records = []
    for src in active_sources:
        name = src["name"]
        entity = env.task.entities.get(name)
        if entity is None:
            raise RuntimeError(f"oracle: entity {name!r} not found in env")
        xpos = entity.get_xpos(env.physics).copy().tolist()
        az, el, dist = gt_az_el(cam_pos, cam_xmat, xpos)
        energy = energy_from_distance(dist, level_db=float(src.get("level_db", 0.0)))
        rec = {
            "name":        name,
            "class_id":    int(src["class_id"]),
            "class_name":  src.get("class_name", ""),
            "xpos":        [float(v) for v in xpos],
            "az_deg":      round(az, 4),
            "el_deg":      round(el, 4),
            "distance_m":  round(dist, 4),
            "energy":      round(energy, 4),
        }
        # Optional time-gating for delayed cues (e.g. the microwave chime
        # in `take_out_microwave_food`). Frames in [0, active_from_frame)
        # are silent for this slot; frames in [active_from_frame, n_frames)
        # carry the GT class+direction. Defaults to 0 → fully active, which
        # matches the legacy radio-task semantics.
        afe = int(src.get("active_from_frame", 0))
        if afe > 0:
            rec["active_from_frame"] = afe
        src_records.append(rec)

    return {
        "schema_version":  1,
        "static":          True,
        "cam_id":          int(cam_id),
        "cam_pos":         [float(v) for v in cam_pos],
        "cam_xmat":        [float(v) for v in cam_xmat],
        "cam_fovy_deg":    round(cam_fovy, 6),
        "active_sources":  src_records,
        "n_frames":        int(n_frames),
    }


# ---------------------------------------------------------------------------
# Noise injection (runtime)
# ---------------------------------------------------------------------------
def build_oracle_topk(
    sources: Sequence[dict],          # [{"class_id", "az_deg", "el_deg"}, ...]
    top_k: int,
    noise_cfg: OracleNoiseConfig,
    rng: np.random.Generator,
) -> dict:
    """Convert GT sources → noisy top-K dict matching SLED's output layout.

    First `n_sources` slots are filled (with noise) from `sources`. Remaining
    slots are either "silence" (cid=-1, conf=0) or, if enabled, low-confidence
    distractors.

    Returns dict with numpy arrays: class_id [K], azimuth_deg [K],
    elevation_deg [K], confidence [K] — exactly the shape SLED produces.
    """
    cid = np.full(top_k, -1, dtype=np.int32)
    az  = np.zeros(top_k, dtype=np.float32)
    el  = np.zeros(top_k, dtype=np.float32)
    cf  = np.zeros(top_k, dtype=np.float32)
    en  = np.zeros(top_k, dtype=np.float32)

    real = []
    for src in list(sources)[:top_k]:
        if noise_cfg.source_drop_prob > 0 and rng.random() < noise_cfg.source_drop_prob:
            continue
        real.append(src)
    for k, src in enumerate(real):
        gt_cid = int(src["class_id"])
        if noise_cfg.class_flip_prob > 0 and rng.random() < noise_cfg.class_flip_prob:
            pool = [c for c in range(noise_cfg.n_classes) if c != gt_cid]
            if pool:
                gt_cid = int(rng.choice(pool))
        cid[k] = gt_cid
        az[k]  = float(src["az_deg"])  + (rng.normal(0.0, noise_cfg.az_std_deg)
                                          if noise_cfg.az_std_deg > 0 else 0.0)
        el[k]  = float(src["el_deg"])  + (rng.normal(0.0, noise_cfg.el_std_deg)
                                          if noise_cfg.el_std_deg > 0 else 0.0)
        cf[k]  = float(rng.uniform(noise_cfg.conf_min, noise_cfg.conf_max)
                       if noise_cfg.conf_max > noise_cfg.conf_min
                       else noise_cfg.conf_max)
        base_energy = float(src.get("energy", 1.0))
        if noise_cfg.energy_std > 0:
            base_energy += float(rng.normal(0.0, noise_cfg.energy_std))
        en[k]  = float(np.clip(base_energy, 0.0, 1.0))

    for k in range(len(real), top_k):
        if noise_cfg.distractor_prob > 0 and rng.random() < noise_cfg.distractor_prob:
            cid[k] = int(rng.integers(0, noise_cfg.n_classes))
            az[k]  = float(rng.uniform(-180.0, 180.0))
            el[k]  = float(rng.uniform(-45.0, 45.0))
            cf[k]  = float(rng.uniform(0.0, noise_cfg.distractor_conf_max))
            en[k]  = float(rng.uniform(0.0, 0.2))

    return {
        "class_id":      cid,
        "azimuth_deg":   az,
        "elevation_deg": el,
        "confidence":    cf,
        "energy":        en,
    }


# ---------------------------------------------------------------------------
# Runtime post-processing: temporal smoother for noisy class predictions
# ---------------------------------------------------------------------------
class TopKClassSmoother:
    """Non-ML smoother that suppresses occasional class flips in the top-K
    audio snapshot fed to the policy at inference time.

    The oracle (and real SLED) pipeline can produce a wrong class ID for a
    single frame even when the underlying source is stable — e.g. a 5%
    `class_flip_prob` realises ~once every 20 frames. The trained policy
    has to be robust to that noise, but at evaluation we can also do a
    cheap, structure-external clean-up: keep a sliding window of recent
    class IDs per slot and emit the *mode* (most-frequent value) instead
    of the raw current frame.

    What it does NOT touch:
      * Confidence (the model still sees the original confidence)
      * Azimuth / elevation (kept frame-fresh; geometry usually moves
        slowly enough that explicit smoothing isn't needed, and over-
        smoothing position would hide real motion of the source).

    The smoother is per-slot, so slot semantics are preserved across
    frames. Slot identity here is positional (slot 0 = first source the
    snapshot reports); for `select_radio_two` both slots are stable
    across the episode because `extract_episode_gt` lists active sources
    in a fixed order.
    """
    def __init__(self, top_k: int, window: int = 5,
                 silence_id: int = -1, prefer_real_class: bool = True):
        from collections import deque
        self.top_k = int(top_k)
        self.window = int(max(1, window))
        self.silence_id = int(silence_id)
        self.prefer_real_class = bool(prefer_real_class)
        self._hist = [deque(maxlen=self.window) for _ in range(self.top_k)]

    def reset(self) -> None:
        for d in self._hist:
            d.clear()

    def smooth(self, class_id_arr: np.ndarray) -> np.ndarray:
        """Append the current frame to history and return the smoothed array.
        Input/output shape: (top_k,) int. Slot ordering is preserved."""
        out = np.array(class_id_arr, dtype=np.int64).copy()
        for k in range(min(self.top_k, out.shape[0])):
            self._hist[k].append(int(out[k]))
            buf = list(self._hist[k])
            if not buf:
                continue
            # Drop silence votes from the tally when at least one real-class
            # vote exists. This stops a single -1 dropout (e.g. SLED missing
            # a window) from flipping the slot back to "silent".
            if self.prefer_real_class:
                real_votes = [c for c in buf if c != self.silence_id]
                if real_votes:
                    buf = real_votes
            vals, counts = np.unique(np.array(buf, dtype=np.int64),
                                     return_counts=True)
            out[k] = int(vals[int(np.argmax(counts))])
        return out


def build_oracle_topk_batched(
    gt_sources_per_sample: list[list[dict]],
    top_k: int,
    noise_cfg: OracleNoiseConfig,
    rng: np.random.Generator,
) -> dict:
    """Batched version for training-time noise injection. Returns numpy arrays
    of shape [B, K] for each field."""
    B = len(gt_sources_per_sample)
    out = {
        "class_id":      np.full((B, top_k), -1, dtype=np.int32),
        "azimuth_deg":   np.zeros((B, top_k), dtype=np.float32),
        "elevation_deg": np.zeros((B, top_k), dtype=np.float32),
        "confidence":    np.zeros((B, top_k), dtype=np.float32),
        "energy":        np.zeros((B, top_k), dtype=np.float32),
    }
    for b in range(B):
        snap = build_oracle_topk(gt_sources_per_sample[b], top_k, noise_cfg, rng)
        for key in out:
            out[key][b] = snap[key]
    return out
