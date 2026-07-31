import os
import random
from functools import partial
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.utils.skill_lib import SkillLib
from VLABench.utils.utils import flatten_list, euler_to_quaternion, distance
from VLABench.algorithms.utils import interpolate_path

# --- candidate-slot taxonomy ---------------------------------------------------
# The hidden object lives in one of 2 (azimuth) x 2 (elevation) drawer slots.
# Two identical cabinets are placed left / right (azimuth cue) and the object
# hides in the TOP or BOTTOM drawer (elevation cue). We deliberately skip the
# middle drawer so the two elevation classes are ~0.25m apart (top handle z~1.04
# vs bottom z~0.79) instead of the ~0.12m adjacent spacing, making the elevation
# audio cue distinguishable; the (still present) middle drawer acts as a visual
# distractor that is never the answer. Vision alone cannot tell which closed
# drawer holds the object, so audio direction (azimuth *and* elevation) is the
# only disambiguating signal.
AZIMUTH_LABELS = ["left", "right"]
ELEVATION_LABELS = ["top", "bottom"]

# task instruction (English, kept in one place so data-gen, the LeRobot
# conversion instruction-fix, and eval all use the exact same string).
FIND_HIDDEN_OPEN_INSTRUCTION = (
    "Open the drawer of the cabinet that contains the object making the sound."
)

# world layout of the two cabinets (tuned in verification against real geometry)
CABINET_X = {"left": -0.30, "right": 0.30}
CABINET_Y = 0.16  # pulled toward the robot to improve reach/grasp (esp. bottom)
CABINET_Z = 0.78  # table surface height

# --- scene variation (per-episode jitter) -------------------------------------
# The task used to be almost fully deterministic (cabinets at fixed x=+-0.30,
# object at a fixed in-drawer pose), so azimuth was always ~+-14 deg and the
# policy could memorise the geometry / mode-average instead of using the audio.
# We now jitter cabinet placement and the in-drawer object pose per episode.
# The |x| range is widened (0.26..0.42) so the azimuth cue also spans a wider,
# more informative range. z stays fixed (grasp-critical). Disable for debugging
# / audio ablation with VLABENCH_HIDDEN_NO_JITTER=1.
# Ranges kept reachable for the Franka so the oracle grasp/pull stays reliable.
# The bottom drawer at full lateral extension is the fragile case, so |x| stays
# modest and yaw small (fixed grasp eulers can't absorb much cabinet yaw). Most
# of the added scene variation comes from the in-drawer object jitter (which
# moves the sound source az/el) and the depth (y) jitter, neither of which hurts
# grasp reach the way a wide |x| does.
CABINET_ABS_X_RANGE = (0.27, 0.33)   # per-side |x|, sign set by left/right
CABINET_Y_RANGE = (0.12, 0.18)       # depth (toward/away from robot)
CABINET_YAW_RANGE = (-0.03, 0.03)    # tiny yaw (rad, ~+-1.7 deg); grasp still ok
# in-drawer object offset jitter (added to DRAWER_LOCAL_POS), world-axis metres.
# x moves the hidden sound source sideways (varies azimuth for a fixed slot),
# y keeps it inside the closed drawer, z nudges elevation without crossing slots.
OBJECT_JITTER_X = (-0.035, 0.035)
OBJECT_JITTER_Y = (-0.02, 0.02)
OBJECT_JITTER_Z = (-0.015, 0.015)


def _jitter_on():
    return os.environ.get("VLABENCH_HIDDEN_NO_JITTER", "0") != "1"


def _u(lo_hi):
    lo, hi = lo_hi
    return float(np.random.uniform(lo, hi))

# per-drawer object placement, as a WORLD-axis offset from the cabinet origin
# (subentity offsets are added to the parent position without the parent's
# rotation). z sets the elevation (world z 1.12 / 0.86); y keeps the object
# inside the closed drawer (hidden). Note: round objects roll toward the drawer
# back during the reset settle, so front-biasing y doesn't reliably keep them
# forward -- use boxy objects if front exposure matters for the retrieve pick.
DRAWER_LOCAL_POS = {
    "top":    [0.0, -0.05, 0.34],
    "bottom": [0.0, -0.05, 0.08],
}
# elevation label -> drawer slide-joint / keypoint index (0=top, 1=middle, 2=bottom),
# consistent across all cabinet assets (XML body/joint order is top/middle/bottom).
ELEVATION_DRAWER_ID = {"top": 0, "bottom": 2}
# small graspable objects that hide inside a drawer. Boxy (non-rolling) snacks
# so a front-biased placement stays put and the object rides out on the opened
# drawer where it can be picked top-down.
DEFAULT_SEEN_OBJECTS = ["boxed_food", "bar", "chocolate"]
DEFAULT_UNSEEN_OBJECTS = ["chips", "bagged_food"]

# Fix the cabinet to a single asset so the oracle grasp is reliable (the 3
# cabinet assets have different handle geometry). Overridable via env for tuning.
_CABINET_ASSETS = {
    "wooden": "assets/obj/meshes/containers/cabinets/wooden_cabinet/wooden_cabinet.xml",
    "white":  "assets/obj/meshes/containers/cabinets/white_cabinet/white_cabinet_fix.xml",
    "short":  "assets/obj/meshes/containers/cabinets/short_cabinet/short_cabinet_fix.xml",
}


def _cabinet_xml():
    # white_cabinet's handle geometry is the one the oracle grasp reliably picks
    # and pulls open (verified); wooden/short fail the handle grasp.
    key = os.environ.get("VLABENCH_CABINET_ASSET", "white")
    rel = _CABINET_ASSETS.get(key, _CABINET_ASSETS["white"])
    return os.path.join(os.getenv("VLABENCH_ROOT", ""), rel)


def _soften_cabinet_drawers(entity, damping=1.0, frictionloss=0.0):
    """Lower the drawer slide-joint damping (assets ship with damping=50, which
    resists the pull hard enough that the gripper slips off the handle before the
    drawer opens). Done on the mjcf before physics compile.
    (frictionloss default 0: it was tried up to 15 to keep the drawer from being
    shoved shut during the composite object-pick, but the position-controlled arm
    overwhelms any usable value AND it hurts the primitive open, so it's off.)"""
    for joint in entity.joints:
        try:
            joint.damping = damping
            if frictionloss:
                joint.frictionloss = frictionloss
        except Exception:
            pass


# --- scene sanity gates (data generation) -------------------------------------
# `_soften_cabinet_drawers` drops the slide damping 50 -> 1 so the oracle can
# actually pull a drawer open. Damping only resists *velocity*, so nothing holds
# a drawer statically any more: the hidden object dropping into place during the
# reset settle can shove its drawer out by several centimetres. When that
# happens on the *target* drawer the `drawer_open` condition is already met
# before the expert moves, the first `env.step` returns a terminal timestep and
# the generator writes out a "successful" episode that contains only the
# stay-still prefix (~2 s, zero task content). It also breaks the task premise
# in two other ways: an ajar drawer is a *visual* giveaway of which drawer to
# open, and an object that rides out with the drawer is no longer hidden and no
# longer sits at the elevation its slot label claims.
#
# The three functions below let the generator repair what is repairable
# (`close_all_drawers`) and reject what is not (`validate_hidden_scene`,
# `validate_hidden_episode`).

# A drawer at or below this open fraction is "shut" for our purposes: 0.02 of a
# 0.32 m travel is 6 mm, below the visible-gap threshold and far below the 0.13
# success threshold.
DRAWER_SHUT_FRACTION = 0.02
# Minimum recorded frames / EE travel for an episode to count as a real
# demonstration. Measured on the 880-episode v2 set: the 10 degenerate episodes
# had 20-39 frames and <= 0.176 m of travel, while every genuine episode had
# >= 64 frames and >= 0.208 m, so these thresholds separate the two cleanly.
MIN_EPISODE_FRAMES = 45
MIN_EE_TRAVEL_M = 0.20
# World-axis tolerances for "the object is still hidden in its labelled drawer",
# checked against the settled position rather than the spawn position (the
# object drops ~2 cm onto the drawer floor during the reset settle).
OBJECT_MAX_ABS_DX = 0.08      # lateral drift inside the drawer
OBJECT_MIN_DEPTH_BEHIND_HANDLE = 0.06   # obj_y - handle_y; < this means it's out
OBJECT_MAX_DZ_ERROR = 0.05    # vertical error vs the labelled drawer level
# Settled height of the object above the cabinet origin, per elevation label.
# Nominal spawn is DRAWER_LOCAL_POS[...][2]; the object then falls onto the
# drawer floor, which lands it ~0.02 m lower (measured: top 0.326 +- 0.046,
# bottom 0.058 +- 0.018 over 880 episodes).
SETTLED_LOCAL_Z = {"top": 0.326, "bottom": 0.058}
# The middle drawer is a distractor that is never the answer, but the object can
# fall into it. Used to reject episodes whose object is nearer the middle drawer
# than its own label -- those carry an elevation cue that points at the wrong
# drawer entirely.
# (0.196 = top floor minus the ~0.13 m drawer pitch; it matches the observed
# cluster of "top"-labelled episodes whose object had fallen one level down.)
ALL_LEVEL_LOCAL_Z = {"top": 0.326, "middle": 0.196, "bottom": 0.058}


def cabinet_entities(env):
    """The task's cabinets, as {entity_name: entity}."""
    return {k: v for k, v in env.task.entities.items() if "cabinet" in k}


def drawer_open_fractions(env):
    """{f"{cabinet}/{joint}": open_fraction} for every drawer in the scene.

    Mirrors `DrawerOpenCondition.open_fraction`: |qpos| / max(|lo|, |hi|), which
    is ~0 shut and ~1 fully open regardless of which way the asset's slide range
    is signed.
    """
    out = {}
    for name, entity in cabinet_entities(env).items():
        for joint in entity.joints:
            bound = env.physics.bind(joint)
            qpos = float(np.asarray(bound.qpos).ravel()[0])
            lo, hi = [float(v) for v in np.asarray(bound.range).ravel()[:2]]
            span = max(abs(lo), abs(hi))
            out[f"{name}/{joint.name}"] = abs(qpos) / span if span else 0.0
    return out


def _snapshot_robot_pose(env):
    """Every robot joint's (joint, qpos) so it can be put back exactly."""
    return [(joint, float(np.asarray(env.physics.bind(joint).qpos).ravel()[0]))
            for joint in env.robot.joints]


def _restore_robot_pose(env, snapshot):
    """Undo any drift the settle steps introduced, then refresh derived state.

    `env.step(None)` commands the arm to its *current* qpos (see
    LM4ManipDMEnv.step), so gravity sag between steps is accepted as the new
    target — a ratchet. Every extra settle step therefore lowers the arm a little,
    and eval performs no such steps. Measured over 6 resets, deterministically:
    the end effector rests at base-frame z=0.4319 straight after `env.reset()`
    (exactly what eval records) but at 0.4137 after `close_all_drawers` — an
    18 mm drop baked into the first recorded frame of every episode. That is the
    same class of train/eval divergence as the 28.8 cm base-frame bug, just
    smaller, so the arm is put back where the reset left it.

    Safe to teleport: at rest the arm is nowhere near the cabinets, so no contact
    is being resolved. `physics.forward()` recomputes xpos/xmat without stepping.
    """
    for joint, qpos in snapshot:
        bound = env.physics.bind(joint)
        bound.qpos = qpos
        bound.qvel = 0.0
    env.physics.forward()


def close_all_drawers(env, settle_steps=10, max_rounds=3,
                      shut_fraction=DRAWER_SHUT_FRACTION):
    """Force every drawer shut and let the scene re-settle. Returns True if all
    drawers ended up shut.

    Iterates because one round is not always enough: the hidden object is still
    in motion right after the reset settle, so it can push its drawer back out
    while it comes to rest. Once it is resting on the drawer floor a re-close
    sticks. Measured over 16 resets of the fragile bottom slot, 4 had drifted
    past the threshold at raw reset (worst 0.118) and *all* of them were shut
    (worst 0.004) after a single round — the extra rounds are for the tail.

    The settle steps would otherwise leave the arm 18 mm lower than eval's rest
    pose, so the robot is snapshotted up front and restored before returning —
    see `_restore_robot_pose`.
    """
    robot_pose = _snapshot_robot_pose(env)
    shut = False
    for _ in range(int(max_rounds)):
        for entity in cabinet_entities(env).values():
            for joint in entity.joints:
                bound = env.physics.bind(joint)
                bound.qpos = 0.0
                bound.qvel = 0.0
        for _ in range(int(settle_steps)):
            env.step()
            # If a drawer drifted far enough to satisfy the success condition,
            # this step returned a terminal timestep — and dm_control's
            # composer.Environment silently calls reset() on the *next* step,
            # re-randomising the whole scene behind our back. Stop stepping and
            # let the caller reject the episode instead.
            if env.task.conditions.is_met(env.physics):
                _restore_robot_pose(env, robot_pose)
                return False
        if max(drawer_open_fractions(env).values(), default=0.0) <= shut_fraction:
            shut = True
            break
    _restore_robot_pose(env, robot_pose)
    return shut


def _target_geometry(env):
    """(cabinet entity, drawer id, cabinet world pos, handle world pos, object)."""
    cm = env.task.config_manager
    cab = env.task.entities[cm.target_cabinet_name]
    drawer_id = ELEVATION_DRAWER_ID[cm.target_elevation]
    cab_pos = np.array(cm.cabinet_positions[cm.target_side], dtype=float)
    handle = np.array(cab.get_drawer_handle_pos(env.physics, drawer_id), dtype=float)
    obj = env.task.entities[cm.target_entity]
    return cab, drawer_id, cab_pos, handle, obj


def _object_placement_problems(env):
    """Reasons the hidden object is not properly hidden in its labelled drawer."""
    cm = env.task.config_manager
    _, _, cab_pos, handle, obj = _target_geometry(env)
    obj_pos = np.array(obj.get_xpos(env.physics), dtype=float)
    d = obj_pos - cab_pos
    problems = []
    if abs(d[0]) > OBJECT_MAX_ABS_DX:
        problems.append(f"object drifted sideways (dx={d[0]:+.3f})")
    depth = float(obj_pos[1] - handle[1])
    if depth < OBJECT_MIN_DEPTH_BEHIND_HANDLE:
        problems.append(f"object is out in front of the drawer (obj_y-handle_y={depth:+.3f})")
    expected_z = SETTLED_LOCAL_Z[cm.target_elevation]
    if abs(d[2] - expected_z) > OBJECT_MAX_DZ_ERROR:
        problems.append(
            f"object left the {cm.target_elevation} drawer "
            f"(dz={d[2]:+.3f}, expected {expected_z:+.3f})"
        )
    nearest = min(ALL_LEVEL_LOCAL_Z, key=lambda k: abs(d[2] - ALL_LEVEL_LOCAL_Z[k]))
    if nearest != cm.target_elevation:
        problems.append(
            f"object is nearest the {nearest} drawer but the label says "
            f"{cm.target_elevation} (dz={d[2]:+.3f})"
        )
    return problems


def validate_hidden_scene(env):
    """Pre-flight check, run after reset (and after `close_all_drawers`).

    Returns a list of human-readable reasons the episode should be discarded;
    empty means the scene is usable. Rejecting here costs one wasted reset,
    which is far cheaper than letting a degenerate episode into the dataset.
    """
    problems = []
    fracs = drawer_open_fractions(env)
    ajar = {k: v for k, v in fracs.items() if v > DRAWER_SHUT_FRACTION}
    if ajar:
        problems.append(
            "drawer(s) not shut at episode start: "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(ajar.items()))
        )
    if env.task.conditions.is_met(env.physics):
        problems.append("success condition already met before the expert moved")
    problems.extend(_object_placement_problems(env))
    return problems


def validate_hidden_episode(env, n_frames, ee_travel_m):
    """Post-hoc check, run once the expert sequence has finished.

    Catches the "drawer opened on its own" episodes that slipped past the
    pre-flight check (the condition can also fire mid-approach, before the
    gripper ever reaches the handle) and episodes where the object left the
    target drawer during the pull, which would mislabel the stored audio cue.
    """
    problems = []
    if n_frames < MIN_EPISODE_FRAMES:
        problems.append(
            f"episode too short: {n_frames} frames < {MIN_EPISODE_FRAMES} "
            f"(drawer likely opened without a real reach+pull)"
        )
    if ee_travel_m < MIN_EE_TRAVEL_M:
        problems.append(
            f"end effector barely moved: {ee_travel_m:.3f} m < {MIN_EE_TRAVEL_M} m"
        )
    return problems


def _pull_open(env, target_pos, n_substep=20):
    """Pull the (already-grasped) drawer handle to `target_pos` with enough
    physics substeps per waypoint that the position-controlled arm actually
    reaches it. SkillLib.pull steps only once per waypoint (max_n_substep=1), so
    the arm lags and the drawer only cracks ~10cm open."""
    start = np.array(env.robot.get_end_effector_pos(env.physics))
    start_quat = np.array(env.robot.get_end_effector_quat(env.physics))
    path, quats = interpolate_path([start, np.array(target_pos)],
                                   [start_quat, start_quat])
    return SkillLib.step_trajectory(env, path, quats, np.zeros(2),
                                    max_n_substep=20, tolerance=0.01)


def _pick_object_topdown(env, obj_name, release_lift=0.18, prepare_height=0.18,
                         grasp_gap=0.10):
    """Release the drawer handle and pick the revealed object with a controlled
    STRAIGHT top-down descent (no RRT swing into the drawer). Crucially we let go
    of the handle and lift STRAIGHT UP (+z only) first: any -y retreat drags the
    drawer shut and pulls the object back to the cabinet mouth. Then we go above
    the object and descend straight down onto it."""
    phys = env.physics
    obj = env.task.entities[obj_name]
    quat = euler_to_quaternion(-np.pi, 0, 0)  # face straight down
    open_g = np.ones(2) * 0.04
    obs_all, wp_all = [], []
    # 1) release the handle and lift straight up in place (no y motion)
    o, w, _, _ = SkillLib.move_offset(env, offset=[0, 0, release_lift],
                                      gripper_state=open_g)
    obs_all += o; wp_all += w
    # read the object AFTER releasing (it stays put, drawer still open)
    obj_pos = np.array(phys.bind(obj.mjcf_model.worldbody).xpos)
    # 2) move clear above the object (RRT is fine well above the drawer)
    o, w, _, _ = SkillLib.moveto(env, obj_pos + np.array([0, 0, prepare_height]),
                                 quat, gripper_state=open_g)
    obs_all += o; wp_all += w
    # 3) straight vertical descent onto the object (no RRT)
    start = np.array(env.robot.get_end_effector_pos(phys))
    target = obj_pos + np.array([0, 0, grasp_gap])
    path, quats = interpolate_path([start, target], [quat, quat])
    o, w, _, _ = SkillLib.step_trajectory(env, path, quats, open_g, max_n_substep=20)
    obs_all += o; wp_all += w
    # 4) close the gripper
    o, w, _, ts = SkillLib.close_gripper(env)
    obs_all += o; wp_all += w
    grasped = env.task.entities[obj_name].is_grasped(phys, env.robot)
    return obs_all, wp_all, grasped, ts


def _forced_slot():
    """Optional override so balanced datasets can force a specific slot.

    VLABENCH_HIDDEN_SLOT_LABEL = "<azimuth>_<elevation>", e.g. "left_top".

    A comma-separated list restricts the draw to that *set* instead of pinning a
    single slot, picking uniformly per episode: "left_top,right_top" evaluates
    the two top drawers with the same balance the generator produces, in one
    run. Generation still passes a single label (one job per slot).
    """
    label = os.environ.get("VLABENCH_HIDDEN_SLOT_LABEL")
    if not label:
        return None
    choices = []
    for item in label.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            side, elevation = item.split("_")
        except ValueError:
            raise ValueError(
                "VLABENCH_HIDDEN_SLOT_LABEL must be '<azimuth>_<elevation>' or a "
                f"comma-separated list of them, got: {label}"
            )
        if side not in AZIMUTH_LABELS or elevation not in ELEVATION_LABELS:
            raise ValueError(
                f"Invalid slot '{item}'. azimuth in {AZIMUTH_LABELS}, "
                f"elevation in {ELEVATION_LABELS}."
            )
        choices.append((side, elevation))
    if not choices:
        raise ValueError(f"VLABENCH_HIDDEN_SLOT_LABEL is empty: {label!r}")
    return random.choice(choices)


@register.add_config_manager("find_hidden_object_open")
class FindHiddenObjectOpenConfigManager(BenchTaskConfigManager):
    """Two identical cabinets (left/right) each with top/middle/bottom drawers.

    The target object is hidden inside one drawer; the robot must localize it by
    sound and open *that* drawer. Only the geometry + target slot are committed
    here; the actual sound-file / class assignment happens downstream
    (trajectory_generation.py / eval_smolvla_audio.py), mirroring the radio tasks.
    """

    def __init__(self, task_name, num_objects=[1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        if self.seen_object is None:
            self.seen_object = DEFAULT_SEEN_OBJECTS
        if self.unseen_object is None:
            self.unseen_object = DEFAULT_UNSEEN_OBJECTS

    def get_seen_task_config(self):
        target_entity = random.choice(flatten_list(self.seen_object))
        return self.get_task_config(target_entity, None, None)

    def get_unseen_task_config(self):
        target_entity = random.choice(flatten_list(self.unseen_object))
        return self.get_task_config(target_entity, None, None)

    def load_containers(self, target_container):
        # place the two identical cabinets left / right (azimuth cue). Each
        # cabinet is independently jittered in x (|x| widened), y (depth) and a
        # small yaw so the scene geometry -- and hence the azimuth cue -- varies
        # per episode instead of being memorisable. z is fixed (grasp-critical).
        jitter = _jitter_on()
        self.cabinet_names = {}
        self.cabinet_positions = {}
        for side in AZIMUTH_LABELS:
            name = f"cabinet_{side}"
            if jitter:
                sign = -1.0 if side == "left" else 1.0
                pos = [sign * _u(CABINET_ABS_X_RANGE), _u(CABINET_Y_RANGE), CABINET_Z]
                yaw = _u(CABINET_YAW_RANGE)
            else:
                pos = [CABINET_X[side], CABINET_Y, CABINET_Z]
                yaw = 0.0
            cabinet_config = self.get_entity_config(
                "cabinet",
                position=pos,
                orientation=[0, 0, yaw],
                specific_name=name,
                xml_path=_cabinet_xml(),
            )
            cabinet_config["subentities"] = []
            self.config["task"]["components"].append(cabinet_config)
            self.cabinet_names[side] = name
            self.cabinet_positions[side] = pos

        # pick the target slot (azimuth x elevation)
        forced = _forced_slot()
        if forced is not None:
            self.target_side, self.target_elevation = forced
        else:
            self.target_side = random.choice(AZIMUTH_LABELS)
            self.target_elevation = random.choice(ELEVATION_LABELS)
        self.target_cabinet_name = self.cabinet_names[self.target_side]
        self.position_label = f"{self.target_side}_{self.target_elevation}"

    def load_objects(self, target_entity):
        cabinet_config = next(
            c for c in self.config["task"]["components"]
            if c.get("name") == self.target_cabinet_name
        )
        base = DRAWER_LOCAL_POS[self.target_elevation]
        if _jitter_on():
            local_pos = [base[0] + _u(OBJECT_JITTER_X),
                         base[1] + _u(OBJECT_JITTER_Y),
                         base[2] + _u(OBJECT_JITTER_Z)]
        else:
            local_pos = list(base)
        object_config = self.get_entity_config(target_entity, position=local_pos)
        cabinet_config["subentities"].append(object_config)
        # (the object is loaded as a cabinet subentity, so the framework already
        # adds it to random_ignored_entities; no explicit append needed.)

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = [FIND_HIDDEN_OPEN_INSTRUCTION]

    def get_condition_config(self, target_entity, **kwargs):
        self.config["task"]["conditions"] = dict(
            drawer_open=dict(
                container=self.target_cabinet_name,
                elevation=self.target_elevation,
                # The fixed-wrist pull hits the Franka's reach limit for the
                # BOTTOM drawer at open_fraction ~0.20 (diagnostic: frac_pull
                # clustered 0.19-0.22 across episodes), so a 0.2 threshold sat
                # right on the achievable max and made success a coin-flip. Set
                # 0.13 (~4cm, unambiguously open, object exposed) so the bottom
                # drawer reliably clears it with margin. Applies to both data
                # generation and eval (both read this task condition).
                open_threshold=0.13,
            )
        )


@register.add_task("find_hidden_object_open")
class FindHiddenObjectOpenTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        # fix the cabinets to the world so they don't tip over when pulled open
        for key, entity in self.entities.items():
            if "cabinet" in key:
                entity.detach()
                self._arena.attach(entity)
                _soften_cabinet_drawers(entity)

    # ------------------------------------------------------------------
    # CompositeTask stubs out intention/progress tracking (no-ops), which
    # breaks env.get_intention_score / get_task_progress for this task. Restore
    # the base behavior here, and — because the target object is an ignored
    # subentity hidden in a drawer — make sure it is still tracked so the
    # intention score (min EE→target distance) is meaningful: it reflects
    # whether the robot approached the *correct* drawer.
    # ------------------------------------------------------------------
    def _target_names(self):
        t = self.target_entity
        return t if isinstance(t, list) else [t]

    def reset_intention_distance(self):
        self.intention_distance = {}
        names = [k for k in self.entities.keys()
                 if k not in self.random_ignored_entities]
        for t in self._target_names():
            if t in self.entities and t not in names:
                names.append(t)
        for n in names:
            self.intention_distance[n] = np.inf

    def update_intention_distance(self, physics):
        ee_pos = self.robot.get_end_effector_pos(physics)
        for key in self.intention_distance:
            entity = self.entities.get(key)
            if entity is None:
                continue
            self.intention_distance[key] = min(
                self.intention_distance[key],
                distance(ee_pos, entity.get_xpos(physics)),
            )

    def reset_task_progress(self):
        self.target_is_grasped = {t: False for t in self._target_names()}

    def update_task_progress(self, physics):
        for t in self.target_is_grasped:
            if self.entities[t].is_grasped(physics, self.robot):
                self.target_is_grasped[t] = True

    def get_expert_skill_sequence(self, physics):
        # Grasp the target drawer's handle (face-forward / vertical gripper) and
        # pull it open *along the drawer's real slide axis*, toward the robot.
        # We avoid SkillLib.open_drawer (its get_drawer_open_trajectory derives
        # the direction from the joint-range sign, which is inverted on these
        # cabinet assets, range like [-0.32, 0.01]) and avoid a pure -y pull
        # (fights the slide constraint and pops the handle out of the grip).
        cm = self.config_manager
        drawer_id = ELEVATION_DRAWER_ID[cm.target_elevation]
        cab = self.entities[cm.target_cabinet_name]
        handle = np.array(cab.get_drawer_handle_pos(physics, drawer_id))
        joint = cab.joints[drawer_id]
        axis = np.array(physics.bind(joint).xaxis, dtype=float)
        rng = np.asarray(physics.bind(joint).range).ravel()[:2]
        span = abs(rng[0]) if abs(rng[0]) > abs(rng[1]) else abs(rng[1])
        # opening moves the drawer toward the robot (-y); pick the axis sign whose
        # y-component is negative so the pull follows the slide out of the cabinet.
        open_dir = axis if axis[1] < 0 else -axis
        target = handle + open_dir * (0.9 * span)
        return [
            partial(SkillLib.pick,
                    target_entity_name=cm.target_cabinet_name,
                    prior_eulers=[[-np.pi / 2, 0, 0], [-np.pi / 2, -np.pi / 2, 0]],
                    specific_keypoint=drawer_id),
            partial(_pull_open, target_pos=target),
        ]
