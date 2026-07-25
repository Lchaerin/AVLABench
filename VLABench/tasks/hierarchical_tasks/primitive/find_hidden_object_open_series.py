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
    """
    label = os.environ.get("VLABENCH_HIDDEN_SLOT_LABEL")
    if not label:
        return None
    try:
        side, elevation = label.split("_")
    except ValueError:
        raise ValueError(
            "VLABENCH_HIDDEN_SLOT_LABEL must be '<azimuth>_<elevation>', "
            f"got: {label}"
        )
    if side not in AZIMUTH_LABELS or elevation not in ELEVATION_LABELS:
        raise ValueError(
            f"Invalid slot '{label}'. azimuth in {AZIMUTH_LABELS}, "
            f"elevation in {ELEVATION_LABELS}."
        )
    return side, elevation


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
