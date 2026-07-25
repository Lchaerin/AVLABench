import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask
from VLABench.utils.skill_lib import SkillLib
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

@register.add_config_manager("heat_food")
class HeatFoodConfigManager(BenchTaskConfigManager):
    """
    Heat the cooked foods instead of raw foods.
    """
    disturbance_objects = ["ingredient", "canned_food"]
    def __init__(self, 
                 task_name,
                 num_objects = [3],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_containers(self, target_container):
        super().load_containers(target_container)
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.15, -0.1), 
                                                             random.uniform(0.3, 0.4), 
                                                             0.8]
    
    def load_init_containers(self, init_container):
        if init_container is not None:
            self.init_container_config = self.get_entity_config(init_container,
                                                           position=[random.uniform(0.25, 0.35), 
                                                                    random.uniform(-0.1, 0.), 
                                                                    0.8],
                                                           orientation=[0, 0, np.pi/2])
            self.config["task"]["components"].append(self.init_container_config)
        
    def load_objects(self, target_entity):
        self.init_container_config["subentities"] = []
        
        objects = []
        objects.append(target_entity)
        objects.extend([random.choice(self.disturbance_objects) for _ in range(self.num_object-1)])
        random.shuffle(objects)
        for i, object in enumerate(objects):
            pos = [-0.1 + 0.1*i + random.uniform(-0.02, 0.02), random.uniform(-0.05, 0.05), 0.05]
            object_config = self.get_entity_config(object, 
                                                   position=pos,
                                                   orientation=[0, 0, -np.pi/2])
            self.init_container_config["subentities"].append(object_config)
    
    def get_instruction(self, target_entity, init_container, **kwargs):
        self.config["task"]["instructions"] = [f"Please heat {target_entity} from {init_container}."]
        return self.config
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        condition_config = dict(
            contain=dict(
                entities=[target_entity],
                container=target_container
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_config_manager("plug_cord_and_heat_food")
class PlugCordAndHeatFoodConfigManager(HeatFoodConfigManager):
    def get_instruction(self, target_entity, init_container, **kwargs):
        self.config["task"]["instructions"] = [f"Please plug the cord and heat {target_entity} from {init_container}."]
        return self.config
    
    def load_containers(self, target_container):
        super().load_containers(target_container)
        cord_config = self.get_entity_config("cord", position=[-0.2, 0.34, 0], randomness=None)
        self.config["task"]["components"][-1]["subentities"] = [cord_config]

@register.add_task("heat_food")
class HeatFoodTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):    
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
                if "microwave" in key:
                    entity.detach()
                    self._arena.attach(entity)
    
    def should_terminate_episode(self, physics):
        condition_met = super().should_terminate_episode(physics)
        is_closed = self.entities[self.config_manager.target_container].is_closed(physics)
        is_active = self.entities[self.config_manager.target_container].is_activate(physics)
        success = condition_met and is_closed and is_active
        return success

    def get_expert_skill_sequence(self, physics):
        target_container_pos = self.entities[self.target_container].get_xpos(physics)
        start_button_pos = np.array(self.entities[self.target_container].get_start_button_pos(physics))
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_container, prior_eulers=[[np.pi, 0, -np.pi/2]]), 
            partial(SkillLib.open_door, target_container_name=self.target_container), 
            partial(SkillLib.lift, gripper_state=np.ones(2)*0.04, lift_height=0.1),
            partial(SkillLib.pull, gripper_state=np.ones(2)*0.04, target_quat=euler_to_quaternion(np.pi, 0, -np.pi/2), pull_distance=0.15),
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi, 0, np.pi/2]]),
            partial(SkillLib.moveto, target_pos=[target_container_pos[0] - 0.1, target_container_pos[1] - 0.2, target_container_pos[2]+0.1], target_quat=euler_to_quaternion(-np.pi*3/4, 0, 0), gripper_state=np.zeros(2)),
            partial(SkillLib.place, target_container_name=self.target_container, target_quat=euler_to_quaternion(-np.pi*3/4, 0, 0)),
            partial(SkillLib.pull, gripper_state=np.ones(2)*0.04),
            partial(SkillLib.lift, gripper_state=np.ones(2)*0.04, lift_height=0.3),
            partial(SkillLib.pick, target_entity_name=self.target_container, prior_eulers=[[-np.pi, 0, 0]]),
            partial(SkillLib.close_door, target_container_name=self.target_container),
            partial(SkillLib.press, target_pos=start_button_pos, target_quat=euler_to_quaternion(-np.pi/2, -np.pi/2, 0), move_vector=[0, -0.2, 0]),
        ]
        return skill_sequence
        
@register.add_task("plug_cord_and_heat_food")
class PlugCordAndHeatFoodTask(HeatFoodTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, config, eval=False):
        super().build_from_config(config, eval)
        for key, entity in self.entities.items():
            if "microwave" in key:
                microwave = entity
            if "cord" in key:
                cord = entity
        cord.detach()
        microwave.attach(cord)

    def after_step(self, physics, random_state):
        # TODO attach plug to the outlet
        return super().after_step(physics, random_state)


# ---------------------------------------------------------------------------
# take_out_microwave_food
#
# Reverse of heat_food: the microwave starts CLOSED with the cooked food
# inside, and an empty tray sits on the table. After a random "cooking
# complete" chime fires (sampled per episode from [min_delay_sec,
# max_delay_sec]), the robot has `react_window_sec` seconds to open the
# microwave door and transfer the food onto the tray.
#
# Failure flags are set in `after_step` but the episode keeps running so
# the dataset / replay still captures the full attempt:
#     * door_opened_before_chime
#     * door_not_opened_in_time
#     * subsequent pick/place failing → contain condition not met → success
#       remains False at the end
# ---------------------------------------------------------------------------

@register.add_config_manager("take_out_microwave_food")
class TakeOutMicrowaveFoodConfigManager(HeatFoodConfigManager):
    """
    Audio-cued microwave unloading.

    Roles inherited from heat_food_series asset layout:
        target_container = microwave (the appliance to operate)
        init_container   = tray      (the destination for the food)
        target_entity    = cooked_food (lives inside the microwave at start)
    """
    disturbance_objects = ["ingredient", "canned_food"]

    def __init__(self,
                 task_name,
                 num_objects=[1],
                 min_delay_sec: float = 0.0,
                 max_delay_sec: float = 20.0,
                 react_window_sec: float = 5.0,
                 step_dt_sec: float = 0.1,
                 **kwargs):
        super().__init__(task_name, num_objects=num_objects, **kwargs)
        # Env-var overrides let shell pipelines (e.g. sh/train_smolvla_take_*.sh)
        # tweak the chime timing without touching code. CLI knobs always win
        # over env vars (kwargs path) — env vars only adjust the *default*.
        import os as _os
        def _envf(name, fallback):
            v = _os.environ.get(name)
            return float(v) if v not in (None, "") else float(fallback)
        self.min_delay_sec    = _envf("VLABENCH_MICROWAVE_MIN_DELAY_SEC",   min_delay_sec)
        self.max_delay_sec    = _envf("VLABENCH_MICROWAVE_MAX_DELAY_SEC",   max_delay_sec)
        self.react_window_sec = _envf("VLABENCH_MICROWAVE_REACT_WINDOW_SEC", react_window_sec)
        # Resolution of the task-side step counter. With the LeRobot/oracle
        # pipeline that ships in this repo, 1 env.step ≈ 1 dataset frame ≈
        # 1/dataset_fps seconds (default 10 fps → 0.1s/step). Override this
        # alongside dataset_fps if you change the pipeline cadence.
        self.step_dt_sec      = _envf("VLABENCH_MICROWAVE_STEP_DT_SEC",     step_dt_sec)
        # Frozen here so the task class, audio config, and expert sequence
        # all agree on a single trigger moment per episode.
        self.trigger_delay_sec = random.uniform(self.min_delay_sec,
                                                self.max_delay_sec)

    @property
    def trigger_step(self) -> int:
        return int(round(self.trigger_delay_sec / max(self.step_dt_sec, 1e-6)))

    @property
    def react_window_steps(self) -> int:
        return int(round(self.react_window_sec / max(self.step_dt_sec, 1e-6)))

    def load_containers(self, target_container):
        # Reuse heat_food's microwave placement, then stash a handle so
        # load_objects() can attach the food as a subentity.
        super().load_containers(target_container)
        self.microwave_config = self.config["task"]["components"][-1]

    def load_init_containers(self, init_container):
        # Empty tray on the table — the food's destination.
        if init_container is not None:
            self.init_container_config = self.get_entity_config(
                init_container,
                position=[random.uniform(0.25, 0.35),
                          random.uniform(-0.1, 0.),
                          0.8],
                orientation=[0, 0, np.pi / 2],
            )
            self.config["task"]["components"].append(self.init_container_config)

    def load_objects(self, target_entity):
        # Place the cooked food (and any distractor ingredients) INSIDE the
        # microwave as subentities. Local z=0.15 matches the height the
        # heat_food expert uses when *placing* food in — i.e. the food sits
        # on the cavity floor once physics settles.
        self.microwave_config["subentities"] = []
        objects = [target_entity]
        objects.extend(
            [random.choice(self.disturbance_objects)
             for _ in range(self.num_object - 1)]
        )
        random.shuffle(objects)
        for i, obj in enumerate(objects):
            # Local (x, y, z) inside the microwave body frame. The internal
            # cavity spans roughly x∈[−0.26, 0.16], y∈[−0.16, 0.22],
            # z∈[−0.12, 0.14]. We bias the spawn towards the *front* of the
            # cavity (smaller y) so the food sits just inside the door
            # opening — much easier for the expert to reach with a single
            # forward-pitched grasp. Local z=0.05 lets physics settle the
            # food onto the cavity floor cleanly.
            pos = [
                -0.03 + 0.05 * i + random.uniform(-0.01, 0.01),
                random.uniform(-0.08, -0.04),
                0.05,
            ]
            cfg = self.get_entity_config(obj,
                                         position=pos,
                                         orientation=[0, 0, -np.pi / 2])
            self.microwave_config["subentities"].append(cfg)

    def get_condition_config(self, target_entity, init_container, **kwargs):
        self.config["task"]["conditions"] = dict(
            contain=dict(
                entities=[target_entity],
                container=init_container,
            )
        )

    def get_instruction(self, target_entity, init_container, **kwargs):
        instruction = (
            f"When the microwave chimes, open the door, take the "
            f"{target_entity} out, and place it on the {init_container}."
        )
        self.config["task"]["instructions"] = [instruction]
        return self.config


@register.add_task("take_out_microwave_food")
class TakeOutMicrowaveFoodTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self._step_count = 0
        self._failure_reason: str | None = None
        self._trigger_step = 0
        self._react_window_steps = 0
        # Sticky flag: once the door has crossed open_threshold at least
        # once *after* the chime, the user's "react in time" obligation is
        # satisfied — even if the door later swings shut (gripper bumps it
        # during reset, physics settling, etc.) we must NOT retroactively
        # fail the episode.
        self._door_was_opened_after_chime = False

    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "microwave" in key:
                entity.detach()
                self._arena.attach(entity)

    def initialize_episode(self, physics, random_state):
        # Reset per-episode counters before the rollout begins. Caching the
        # trigger / window values here makes the task self-contained — callers
        # don't need to keep the config_manager alive.
        self._step_count = 0
        self._failure_reason = None
        self._door_was_opened_after_chime = False
        self._trigger_step = self.config_manager.trigger_step
        self._react_window_steps = self.config_manager.react_window_steps
        # `CompositeTask.reset_task_progress` / reset_intention_distance
        # are no-ops, but the base `get_task_progress` /
        # `get_intention_score_to_entity` (used by the eval script) read
        # `self.target_is_grasped` and `self.intention_distance`. Seed
        # both here so eval doesn't crash with AttributeError.
        self.target_is_grasped = {self.target_entity: False}
        self.intention_distance = {
            key: float("inf") for key in self.entities.keys()
        }
        return super().initialize_episode(physics, random_state)

    def update_task_progress(self, physics):
        # Mirror LM4ManipBaseTask.update_task_progress so the eval's
        # intention/progress metrics reflect whether the food has been
        # grasped (the only graspable target in this task).
        food = self.entities.get(self.target_entity)
        if food is not None and food.is_grasped(physics, self.robot):
            self.target_is_grasped[self.target_entity] = True

    def update_intention_distance(self, physics):
        # Track minimum distance from the EE to each entity, matching
        # LM4ManipBaseTask. The intention score reads `intention_distance`
        # for the target entity to credit the policy with "tried to reach".
        from VLABench.utils.utils import distance as _dist
        ee_pos = self.robot.get_end_effector_pos(physics)
        for key, entity in self.entities.items():
            self.intention_distance[key] = min(
                self.intention_distance[key],
                _dist(ee_pos, entity.get_xpos(physics)),
            )

    # ------------------------------------------------------------------
    # Public accessors used by the audio config builder + expert sequence.
    # ------------------------------------------------------------------
    @property
    def audio_trigger_step(self) -> int:
        """Env.step index (counting from episode start) at which the
        chime should begin playing."""
        return self._trigger_step

    @property
    def react_window_steps(self) -> int:
        return self._react_window_steps

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def failure_reason(self):
        return self._failure_reason

    def after_step(self, physics, random_state):
        super().after_step(physics, random_state)
        self._step_count += 1

        if self._failure_reason is not None:
            return

        microwave = self.entities[self.config_manager.target_container]
        door_open = microwave.is_open(physics)

        # Record the once-and-done "user reacted in time" event so we
        # don't penalise later door-close events caused by the gripper
        # passing near the open door.
        if self._step_count > self._trigger_step and door_open:
            self._door_was_opened_after_chime = True

        if self._step_count <= self._trigger_step:
            if door_open:
                self._failure_reason = "door_opened_before_chime"
        elif not self._door_was_opened_after_chime:
            steps_since_chime = self._step_count - self._trigger_step
            if steps_since_chime >= self._react_window_steps:
                self._failure_reason = "door_not_opened_in_time"

    def should_terminate_episode(self, physics):
        if self._failure_reason is not None:
            return False
        # Temporary simplified mode (VLABENCH_MICROWAVE_DOOROPEN_ONLY): the
        # episode is "done" the moment the door has been opened after the chime,
        # so generation/training/eval all stop before the food take-out phase.
        # Lets a quick end-to-end run verify audio-timing reactivity cheaply.
        import os as _os
        if _os.environ.get("VLABENCH_MICROWAVE_DOOROPEN_ONLY", "0") \
                not in ("0", "", "false", "False", "no"):
            return bool(self._door_was_opened_after_chime)
        return super().should_terminate_episode(physics)

    # ------------------------------------------------------------------
    # Expert sequence
    # ------------------------------------------------------------------
    @staticmethod
    def _wait_for_chime(env, buffer_steps: int = 2, max_wait: int = 600):
        """Hold the arm still until the chime has triggered + a small buffer.

        Polls task.step_count instead of waiting a fixed number of steps so
        the wait stays correct even when there are preceding idle frames
        (e.g. --start-idle-seconds in trajectory_generation.py).
        """
        task = env.task
        target_step = task.audio_trigger_step + max(0, int(buffer_steps))
        current_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        gripper_state = np.zeros(2) if gripper_closed else np.ones(2) * 0.04

        observations, waypoints = [env.get_observation()], []
        task_success = False
        for _ in range(max_wait):
            if task.step_count >= target_step:
                break
            action = np.concatenate([current_qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([
                env.robot.get_end_effector_pos(env.physics),
                quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                gripper_state,
            ]))
        observations.pop(-1)
        assert len(observations) == len(waypoints)
        return observations, waypoints, True, task_success

    def get_expert_skill_sequence(self, physics):
        # The microwave door hinge is at the cavity's LEFT-FRONT corner
        # (local pos -0.27, -0.22, 0) and the joint range is 0→1.57 rad,
        # so when fully open the door extends FORWARD-LEFT — i.e. the
        # gripper, which was holding the handle, ends up far in front of
        # and to the left of the workspace. Any further sidestep +X from
        # there cuts straight through the open door (x∈[−0.31, −0.27] in
        # local). The trick is to *release and reset* before trying to
        # reach the food: from the robot's default pose, `pick(food)`
        # plans a clean RRT that doesn't traverse the door arc at all.
        # ---- Door phase ordering -----------------------------------------
        # Two strategies for the chime-gated door opening, selected by the
        # VLABENCH_MICROWAVE_PREGRIP env var:
        #
        #   pregrip=0 (DEFAULT): the robot stays STILL at its home pose until
        #     the chime, then starts the whole reaction — approach + grasp the
        #     handle + open the door — entirely AFTER the sound. This is the
        #     "react to the cue from scratch" behaviour: motion onset == chime.
        #     Needs a react window long enough to cover grasp+open (~5-6 s);
        #     set VLABENCH_MICROWAVE_REACT_WINDOW_SEC >= ~10 (the shipped
        #     microwave pipeline uses 15 s).
        #
        #   pregrip=1: the legacy optimisation — pre-grip the handle BEFORE the
        #     chime and only trigger `open_door` once it fires. Lets a SHORT
        #     react window (default 5 s) still succeed because the slow grasp
        #     is already done. Grasping alone doesn't pass `open_threshold=π/3`,
        #     so the `door_opened_before_chime` check stays safe.
        import os as _os
        _pregrip = _os.environ.get("VLABENCH_MICROWAVE_PREGRIP", "0") \
            not in ("0", "", "false", "False", "no")
        _handle_pick = partial(SkillLib.pick, target_entity_name=self.target_container,
                               prior_eulers=[[np.pi, 0, -np.pi / 2]])
        _wait = partial(self._wait_for_chime, buffer_steps=2)
        _open = partial(SkillLib.open_door, target_container_name=self.target_container)
        _door_phase = [_handle_pick, _wait, _open] if _pregrip else [_wait, _handle_pick, _open]

        # Temporary simplified mode: stop the demo right after the door opens
        # (matches should_terminate_episode under VLABENCH_MICROWAVE_DOOROPEN_ONLY).
        if _os.environ.get("VLABENCH_MICROWAVE_DOOROPEN_ONLY", "0") \
                not in ("0", "", "false", "False", "no"):
            return list(_door_phase)

        skill_sequence = [
            # Door phase: wait for the chime, then (or before) grasp the handle
            # and open the door. See the VLABENCH_MICROWAVE_PREGRIP note above.
            *_door_phase,
            # Release the door and pull straight back so the gripper isn't
            # still in contact with the open door when we reset.
            partial(SkillLib.pull, gripper_state=np.ones(2) * 0.04,
                    pull_distance=0.2),
            # Return the arm to its home pose. Subsequent pick/place RRTs
            # start from a known-safe configuration well away from the
            # door's swing arc.
            partial(SkillLib.reset),
            # Approach the food using the same `[-3π/4, 0, 0]` gripper pose
            # heat_food uses to PLACE food at the microwave place_point —
            # the inverse action (grasping food sitting at that point) is
            # geometrically the same problem. Extra yaw variants act as
            # fallbacks when the primary pose's prepare-grasp pcd collides
            # with the cavity ceiling (find_keypoint_and_prepare_grasp
            # walks this list until it finds a collision-free pose).
            partial(SkillLib.pick, target_entity_name=self.target_entity,
                    prior_eulers=[
                        [-np.pi * 3 / 4, 0, 0],
                        [-np.pi * 3 / 4, 0, np.pi / 4],
                        [-np.pi * 3 / 4, 0, -np.pi / 4],
                        [-np.pi, 0, 0],
                    ]),
            # Extract straight back out of the cavity (lift first so the
            # food doesn't drag on the cavity floor) before moving on.
            partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.1),
            partial(SkillLib.pull, pull_distance=0.3),
            # Carry over to the tray. Let `place` query the tray entity's
            # own `place_point` site so the drop lands inside the tray's
            # contain bbox — that's what the success condition checks.
            partial(SkillLib.place, target_container_name=self.init_container,
                    target_quat=euler_to_quaternion(-np.pi * 3 / 4, 0, 0)),
            # Hold the arm still for ~3 s after release so the food has
            # time to fall + settle on the tray. `place` ends right after
            # the gripper finishes opening (~1 s), which is too early for
            # `should_terminate_episode` to see the food inside the tray
            # bbox — without this wait the episode is logged as success=
            # False even when the video clearly shows a successful drop.
            partial(SkillLib.wait, wait_time=30),
        ]
        return skill_sequence