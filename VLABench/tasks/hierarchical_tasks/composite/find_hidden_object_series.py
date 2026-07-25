from functools import partial
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.find_hidden_object_open_series import (
    FindHiddenObjectOpenConfigManager,
    FindHiddenObjectOpenTask,
    _pick_object_topdown,
)
from VLABench.utils.register import register
from VLABench.utils.skill_lib import SkillLib


@register.add_config_manager("find_hidden_object")
class FindHiddenObjectConfigManager(FindHiddenObjectOpenConfigManager):
    """Same layout as `find_hidden_object_open` but success requires actually
    retrieving (grasping) the hidden object, not just opening its drawer."""

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = [
            "Open the cabinet drawer that contains the object making the sound "
            "and pick the object up."
        ]

    def get_condition_config(self, target_entity, **kwargs):
        self.config["task"]["conditions"] = dict(
            is_grasped=dict(
                entities=[target_entity],
                robot="franka",
            )
        )


@register.add_task("find_hidden_object")
class FindHiddenObjectTask(FindHiddenObjectOpenTask):
    """Reuses the open-drawer setup/oracle, then adds picking the revealed object."""

    def get_expert_skill_sequence(self, physics):
        skills = super().get_expert_skill_sequence(physics)  # pick handle + pull open
        cm = self.config_manager
        # The object rides out on the open drawer. _pick_object_topdown releases
        # the handle, lifts straight up (no -y drag that would re-close the
        # drawer), then descends straight onto the revealed object.
        skills += [
            partial(_pick_object_topdown, obj_name=cm.target_entity),
            partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.15),
        ]
        return skills
