import os
import random
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PressButtonTask
from VLABench.tasks.config_manager import PressButtonConfigManager
from VLABench.utils.register import register
from VLABench.tasks.components import Button
from VLABench.configs.constant import name2class_xml

POSITION_LABELS = ["left", "middle", "right"]
SELECT_RADIO_INSTRUCTION = (
    "primitive: Press the button in front of the radio that is making sound."
)


def _forced_target_index():
    """Optional direction override (VLABENCH_TARGET_POSITION_LABEL) so the
    generator can balance the left/middle/right distribution. Mirrors the
    helper in select_radio_two_series / select_radio_silent_series."""
    forced_label = os.environ.get("VLABENCH_TARGET_POSITION_LABEL")
    if not forced_label:
        return None
    if forced_label not in POSITION_LABELS:
        raise ValueError(
            "VLABENCH_TARGET_POSITION_LABEL must be one of: "
            f"{', '.join(POSITION_LABELS)}"
        )
    return POSITION_LABELS.index(forced_label)


@register.add_config_manager("select_radio")
class SelectRadioConfigManager(PressButtonConfigManager):
    """
    3 buttons on the table, each with a radio. Robot must press the button
    in front of the radio that is making sound. The target position is kept
    in metadata/config state only; language must not reveal left/middle/right.
    """
    def __init__(self, task_name, num_objects=[3], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def get_seen_task_config(self):
        return self.get_task_config()

    def get_unseen_task_config(self):
        return self.get_task_config()

    def get_instruction(self, position_label=None, **kwargs):
        instruction = [SELECT_RADIO_INSTRUCTION]
        self.config["task"]["instructions"] = instruction
        return self.config

    def get_task_config(self):
        self.load_buttons()
        self.target_entity, self.target_container, self.init_container = "radio", None, None

        buttons_config = [cfg for cfg in self.config["task"]["components"] if cfg["class"] == Button]
        forced_idx = _forced_target_index()
        if forced_idx is not None and forced_idx < len(buttons_config):
            target_idx = forced_idx
        else:
            target_idx = random.randint(0, len(buttons_config) - 1)
        self.active_radio_idx = target_idx
        self.active_position_label = POSITION_LABELS[target_idx]

        for i, button_config in enumerate(buttons_config):
            subentity_config = self._get_radio_config(name=f"radio_{i}")
            button_config["subentities"] = [subentity_config]
            if i == target_idx:
                self.get_condition_config(button_config["name"])

        self.get_instruction(POSITION_LABELS[target_idx])
        return self.config

    def _get_radio_config(self, name="radio"):
        cfg = dict(
            name=name,
            xml_path=name2class_xml["radio"][-1],
            position=[random.uniform(-0.03, 0.03), random.uniform(0.12, 0.16), 0.0],
        )
        cfg["class"] = name2class_xml["radio"][0]
        return cfg


@register.add_task("select_radio")
class SelectRadioTask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if key not in ["table", "button0", "button1", "button2"]:
                entity.detach()
                self._arena.attach(entity)

    def reset_camera_views(self, index=1):
        return super().reset_camera_views(index)
