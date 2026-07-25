import random
import os
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PressButtonTask
from VLABench.tasks.config_manager import PressButtonConfigManager
from VLABench.utils.register import register
from VLABench.tasks.components import Button
from VLABench.configs.constant import name2class_xml

POSITION_LABELS = ["left", "middle", "right"]


def _forced_target_index():
    forced_label = os.environ.get("VLABENCH_TARGET_POSITION_LABEL")
    if not forced_label:
        return None
    if forced_label not in POSITION_LABELS:
        raise ValueError(
            "VLABENCH_TARGET_POSITION_LABEL must be one of: "
            f"{', '.join(POSITION_LABELS)}"
        )
    return POSITION_LABELS.index(forced_label)


@register.add_config_manager("select_radio_two")
class SelectRadioTwoConfigManager(PressButtonConfigManager):
    """3 buttons + radios on the table. Two of the three radios are randomly
    selected to be active and emit sounds from two *different* classes; the
    instruction names one of the two playing sound classes and the robot must
    press the button in front of the radio that is playing it.

    The actual sound files / class assignments are picked downstream (in
    trajectory_generation.py / eval_smolvla_audio.py) so the taxonomy stays
    out of the geometry-only config manager. This class only commits to:
      * which two radio indices are active            (active_radio_indices)
      * which of those two is the instruction target  (target_radio_idx)
      * the position label of the target              (active_position_label)
    """
    def __init__(self, task_name, num_objects=[3], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def get_seen_task_config(self):
        return self.get_task_config()

    def get_unseen_task_config(self):
        return self.get_task_config()

    def get_instruction(self, sound_class_name=None, **kwargs):
        if sound_class_name is None:
            instruction = ["Please press the button in front of the radio that is playing the target sound."]
        else:
            instruction = [
                f"primitive: Press the button in front of the radio playing the {sound_class_name} sound."
            ]
        self.config["task"]["instructions"] = instruction
        return self.config

    def get_task_config(self):
        self.load_buttons()
        self.target_entity, self.target_container, self.init_container = "radio", None, None

        buttons_config = [cfg for cfg in self.config["task"]["components"] if cfg["class"] == Button]
        n = len(buttons_config)

        forced_target_idx = _forced_target_index()
        if forced_target_idx is None:
            active_indices = sorted(random.sample(range(n), 2))
            target_idx = random.choice(active_indices)
        else:
            target_idx = forced_target_idx
            candidates = [i for i in range(n) if i != target_idx]
            active_indices = sorted([target_idx, random.choice(candidates)])
        other_idx = active_indices[0] if active_indices[1] == target_idx else active_indices[1]

        self.active_radio_indices = active_indices
        self.target_radio_idx = target_idx
        self.other_radio_idx = other_idx
        # Mirror SelectRadioConfigManager so existing trajectory_generation
        # helpers (active_radio_idx / active_position_label) still work for
        # the *target* radio.
        self.active_radio_idx = target_idx
        self.active_position_label = POSITION_LABELS[target_idx]
        self.other_position_label = POSITION_LABELS[other_idx]

        for i, button_config in enumerate(buttons_config):
            subentity_config = self._get_radio_config(name=f"radio_{i}")
            button_config["subentities"] = [subentity_config]
            if i == target_idx:
                self.get_condition_config(button_config["name"])

        self.get_instruction(sound_class_name=None)
        return self.config

    def _get_radio_config(self, name="radio"):
        cfg = dict(
            name=name,
            xml_path=name2class_xml["radio"][-1],
            position=[random.uniform(-0.03, 0.03), random.uniform(0.12, 0.16), 0.0],
        )
        cfg["class"] = name2class_xml["radio"][0]
        return cfg


@register.add_task("select_radio_two")
class SelectRadioTwoTask(PressButtonTask):
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
