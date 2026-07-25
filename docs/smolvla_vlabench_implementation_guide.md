# Implementing SmolVLA Training and Evaluation for a Clean VLABench Clone

This guide is for an AI coding agent that starts from the upstream VLABench
repository and needs to add SmolVLA support without any audio-specific code.

The goal is to make VLABench work with Hugging Face LeRobot's SmolVLA in two
ways:

1. Fine-tune SmolVLA on VLABench trajectories converted to LeRobot format.
2. Evaluate a SmolVLA checkpoint inside the VLABench simulator.

Do not add SLED, spatial audio, oracle audio, audio tokens, or any `observation.audio.*`
features. This guide describes only vision-language-action SmolVLA integration.

## Target Architecture

Use the existing VLABench evaluator interface. SmolVLA should be implemented as a
normal VLABench policy:

```text
VLABench Evaluator
  -> env.get_observation()
  -> SmolVLAPolicy wrapper builds a LeRobot-style batch
  -> LeRobot SmolVLAPolicy.select_action() or predict_action_chunk()
  -> VLABench policy returns EE target: (position, euler, gripper)
  -> VLABench evaluator converts EE target to joint action through IK
  -> env.step()
```

Training should use a LeRobot dataset whose feature names and shapes match the
SmolVLA VLABench checkpoint contract:

```text
observation.images.image         (224, 224, 3) uint8/video, front camera
observation.images.second_image  (224, 224, 3) uint8/video, side camera
observation.images.wrist_image   (224, 224, 3) uint8/video, wrist camera
observation.state                (7,) float32 [x, y, z, rx, ry, rz, gripper]
action                           (7,) float32 [x, y, z, rx, ry, rz, gripper_cmd]
task                             string instruction
```

The recommended base checkpoint is:

```text
lerobot/smolvla_vlabench
```

## Dependencies

Add LeRobot to the environment used for SmolVLA. If the repository already has a
VLABench conda/docker environment, install LeRobot there or create a separate
environment that can import both `VLABench` and `lerobot`.

Recommended dependency entry:

```text
git+https://github.com/huggingface/lerobot@6674e368249472c91382eb54bb8501c94c7f0c56#egg=lerobot
```

Typical runtime dependencies also include:

```text
torch
torchvision
transformers
accelerate
datasets
opencv-python
mediapy
tqdm
```

Use `MUJOCO_GL=egl` for headless evaluation.

## Files to Add

Add these files to a clean VLABench clone:

```text
scripts/convert_vlabench_to_lerobot_smolvla.py
scripts/train_smolvla_vlabench.py
VLABench/evaluation/model/policy/smolvla.py
sh/train_smolvla_vlabench.sh
sh/eval_smolvla_vlabench.sh
```

Also update:

```text
scripts/evaluate_policy.py
```

to instantiate `SmolVLAPolicy` when `--policy smolvla` is selected.

## Data Conversion

SmolVLA should be trained from a LeRobot dataset. VLABench trajectories are
usually stored as HDF5 episodes with camera frames and actions. Convert them into
LeRobot format with the same camera, state, action, and task fields expected by
`lerobot/smolvla_vlabench`.

### Camera Mapping

Use the same camera convention during training and evaluation.

Recommended default:

```python
DEFAULT_CAM_MAP = {
    "image": 2,         # front/forward camera
    "second_image": 0,  # side camera
    "wrist_image": 3,   # wrist camera
}
```

If the upstream VLABench XML camera order differs for a task, inspect
`observation["rgb"]` visually and adjust the indices. The important part is that
the same semantic camera views are used in both dataset conversion and live
evaluation.

### State and Action Contract

SmolVLA should receive a 7D end-effector state/action:

```text
[x, y, z, rx, ry, rz, gripper]
```

where:

- `x, y, z` are end-effector position in the robot base frame.
- `rx, ry, rz` are Euler angles.
- `gripper` is scalar open/closed state or command.

If a VLABench trajectory stores an 8D action like:

```text
[x, y, z, rx, ry, rz, left_finger, right_finger]
```

collapse the two finger dimensions:

```python
def binarize_gripper(g: float) -> float:
    return 1.0 if g > 0.02 else 0.0

def compute_state_action(action_8d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((action_8d.shape[0], 7), dtype=np.float32)
    out[:, :6] = action_8d[:, :6]
    out[:, 6] = [binarize_gripper(float(x)) for x in action_8d[:, 6]]
    state = out.copy()
    action = out.copy()
    return state, action
```

If the dataset has true recorded end-effector state, prefer that for
`observation.state`; otherwise using the executed target pose as the state is an
acceptable first implementation and matches the simple VLABench trajectory
generation style.

### Converter Skeleton

Create `scripts/convert_vlabench_to_lerobot_smolvla.py`.

```python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_CAM_MAP = {"image": 2, "second_image": 0, "wrist_image": 3}
DEFAULT_INSTRUCTION_PREFIX = "primitive: "


def resize_uint8(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def binarize_gripper(g: float) -> float:
    return 1.0 if g > 0.02 else 0.0


def compute_state_action(action_8d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action_7d = np.zeros((action_8d.shape[0], 7), dtype=np.float32)
    action_7d[:, :6] = action_8d[:, :6]
    action_7d[:, 6] = [binarize_gripper(float(x)) for x in action_8d[:, 6]]
    state_7d = action_7d.copy()
    return state_7d, action_7d


def build_features(image_hw: tuple[int, int], use_video: bool = True) -> dict:
    h, w = image_hw
    image_dtype = "video" if use_video else "image"
    return {
        "observation.images.image": {
            "dtype": image_dtype,
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.second_image": {
            "dtype": image_dtype,
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_image": {
            "dtype": image_dtype,
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
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
    }


def read_instruction(ep, fallback: str) -> str:
    try:
        raw = ep["instruction"][()]
    except KeyError:
        return fallback
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, list):
        raw = raw[0] if raw else fallback
    text = str(raw) if raw else fallback
    if not text.startswith(DEFAULT_INSTRUCTION_PREFIX):
        text = DEFAULT_INSTRUCTION_PREFIX + text
    return text


def convert(src_dir: Path, out_dir: Path, repo_id: str, fps: int, image_hw: tuple[int, int]):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; remove it before conversion.")

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_features(image_hw, use_video=True),
        root=str(out_dir),
        robot_type="franka",
        use_videos=True,
        vcodec="h264",
        image_writer_threads=4,
    )

    h5_files = sorted(src_dir.glob("data_*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No data_*.hdf5 files found in {src_dir}")

    for h5_path in tqdm(h5_files, desc="episodes"):
        with h5py.File(h5_path, "r") as f:
            grp = f["data"]
            episode = grp[list(grp.keys())[0]]

            rgb = episode["observation/rgb"][...]  # [T, C, H, W, 3]
            action_8d = episode["action"][...]     # [T, 8]
            state_7d, action_7d = compute_state_action(action_8d)
            instruction = read_instruction(episode, fallback="primitive: complete the task.")

            for t in range(rgb.shape[0]):
                frame = {
                    "observation.state": state_7d[t].astype(np.float32),
                    "action": action_7d[t].astype(np.float32),
                    "task": instruction,
                }
                for key, cam_idx in DEFAULT_CAM_MAP.items():
                    frame[f"observation.images.{key}"] = resize_uint8(
                        np.ascontiguousarray(rgb[t, cam_idx]), image_hw
                    )
                ds.add_frame(frame)
            ds.save_episode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-id", default="local/vlabench_smolvla")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    convert(
        src_dir=Path(args.src_dir),
        out_dir=Path(args.out_dir),
        repo_id=args.repo_id,
        fps=args.fps,
        image_hw=(args.image_size, args.image_size),
    )


if __name__ == "__main__":
    main()
```

Run it after generating VLABench HDF5 trajectories:

```bash
python scripts/convert_vlabench_to_lerobot_smolvla.py \
  --src-dir ~/data/vlabench/trajectory/dataset/select_toy \
  --out-dir ~/data/vlabench/lerobot/select_toy_smolvla \
  --repo-id local/select_toy_smolvla
```

## Training

Implement a small trainer around LeRobot's `SmolVLAPolicy`. Keep the first
version simple:

- Load `lerobot/smolvla_vlabench`.
- Build a `LeRobotDataset` with action chunk timestamps.
- Convert raw LeRobot samples into the policy's expected batch.
- Call `policy.forward(batch)`.
- Save `policy.state_dict()` and minimal metadata.

### Trainer Skeleton

Create `scripts/train_smolvla_vlabench.py`.

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate(samples: list[dict]) -> dict:
    out = {}
    for key in samples[0].keys():
        vals = [sample[key] for sample in samples]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out


def prep_image(img: torch.Tensor) -> torch.Tensor:
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    if img.ndim == 4 and img.shape[1] != 3:
        img = img.permute(0, 3, 1, 2).contiguous()
    elif img.ndim == 5 and img.shape[2] != 3:
        img = img.permute(0, 1, 4, 2, 3)[:, -1].contiguous()
    return img


def build_dataset(dataset_root: str, repo_id: str, chunk_size: int, fps: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION

    delta_timestamps = {ACTION: [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )


def prep_batch(batch: dict, tokenizer, device: torch.device, tokenizer_max_length: int):
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    out = {}
    for key in (
        "observation.images.image",
        "observation.images.second_image",
        "observation.images.wrist_image",
    ):
        out[key] = prep_image(batch[key]).to(device)

    state = batch["observation.state"].float().to(device)
    if state.ndim == 3:
        state = state[:, -1]
    out[OBS_STATE] = state

    out[ACTION] = batch[ACTION].float().to(device)
    if "action_is_pad" in batch:
        out["actions_id_pad"] = batch["action_is_pad"].to(device)

    tasks = batch["task"]
    tasks = [task if task.endswith("\n") else task + "\n" for task in tasks]
    enc = tokenizer(
        tasks,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_max_length,
        return_tensors="pt",
    )
    out[OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(device)
    out[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].bool().to(device)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="local/vlabench_smolvla")
    parser.add_argument("--pretrained", default="lerobot/smolvla_vlabench")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    policy = SmolVLAPolicy.from_pretrained(args.pretrained)

    # Keep these explicit so the converter, trainer, and evaluator agree.
    policy.config.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.second_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    policy.to(device)
    policy.train()

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    ds = build_dataset(
        args.dataset_root,
        args.repo_id,
        chunk_size=policy.config.chunk_size,
        fps=10,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    step = 0
    data_iter = iter(loader)
    pbar = tqdm(total=args.steps, desc="train")
    while step < args.steps:
        try:
            raw_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            raw_batch = next(data_iter)

        batch = prep_batch(
            raw_batch,
            tokenizer,
            device,
            tokenizer_max_length=policy.config.tokenizer_max_length,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, loss_dict = policy.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            policy.config.optimizer_grad_clip_norm,
        )
        optimizer.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = out_dir / f"ckpt_step{step:07d}.pt"
            torch.save(
                {
                    "step": step,
                    "pretrained": args.pretrained,
                    "model_state_dict": policy.state_dict(),
                },
                ckpt_path,
            )
            print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
```

Training command:

```bash
python scripts/train_smolvla_vlabench.py \
  --dataset-root ~/data/vlabench/lerobot/select_toy_smolvla \
  --repo-id local/select_toy_smolvla \
  --pretrained lerobot/smolvla_vlabench \
  --output-dir outputs/smolvla_select_toy \
  --steps 20000 \
  --batch-size 32
```

## Evaluation Policy

Create `VLABench/evaluation/model/policy/smolvla.py`. This wrapper should adapt
the VLABench observation dictionary into the batch keys expected by LeRobot's
SmolVLA policy, then return an EE target in VLABench's `Policy.predict()`
format.

Important behavior:

- Implement `reset()` and call the underlying SmolVLA policy reset so action
  queues do not leak across episodes.
- Render/use the same three camera views used during conversion.
- Convert the current EE state to the robot base frame.
- Tokenize `obs["instruction"]`.
- Call `policy.select_action(batch)` if available. If the LeRobot version only
  exposes chunk APIs, call the chunk method and consume the first action.

### Policy Wrapper Skeleton

```python
from __future__ import annotations

import numpy as np
import torch

from VLABench.evaluation.model.policy.base import Policy
from VLABench.utils.utils import quaternion_to_euler


GRIPPER_OPEN = np.full(2, 0.04)
GRIPPER_CLOSED = np.zeros(2)


class SmolVLA(Policy):
    def __init__(
        self,
        model_ckpt: str | None = None,
        pretrained: str = "lerobot/smolvla_vlabench",
        device: str = "cuda",
        use_bfloat16: bool = True,
        **kwargs,
    ):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_bfloat16 = use_bfloat16 and self.device.type == "cuda"

        self.policy = SmolVLAPolicy.from_pretrained(pretrained)
        self.policy.config.input_features = {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.second_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        }
        self.policy.config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }

        if model_ckpt:
            ckpt = torch.load(model_ckpt, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.policy.load_state_dict(state_dict, strict=True)

        self.policy.to(self.device)
        self.policy.eval()
        self.tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
        super().__init__(self.policy)

    @property
    def name(self):
        return "SmolVLA"

    @property
    def control_mode(self):
        return "ee"

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _to_chw_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
        return x.permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device)

    def _state_7d(self, obs: dict) -> torch.Tensor:
        ee_state = np.asarray(obs["ee_state"], dtype=np.float32)
        robot_frame = np.asarray(obs.get("robot_frame", np.zeros(3)), dtype=np.float32)

        pos_world = ee_state[:3]
        pos_base = pos_world - robot_frame
        quat = ee_state[3:7]
        euler = np.asarray(quaternion_to_euler(quat), dtype=np.float32)

        # If the upstream environment exposes an explicit gripper state, use it.
        # Otherwise default to open; the policy will still predict the next command.
        gripper = float(obs.get("gripper_open", 1.0))
        state = np.concatenate([pos_base, euler, [gripper]]).astype(np.float32)
        return torch.from_numpy(state).unsqueeze(0).to(self.device)

    def _build_batch(self, obs: dict) -> dict:
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            OBS_STATE,
        )

        rgb = obs["rgb"]
        batch = {
            "observation.images.image": self._to_chw_tensor(rgb[2]),
            "observation.images.second_image": self._to_chw_tensor(rgb[0]),
            "observation.images.wrist_image": self._to_chw_tensor(rgb[3]),
            OBS_STATE: self._state_7d(obs),
        }

        instruction = obs["instruction"]
        if not instruction.startswith("primitive:"):
            instruction = "primitive: " + instruction
        if not instruction.endswith("\n"):
            instruction += "\n"

        enc = self.tokenizer(
            [instruction],
            padding="max_length",
            truncation=True,
            max_length=self.policy.config.tokenizer_max_length,
            return_tensors="pt",
        )
        batch[OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(self.device)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].bool().to(self.device)
        return batch

    @torch.no_grad()
    def predict(self, obs, **kwargs):
        batch = self._build_batch(obs)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bfloat16):
            if hasattr(self.policy, "select_action"):
                action = self.policy.select_action(batch)
            elif hasattr(self.policy, "predict_action_chunk"):
                action = self.policy.predict_action_chunk(batch)[:, 0]
            else:
                raise RuntimeError("Unsupported SmolVLA policy API: no select_action or predict_action_chunk")

        action = action[0].float().cpu().numpy()
        pos_base = action[:3]
        euler = action[3:6]
        gripper_scalar = float(action[6])

        robot_frame = np.asarray(obs.get("robot_frame", np.zeros(3)), dtype=np.float32)
        pos_world = pos_base + robot_frame
        gripper_state = GRIPPER_OPEN if gripper_scalar > 0.5 else GRIPPER_CLOSED
        return pos_world, euler, gripper_state
```

If the output appears to be consistently offset, inspect whether the checkpoint
predicts position in world frame or robot-base frame. The wrapper above assumes
robot-base-frame positions because that is the preferred training contract in
this guide.

## Wire SmolVLA into `scripts/evaluate_policy.py`

Add an import branch:

```python
elif args.policy.lower() == "smolvla":
    from VLABench.evaluation.model.policy.smolvla import SmolVLA
    policy = SmolVLA(
        pretrained=args.model_ckpt,
        model_ckpt=args.lora_ckpt,
        device=args.device,
    )
```

Also add `--device` if the script does not already expose it:

```python
parser.add_argument("--device", default="cuda")
```

A cleaner CLI is to add separate SmolVLA names:

```python
parser.add_argument("--smolvla-pretrained", default="lerobot/smolvla_vlabench")
parser.add_argument("--smolvla-ckpt", default=None)
```

and instantiate:

```python
policy = SmolVLA(
    pretrained=args.smolvla_pretrained,
    model_ckpt=args.smolvla_ckpt,
    device=args.device,
)
```

## Evaluation Command

Create `sh/eval_smolvla_vlabench.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export VLABENCH_ROOT="${VLABENCH_ROOT:-$(pwd)/VLABench}"

PRETRAINED="${PRETRAINED:-lerobot/smolvla_vlabench}"
CKPT="${CKPT:-}"
TASKS="${TASKS:-select_toy}"
N_EPISODE="${N_EPISODE:-20}"
SAVE_DIR="${SAVE_DIR:-logs/smolvla_eval}"
DEVICE="${DEVICE:-cuda}"

ckpt_args=()
if [[ -n "${CKPT}" ]]; then
  ckpt_args+=(--smolvla-ckpt "${CKPT}")
fi

python scripts/evaluate_policy.py \
  --tasks ${TASKS} \
  --n-episode "${N_EPISODE}" \
  --policy smolvla \
  --smolvla-pretrained "${PRETRAINED}" \
  "${ckpt_args[@]}" \
  --device "${DEVICE}" \
  --save-dir "${SAVE_DIR}" \
  --metrics success_rate intention_score progress_score
```

Run base SmolVLA:

```bash
bash sh/eval_smolvla_vlabench.sh
```

Run a fine-tuned checkpoint:

```bash
CKPT=outputs/smolvla_select_toy/ckpt_step0020000.pt \
bash sh/eval_smolvla_vlabench.sh
```

## Training Shell Script

Create `sh/train_smolvla_vlabench.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-$HOME/data/vlabench/lerobot/select_toy_smolvla}"
REPO_ID="${REPO_ID:-local/select_toy_smolvla}"
PRETRAINED="${PRETRAINED:-lerobot/smolvla_vlabench}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_select_toy}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"

python scripts/train_smolvla_vlabench.py \
  --dataset-root "${DATASET_ROOT}" \
  --repo-id "${REPO_ID}" \
  --pretrained "${PRETRAINED}" \
  --output-dir "${OUTPUT_DIR}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"
```

## Implementation Checks

Before running long training, verify these items:

- `python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy"` works.
- `SmolVLAPolicy.from_pretrained("lerobot/smolvla_vlabench")` loads successfully.
- The converted LeRobot dataset can be opened with `LeRobotDataset`.
- A sampled batch contains all required keys:
  `observation.images.image`, `observation.images.second_image`,
  `observation.images.wrist_image`, `observation.state`, `action`, `task`.
- Image tensors entering the policy are `[B, 3, H, W]`, float, in `[0, 1]`.
- State and action tensors are `[B, 7]` and `[B, chunk_size, 7]` respectively.
- The same camera indices are used in converter and evaluator.
- `policy.reset()` is called at the start of every episode.
- Gripper conversion is consistent: scalar policy output becomes two VLABench
  finger commands, usually `[0.04, 0.04]` for open and `[0.0, 0.0]` for closed.

## Common Failure Modes

### `load_state_dict` shape mismatch

The checkpoint was trained with different feature dimensions or a different
SmolVLA config. Make sure all code uses:

```text
3 camera inputs
7D state
7D action
same pretrained base checkpoint
```

### Bad actions or constant offset

Check the coordinate frame. If training stored world-frame EE positions but
evaluation interprets outputs as robot-base-frame positions, every action will be
offset. Pick one convention and use it in both conversion and evaluation.

### Model ignores instruction

Make sure task strings are present in the LeRobot dataset and evaluation wrapper,
and normalize them with the same prefix:

```text
primitive: ...
```

### Evaluation works for some tasks but not others

Inspect camera order for each task. Some tasks may need a different primary
camera. Either standardize the VLABench XML camera order or add a per-task camera
map in the SmolVLA policy wrapper.

### Training is slow or unstable

Start with a smoke run:

```bash
STEPS=20 BATCH_SIZE=2 NUM_WORKERS=0 bash sh/train_smolvla_vlabench.sh
```

Then run one evaluation episode:

```bash
N_EPISODE=1 CKPT=outputs/smolvla_select_toy/ckpt_step0000020.pt \
bash sh/eval_smolvla_vlabench.sh
```

Only scale up after both commands complete.

## Minimal End-to-End Order

1. Install LeRobot dependencies.
2. Generate VLABench HDF5 trajectories.
3. Convert trajectories:

```bash
python scripts/convert_vlabench_to_lerobot_smolvla.py \
  --src-dir ~/data/vlabench/trajectory/dataset/select_toy \
  --out-dir ~/data/vlabench/lerobot/select_toy_smolvla \
  --repo-id local/select_toy_smolvla
```

4. Train:

```bash
DATASET_ROOT=~/data/vlabench/lerobot/select_toy_smolvla \
REPO_ID=local/select_toy_smolvla \
bash sh/train_smolvla_vlabench.sh
```

5. Evaluate:

```bash
TASKS=select_toy \
CKPT=outputs/smolvla_select_toy/ckpt_step0020000.pt \
bash sh/eval_smolvla_vlabench.sh
```

This gives a clean SmolVLA-only path: VLABench trajectories are converted to
LeRobot format, SmolVLA is fine-tuned with standard visual/state/language/action
features, and evaluation runs through the normal VLABench policy interface.
