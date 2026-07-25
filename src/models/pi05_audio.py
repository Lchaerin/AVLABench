"""Audio-conditioned pi0.5/openpi adapters.

These adapters keep the surrounding AVLABench audio pipeline unchanged by
exposing a SmolVLA-like `predict_action_chunk(batch)` method and appending the
audio scene to `prompt`. Both local openpi loading and websocket serving are
supported.
"""
from __future__ import annotations

from dataclasses import dataclass
import collections
import functools
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import msgpack
import numpy as np
import torch

from src.audio.audio_prompt_builder import AudioPromptBuilder, AudioPromptConfig

AUDIO_AZ_KEY = "observation.audio.azimuth_deg"
AUDIO_EL_KEY = "observation.audio.elevation_deg"
AUDIO_CONF_KEY = "observation.audio.confidence"
AUDIO_CLASS_KEY = "observation.audio.class_id"
AUDIO_ENERGY_KEY = "observation.audio.energy"
AUDIO_UV_KEY = "observation.audio.uv"


def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class _OpenPiClient:
    def __init__(self, host: str, port: int, replan_steps: int) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = _Packer()
        self._ws, self._server_metadata = self._wait_for_server()
        self.action_plan = collections.deque(maxlen=replan_steps)
        self.replan_steps = replan_steps
        self.timestep = 0

    def _wait_for_server(self) -> Tuple[Any, Dict]:
        try:
            import websockets.sync.client
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "AudioAwarePi05Policy requires the 'websockets' package to "
                "connect to the openpi server. Install/run inside the openpi "
                "environment before evaluating pi0.5."
            ) from exc

        print(f"Waiting for openpi server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None
                )
                metadata = _unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                print("Still waiting for openpi server...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in openpi inference server:\n{response}")
        return _unpackb(response)

    def reset(self) -> None:
        self.timestep = 0
        self.action_plan = collections.deque(maxlen=self.replan_steps)


class _LocalOpenPiPolicy:
    """Lazy local openpi policy loader.

    Expected API, matching openpi's inference docs:
      config = openpi.training.config.get_config(name)
      policy = openpi.policies.policy_config.create_trained_policy(config, ckpt)
      policy.infer(obs)["actions"]
    """

    def __init__(
        self,
        policy_config_name: str,
        checkpoint_dir: str,
        openpi_root: str | None = None,
    ) -> None:
        if openpi_root:
            root = Path(openpi_root).resolve()
            for candidate in (root / "src", root):
                s = str(candidate)
                if candidate.exists() and s not in sys.path:
                    sys.path.insert(0, s)

        try:
            from openpi.policies import policy_config
            from openpi.training import config as train_config
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Local pi0.5 inference requires the openpi package. "
                "Populate third_party/openpi or set --openpi-root to an "
                "openpi checkout, then run inside its environment."
            ) from exc

        checkpoint = checkpoint_dir
        try:
            from openpi.shared import download

            checkpoint = download.maybe_download(checkpoint_dir)
        except Exception:
            # Local paths do not need download resolution; keep the caller's path.
            checkpoint = checkpoint_dir

        cfg = train_config.get_config(policy_config_name)
        self.policy = policy_config.create_trained_policy(cfg, checkpoint)

    def infer(self, obs: Dict) -> Dict:
        return self.policy.infer(obs)

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()


@dataclass
class Pi05AudioConfig:
    taxonomy_path: str
    top_k: int = 3
    tokenizer_max_length: int = 256
    chunk_size: int = 50
    n_classes: int = 38
    prompt_joiner: str = "\n"
    # "text"  - serialise audio into the prompt (AudioPromptBuilder).
    # "slots" - send raw per-step SELD arrays as observation/audio/* so the
    #           openpi BuildAudioSlots transform packs them into audio_slots and
    #           the model's continuous audio tokens consume them (parity with
    #           SmolVLA). The prompt stays instruction-only. Must match the
    #           training config's data.audio_mode.
    # "slots_uv" - SELD-VLA SlotEncoder path: additionally send energy + the
    #           projected (u,v). The server InjectAudioBlockText builds the
    #           "<audio> ... </audio>" text and BuildAudioSlotsUV packs [K,6]
    #           slots, so the prompt stays instruction-only here.
    audio_mode: str = "text"


def _chw_float_to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim != 3:
        raise ValueError(f"expected CHW image tensor, got shape={tuple(x.shape)}")
    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    x = x.clamp(0.0, 1.0).numpy()
    return (x * 255.0).round().astype(np.uint8)


class AudioAwarePi05Policy:
    """pi0.5/openpi policy with audio-as-text conditioning."""

    def __init__(
        self,
        audio_config: Pi05AudioConfig,
        host: str = "localhost",
        port: int = 8000,
        replan_steps: int = 4,
        local: bool = False,
        policy_config_name: str | None = None,
        checkpoint_dir: str | None = None,
        openpi_root: str | None = None,
    ) -> None:
        if local:
            if not policy_config_name:
                raise ValueError("--policy-config is required for local pi0.5")
            if not checkpoint_dir:
                raise ValueError("--policy-dir is required for local pi0.5")
            self.client = _LocalOpenPiPolicy(
                policy_config_name=policy_config_name,
                checkpoint_dir=checkpoint_dir,
                openpi_root=openpi_root,
            )
        else:
            self.client = _OpenPiClient(host=host, port=port, replan_steps=replan_steps)
        self.audio_config = audio_config
        self.audio_prompt_builder = AudioPromptBuilder(
            AudioPromptConfig(
                taxonomy_path=audio_config.taxonomy_path,
                top_k=audio_config.top_k,
                n_classes=audio_config.n_classes,
            )
        )
        # Minimal config surface used by the shared eval loop.
        self.config = SimpleNamespace(
            tokenizer_max_length=audio_config.tokenizer_max_length,
            chunk_size=audio_config.chunk_size,
        )

    def eval(self):
        return self

    def to(self, _device):
        return self

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        state = batch["observation.state"].detach().float().cpu()
        class_id = batch[AUDIO_CLASS_KEY]
        az = batch[AUDIO_AZ_KEY]
        el = batch[AUDIO_EL_KEY]
        conf = batch[AUDIO_CONF_KEY]
        uv_mode = self.audio_config.audio_mode == "slots_uv"
        slots_mode = self.audio_config.audio_mode in ("slots", "slots_uv")
        energy = batch.get(AUDIO_ENERGY_KEY)
        uv = batch.get(AUDIO_UV_KEY)

        def _np1d(x, b):
            return np.asarray(x[b].detach().cpu()).reshape(-1)

        if slots_mode:
            audio_prompts = [None] * state.shape[0]   # prompt stays instruction-only
        else:
            audio_prompts = self.audio_prompt_builder.build_batch(class_id, az, el, conf)

        actions = []
        for b, audio_text in enumerate(audio_prompts):
            base_prompt = batch.get("raw_instruction", [""])[b]
            if batch.get("no_instruction", False):
                base_prompt = ""
            if slots_mode:
                prompt = base_prompt
            else:
                prompt = audio_text if not base_prompt else (
                    base_prompt.rstrip() + self.audio_config.prompt_joiner + audio_text
                )

            policy_input = {
                "observation/image": _chw_float_to_hwc_uint8(
                    batch["observation.images.image"][b]
                ),
                "observation/second_image": _chw_float_to_hwc_uint8(
                    batch["observation.images.second_image"][b]
                ),
                "observation/wrist_image": _chw_float_to_hwc_uint8(
                    batch["observation.images.wrist_image"][b]
                ),
                "observation/state": state[b].numpy(),
                "prompt": prompt,
            }
            if slots_mode:
                # Same column names the openpi BuildAudioSlots transform reads at
                # training time, so train/eval produce identical audio_slots.
                policy_input["observation/audio/class_id"] = _np1d(class_id, b).astype(np.int32)
                policy_input["observation/audio/azimuth_deg"] = _np1d(az, b).astype(np.float32)
                policy_input["observation/audio/elevation_deg"] = _np1d(el, b).astype(np.float32)
                policy_input["observation/audio/confidence"] = _np1d(conf, b).astype(np.float32)
            if uv_mode:
                # SlotEncoder path also needs loudness + projected (u,v). The
                # server builds the audio text block and packs [K,6] slots.
                if energy is not None:
                    policy_input["observation/audio/energy"] = _np1d(energy, b).astype(np.float32)
                if uv is not None:
                    policy_input["observation/audio/uv"] = (
                        np.asarray(uv[b].detach().cpu()).reshape(-1, 2).astype(np.float32)
                    )
            out = self.client.infer(policy_input)
            action = np.asarray(out["actions"], dtype=np.float32)
            if action.ndim != 2:
                raise RuntimeError(f"openpi server returned invalid actions shape={action.shape}")
            actions.append(torch.from_numpy(action))

        max_t = max(a.shape[0] for a in actions)
        padded = []
        for a in actions:
            if a.shape[0] < max_t:
                pad = a[-1:].expand(max_t - a.shape[0], -1)
                a = torch.cat([a, pad], dim=0)
            padded.append(a)
        return torch.stack(padded, dim=0)

    def reset(self) -> None:
        self.client.reset()
