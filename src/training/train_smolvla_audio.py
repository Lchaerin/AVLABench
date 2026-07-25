"""Fine-tune the audio-aware SmolVLA on the converted AVLABench dataset.

Example
-------
    python src/training/train_smolvla_audio.py \
        --dataset-root  ./dataset_v5_lerobot \
        --pretrained    lerobot/smolvla_vlabench \
        --taxonomy      ./class_taxonomy.yaml \
        --output-dir    ./outputs/smolvla_audio_v0 \
        --batch-size    8 \
        --steps         5000 \
        --lr            1e-4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.smolvla_audio import (  # noqa: E402
    AudioAwareSmolVLAPolicy,
    AudioConfig,
    AUDIO_AZ_KEY, AUDIO_EL_KEY, AUDIO_CONF_KEY, AUDIO_CLASS_KEY,
)
from src.audio.oracle_sled import OracleNoiseConfig  # noqa: E402


# ----------------------------------------------------------------------------
def build_dataset(dataset_root: str, repo_id: str, chunk_size: int, fps: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION

    delta_timestamps = {ACTION: [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
        # torchcodec can't load against ffmpeg 8 (libavutil.60) in this env;
        # pyav is installed and decodes the h264 frames fine.
        video_backend="pyav",
    )


# ----------------------------------------------------------------------------
def collate(samples: list[dict]) -> dict:
    """Stack a list of LeRobot dataset samples into a batch.

    LeRobotDataset returns a dict per sample. Tensors are stacked along dim 0,
    images are turned into (B, C, H, W) float in [0, 1], and the task string
    is left as a list for the tokenizer step.
    """
    out: dict = {}
    keys = samples[0].keys()
    for key in keys:
        vals = [s[key] for s in samples]
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        elif isinstance(v0, (int, float, np.integer, np.floating)):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals
    return out


# ----------------------------------------------------------------------------
def _prep_image(img: torch.Tensor) -> torch.Tensor:
    """uint8/HWC → float [0,1] CHW (handles batch shapes (B,H,W,C) and (B,T,H,W,C))."""
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    if img.ndim == 4 and img.shape[1] != 3:
        img = img.permute(0, 3, 1, 2).contiguous()
    elif img.ndim == 5 and img.shape[2] != 3:
        img = img.permute(0, 1, 4, 2, 3)[:, -1].contiguous()
    return img


def _augment_image(img: torch.Tensor, color_jitter: float, translate_px: int) -> torch.Tensor:
    """Per-sample brightness/contrast jitter + circular pixel shift.

    Both ops are GPU-only and shape-preserving. Designed to break the policy's
    over-reliance on exact pixel statistics (the dominant ver1-era failure mode
    when the env render drifted by a few RGB units from the dataset videos).
    """
    if img.ndim != 4:
        return img
    B = img.shape[0]
    device = img.device
    if color_jitter > 0:
        bright   = 1 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * color_jitter
        contrast = 1 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * color_jitter
        img = img * bright
        img = (img - 0.5) * contrast + 0.5
        img = img.clamp(0.0, 1.0)
    if translate_px > 0:
        sx = torch.randint(-translate_px, translate_px + 1, (B,), device=device)
        sy = torch.randint(-translate_px, translate_px + 1, (B,), device=device)
        if (sx.abs().sum() + sy.abs().sum()).item() > 0:
            shifted = torch.empty_like(img)
            for b in range(B):
                shifted[b] = torch.roll(img[b], shifts=(int(sy[b]), int(sx[b])), dims=(1, 2))
            img = shifted
    return img


def _apply_oracle_noise(
    cid: torch.Tensor,        # [B, K] int
    az:  torch.Tensor,        # [B, K] float
    el:  torch.Tensor,        # [B, K] float
    cf:  torch.Tensor,        # [B, K] float  (clean GT = 1.0 present, 0.0 empty)
    cfg: OracleNoiseConfig,
    rng: np.random.Generator,
):
    """Return (cid', az', el', cf') with fresh noise applied to present slots.

    "Present" = cid >= 0 in the input. Absent slots are passed through
    unchanged so the audio token builder still renders them as 'silence'.
    Noise is drawn per-call so every batch sees a different realisation —
    this is the main regularisation signal in oracle training.
    """
    B, K = cid.shape
    shape = (B, K)
    device = cid.device

    present_np = (cid.cpu().numpy() >= 0)
    az_np = az.cpu().numpy().copy()
    el_np = el.cpu().numpy().copy()
    cf_np = cf.cpu().numpy().copy()
    cid_np = cid.cpu().numpy().copy()
    target_cid_np = cid_np[:, 0].copy() if K > 0 else np.full((B,), -1, dtype=cid_np.dtype)

    if cfg.az_std_deg > 0:
        az_np[present_np] += rng.normal(
            0.0, cfg.az_std_deg, size=int(present_np.sum())
        ).astype(az_np.dtype)
    if cfg.el_std_deg > 0:
        el_np[present_np] += rng.normal(
            0.0, cfg.el_std_deg, size=int(present_np.sum())
        ).astype(el_np.dtype)
    if cfg.conf_max > cfg.conf_min:
        conf_samples = rng.uniform(
            cfg.conf_min, cfg.conf_max, size=int(present_np.sum())
        ).astype(cf_np.dtype)
        cf_np[present_np] = conf_samples
    else:
        cf_np[present_np] = cfg.conf_max
    if cfg.class_flip_prob > 0:
        flip_mask = (rng.random(size=shape) < cfg.class_flip_prob) & present_np
        # Target-slot protection: the dataset always writes the target to
        # slot 0 BEFORE we shuffle below, so freezing that column keeps
        # instruction-↔-class matching loss-consistent (the model never
        # sees an episode where the instruction's target class word is
        # absent from every audio slot).
        if cfg.target_slot_protect and K > 0:
            flip_mask[:, 0] = False
        if flip_mask.any():
            new_cid = rng.integers(0, cfg.n_classes, size=shape).astype(cid_np.dtype)
            # Ensure we actually flip to a different class
            same = (new_cid == cid_np) & flip_mask
            if same.any():
                new_cid[same] = (new_cid[same] + 1) % cfg.n_classes
            cid_np[flip_mask] = new_cid[flip_mask]

    # Slot shuffle: per-sample random permutation of the K slots, applied
    # AFTER noise so the per-slot semantics (target-protect above) still
    # work. This breaks the positional shortcut where slot 0 was always the
    # target, forcing the policy to use class-text matching from the
    # instruction to pick the right direction at inference time.
    if cfg.shuffle_slots and K > 1:
        for b in range(B):
            perm = rng.permutation(K)
            cid_np[b] = cid_np[b, perm]
            az_np[b]  = az_np[b, perm]
            el_np[b]  = el_np[b, perm]
            cf_np[b]  = cf_np[b, perm]

    # Optional SELD canonicalization kept for ablation/backward compatibility.
    # It is disabled in the two-radio wrapper by default because it uses the
    # instruction target to reorder slots and can mask whether the policy has
    # actually learned class-name matching.
    if cfg.target_first_slots and K > 1:
        for b in range(B):
            target_cid = int(target_cid_np[b])
            if target_cid < 0:
                continue
            matches = np.flatnonzero((cid_np[b] == target_cid) & (cf_np[b] > 0))
            if matches.size == 0:
                continue
            best = int(matches[np.argmax(cf_np[b, matches])])
            if best == 0:
                continue
            for arr in (cid_np, az_np, el_np, cf_np):
                arr[b, [0, best]] = arr[b, [best, 0]]

    if cfg.canonicalize_slots == "azimuth" and K > 1:
        _canonicalize_slots_by_azimuth(cid_np, az_np, el_np, cf_np)

    return (
        torch.from_numpy(cid_np).long().to(device),
        torch.from_numpy(az_np).float().to(device),
        torch.from_numpy(el_np).float().to(device),
        torch.from_numpy(cf_np).float().to(device),
    )


def _canonicalize_slots_by_azimuth(
    cid_np: np.ndarray,
    az_np: np.ndarray,
    el_np: np.ndarray,
    cf_np: np.ndarray,
) -> None:
    """Order present slots from task-left to task-right in-place.

    In this benchmark positive stored azimuth corresponds to the left side of
    the table, negative to the right. Empty slots stay at the end. This uses
    only SELD geometry, not the instruction target class.
    """
    B, _ = cid_np.shape
    for b in range(B):
        present = np.flatnonzero((cid_np[b] >= 0) & (cf_np[b] > 0))
        if present.size < 2:
            continue
        ordered = present[np.argsort(-az_np[b, present])]
        if np.array_equal(present, ordered):
            continue
        for arr in (cid_np, az_np, el_np, cf_np):
            arr[b, present] = arr[b, ordered]


def prep_batch_for_policy(
    batch: dict,
    tokenizer,
    device: torch.device,
    image_keys: tuple[str, ...] = (
        "observation.images.image",
        "observation.images.second_image",
        "observation.images.wrist_image",
    ),
    state_key: str = "observation.state",
    tokenizer_max_length: int = 48,
    oracle_noise_cfg: OracleNoiseConfig | None = None,
    oracle_rng: np.random.Generator | None = None,
    training: bool = False,
    image_color_jitter: float = 0.0,
    image_translate_px: int = 0,
    state_noise_std: float = 0.0,
    audio_conf_dropout: float = 0.0,
):
    """Convert a raw collated LeRobot batch into the dict that the policy expects.

    If `oracle_noise_cfg` is set, the audio fields are treated as clean GT and
    noise is re-sampled each call (fresh augmentation per batch).

    Augmentation knobs (active only when `training=True`):
      * image_color_jitter — symmetric ± brightness/contrast jitter range.
      * image_translate_px — random circular pixel shift in (H, W).
      * state_noise_std    — Gaussian noise on the first 6 dims of state
                             (gripper dim left alone).
      * audio_conf_dropout — probability of fully silencing audio (conf→0)
                             per sample, exposing the policy to visual-only
                             fallback at inference time.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE, ACTION

    out: dict = {}

    for k in image_keys:
        if k not in batch:
            continue
        img = _prep_image(batch[k]).to(device)
        if training and (image_color_jitter > 0 or image_translate_px > 0):
            img = _augment_image(img, image_color_jitter, image_translate_px)
        out[k] = img

    state = batch[state_key].to(device).float()
    if state.ndim == 3:
        state = state[:, -1]
    if training and state_noise_std > 0 and state.shape[-1] >= 6:
        noise = torch.randn_like(state[..., :6]) * state_noise_std
        state = torch.cat([state[..., :6] + noise, state[..., 6:]], dim=-1)
    out[OBS_STATE] = state

    action = batch[ACTION].to(device).float()
    out[ACTION] = action
    if "action_is_pad" in batch:
        out["actions_id_pad"] = batch["action_is_pad"].to(device)

    # Audio features (all (B, K) or (B, T, K))
    audio_bk = {}
    for k in [AUDIO_AZ_KEY, AUDIO_EL_KEY, AUDIO_CONF_KEY, AUDIO_CLASS_KEY]:
        v = batch[k]
        if v.ndim == 3:
            v = v[:, -1]
        audio_bk[k] = v.to(device)

    if oracle_noise_cfg is not None:
        assert oracle_rng is not None, "oracle_rng must be provided when oracle_noise_cfg is set"
        cid2, az2, el2, cf2 = _apply_oracle_noise(
            audio_bk[AUDIO_CLASS_KEY],
            audio_bk[AUDIO_AZ_KEY],
            audio_bk[AUDIO_EL_KEY],
            audio_bk[AUDIO_CONF_KEY],
            oracle_noise_cfg,
            oracle_rng,
        )
        out[AUDIO_CLASS_KEY] = cid2
        out[AUDIO_AZ_KEY]    = az2
        out[AUDIO_EL_KEY]    = el2
        out[AUDIO_CONF_KEY]  = cf2
    else:
        out.update(audio_bk)

    if training and audio_conf_dropout > 0:
        B = out[AUDIO_CONF_KEY].shape[0]
        mask = (torch.rand(B, device=device) < audio_conf_dropout).float().unsqueeze(-1)
        out[AUDIO_CONF_KEY] = out[AUDIO_CONF_KEY] * (1.0 - mask)

    # Language tokens (the tokenizer expects a list of strings)
    tasks = batch["task"]
    if not isinstance(tasks, list):
        tasks = list(tasks)
    tasks = [t if t.endswith("\n") else t + "\n" for t in tasks]
    enc = tokenizer(
        tasks,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_max_length,
        return_tensors="pt",
    )
    out[OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(device)
    out[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].to(device).bool()
    return out


# ----------------------------------------------------------------------------
_LM_LAYER_RE = __import__("re").compile(r"(?:^|\.)layers\.(\d+)\.")
_LORA_LAYER_RE = __import__("re").compile(r"\.layers\.\d+\.")


def apply_vlm_lora(
    policy,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
) -> list[str]:
    """Inject LoRA adapters into VLM text-model layers (skips action expert & vision encoder).

    Each target Linear gets lora_A / lora_B parameters registered directly on
    the module; a forward hook adds the low-rank delta.  Returns the list of
    new parameter names so the optimizer can assign them to the lm group.

    Checkpoint saving should call _save_merged_checkpoint() which merges the
    LoRA delta into the base weights before writing so the saved file is
    compatible with the standard (non-LoRA) eval loader.
    """
    import torch.nn as nn

    lora_names: list[str] = []
    for name, module in policy.named_modules():
        if "lm_expert" in name or "vision" in name:
            continue
        if not _LORA_LAYER_RE.search(name):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in target_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue

        dev = module.weight.device
        fin, fout = module.in_features, module.out_features
        module.lora_A = torch.nn.Parameter(
            torch.randn(r, fin, device=dev, dtype=torch.float32) * 0.02
        )
        module.lora_B = torch.nn.Parameter(
            torch.zeros(fout, r, device=dev, dtype=torch.float32)
        )
        module.lora_scale = alpha / r
        module.lora_drop = torch.nn.Dropout(p=dropout) if dropout > 0 else None

        def _make_hook(m):
            def _h(_, inp, out):
                x = inp[0].float()
                delta = x @ m.lora_A.T @ m.lora_B.T * m.lora_scale
                if m.lora_drop is not None:
                    delta = m.lora_drop(delta)
                return out + delta.to(out.dtype)
            return _h

        module.register_forward_hook(_make_hook(module))
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)

        lora_names += [f"{name}.lora_A", f"{name}.lora_B"]

    n = len(lora_names) // 2
    targets = list(target_modules)
    print(f"[lora] r={r} alpha={alpha} dropout={dropout}  "
          f"targets={targets}  injected into {n} VLM linear modules")
    return lora_names


def _save_merged_checkpoint(policy, lora_active: bool, path, meta: dict):
    """Save a checkpoint.  When LoRA is active, merge adapters → save → unmerge
    so the file is loadable by the standard (non-LoRA) eval script."""
    if lora_active:
        for module in policy.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                delta = (module.lora_B @ module.lora_A).to(module.weight.dtype) * module.lora_scale
                module.weight.data += delta
        sd = {k: v for k, v in policy.state_dict().items()
              if not (k.endswith(".lora_A") or k.endswith(".lora_B"))}
        torch.save({**meta, "model_state_dict": sd}, path)
        for module in policy.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                delta = (module.lora_B @ module.lora_A).to(module.weight.dtype) * module.lora_scale
                module.weight.data -= delta
    else:
        torch.save({**meta, "model_state_dict": policy.state_dict()}, path)


def _vlm_layer_params(policy) -> dict[int, list[str]]:
    """Return {layer_idx: [param_names]} for the VLM's LLM stack only
    (the action-expert's own transformer layers, nested under `lm_expert`, are
    excluded). Used to selectively unfreeze the *top* of the frozen LLM."""
    groups: dict[int, list[str]] = {}
    for name, _ in policy.named_parameters():
        if "lm_expert" in name:
            continue
        m = _LM_LAYER_RE.search(name)
        if m:
            groups.setdefault(int(m.group(1)), []).append(name)
    return groups


def configure_trainable(
    policy: AudioAwareSmolVLAPolicy,
    train_audio_only: bool,
    unfreeze_last_n_lm_layers: int = 0,
) -> list[str]:
    """Freeze parameters according to the chosen recipe; return the list of
    parameter-name prefixes that end up trainable (used for optimizer groups).

    Recipes
    -------
    * default                  : action expert + state_proj + action projections
                                 + direction_encoder.
                                 Frozen = SigLIP + LLM.
    * --train-audio-only       : only direction_encoder (alignment stage).
    * --unfreeze-last-n-lm ... : additionally mark the top-N layers of the
                                 VLM's LLM stack as trainable. Use a small N
                                 (1-2) with a small LR to adapt the frozen LM
                                 to audio tokens without full-finetuning.
    """
    for _, p in policy.named_parameters():
        p.requires_grad = False

    trainable_subnames = [
        "lm_expert",                # action expert
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "direction_encoder",
        "class_id_embedding",
    ]
    _audio_subnames = {"direction_encoder", "class_id_embedding"}
    if not train_audio_only:
        for name, p in policy.named_parameters():
            if any(s in name for s in trainable_subnames):
                p.requires_grad = True
    else:
        for name, p in policy.named_parameters():
            if any(s in name for s in _audio_subnames):
                p.requires_grad = True

    unfrozen_lm_names: list[str] = []
    if unfreeze_last_n_lm_layers > 0 and not train_audio_only:
        groups = _vlm_layer_params(policy)
        if groups:
            max_idx = max(groups.keys())
            target = range(max_idx - unfreeze_last_n_lm_layers + 1, max_idx + 1)
            all_params = dict(policy.named_parameters())
            for idx in target:
                for name in groups.get(idx, []):
                    all_params[name].requires_grad = True
                    unfrozen_lm_names.append(name)
            print(f"[unfreeze] top {unfreeze_last_n_lm_layers} LM layer(s): "
                  f"indices {list(target)}, {len(unfrozen_lm_names)} tensors")

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"[trainable] {n_train/1e6:.2f}M / {n_total/1e6:.2f}M params")
    return unfrozen_lm_names


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", default="local/avla_select_radio")
    ap.add_argument("--pretrained", default="lerobot/smolvla_vlabench")
    ap.add_argument("--taxonomy", default=str(REPO_ROOT / "class_taxonomy.yaml"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--audio-lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-10)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--audio-max-len", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-audio-only", action="store_true",
                    help="Freeze action expert, only train audio modules (alignment phase)")
    # ---- Oracle / noise injection --------------------------------------
    ap.add_argument("--oracle-noise", type=str, default="auto",
                    choices=["auto", "on", "off"],
                    help="auto (default): enable if dataset-root/oracle_mode.json exists. "
                         "on/off: force.  Treats observation.audio.* as clean GT and "
                         "samples fresh noise per batch.")
    ap.add_argument("--noise-az-std", type=float, default=3.0)
    ap.add_argument("--noise-el-std", type=float, default=5.0)
    ap.add_argument("--noise-conf-min", type=float, default=0.85)
    ap.add_argument("--noise-conf-max", type=float, default=0.98)
    ap.add_argument("--noise-class-flip-prob", type=float, default=0.02)
    ap.add_argument("--noise-n-classes", type=int, default=38)
    ap.add_argument("--shuffle-slots", type=str, default="off",
                    choices=["on", "off"],
                    help="Permute the K audio slot positions per sample. "
                         "Strongly recommended for multi-source tasks "
                         "(select_radio_two): the dataset writes the target "
                         "to slot 0 by convention, and without shuffling the "
                         "policy learns to shortcut on slot index instead of "
                         "matching instruction class names against audio "
                         "slot class names.")
    ap.add_argument("--target-slot-protect", type=str, default="off",
                    choices=["on", "off"],
                    help="When applying class flips, never flip slot 0 "
                         "(the target, by dataset convention, before the "
                         "shuffle above). Keeps the matching supervision "
                         "consistent — every training sample has the "
                         "instruction's target class word present in some "
                         "audio slot.")
    ap.add_argument("--target-first-slots", type=str, default="off",
                    choices=["on", "off"],
                    help="After raw multi-source slot shuffling/noise, move "
                         "the instruction-target class back to slot 0 before "
                         "feeding the policy. This is SELD-level canonical "
                         "ordering; SmolVLA still predicts the trajectory.")
    ap.add_argument("--canonicalize-slots", type=str, default="none",
                    choices=["none", "azimuth"],
                    help="Target-agnostic SELD slot ordering. 'azimuth' sorts "
                         "present detections from task-left to task-right "
                         "using only their reported direction.")
    # ---- LLM partial unfreeze ------------------------------------------
    ap.add_argument("--unfreeze-last-n-lm-layers", type=int, default=2,
                    help="Unfreeze the top N layers of the frozen VLM's LLM. "
                         "0 = keep LLM fully frozen (original recipe). "
                         "Ignored when --vlm-lora is set.")
    ap.add_argument("--lm-lr", type=float, default=None,
                    help="LR for unfrozen LM layers or LoRA adapters. "
                         "Default = lr (LoRA mode) or lr/5 (unfreeze mode).")
    # ---- VLM LoRA ----------------------------------------------------------
    ap.add_argument("--vlm-lora", action="store_true",
                    help="Fine-tune the full VLM text-model via LoRA instead of "
                         "unfreezing top-N layers. Adapters are merged into base "
                         "weights on save so checkpoints are eval-compatible.")
    ap.add_argument("--lora-r", type=int, default=16,
                    help="LoRA rank.")
    ap.add_argument("--lora-alpha", type=float, default=32.0,
                    help="LoRA alpha (scaling = alpha / r).")
    ap.add_argument("--lora-dropout", type=float, default=0.05,
                    help="Dropout applied inside LoRA adapters.")
    ap.add_argument("--lora-target-modules", type=str, default="q_proj,v_proj",
                    help="Comma-separated list of Linear module suffixes to wrap "
                         "with LoRA. E.g. 'q_proj,v_proj' or "
                         "'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'.")
    # ---- Train-time augmentation / regularisation ----------------------
    ap.add_argument("--image-color-jitter", type=float, default=0.0,
                    help="Per-sample ±brightness/contrast jitter range "
                         "(0 = off). Mitigates ver1-era pixel-level "
                         "overfitting to specific dataset render statistics.")
    ap.add_argument("--image-translate-px", type=int, default=0,
                    help="Random circular pixel shift in (H, W). 0 = off.")
    ap.add_argument("--state-noise-std", type=float, default=0.0,
                    help="Gaussian noise std on the first 6 state dims "
                         "(pos + euler); gripper dim left alone. 0 = off.")
    ap.add_argument("--audio-conf-dropout", type=float, default=0.0,
                    help="Probability of fully silencing audio (conf → 0) per "
                         "sample. Forces a visual fallback path. 0 = off.")
    ap.add_argument("--direction-dropout", type=float, default=0.0,
                    help="Dropout probability applied to direction_encoder "
                         "output during training. 0 = off.")
    ap.add_argument("--direction-encoder-type", type=str, default="mlp",
                    choices=["mlp", "fixed_fourier"],
                    help="How to map azimuth/elevation into the VLM hidden "
                         "dimension. 'mlp' is trainable; 'fixed_fourier' is a "
                         "deterministic sinusoidal encoding with no trainable "
                         "direction_encoder parameters.")
    ap.add_argument("--audio-fusion-mode", type=str, default="inline",
                    choices=["inline", "class_tokens", "natural_language"],
                    help="inline: replace @ token embeddings with direction "
                         "features. class_tokens: keep audio text intact and "
                         "append one learned semantic direction token per slot. "
                         "natural_language: describe each detected source in "
                         "plain spatial language and append the same learned "
                         "continuous direction tokens.")
    ap.add_argument("--class-token-scale", type=float, default=0.1,
                    help="Initial scale for VLM-text-seeded class direction "
                         "tokens when --audio-fusion-mode=class_tokens.")
    # ---- Resume from checkpoint ----------------------------------------
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Path to a previous ckpt_step*.pt file. Loads the "
                         "model state_dict (strict=True) and starts the step "
                         "counter at the saved step. Optimizer/scheduler "
                         "momentum is reset (only model weights are saved in "
                         "the checkpoint format).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)

    # ---- 1. Audio config & policy ---------------------------------------
    audio_cfg = AudioConfig(
        taxonomy_path=args.taxonomy,
        top_k=args.top_k,
        audio_max_len=args.audio_max_len,
        direction_dropout=args.direction_dropout,
        direction_encoder_type=args.direction_encoder_type,
        audio_fusion_mode=args.audio_fusion_mode,
        class_token_scale=args.class_token_scale,
    )
    print(f"[load] {args.pretrained}")
    policy = AudioAwareSmolVLAPolicy.from_pretrained_with_audio(
        args.pretrained, audio_config=audio_cfg
    )
    # Our converter now matches lerobot/vlabench_unified exactly (3 cameras,
    # 7-dim state, 7-dim action), so the pretrained config's input_features
    # already line up — no override needed. We only need to remap the
    # checkpoint's `camera1/2/3` keys to our `image/second_image/wrist_image`.
    from lerobot.configs.types import FeatureType, PolicyFeature
    policy.config.input_features = {
        "observation.images.image":        PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.second_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.wrist_image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state":               PolicyFeature(type=FeatureType.STATE,  shape=(7,)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    policy.to(device)
    # When LoRA is enabled, unfreezing top-N layers is redundant — disable it.
    n_unfreeze = 0 if args.vlm_lora else args.unfreeze_last_n_lm_layers
    unfrozen_lm_names = configure_trainable(
        policy,
        train_audio_only=args.train_audio_only,
        unfreeze_last_n_lm_layers=n_unfreeze,
    )

    lora_param_names: list[str] = []
    if args.vlm_lora and not args.train_audio_only:
        target_mods = tuple(m.strip() for m in args.lora_target_modules.split(",") if m.strip())
        lora_param_names = apply_vlm_lora(
            policy,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_mods,
        )

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    # ---- Oracle noise detection ---------------------------------------
    oracle_marker = Path(args.dataset_root) / "oracle_mode.json"
    if args.oracle_noise == "auto":
        use_oracle_noise = oracle_marker.exists()
    else:
        use_oracle_noise = (args.oracle_noise == "on")
    oracle_noise_cfg = None
    oracle_rng = None
    if use_oracle_noise:
        oracle_noise_cfg = OracleNoiseConfig(
            az_std_deg=args.noise_az_std,
            el_std_deg=args.noise_el_std,
            conf_min=args.noise_conf_min,
            conf_max=args.noise_conf_max,
            class_flip_prob=args.noise_class_flip_prob,
            n_classes=args.noise_n_classes,
            shuffle_slots=(args.shuffle_slots == "on"),
            target_slot_protect=(args.target_slot_protect == "on"),
            target_first_slots=(args.target_first_slots == "on"),
            canonicalize_slots=args.canonicalize_slots,
        )
        oracle_rng = np.random.default_rng(args.seed + 12345)
        print(f"[oracle] runtime noise ON  marker={oracle_marker.exists()}  "
              f"shuffle_slots={oracle_noise_cfg.shuffle_slots}  "
              f"target_slot_protect={oracle_noise_cfg.target_slot_protect}  "
              f"target_first_slots={oracle_noise_cfg.target_first_slots}  "
              f"canonicalize_slots={oracle_noise_cfg.canonicalize_slots}  "
              f"az_std={oracle_noise_cfg.az_std_deg}° "
              f"el_std={oracle_noise_cfg.el_std_deg}° "
              f"conf~U[{oracle_noise_cfg.conf_min},{oracle_noise_cfg.conf_max}] "
              f"flip={oracle_noise_cfg.class_flip_prob}")
    else:
        print("[oracle] runtime noise OFF (dataset treated as SLED outputs)")

    # ---- 2. Dataset & loader -------------------------------------------
    chunk_size = policy.config.chunk_size
    fps = 10
    print(f"[data] chunk_size={chunk_size} fps={fps}")
    ds = build_dataset(args.dataset_root, args.repo_id, chunk_size=chunk_size, fps=fps)
    print(f"[data] num samples = {len(ds)} ({ds.num_episodes} episodes)")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    # ---- 3. Optimizer / scheduler --------------------------------------
    unfrozen_lm_set = set(unfrozen_lm_names)
    lora_param_set = set(lora_param_names)
    _audio_keys = {"direction_encoder", "class_id_embedding"}
    audio_params = [p for n, p in policy.named_parameters()
                    if p.requires_grad and any(k in n for k in _audio_keys)]
    lm_params    = [p for n, p in policy.named_parameters()
                    if p.requires_grad and (n in unfrozen_lm_set or n in lora_param_set)]
    other_params = [p for n, p in policy.named_parameters()
                    if p.requires_grad
                    and not any(k in n for k in _audio_keys)
                    and n not in unfrozen_lm_set
                    and n not in lora_param_set]
    # LoRA adapters start from random init → full LR is appropriate.
    # Unfrozen LM layers retain pretrained weights → use lr/5 (conservative).
    if args.lm_lr is not None:
        lm_lr = args.lm_lr
    elif args.vlm_lora:
        lm_lr = args.lr
    else:
        lm_lr = args.lr / 5.0
    lm_group_name = "vlm_lora" if args.vlm_lora else "lm_top_layers"
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr,      "name": "action_expert_etc"})
    if lm_params:
        param_groups.append({"params": lm_params,    "lr": lm_lr,        "name": lm_group_name})
    if audio_params:
        param_groups.append({"params": audio_params, "lr": args.audio_lr,"name": "audio_modules"})
    print(f"[optim] groups: " + ", ".join(
        f"{g['name']}({sum(p.numel() for p in g['params'])/1e6:.2f}M@lr={g['lr']:.2e})"
        for g in param_groups
    ))
    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return max(0.025, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # GradScaler is fp16-only; bf16 has the dynamic range to skip it entirely.

    # ---- Resume: load weights + fast-forward step counter --------------
    resume_step = 0
    if args.resume_from is not None:
        if not Path(args.resume_from).is_file():
            raise FileNotFoundError(f"--resume-from points to a missing file: {args.resume_from}")
        print(f"[resume] loading {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        missing, unexpected = policy.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"[resume] first missing: {missing[:3]}")
            if unexpected:
                print(f"[resume] first unexpected: {unexpected[:3]}")
        resume_step = int(ckpt.get("step", 0))
        # Fast-forward the LR scheduler so the current step's LR matches what
        # the original run had at that point.
        for _ in range(resume_step):
            scheduler.step()
        print(f"[resume] resuming from step {resume_step}")

    # ---- 4. Training loop ----------------------------------------------
    policy.train()
    step = resume_step
    log_buf = []
    pbar = tqdm(total=args.steps, initial=resume_step, desc="train")
    t0 = time.time()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        prepped = prep_batch_for_policy(
            batch, tokenizer, device,
            tokenizer_max_length=policy.config.tokenizer_max_length,
            oracle_noise_cfg=oracle_noise_cfg,
            oracle_rng=oracle_rng,
            training=True,
            image_color_jitter=args.image_color_jitter,
            image_translate_px=args.image_translate_px,
            state_noise_std=args.state_noise_std,
            audio_conf_dropout=args.audio_conf_dropout,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, loss_dict = policy.forward(prepped)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            policy.config.optimizer_grad_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        log_buf.append(float(loss.item()))
        step += 1
        pbar.update(1)
        if step % args.log_every == 0:
            mean_loss = float(np.mean(log_buf))
            log_buf = []
            pbar.set_postfix({"loss": f"{mean_loss:.4f}",
                              "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                              "step/s": f"{(step - resume_step) / max(1.0, time.time()-t0):.2f}"})
        if step % args.save_every == 0 or step == args.steps:
            ckpt = out_dir / f"ckpt_step{step:07d}.pt"
            _save_merged_checkpoint(
                policy,
                lora_active=args.vlm_lora,
                path=ckpt,
                meta={"step": step, "audio_config": asdict(audio_cfg)},
            )
            print(f"[ckpt] saved {ckpt}")

    pbar.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
