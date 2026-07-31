#!/usr/bin/env python
"""Teacher-forced action-chunk error, per chunk index.

The question this answers
-------------------------
Raising EVAL_HORIZON from 5 to 20 tripled find_hidden success (0.100 -> 0.300,
Fisher p=0.00065) without touching a single weight. Two explanations survive:

  (SNR)          the chunk prediction is fine, but the *executed prefix* is tiny
                 (delta[0]=1.3cm, delta[4]=3.3cm measured on the training set) so
                 prediction error is the same size as the commanded motion, and
                 the arm random-walks instead of advancing.
  (COLLAPSE)     the prediction itself degrades once the arm leaves the demo
                 manifold, shrinking toward zero delta.

They call for different fixes -- (SNR) -> temporal ensembling / coarser action
timestep; (COLLAPSE) -> absolute actions / state-noise augmentation -- so
guessing is expensive.

This script measures the first half: feed the policy *training* observations
through the exact training pipeline (openpi's own data loader, so every
transform matches bit-for-bit) and compare the sampled action chunk against the
ground-truth chunk, broken down by chunk index. If relative error is high at
index 0 and falls with index, (SNR) is supported: the model knows where to go,
it just cannot express it in the first few steps.

Errors are reported in metres, un-normalised via the checkpoint's own norm
stats, so the numbers are directly comparable to the delta magnitudes above.

Usage
-----
  uv --project third_party/openpi run python scripts/teacher_forced_chunk_error.py \
      --config pi05_ft_vlabench_find_hidden_v3_lora \
      --checkpoint outputs/pi05_.../11000 --batches 8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import dataclasses

    import openpi.training.config as _config
    import openpi.training.data_loader as _data

    cfg = _config.get_config(args.config)
    cfg = dataclasses.replace(cfg, batch_size=args.batch_size)

    ckpt = pathlib.Path(args.checkpoint)
    weight_path = ckpt / "model.safetensors"
    if not weight_path.exists():
        raise SystemExit(f"no model.safetensors under {ckpt}")

    # Record the state convention this run used. The eval loop has changed
    # underneath a running sweep before (roll folding landed mid-sweep, which
    # silently made one horizon point incomparable), so make it explicit.
    fold = None
    try:
        from VLABench.utils.utils import fold_roll_to_negative_branch  # noqa: F401

        fold = "fold_roll_to_negative_branch AVAILABLE"
    except Exception:  # noqa: BLE001
        fold = "fold_roll_to_negative_branch ABSENT"
    print(f"[ctx] {fold}")
    print(f"[ctx] config={args.config}  checkpoint={ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg.model.load_pytorch(cfg, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # That cast sweeps the LoRA adapters into bf16, but LoRALinear.forward casts
    # its input to float32 and then does F.linear(lx, lora_A) -- in eager mode
    # that is a dtype error. The eval path only survives it because torch.compile
    # promotes silently; training kept the adapter math in float32 by design.
    # Restore fp32 so this diagnostic matches the training numerics.
    n_restored = 0
    for pname, param in model.named_parameters():
        if ".lora_A" in pname or ".lora_B" in pname:
            param.data = param.data.to(torch.float32)
            n_restored += 1
    print(f"[ctx] restored {n_restored} LoRA tensors to float32 (LoRALinear contract)")
    model = model.to(device).eval()

    # Same loader the trainer uses -> identical repack/transform/normalisation.
    loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=False,
                                      num_batches=args.batches)

    # Un-normalise deltas back to metres using the checkpoint's own stats.
    # Un-normalise as x*std + mean. The mean cancels in the *error* (a
    # difference) but NOT in |truth|, so omitting it silently reports the
    # magnitude of the mean-centred delta instead of the physical displacement.
    asset_dirs = list((ckpt / "assets").glob("*/*/norm_stats.json"))
    scale = offset = None
    if asset_dirs:
        ns = json.loads(asset_dirs[0].read_text())["norm_stats"]
        key = "actions" if "actions" in ns else "action"
        scale = np.asarray(ns[key]["std"], dtype=np.float64)[:3]
        offset = np.asarray(ns[key]["mean"], dtype=np.float64)[:3]
        print(f"[ctx] norm stats: {asset_dirs[0].parent.name}")
        print(f"[ctx]   xyz std ={np.round(scale,4)}  xyz mean={np.round(offset,4)}")

    idxs = [0, 4, 9, 19, 24, 49]
    err = {k: [] for k in idxs}
    mag = {k: [] for k in idxs}

    with torch.no_grad():
        for bi, (obs, actions) in enumerate(loader):
            obs = _to_device(obs, device)
            actions = actions.to(device, dtype=torch.float32)
            # PyTorch pi0's signature is sample_actions(device, observation, ...)
            # -- the first argument is the device, not an RNG (the JAX path takes
            # a key there, which is what makes this easy to get wrong).
            pred = model.sample_actions(device, obs)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            p = pred.detach().float().cpu().numpy()
            a = actions.detach().float().cpu().numpy()
            H = a.shape[1]
            for k in idxs:
                if k >= H:
                    continue
                if scale is not None:
                    pred_m = p[:, k, :3] * scale + offset
                    truth = a[:, k, :3] * scale + offset
                else:
                    pred_m, truth = p[:, k, :3], a[:, k, :3]
                d = pred_m - truth
                err[k].extend(np.linalg.norm(d, axis=-1).tolist())
                mag[k].extend(np.linalg.norm(truth, axis=-1).tolist())
            print(f"  batch {bi+1}/{args.batches}", flush=True)

    print()
    print("Teacher-forced chunk error on TRAINING observations")
    print("  %-10s %12s %12s %10s" % ("chunk idx", "|error| (m)", "|truth| (m)", "rel err"))
    for k in idxs:
        if not err[k]:
            continue
        e = float(np.mean(err[k]))
        m = float(np.mean(mag[k]))
        tag = "  <- executed at horizon 5" if k <= 4 else ""
        print("  delta[%-4d] %12.4f %12.4f %10.2f%s" % (k, e, m, e / m if m else float("nan"), tag))
    print()
    print("Read: relative error >~1 at small k means the executed prefix is dominated")
    print("by prediction noise (SNR). Flat-and-low relative error would instead point")
    print("at the eval-time observation distribution, not the chunk interface.")
    return 0


def _to_device(obs, device):
    import jax

    return jax.tree.map(lambda x: x.to(device) if hasattr(x, "to") else x, obs)


if __name__ == "__main__":
    raise SystemExit(main())
